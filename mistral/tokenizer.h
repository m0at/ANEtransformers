// tokenizer.h — SentencePiece BPE tokenizer for Mistral 7B
// Loads vocab from GGUFFile, encodes/decodes with max-heap BPE merge
#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "gguf_loader.h"

// ─── Token types from SentencePiece ─────────────────────────────────────────
enum {
    TOK_NORMAL = 1,
    TOK_UNKNOWN = 2,
    TOK_CONTROL = 3,
    TOK_UNUSED = 4,
    TOK_BYTE = 6,
};

// ─── Tokenizer ──────────────────────────────────────────────────────────────
typedef struct {
    char **tokens;       // [vocab_size] token strings (owned)
    float *scores;       // [vocab_size]
    int *types;          // [vocab_size]
    int vocab_size;
    int bos_id, eos_id;
    uint32_t *hash_keys;
    int32_t *hash_vals;
    int hash_cap;
} Tokenizer;

// ─── FNV-1a hash ────────────────────────────────────────────────────────────
static inline uint32_t fnv1a(const char *s, int len) {
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 0x01000193u;
    }
    return h;
}

static void tok_hash_insert(Tokenizer *t, const char *key, int32_t val) {
    uint32_t h = fnv1a(key, (int)strlen(key));
    uint32_t idx = h & (uint32_t)(t->hash_cap - 1);
    for (;;) {
        if (t->hash_vals[idx] == -1) {
            t->hash_keys[idx] = h;
            t->hash_vals[idx] = val;
            return;
        }
        idx = (idx + 1) & (uint32_t)(t->hash_cap - 1);
    }
}

static int32_t tok_hash_lookup(Tokenizer *t, const char *key, int len) {
    uint32_t h = fnv1a(key, len);
    uint32_t idx = h & (uint32_t)(t->hash_cap - 1);
    for (;;) {
        if (t->hash_vals[idx] == -1) return -1;
        if (t->hash_keys[idx] == h) {
            // Verify match against stored token
            int32_t id = t->hash_vals[idx];
            if (id >= 0 && id < t->vocab_size &&
                (int)strlen(t->tokens[id]) == len &&
                memcmp(t->tokens[id], key, len) == 0)
                return id;
        }
        idx = (idx + 1) & (uint32_t)(t->hash_cap - 1);
    }
}

// ─── Max-heap for BPE merges ────────────────────────────────────────────────
typedef struct {
    float score;
    int pos;      // position of left token in sequence
    int left_id;  // token id of left piece (for invalidation check)
    int right_id; // token id of right piece
} BPEMerge;

typedef struct {
    BPEMerge *data;
    int size;
    int cap;
} BPEHeap;

static inline void heap_swap(BPEMerge *a, BPEMerge *b) {
    BPEMerge tmp = *a; *a = *b; *b = tmp;
}

static void heap_push(BPEHeap *h, BPEMerge m) {
    if (h->size >= h->cap) {
        h->cap = h->cap ? h->cap * 2 : 256;
        h->data = (BPEMerge *)realloc(h->data, h->cap * sizeof(BPEMerge));
    }
    h->data[h->size] = m;
    int i = h->size++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].score >= h->data[i].score) break;
        heap_swap(&h->data[p], &h->data[i]);
        i = p;
    }
}

static BPEMerge heap_pop(BPEHeap *h) {
    BPEMerge top = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    for (;;) {
        int best = i, l = 2*i+1, r = 2*i+2;
        if (l < h->size && h->data[l].score > h->data[best].score) best = l;
        if (r < h->size && h->data[r].score > h->data[best].score) best = r;
        if (best == i) break;
        heap_swap(&h->data[i], &h->data[best]);
        i = best;
    }
    return top;
}

// ─── Init ───────────────────────────────────────────────────────────────────
static Tokenizer *tokenizer_init(GGUFFile *gguf) {
    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    t->vocab_size = gguf->vocab_size;
    t->bos_id = gguf->bos_id;
    t->eos_id = gguf->eos_id;

    // Copy vocab data
    t->tokens = (char **)malloc(t->vocab_size * sizeof(char *));
    t->scores = (float *)malloc(t->vocab_size * sizeof(float));
    t->types = (int *)malloc(t->vocab_size * sizeof(int));

    for (int i = 0; i < t->vocab_size; i++) {
        t->tokens[i] = strdup(gguf->vocab_tokens[i]);
        t->scores[i] = gguf->vocab_scores ? gguf->vocab_scores[i] : 0.0f;
        t->types[i] = gguf->vocab_types ? gguf->vocab_types[i] : TOK_NORMAL;
    }

    // Build hash table (power-of-2 capacity, ~50% load)
    t->hash_cap = 1;
    while (t->hash_cap < t->vocab_size * 2) t->hash_cap <<= 1;
    t->hash_keys = (uint32_t *)malloc(t->hash_cap * sizeof(uint32_t));
    t->hash_vals = (int32_t *)malloc(t->hash_cap * sizeof(int32_t));
    memset(t->hash_keys, 0, t->hash_cap * sizeof(uint32_t));
    for (int i = 0; i < t->hash_cap; i++) t->hash_vals[i] = -1;

    for (int i = 0; i < t->vocab_size; i++) {
        if (t->types[i] != TOK_CONTROL && t->types[i] != TOK_UNKNOWN)
            tok_hash_insert(t, t->tokens[i], i);
    }

    return t;
}

// ─── Byte fallback: find <0xNN> token ───────────────────────────────────────
static int tok_byte_fallback(Tokenizer *t, uint8_t byte) {
    char buf[8];
    snprintf(buf, sizeof(buf), "<0x%02X>", byte);
    int32_t id = tok_hash_lookup(t, buf, (int)strlen(buf));
    return id >= 0 ? id : 0;
}

// ─── UTF-8 helpers ──────────────────────────────────────────────────────────
static int utf8_char_len(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

// ─── BPE Encode ─────────────────────────────────────────────────────────────
// Linked list node for the token sequence during merging
typedef struct _BPENode {
    int token_id;
    int prev, next; // indices into array, -1 = end
} BPENode;

static void bpe_try_merge(Tokenizer *t, BPENode *nodes, int left, int right, BPEHeap *heap) {
    if (left < 0 || right < 0) return;
    const char *s1 = t->tokens[nodes[left].token_id];
    const char *s2 = t->tokens[nodes[right].token_id];
    int l1 = (int)strlen(s1), l2 = (int)strlen(s2);
    int total = l1 + l2;
    if (total > 256) return;
    char buf[257];
    memcpy(buf, s1, l1);
    memcpy(buf + l1, s2, l2);
    buf[total] = '\0';

    int32_t merged_id = tok_hash_lookup(t, buf, total);
    if (merged_id >= 0) {
        BPEMerge m;
        m.score = t->scores[merged_id];
        m.pos = left;
        m.left_id = nodes[left].token_id;
        m.right_id = nodes[right].token_id;
        heap_push(heap, m);
    }
}

static int tokenizer_encode(Tokenizer *t, const char *text, int *out, int max_tokens, int add_bos) {
    int n = 0;
    if (add_bos && n < max_tokens) out[n++] = t->bos_id;

    if (!text || !*text) return n;

    // Prepend space for SentencePiece (U+2581 = 0xE2 0x96 0x81 represents space)
    // Build working string: " " + text, then tokenize
    int text_len = (int)strlen(text);
    int buf_len = text_len + 2;
    char *work = (char *)malloc(buf_len);
    work[0] = ' ';
    memcpy(work + 1, text, text_len + 1);
    text_len = buf_len - 1;

    // Split into individual UTF-8 characters, converting spaces to U+2581
    // Max possible chars = text_len (one byte each in worst case)
    int max_chars = text_len * 3 + 1; // space->3byte expansion
    char *char_buf = (char *)malloc(max_chars + text_len);
    int *char_starts = (int *)malloc((text_len + 1) * sizeof(int));
    int *char_lens = (int *)malloc((text_len + 1) * sizeof(int));
    int n_chars = 0;
    int write_pos = 0;

    const uint8_t *src = (const uint8_t *)work;
    int i = 0;
    while (i < text_len) {
        char_starts[n_chars] = write_pos;
        if (src[i] == ' ') {
            // Replace space with U+2581 (▁)
            char_buf[write_pos++] = (char)0xE2;
            char_buf[write_pos++] = (char)0x96;
            char_buf[write_pos++] = (char)0x81;
            char_lens[n_chars] = 3;
            i++;
        } else {
            int clen = utf8_char_len(src[i]);
            if (i + clen > text_len) clen = text_len - i;
            memcpy(char_buf + write_pos, src + i, clen);
            char_lens[n_chars] = clen;
            write_pos += clen;
            i += clen;
        }
        n_chars++;
    }

    if (n_chars == 0) {
        free(work); free(char_buf); free(char_starts); free(char_lens);
        return n;
    }

    // Initialize linked list of tokens (one per UTF-8 char)
    BPENode *nodes = (BPENode *)malloc(n_chars * sizeof(BPENode));
    for (int c = 0; c < n_chars; c++) {
        // Look up single-char token
        char tmp[8];
        int cl = char_lens[c];
        memcpy(tmp, char_buf + char_starts[c], cl);
        tmp[cl] = '\0';
        int32_t id = tok_hash_lookup(t, tmp, cl);
        if (id >= 0) {
            nodes[c].token_id = id;
        } else {
            // Byte fallback for first byte; remaining bytes handled at output
            nodes[c].token_id = tok_byte_fallback(t, (uint8_t)char_buf[char_starts[c]]);
        }
        nodes[c].prev = c - 1;
        nodes[c].next = c + 1;
    }
    nodes[n_chars - 1].next = -1;

    // Build initial heap of adjacent pair merges
    BPEHeap heap = {0};
    for (int c = 0; c < n_chars - 1; c++) {
        bpe_try_merge(t, nodes, c, nodes[c].next, &heap);
    }

    // Iteratively apply highest-scoring merge
    while (heap.size > 0) {
        BPEMerge m = heap_pop(&heap);
        int pos = m.pos;
        // Validate: node still exists and tokens match
        if (nodes[pos].token_id != m.left_id) continue;
        int right = nodes[pos].next;
        if (right < 0 || nodes[right].token_id != m.right_id) continue;

        // Merge: concat the two token strings, look up
        const char *s1 = t->tokens[m.left_id];
        const char *s2 = t->tokens[m.right_id];
        int l1 = (int)strlen(s1), l2 = (int)strlen(s2);
        char buf2[257];
        memcpy(buf2, s1, l1);
        memcpy(buf2 + l1, s2, l2);
        buf2[l1 + l2] = '\0';
        int32_t merged_id = tok_hash_lookup(t, buf2, l1 + l2);
        if (merged_id < 0) continue;

        // Apply merge
        nodes[pos].token_id = merged_id;
        nodes[pos].next = nodes[right].next;
        if (nodes[right].next >= 0)
            nodes[nodes[right].next].prev = pos;

        // Try new merges with neighbors
        if (nodes[pos].prev >= 0)
            bpe_try_merge(t, nodes, nodes[pos].prev, pos, &heap);
        if (nodes[pos].next >= 0)
            bpe_try_merge(t, nodes, pos, nodes[pos].next, &heap);
    }

    // Collect tokens from linked list
    int cur = 0;
    // Find start of list
    while (cur >= 0 && nodes[cur].prev >= 0) cur = nodes[cur].prev;
    // Actually start is always 0 since we built it linearly, but be safe
    while (cur >= 0 && n < max_tokens) {
        int tid = nodes[cur].token_id;
        // If this was a byte fallback for a multi-byte char, emit all bytes
        if (t->types[tid] == TOK_BYTE) {
            // The original char might have been multi-byte
            // We already stored the first byte's fallback token
            // For proper byte fallback, find the original char data
            // Look up which original char index this node was
            int orig_idx = cur; // node index == original char index
            if (orig_idx < n_chars) {
                int cl = char_lens[orig_idx];
                for (int b = 0; b < cl && n < max_tokens; b++) {
                    out[n++] = tok_byte_fallback(t, (uint8_t)char_buf[char_starts[orig_idx] + b]);
                }
            } else {
                out[n++] = tid;
            }
        } else {
            out[n++] = tid;
        }
        cur = nodes[cur].next;
    }

    free(nodes);
    free(heap.data);
    free(work);
    free(char_buf);
    free(char_starts);
    free(char_lens);
    return n;
}

// ─── Decode single token ────────────────────────────────────────────────────
static const char *tokenizer_decode_token(Tokenizer *t, int token_id) {
    if (token_id < 0 || token_id >= t->vocab_size) return "";
    // Skip BOS/EOS
    if (token_id == t->bos_id || token_id == t->eos_id) return "";
    // Byte token: <0xNN>
    if (t->types[token_id] == TOK_BYTE) {
        const char *s = t->tokens[token_id];
        // Parse hex value from <0xNN>
        if (s[0] == '<' && s[1] == '0' && s[2] == 'x') {
            static __thread char byte_buf[2];
            unsigned val = 0;
            for (int i = 3; s[i] && s[i] != '>'; i++) {
                val <<= 4;
                if (s[i] >= '0' && s[i] <= '9') val += s[i] - '0';
                else if (s[i] >= 'A' && s[i] <= 'F') val += s[i] - 'A' + 10;
                else if (s[i] >= 'a' && s[i] <= 'f') val += s[i] - 'a' + 10;
            }
            byte_buf[0] = (char)val;
            byte_buf[1] = '\0';
            return byte_buf;
        }
    }
    return t->tokens[token_id];
}

// ─── Decode token sequence to string ────────────────────────────────────────
// Caller must free the returned string.
static char *tokenizer_decode(Tokenizer *t, const int *tokens, int n_tokens) {
    // First pass: compute total length
    int total = 0;
    for (int i = 0; i < n_tokens; i++) {
        const char *s = tokenizer_decode_token(t, tokens[i]);
        total += (int)strlen(s);
    }

    // Allocate with extra room for U+2581 -> space conversion (shrinks, never grows)
    char *out = (char *)malloc(total + 1);
    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        const char *s = tokenizer_decode_token(t, tokens[i]);
        int slen = (int)strlen(s);
        // Copy while replacing U+2581 (0xE2 0x96 0x81) with space
        int j = 0;
        while (j < slen) {
            if (j + 2 < slen &&
                (uint8_t)s[j] == 0xE2 &&
                (uint8_t)s[j+1] == 0x96 &&
                (uint8_t)s[j+2] == 0x81) {
                out[pos++] = ' ';
                j += 3;
            } else {
                out[pos++] = s[j++];
            }
        }
    }
    out[pos] = '\0';

    // Strip leading space (SentencePiece artifact from prepended space)
    if (pos > 0 && out[0] == ' ') {
        memmove(out, out + 1, pos);
    }

    return out;
}

// ─── Chat template ──────────────────────────────────────────────────────────
// Format user message with Mistral instruct template: [INST] {msg} [/INST]
// Caller must free.
static char *tokenizer_apply_chat(const char *user_msg) {
    const char *prefix = "[INST] ";
    const char *suffix = " [/INST]";
    int plen = (int)strlen(prefix);
    int mlen = (int)strlen(user_msg);
    int slen = (int)strlen(suffix);
    char *out = (char *)malloc(plen + mlen + slen + 1);
    memcpy(out, prefix, plen);
    memcpy(out + plen, user_msg, mlen);
    memcpy(out + plen + mlen, suffix, slen);
    out[plen + mlen + slen] = '\0';
    return out;
}

// ─── Free ───────────────────────────────────────────────────────────────────
static void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    if (t->tokens) {
        for (int i = 0; i < t->vocab_size; i++) free(t->tokens[i]);
        free(t->tokens);
    }
    free(t->scores);
    free(t->types);
    free(t->hash_keys);
    free(t->hash_vals);
    free(t);
}
