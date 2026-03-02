#ifndef GPT2_TOKENIZER_H
#define GPT2_TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Config ---
#define GPT2_MAX_VOCAB   50257
#define GPT2_MAX_MERGES  50000
#define GPT2_HT_SIZE     131072  // power of 2, > 2*vocab
#define GPT2_MAX_TOK_LEN 256

typedef struct {
    int   id;
    char *bytes;    // raw byte string this token represents
    int   len;      // length in bytes
} GPT2Entry;

typedef struct {
    char *a;
    char *b;
} GPT2Merge;

// Hash table entry: token bytes -> id
typedef struct {
    char *key;
    int   key_len;
    int   val;
    int   occupied;
} GPT2HTEntry;

typedef struct GPT2Tok {
    GPT2Entry   vocab[GPT2_MAX_VOCAB];
    int         vocab_size;
    // Reverse lookup: id -> index into vocab array
    int         id_to_idx[GPT2_MAX_VOCAB];
    GPT2Merge   merges[GPT2_MAX_MERGES];
    int         n_merges;
    GPT2HTEntry ht[GPT2_HT_SIZE];
    // byte_to_entry: for each byte 0-255, index into vocab (-1 if none)
    int         byte_map[256];
    // merge lookup: hash(a,b) -> priority (index into merges)
    int         merge_ht_keys[GPT2_HT_SIZE]; // packed index, or -1
    // We store merge pairs as concatenated "a\0b\0" strings
    char       *merge_ht_a[GPT2_HT_SIZE];
    char       *merge_ht_b[GPT2_HT_SIZE];
    int         merge_ht_pri[GPT2_HT_SIZE];
    int         merge_ht_occ[GPT2_HT_SIZE];
} GPT2Tok;

// --- Base64 decode ---
static int gpt2__b64_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Decode base64 string into buf, return length. buf must be large enough.
static int gpt2__b64_decode(const char *src, int src_len, char *buf) {
    int out = 0;
    int accum = 0, bits = 0;
    for (int i = 0; i < src_len; i++) {
        int v = gpt2__b64_val(src[i]);
        if (v < 0) continue; // skip padding '='
        accum = (accum << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf[out++] = (char)((accum >> bits) & 0xFF);
        }
    }
    return out;
}

// --- Hash functions ---
static unsigned gpt2__hash_bytes(const char *s, int len) {
    unsigned h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static unsigned gpt2__hash_pair(const char *a, int alen, const char *b, int blen) {
    unsigned h = 2166136261u;
    for (int i = 0; i < alen; i++) { h ^= (unsigned char)a[i]; h *= 16777619u; }
    h ^= 0xFF; h *= 16777619u; // separator
    for (int i = 0; i < blen; i++) { h ^= (unsigned char)b[i]; h *= 16777619u; }
    return h;
}

// Insert into token->id hash table
static void gpt2__ht_insert(GPT2Tok *tok, const char *key, int key_len, int val) {
    unsigned idx = gpt2__hash_bytes(key, key_len) & (GPT2_HT_SIZE - 1);
    while (tok->ht[idx].occupied) {
        if (tok->ht[idx].key_len == key_len && memcmp(tok->ht[idx].key, key, key_len) == 0) {
            tok->ht[idx].val = val;
            return;
        }
        idx = (idx + 1) & (GPT2_HT_SIZE - 1);
    }
    tok->ht[idx].key = (char *)malloc(key_len);
    memcpy(tok->ht[idx].key, key, key_len);
    tok->ht[idx].key_len = key_len;
    tok->ht[idx].val = val;
    tok->ht[idx].occupied = 1;
}

// Lookup token bytes -> id, returns -1 if not found
static int gpt2__ht_lookup(GPT2Tok *tok, const char *key, int key_len) {
    unsigned idx = gpt2__hash_bytes(key, key_len) & (GPT2_HT_SIZE - 1);
    while (tok->ht[idx].occupied) {
        if (tok->ht[idx].key_len == key_len && memcmp(tok->ht[idx].key, key, key_len) == 0)
            return tok->ht[idx].val;
        idx = (idx + 1) & (GPT2_HT_SIZE - 1);
    }
    return -1;
}

// Insert into merge hash table
static void gpt2__merge_insert(GPT2Tok *tok, const char *a, int alen, const char *b, int blen, int pri) {
    unsigned idx = gpt2__hash_pair(a, alen, b, blen) & (GPT2_HT_SIZE - 1);
    while (tok->merge_ht_occ[idx]) {
        // check match
        if (strlen(tok->merge_ht_a[idx]) == (size_t)alen &&
            strlen(tok->merge_ht_b[idx]) == (size_t)blen &&
            memcmp(tok->merge_ht_a[idx], a, alen) == 0 &&
            memcmp(tok->merge_ht_b[idx], b, blen) == 0) {
            return; // already present
        }
        idx = (idx + 1) & (GPT2_HT_SIZE - 1);
    }
    tok->merge_ht_a[idx] = (char *)malloc(alen + 1);
    memcpy(tok->merge_ht_a[idx], a, alen);
    tok->merge_ht_a[idx][alen] = '\0';
    tok->merge_ht_b[idx] = (char *)malloc(blen + 1);
    memcpy(tok->merge_ht_b[idx], b, blen);
    tok->merge_ht_b[idx][blen] = '\0';
    tok->merge_ht_pri[idx] = pri;
    tok->merge_ht_occ[idx] = 1;
}

// Lookup merge priority, returns -1 if not found
static int gpt2__merge_lookup(GPT2Tok *tok, const char *a, int alen, const char *b, int blen) {
    unsigned idx = gpt2__hash_pair(a, alen, b, blen) & (GPT2_HT_SIZE - 1);
    while (tok->merge_ht_occ[idx]) {
        if (strlen(tok->merge_ht_a[idx]) == (size_t)alen &&
            strlen(tok->merge_ht_b[idx]) == (size_t)blen &&
            memcmp(tok->merge_ht_a[idx], a, alen) == 0 &&
            memcmp(tok->merge_ht_b[idx], b, blen) == 0)
            return tok->merge_ht_pri[idx];
        idx = (idx + 1) & (GPT2_HT_SIZE - 1);
    }
    return -1;
}

// --- GPT-2 byte encoder mapping ---
// GPT-2 maps bytes to unicode chars to avoid control chars.
// We need this to convert raw bytes into the token strings used in merges.
static void gpt2__init_byte_encoder(int byte_to_unicode[256]) {
    int n = 0;
    // Printable ranges that map to themselves
    for (int i = 33; i <= 126; i++) byte_to_unicode[i] = i, n++;
    for (int i = 161; i <= 172; i++) byte_to_unicode[i] = i, n++;
    for (int i = 174; i <= 255; i++) byte_to_unicode[i] = i, n++;
    // Everything else maps to 256+
    int extra = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode[i] == 0) {
            byte_to_unicode[i] = extra++;
        }
    }
    // Fix: the printable chars were set above, but byte 0 might have been
    // set to 0 meaning "unset". Use a sentinel approach instead.
    // Re-do properly:
    memset(byte_to_unicode, -1, 256 * sizeof(int));
    for (int i = 33; i <= 126; i++) byte_to_unicode[i] = i;
    for (int i = 161; i <= 172; i++) byte_to_unicode[i] = i;
    for (int i = 174; i <= 255; i++) byte_to_unicode[i] = i;
    extra = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode[i] == -1)
            byte_to_unicode[i] = extra++;
    }
}

// Encode a unicode codepoint as UTF-8 into buf, return bytes written
static int gpt2__utf8_encode(int cp, char *buf) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// Build a table: for each byte (0-255), what is its GPT-2 token string (UTF-8)?
// byte_tokens[i] is a null-terminated string, byte_token_lens[i] is its length
static void gpt2__build_byte_tokens(char byte_tokens[256][8], int byte_token_lens[256]) {
    int byte_to_unicode[256];
    gpt2__init_byte_encoder(byte_to_unicode);
    for (int i = 0; i < 256; i++) {
        int len = gpt2__utf8_encode(byte_to_unicode[i], byte_tokens[i]);
        byte_tokens[i][len] = '\0';
        byte_token_lens[i] = len;
    }
}

// Forward declaration
static void gpt2_tok_free(GPT2Tok *tok);

// --- Load tokenizer ---
static GPT2Tok *gpt2_tok_load(const char *dir) {
    GPT2Tok *tok = (GPT2Tok *)calloc(1, sizeof(GPT2Tok));
    if (!tok) return NULL;

    memset(tok->byte_map, -1, sizeof(tok->byte_map));
    memset(tok->id_to_idx, -1, sizeof(tok->id_to_idx));

    char path[1024];

    // Load encoder.txt
    snprintf(path, sizeof(path), "%s/encoder.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "gpt2_tok: cannot open %s\n", path); free(tok); return NULL; }

    char line[4096];
    tok->vocab_size = 0;
    while (fgets(line, sizeof(line), f)) {
        // format: ID\tBASE64\n
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        int id = atoi(line);
        char *b64 = tab + 1;
        // strip newline
        int b64_len = (int)strlen(b64);
        while (b64_len > 0 && (b64[b64_len-1] == '\n' || b64[b64_len-1] == '\r'))
            b64[--b64_len] = '\0';

        char decoded[GPT2_MAX_TOK_LEN];
        int dec_len = gpt2__b64_decode(b64, b64_len, decoded);

        int idx = tok->vocab_size++;
        tok->vocab[idx].id = id;
        tok->vocab[idx].bytes = (char *)malloc(dec_len + 1);
        memcpy(tok->vocab[idx].bytes, decoded, dec_len);
        tok->vocab[idx].bytes[dec_len] = '\0';
        tok->vocab[idx].len = dec_len;

        // Build hash: raw_bytes -> id
        gpt2__ht_insert(tok, decoded, dec_len, id);

        // Reverse lookup: id -> vocab index
        if (id >= 0 && id < GPT2_MAX_VOCAB)
            tok->id_to_idx[id] = idx;

        // Track single-byte tokens
        if (dec_len == 1) {
            tok->byte_map[(unsigned char)decoded[0]] = id;
        }
    }
    fclose(f);

    // Now we need a reverse mapping: for encoding, we need to convert raw bytes
    // into the GPT-2 unicode token strings used in merges.txt.
    // Build byte_to_token_str: for each byte, what UTF-8 string does GPT-2 use?
    // Then we also need a hash from that UTF-8 string -> raw byte(s) for token lookup.

    // Load merges.txt
    // Merges are in terms of GPT-2's unicode token strings.
    // We need to store them as-is for the BPE algorithm.
    snprintf(path, sizeof(path), "%s/merges.txt", dir);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "gpt2_tok: cannot open %s\n", path); gpt2_tok_free(tok); return NULL; }

    tok->n_merges = 0;
    while (fgets(line, sizeof(line), f)) {
        // skip comment lines
        if (line[0] == '#') continue;
        // strip newline
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        // Split on first space
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = '\0';
        char *a = line;
        char *b = sp + 1;
        int alen = (int)strlen(a);
        int blen = (int)strlen(b);

        int mi = tok->n_merges;
        tok->merges[mi].a = (char *)malloc(alen + 1);
        memcpy(tok->merges[mi].a, a, alen + 1);
        tok->merges[mi].b = (char *)malloc(blen + 1);
        memcpy(tok->merges[mi].b, b, blen + 1);

        gpt2__merge_insert(tok, a, alen, b, blen, mi);
        tok->n_merges++;
        if (tok->n_merges >= GPT2_MAX_MERGES) break;
    }
    fclose(f);

    return tok;
}

// --- Free tokenizer ---
static void gpt2_tok_free(GPT2Tok *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++)
        free(tok->vocab[i].bytes);
    for (int i = 0; i < tok->n_merges; i++) {
        free(tok->merges[i].a);
        free(tok->merges[i].b);
    }
    for (int i = 0; i < GPT2_HT_SIZE; i++) {
        if (tok->ht[i].occupied) free(tok->ht[i].key);
        if (tok->merge_ht_occ[i]) {
            free(tok->merge_ht_a[i]);
            free(tok->merge_ht_b[i]);
        }
    }
    free(tok);
}

// --- BPE token representation for encoding ---
// During BPE, we represent each token as a GPT-2 unicode string
// (the same strings used in merges.txt).

typedef struct {
    char *str;   // GPT-2 unicode string (allocated)
    int   len;   // length of str
} GPT2BPETok;

// Convert raw bytes to GPT-2 unicode token strings (one per byte)
static int gpt2__bytes_to_bpe_tokens(const char *bytes, int n,
                                      GPT2BPETok *out, int max_out) {
    static int inited = 0;
    static char byte_tokens[256][8];
    static int  byte_token_lens[256];
    if (!inited) {
        gpt2__build_byte_tokens(byte_tokens, byte_token_lens);
        inited = 1;
    }

    int count = 0;
    for (int i = 0; i < n && count < max_out; i++) {
        unsigned char b = (unsigned char)bytes[i];
        int len = byte_token_lens[b];
        out[count].str = (char *)malloc(len + 1);
        memcpy(out[count].str, byte_tokens[b], len);
        out[count].str[len] = '\0';
        out[count].len = len;
        count++;
    }
    return count;
}

// Apply BPE merges to a sequence of GPT-2 unicode tokens
static int gpt2__bpe_merge(GPT2Tok *tok, GPT2BPETok *tokens, int n) {
    while (n > 1) {
        // Find best merge
        int best_i = -1, best_pri = tok->n_merges;
        for (int i = 0; i < n - 1; i++) {
            int pri = gpt2__merge_lookup(tok,
                tokens[i].str, tokens[i].len,
                tokens[i+1].str, tokens[i+1].len);
            if (pri >= 0 && pri < best_pri) {
                best_pri = pri;
                best_i = i;
            }
        }
        if (best_i < 0) break;

        // Merge tokens[best_i] and tokens[best_i+1]
        int new_len = tokens[best_i].len + tokens[best_i+1].len;
        char *merged = (char *)malloc(new_len + 1);
        memcpy(merged, tokens[best_i].str, tokens[best_i].len);
        memcpy(merged + tokens[best_i].len, tokens[best_i+1].str, tokens[best_i+1].len);
        merged[new_len] = '\0';

        free(tokens[best_i].str);
        free(tokens[best_i+1].str);
        tokens[best_i].str = merged;
        tokens[best_i].len = new_len;

        // Shift remaining tokens down
        for (int i = best_i + 1; i < n - 1; i++)
            tokens[i] = tokens[i + 1];
        n--;
    }
    return n;
}

// Build a reverse lookup: GPT-2 unicode string -> token ID
// We need this because the BPE tokens are in unicode form, but our
// hash table maps raw bytes -> id. So we need unicode_str -> raw_bytes -> id.
// Approach: for each vocab entry, compute its GPT-2 unicode string and
// build a second hash table. But that's expensive. Instead, we can
// convert the BPE unicode string back to raw bytes and look up in our
// existing hash table.

// Convert a GPT-2 unicode string back to raw bytes
// The unicode_to_byte mapping is the inverse of byte_to_unicode
static int gpt2__unicode_to_bytes(const char *ustr, int ulen, char *out, int max_out) {
    // Build inverse mapping on first call
    static int inited = 0;
    static int unicode_to_byte[512]; // codepoints up to ~323
    if (!inited) {
        memset(unicode_to_byte, -1, sizeof(unicode_to_byte));
        int byte_to_unicode[256];
        gpt2__init_byte_encoder(byte_to_unicode);
        for (int i = 0; i < 256; i++) {
            if (byte_to_unicode[i] < 512)
                unicode_to_byte[byte_to_unicode[i]] = i;
        }
        inited = 1;
    }

    int out_len = 0;
    int i = 0;
    while (i < ulen && out_len < max_out) {
        // Decode one UTF-8 codepoint
        unsigned char c = (unsigned char)ustr[i];
        int cp = 0;
        int bytes = 1;
        if (c < 0x80) {
            cp = c; bytes = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F; bytes = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; bytes = 3;
        } else {
            cp = c & 0x07; bytes = 4;
        }
        for (int j = 1; j < bytes && (i + j) < ulen; j++)
            cp = (cp << 6) | ((unsigned char)ustr[i + j] & 0x3F);
        i += bytes;

        if (cp < 512 && unicode_to_byte[cp] >= 0)
            out[out_len++] = (char)unicode_to_byte[cp];
        else
            out[out_len++] = '?'; // fallback
    }
    return out_len;
}

// --- Word splitting (simplified GPT-2 pre-tokenization) ---
// Splits text into words. Spaces attach to the following word.
// Returns array of (start, length) pairs into the original text.
typedef struct { int start; int len; } GPT2Span;

static int gpt2__is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static int gpt2__is_digit(char c) {
    return c >= '0' && c <= '9';
}
static int gpt2__is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Simplified word splitter: groups of alphanumeric, groups of spaces,
// individual punctuation. Spaces before alphanumeric attach to the next word.
static int gpt2__split_words(const char *text, int text_len, GPT2Span *spans, int max_spans) {
    int n = 0;
    int i = 0;
    while (i < text_len && n < max_spans) {
        // Consume leading spaces (they attach to the next token)
        int space_start = i;
        while (i < text_len && text[i] == ' ') i++;
        int had_spaces = (i > space_start);

        if (i >= text_len) {
            // Trailing spaces become their own token
            if (had_spaces) {
                spans[n].start = space_start;
                spans[n].len = i - space_start;
                n++;
            }
            break;
        }

        // If we had spaces, the word starts at space_start (spaces attach to next word)
        int word_start = had_spaces ? space_start : i;

        if (gpt2__is_alpha(text[i])) {
            // Consume letters
            while (i < text_len && gpt2__is_alpha(text[i])) i++;
        } else if (gpt2__is_digit(text[i])) {
            // Consume digits (GPT-2 groups up to 3 digits but we simplify)
            while (i < text_len && gpt2__is_digit(text[i])) i++;
        } else if (text[i] == '\n') {
            i++;
        } else if (!gpt2__is_space(text[i])) {
            // Single punctuation/symbol
            i++;
        } else {
            // Other whitespace
            while (i < text_len && gpt2__is_space(text[i]) && text[i] != '\n') i++;
        }

        spans[n].start = word_start;
        spans[n].len = i - word_start;
        n++;
    }
    return n;
}

// --- Encode ---
static int *gpt2_tok_encode(GPT2Tok *tok, const char *text, int *out_n) {
    int text_len = (int)strlen(text);
    if (text_len == 0) { *out_n = 0; return NULL; }

    // Split into words
    int max_spans = text_len + 1;
    GPT2Span *spans = (GPT2Span *)malloc(max_spans * sizeof(GPT2Span));
    int n_spans = gpt2__split_words(text, text_len, spans, max_spans);

    // Allocate output (worst case: one token per byte)
    int *ids = (int *)malloc((text_len + 1) * sizeof(int));
    int n_ids = 0;

    // Temp buffer for BPE tokens per word
    int max_word_tokens = 1024;
    GPT2BPETok *bpe_tokens = (GPT2BPETok *)malloc(max_word_tokens * sizeof(GPT2BPETok));

    for (int w = 0; w < n_spans; w++) {
        const char *word = text + spans[w].start;
        int word_len = spans[w].len;

        // Convert word bytes to GPT-2 unicode tokens
        int n_tok = gpt2__bytes_to_bpe_tokens(word, word_len, bpe_tokens, max_word_tokens);

        // Apply BPE merges
        n_tok = gpt2__bpe_merge(tok, bpe_tokens, n_tok);

        // Convert each merged unicode token to raw bytes, then look up ID
        for (int t = 0; t < n_tok; t++) {
            char raw[GPT2_MAX_TOK_LEN];
            int raw_len = gpt2__unicode_to_bytes(bpe_tokens[t].str, bpe_tokens[t].len, raw, GPT2_MAX_TOK_LEN);

            int id = gpt2__ht_lookup(tok, raw, raw_len);
            if (id >= 0) {
                ids[n_ids++] = id;
            } else {
                // Fallback: encode each byte individually
                for (int b = 0; b < raw_len; b++) {
                    int byte_id = tok->byte_map[(unsigned char)raw[b]];
                    if (byte_id >= 0)
                        ids[n_ids++] = byte_id;
                }
            }
            free(bpe_tokens[t].str);
        }
    }

    free(bpe_tokens);
    free(spans);
    *out_n = n_ids;
    return ids;
}

// --- Decode ---
static char *gpt2_tok_decode(GPT2Tok *tok, const int *ids, int n) {
    // First pass: compute total length
    int total = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id >= 0 && id < GPT2_MAX_VOCAB && tok->id_to_idx[id] >= 0)
            total += tok->vocab[tok->id_to_idx[id]].len;
    }

    char *out = (char *)malloc(total + 1);
    int pos = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id >= 0 && id < GPT2_MAX_VOCAB && tok->id_to_idx[id] >= 0) {
            GPT2Entry *e = &tok->vocab[tok->id_to_idx[id]];
            memcpy(out + pos, e->bytes, e->len);
            pos += e->len;
        }
    }
    out[pos] = '\0';
    return out;
}

#endif // GPT2_TOKENIZER_H
