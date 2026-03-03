// gguf_loader.h — GGUF v3 parser with mmap, model config + vocab extraction
// Zero-copy access to tensor data for Mistral 7B Q4_0/Q4_K_M inference
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747 // "GGUF" as little-endian uint32
#define GGUF_MAX_TENSORS 512
#define GGUF_MAX_NAME 256

// ─── GGUF value types ────────────────────────────────────────────────────────
enum {
    GGUF_VAL_UINT8   = 0,  GGUF_VAL_INT8    = 1,
    GGUF_VAL_UINT16  = 2,  GGUF_VAL_INT16   = 3,
    GGUF_VAL_UINT32  = 4,  GGUF_VAL_INT32   = 5,
    GGUF_VAL_FLOAT32 = 6,  GGUF_VAL_BOOL    = 7,
    GGUF_VAL_STRING  = 8,  GGUF_VAL_ARRAY   = 9,
    GGUF_VAL_UINT64  = 10, GGUF_VAL_INT64   = 11,
    GGUF_VAL_FLOAT64 = 12,
};

// ─── GGML quant types ────────────────────────────────────────────────────────
enum {
    GGML_TYPE_F32  = 0,  GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,  GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,  GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,  GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14, GGML_TYPE_Q8_K = 15,
};

// ─── Tensor descriptor ───────────────────────────────────────────────────────
typedef struct {
    char     name[GGUF_MAX_NAME];
    uint32_t n_dims;
    uint64_t ne[4];       // shape (ne[0]=cols, ne[1]=rows for 2D)
    uint32_t type;        // GGML_TYPE_*
    uint64_t offset;      // relative to data section start
} GGUFTensor;

// ─── Model config (extracted from GGUF metadata) ─────────────────────────────
typedef struct {
    int dim;              // embedding dimension (4096 for Mistral 7B)
    int hidden_dim;       // FFN intermediate (14336)
    int n_layers;         // 32
    int n_heads;          // 32
    int n_kv_heads;       // 8 (GQA)
    int head_dim;         // 128
    int vocab_size;       // 32000
    int max_seq_len;      // 32768
    float rope_theta;     // 10000.0
    float rms_eps;        // 1e-5
    int sliding_window;   // 4096
} MistralConfig;

// ─── Main GGUF file handle ───────────────────────────────────────────────────
typedef struct {
    int       fd;
    void     *mmap_base;
    size_t    mmap_len;
    uint8_t  *data_start;   // pointer to tensor data section

    uint32_t  version;
    uint32_t  n_tensors;
    GGUFTensor *tensors;

    MistralConfig cfg;

    // Tokenizer vocab (pointers into mmap'd metadata region)
    int     vocab_size;
    char  **vocab_tokens;   // [vocab_size] allocated strings
    float  *vocab_scores;   // [vocab_size]
    int    *vocab_types;    // [vocab_size]
    int     bos_id, eos_id, unk_id;
} GGUFFile;

// ─── Reader helpers ──────────────────────────────────────────────────────────
typedef struct {
    const uint8_t *base;
    size_t pos;
    size_t size;
} gguf_rd;

static inline uint8_t  rd8(gguf_rd *r)  { uint8_t  v; memcpy(&v, r->base+r->pos, 1); r->pos+=1; return v; }
static inline uint32_t rd32(gguf_rd *r) { uint32_t v; memcpy(&v, r->base+r->pos, 4); r->pos+=4; return v; }
static inline uint64_t rd64(gguf_rd *r) { uint64_t v; memcpy(&v, r->base+r->pos, 8); r->pos+=8; return v; }
static inline float    rdf32(gguf_rd *r){ float    v; memcpy(&v, r->base+r->pos, 4); r->pos+=4; return v; }
static inline double   rdf64(gguf_rd *r){ double   v; memcpy(&v, r->base+r->pos, 8); r->pos+=8; return v; }

static inline const char *rd_str(gguf_rd *r, uint64_t *out_len) {
    *out_len = rd64(r);
    const char *s = (const char *)(r->base + r->pos);
    r->pos += *out_len;
    return s;
}

// Read a value and return it, or skip it. Used during metadata parsing.
static uint64_t gguf_read_uint(gguf_rd *r, uint32_t type) {
    switch (type) {
        case GGUF_VAL_UINT8:   return rd8(r);
        case GGUF_VAL_UINT16:  { uint16_t v; memcpy(&v, r->base+r->pos, 2); r->pos+=2; return v; }
        case GGUF_VAL_UINT32:  return rd32(r);
        case GGUF_VAL_UINT64:  return rd64(r);
        case GGUF_VAL_INT8:    { int8_t v; memcpy(&v, r->base+r->pos, 1); r->pos+=1; return (uint64_t)v; }
        case GGUF_VAL_INT16:   { int16_t v; memcpy(&v, r->base+r->pos, 2); r->pos+=2; return (uint64_t)v; }
        case GGUF_VAL_INT32:   { int32_t v; memcpy(&v, r->base+r->pos, 4); r->pos+=4; return (uint64_t)v; }
        case GGUF_VAL_INT64:   { int64_t v; memcpy(&v, r->base+r->pos, 8); r->pos+=8; return (uint64_t)v; }
        case GGUF_VAL_BOOL:    return rd8(r);
        default: return 0;
    }
}

static float gguf_read_float(gguf_rd *r, uint32_t type) {
    if (type == GGUF_VAL_FLOAT32) return rdf32(r);
    if (type == GGUF_VAL_FLOAT64) return (float)rdf64(r);
    return (float)gguf_read_uint(r, type);
}

static void gguf_skip_value(gguf_rd *r, uint32_t type) {
    switch (type) {
        case GGUF_VAL_UINT8:  case GGUF_VAL_INT8:  case GGUF_VAL_BOOL: r->pos += 1; break;
        case GGUF_VAL_UINT16: case GGUF_VAL_INT16:  r->pos += 2; break;
        case GGUF_VAL_UINT32: case GGUF_VAL_INT32: case GGUF_VAL_FLOAT32: r->pos += 4; break;
        case GGUF_VAL_UINT64: case GGUF_VAL_INT64: case GGUF_VAL_FLOAT64: r->pos += 8; break;
        case GGUF_VAL_STRING: { uint64_t len; rd_str(r, &len); break; }
        case GGUF_VAL_ARRAY: {
            uint32_t atype = rd32(r);
            uint64_t count = rd64(r);
            for (uint64_t i = 0; i < count; i++) gguf_skip_value(r, atype);
            break;
        }
    }
}

// ─── Key matching helper ─────────────────────────────────────────────────────
static inline int key_match(const char *key, uint64_t klen, const char *target) {
    size_t tlen = strlen(target);
    return klen == tlen && memcmp(key, target, tlen) == 0;
}

// ─── Open + parse GGUF file ──────────────────────────────────────────────────
static GGUFFile *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("gguf_open"); return NULL; }

    struct stat st;
    fstat(fd, &st);
    size_t fsize = st.st_size;

    void *base = mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return NULL; }
    madvise(base, fsize, MADV_SEQUENTIAL);

    GGUFFile *f = (GGUFFile *)calloc(1, sizeof(GGUFFile));
    f->fd = fd;
    f->mmap_base = base;
    f->mmap_len = fsize;

    // Defaults
    f->cfg.rope_theta = 10000.0f;
    f->cfg.rms_eps = 1e-5f;
    f->cfg.sliding_window = 4096;
    f->bos_id = 1;
    f->eos_id = 2;
    f->unk_id = 0;

    gguf_rd r = { .base = (const uint8_t *)base, .pos = 0, .size = fsize };

    // Header
    uint32_t magic = rd32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Not a GGUF file (magic=0x%08X)\n", magic);
        munmap(base, fsize); close(fd); free(f); return NULL;
    }
    f->version = rd32(&r);
    uint64_t n_tensors = rd64(&r);
    uint64_t n_kv = rd64(&r);
    f->n_tensors = (uint32_t)n_tensors;

    // ── Parse metadata KV pairs ──────────────────────────────────────────
    // We need to extract: model config, tokenizer vocab/scores/types
    // Pointers for deferred array parsing
    size_t vocab_tokens_pos = 0, vocab_scores_pos = 0, vocab_types_pos = 0;
    uint64_t vocab_tokens_count = 0, vocab_scores_count = 0, vocab_types_count = 0;

    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen;
        const char *key = rd_str(&r, &klen);
        uint32_t vtype = rd32(&r);

        // Model architecture params
        if (key_match(key, klen, "llama.embedding_length") ||
            key_match(key, klen, "mistral.embedding_length")) {
            f->cfg.dim = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "llama.feed_forward_length") ||
                   key_match(key, klen, "mistral.feed_forward_length")) {
            f->cfg.hidden_dim = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "llama.block_count") ||
                   key_match(key, klen, "mistral.block_count")) {
            f->cfg.n_layers = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "llama.attention.head_count") ||
                   key_match(key, klen, "mistral.attention.head_count")) {
            f->cfg.n_heads = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "llama.attention.head_count_kv") ||
                   key_match(key, klen, "mistral.attention.head_count_kv")) {
            f->cfg.n_kv_heads = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "llama.context_length") ||
                   key_match(key, klen, "mistral.context_length")) {
            f->cfg.max_seq_len = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "llama.rope.freq_base") ||
                   key_match(key, klen, "mistral.rope.freq_base")) {
            f->cfg.rope_theta = gguf_read_float(&r, vtype);
        } else if (key_match(key, klen, "llama.attention.layer_norm_rms_epsilon") ||
                   key_match(key, klen, "mistral.attention.layer_norm_rms_epsilon")) {
            f->cfg.rms_eps = gguf_read_float(&r, vtype);
        } else if (key_match(key, klen, "mistral.attention.sliding_window")) {
            f->cfg.sliding_window = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "tokenizer.ggml.bos_token_id")) {
            f->bos_id = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "tokenizer.ggml.eos_token_id")) {
            f->eos_id = (int)gguf_read_uint(&r, vtype);
        } else if (key_match(key, klen, "tokenizer.ggml.unknown_token_id")) {
            f->unk_id = (int)gguf_read_uint(&r, vtype);
        }
        // Tokenizer arrays — save position for deferred parsing
        else if (key_match(key, klen, "tokenizer.ggml.tokens") && vtype == GGUF_VAL_ARRAY) {
            uint32_t atype = rd32(&r);
            vocab_tokens_count = rd64(&r);
            vocab_tokens_pos = r.pos;
            for (uint64_t j = 0; j < vocab_tokens_count; j++) gguf_skip_value(&r, atype);
        } else if (key_match(key, klen, "tokenizer.ggml.scores") && vtype == GGUF_VAL_ARRAY) {
            uint32_t atype = rd32(&r);
            vocab_scores_count = rd64(&r);
            vocab_scores_pos = r.pos;
            for (uint64_t j = 0; j < vocab_scores_count; j++) gguf_skip_value(&r, atype);
        } else if (key_match(key, klen, "tokenizer.ggml.token_type") && vtype == GGUF_VAL_ARRAY) {
            uint32_t atype = rd32(&r);
            vocab_types_count = rd64(&r);
            vocab_types_pos = r.pos;
            for (uint64_t j = 0; j < vocab_types_count; j++) gguf_skip_value(&r, atype);
        } else {
            gguf_skip_value(&r, vtype);
        }
    }

    // Derived config
    if (f->cfg.dim > 0 && f->cfg.n_heads > 0)
        f->cfg.head_dim = f->cfg.dim / f->cfg.n_heads;
    if (f->cfg.vocab_size == 0 && vocab_tokens_count > 0)
        f->cfg.vocab_size = (int)vocab_tokens_count;

    // ── Parse tensor info ────────────────────────────────────────────────
    f->tensors = (GGUFTensor *)calloc(n_tensors, sizeof(GGUFTensor));
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &f->tensors[i];
        uint64_t nlen;
        const char *name = rd_str(&r, &nlen);
        size_t cpy = nlen < GGUF_MAX_NAME - 1 ? nlen : GGUF_MAX_NAME - 1;
        memcpy(t->name, name, cpy);
        t->name[cpy] = '\0';
        t->n_dims = rd32(&r);
        for (uint32_t d = 0; d < t->n_dims; d++)
            t->ne[d] = rd64(&r);
        for (uint32_t d = t->n_dims; d < 4; d++)
            t->ne[d] = 1;
        t->type = rd32(&r);
        t->offset = rd64(&r);
    }

    // Data section starts at aligned position after all metadata + tensor info
    uint64_t alignment = 32;
    f->data_start = (uint8_t *)base + ((r.pos + alignment - 1) & ~(alignment - 1));

    // ── Parse tokenizer vocab arrays (deferred) ──────────────────────────
    if (vocab_tokens_count > 0) {
        f->vocab_size = (int)vocab_tokens_count;
        f->cfg.vocab_size = f->vocab_size;
        f->vocab_tokens = (char **)calloc(vocab_tokens_count, sizeof(char *));

        gguf_rd vr = { .base = (const uint8_t *)base, .pos = vocab_tokens_pos, .size = fsize };
        for (uint64_t i = 0; i < vocab_tokens_count; i++) {
            uint64_t slen;
            const char *s = rd_str(&vr, &slen);
            f->vocab_tokens[i] = (char *)malloc(slen + 1);
            memcpy(f->vocab_tokens[i], s, slen);
            f->vocab_tokens[i][slen] = '\0';
        }
    }
    if (vocab_scores_count > 0) {
        f->vocab_scores = (float *)malloc(vocab_scores_count * sizeof(float));
        gguf_rd vr = { .base = (const uint8_t *)base, .pos = vocab_scores_pos, .size = fsize };
        for (uint64_t i = 0; i < vocab_scores_count; i++)
            f->vocab_scores[i] = rdf32(&vr);
    }
    if (vocab_types_count > 0) {
        f->vocab_types = (int *)malloc(vocab_types_count * sizeof(int));
        gguf_rd vr = { .base = (const uint8_t *)base, .pos = vocab_types_pos, .size = fsize };
        for (uint64_t i = 0; i < vocab_types_count; i++)
            f->vocab_types[i] = (int)rd32(&vr);
    }

    return f;
}

// ─── Tensor lookup ───────────────────────────────────────────────────────────
static GGUFTensor *gguf_find(GGUFFile *f, const char *name) {
    for (uint32_t i = 0; i < f->n_tensors; i++)
        if (strcmp(f->tensors[i].name, name) == 0)
            return &f->tensors[i];
    return NULL;
}

// Get raw pointer to tensor data (into mmap'd region)
static const void *gguf_data(GGUFFile *f, GGUFTensor *t) {
    return f->data_start + t->offset;
}

// Number of elements in tensor
static uint64_t gguf_nelements(GGUFTensor *t) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n *= t->ne[d];
    return n;
}

// Bytes per element for a given type (approximate for quantized)
static size_t gguf_type_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 18;  // per block of 32
        case GGML_TYPE_Q4_K: return 144; // per block of 256
        case GGML_TYPE_Q8_0: return 34;  // per block of 32
        default: return 0;
    }
}

static uint32_t gguf_block_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K: return 256;
        default: return 1;
    }
}

// Total bytes for a tensor in the file
static size_t gguf_tensor_bytes(GGUFTensor *t) {
    uint64_t nel = gguf_nelements(t);
    uint32_t bs = gguf_block_size(t->type);
    uint64_t n_blocks = nel / bs;
    return n_blocks * gguf_type_size(t->type);
}

// ─── Print model info ────────────────────────────────────────────────────────
static void gguf_print_info(GGUFFile *f) {
    MistralConfig *c = &f->cfg;
    fprintf(stderr, "GGUF v%u: %u tensors, vocab=%d\n", f->version, f->n_tensors, f->vocab_size);
    fprintf(stderr, "  dim=%d hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d\n",
            c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->head_dim);
    fprintf(stderr, "  vocab=%d max_seq=%d rope_theta=%.1f rms_eps=%.0e window=%d\n",
            c->vocab_size, c->max_seq_len, c->rope_theta, c->rms_eps, c->sliding_window);
    fprintf(stderr, "  bos=%d eos=%d unk=%d\n", f->bos_id, f->eos_id, f->unk_id);
}

// ─── Cleanup ─────────────────────────────────────────────────────────────────
static void gguf_close(GGUFFile *f) {
    if (!f) return;
    if (f->vocab_tokens) {
        for (int i = 0; i < f->vocab_size; i++) free(f->vocab_tokens[i]);
        free(f->vocab_tokens);
    }
    free(f->vocab_scores);
    free(f->vocab_types);
    free(f->tensors);
    if (f->mmap_base && f->mmap_base != MAP_FAILED) munmap(f->mmap_base, f->mmap_len);
    if (f->fd >= 0) close(f->fd);
    free(f);
}
