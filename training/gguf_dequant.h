// gguf_dequant.h — GGUF Q4_0 / Q4_K_M → fp16 dequantization via ARM NEON
// For loading quantized Mistral/LLaMA weights and baking into ANE MIL blobs
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

// ─── GGUF format constants ───────────────────────────────────────────────────

#define GGUF_MAGIC 0x46475547 // "GGUF"
#define QK4_0  32
#define QK_K  256

// ─── Block layouts (match ggml exactly) ──────────────────────────────────────

// Q4_0: 32 weights per block, 18 bytes
// Layout: fp16 scale (2B) + 16B nibbles
// Nibble layout: byte[j] low 4 bits → val[j], byte[j] high 4 bits → val[j+16]
typedef struct {
    _Float16 d;
    uint8_t  qs[QK4_0/2];
} block_q4_0;
_Static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size");

// Q4_K: 256 weights per super-block, 144 bytes
// 8 sub-blocks of 32 values, paired into 4 groups of 64 for nibble packing
typedef struct {
    _Float16 d;
    _Float16 dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K/2];
} block_q4_K;
_Static_assert(sizeof(block_q4_K) == 144, "block_q4_K size");

// ─── Q4_0 dequantization (NEON) ─────────────────────────────────────────────
//
// ggml layout: for byte qs[j] (j=0..15):
//   val[j]    = d * ((qs[j] & 0xF) - 8)
//   val[j+16] = d * ((qs[j] >> 4)  - 8)
//
// So low nibbles → first 16 values, high nibbles → last 16 values.
// We process all 16 bytes at once with NEON.

static void dequant_q4_0_block_neon(const block_q4_0 *block, _Float16 *out) {
    float16x8_t vscale = vdupq_n_f16(block->d);
    int8x16_t v8 = vdupq_n_s8(8);

    uint8x16_t raw = vld1q_u8(block->qs);

    // Low nibbles → out[0..15]
    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), v8);
    // High nibbles → out[16..31]
    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8);

    // Low nibbles: convert s8→s16→f16, multiply by scale
    {
        int16x8_t a = vmovl_s8(vget_low_s8(lo));
        int16x8_t b = vmovl_s8(vget_high_s8(lo));
        vst1q_f16((__fp16*)(out + 0), vmulq_f16(vcvtq_f16_s16(a), vscale));
        vst1q_f16((__fp16*)(out + 8), vmulq_f16(vcvtq_f16_s16(b), vscale));
    }
    // High nibbles
    {
        int16x8_t a = vmovl_s8(vget_low_s8(hi));
        int16x8_t b = vmovl_s8(vget_high_s8(hi));
        vst1q_f16((__fp16*)(out + 16), vmulq_f16(vcvtq_f16_s16(a), vscale));
        vst1q_f16((__fp16*)(out + 24), vmulq_f16(vcvtq_f16_s16(b), vscale));
    }
}

static void dequant_q4_0_row(const block_q4_0 *blocks, _Float16 *out, int n_blocks) {
    for (int i = 0; i < n_blocks; i++)
        dequant_q4_0_block_neon(&blocks[i], out + i * QK4_0);
}

static void dequant_q4_0_matrix(const void *src, _Float16 *dst, int rows, int cols) {
    int bpr = cols / QK4_0;
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    for (int r = 0; r < rows; r++)
        dequant_q4_0_row(blocks + r * bpr, dst + r * cols, bpr);
}

// ─── Q4_K dequantization (NEON) ─────────────────────────────────────────────
//
// ggml's get_scale_min_k4 unpacking (verbatim from ggml-quants.c):
//   j < 4:  sc = q[j] & 63,        m = q[j+4] & 63
//   j >= 4: sc = (q[j+4]&0xF) | ((q[j-4]>>6)<<4),
//           m  = (q[j+4]>>4)  | ((q[j]>>6)<<4)
//
// ggml iteration over 256 values:
//   for j in [0, 64, 128, 192]:
//     get_scale_min_k4(is+0, ...) → d1, m1  (low nibbles of 32 bytes)
//     get_scale_min_k4(is+1, ...) → d2, m2  (high nibbles of same 32 bytes)
//     for l in 0..31: out = d1 * (qs[l] & 0xF) - m1
//     for l in 0..31: out = d2 * (qs[l] >> 4) - m2
//     qs += 32, is += 2

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

static void dequant_q4_K_block_neon(const block_q4_K *block, _Float16 *out) {
    float d    = (float)block->d;
    float dmin = (float)block->dmin;
    const uint8_t *qs = block->qs;
    int is = 0;

    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4(is + 0, block->scales, &sc1, &m1);
        get_scale_min_k4(is + 1, block->scales, &sc2, &m2);

        float16x8_t vd1   = vdupq_n_f16((_Float16)(d * sc1));
        float16x8_t vm1   = vdupq_n_f16((_Float16)(dmin * m1));
        float16x8_t vd2   = vdupq_n_f16((_Float16)(d * sc2));
        float16x8_t vm2   = vdupq_n_f16((_Float16)(dmin * m2));

        // Load 32 bytes of nibble data
        uint8x16_t raw0 = vld1q_u8(qs);
        uint8x16_t raw1 = vld1q_u8(qs + 16);

        // Low nibbles → first 32 values (scale d1, min m1)
        uint8x16_t lo0 = vandq_u8(raw0, vdupq_n_u8(0x0F));
        uint8x16_t lo1 = vandq_u8(raw1, vdupq_n_u8(0x0F));
        // High nibbles → next 32 values (scale d2, min m2)
        uint8x16_t hi0 = vshrq_n_u8(raw0, 4);
        uint8x16_t hi1 = vshrq_n_u8(raw1, 4);

        // Low nibbles: val = d1 * q - m1
        {
            float16x8_t f0 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(lo0)));
            float16x8_t f1 = vcvtq_f16_u16(vmovl_u8(vget_high_u8(lo0)));
            float16x8_t f2 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(lo1)));
            float16x8_t f3 = vcvtq_f16_u16(vmovl_u8(vget_high_u8(lo1)));
            vst1q_f16((__fp16*)(out + j + 0),  vsubq_f16(vmulq_f16(f0, vd1), vm1));
            vst1q_f16((__fp16*)(out + j + 8),  vsubq_f16(vmulq_f16(f1, vd1), vm1));
            vst1q_f16((__fp16*)(out + j + 16), vsubq_f16(vmulq_f16(f2, vd1), vm1));
            vst1q_f16((__fp16*)(out + j + 24), vsubq_f16(vmulq_f16(f3, vd1), vm1));
        }
        // High nibbles: val = d2 * q - m2
        {
            float16x8_t f0 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(hi0)));
            float16x8_t f1 = vcvtq_f16_u16(vmovl_u8(vget_high_u8(hi0)));
            float16x8_t f2 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(hi1)));
            float16x8_t f3 = vcvtq_f16_u16(vmovl_u8(vget_high_u8(hi1)));
            vst1q_f16((__fp16*)(out + j + 32), vsubq_f16(vmulq_f16(f0, vd2), vm2));
            vst1q_f16((__fp16*)(out + j + 40), vsubq_f16(vmulq_f16(f1, vd2), vm2));
            vst1q_f16((__fp16*)(out + j + 48), vsubq_f16(vmulq_f16(f2, vd2), vm2));
            vst1q_f16((__fp16*)(out + j + 56), vsubq_f16(vmulq_f16(f3, vd2), vm2));
        }
        qs += 32;
        is += 2;
    }
}

static void dequant_q4_K_row(const block_q4_K *blocks, _Float16 *out, int n_blocks) {
    for (int i = 0; i < n_blocks; i++)
        dequant_q4_K_block_neon(&blocks[i], out + i * QK_K);
}

static void dequant_q4_K_matrix(const void *src, _Float16 *dst, int rows, int cols) {
    int bpr = cols / QK_K;
    const block_q4_K *blocks = (const block_q4_K *)src;
    for (int r = 0; r < rows; r++)
        dequant_q4_K_row(blocks + r * bpr, dst + r * cols, bpr);
}

// ─── GGUF file parsing (minimal, mmap-based) ────────────────────────────────

typedef struct {
    int      fd;
    void    *data;
    size_t   size;
} gguf_mmap;

static int gguf_mmap_open(gguf_mmap *g, const char *path) {
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) { perror("open"); return -1; }
    struct stat st;
    fstat(g->fd, &st);
    g->size = st.st_size;
    g->data = mmap(NULL, g->size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (g->data == MAP_FAILED) { perror("mmap"); close(g->fd); return -1; }
    madvise(g->data, g->size, MADV_SEQUENTIAL);
    return 0;
}

static void gguf_mmap_close(gguf_mmap *g) {
    if (g->data && g->data != MAP_FAILED) munmap(g->data, g->size);
    if (g->fd >= 0) close(g->fd);
}

// GGUF value types
enum gguf_type {
    GGUF_TYPE_UINT8   = 0, GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2, GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4, GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6, GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8, GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64  = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// GGML quant types
enum ggml_type {
    GGML_TYPE_F32  = 0,  GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,  GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,  GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,  GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14, GGML_TYPE_Q8_K = 15,
};

typedef struct {
    const uint8_t *base;
    size_t         pos;
    size_t         size;
} gguf_reader;

static inline uint32_t rd_u32(gguf_reader *r) { uint32_t v; memcpy(&v, r->base+r->pos, 4); r->pos+=4; return v; }
static inline uint64_t rd_u64(gguf_reader *r) { uint64_t v; memcpy(&v, r->base+r->pos, 8); r->pos+=8; return v; }

static inline const char *rd_str(gguf_reader *r, uint64_t *len) {
    *len = rd_u64(r);
    const char *s = (const char*)(r->base + r->pos);
    r->pos += *len;
    return s;
}

static void skip_gguf_value(gguf_reader *r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: r->pos += 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:  r->pos += 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: r->pos += 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: r->pos += 8; break;
        case GGUF_TYPE_STRING: { uint64_t len; rd_str(r, &len); break; }
        case GGUF_TYPE_ARRAY: {
            uint32_t atype = rd_u32(r);
            uint64_t count = rd_u64(r);
            for (uint64_t i = 0; i < count; i++) skip_gguf_value(r, atype);
            break;
        }
    }
}

typedef struct {
    char     name[128];
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
} gguf_tensor_info;

typedef struct {
    uint32_t          version;
    uint64_t          n_tensors;
    uint64_t          n_kv;
    gguf_tensor_info *tensors;
    size_t            data_offset;
} gguf_header;

static int gguf_parse_header(const gguf_mmap *g, gguf_header *hdr) {
    gguf_reader r = { .base = (const uint8_t*)g->data, .pos = 0, .size = g->size };

    uint32_t magic = rd_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Not a GGUF file (magic=0x%08X)\n", magic);
        return -1;
    }
    hdr->version   = rd_u32(&r);
    hdr->n_tensors = rd_u64(&r);
    hdr->n_kv      = rd_u64(&r);

    for (uint64_t i = 0; i < hdr->n_kv; i++) {
        uint64_t klen;
        rd_str(&r, &klen);
        uint32_t vtype = rd_u32(&r);
        skip_gguf_value(&r, vtype);
    }

    hdr->tensors = (gguf_tensor_info*)calloc(hdr->n_tensors, sizeof(gguf_tensor_info));
    for (uint64_t i = 0; i < hdr->n_tensors; i++) {
        gguf_tensor_info *t = &hdr->tensors[i];
        uint64_t nlen;
        const char *name = rd_str(&r, &nlen);
        size_t cpy = nlen < 127 ? nlen : 127;
        memcpy(t->name, name, cpy);
        t->name[cpy] = '\0';
        t->n_dims = rd_u32(&r);
        for (uint32_t d = 0; d < t->n_dims; d++)
            t->ne[d] = rd_u64(&r);
        for (uint32_t d = t->n_dims; d < 4; d++)
            t->ne[d] = 1;
        t->type   = rd_u32(&r);
        t->offset = rd_u64(&r);
    }

    // GGUF v2/v3 alignment is 32 bytes
    size_t alignment = 32;
    hdr->data_offset = (r.pos + alignment - 1) & ~(alignment - 1);
    return 0;
}

static const gguf_tensor_info *gguf_find_tensor(const gguf_header *hdr, const char *name) {
    for (uint64_t i = 0; i < hdr->n_tensors; i++)
        if (strcmp(hdr->tensors[i].name, name) == 0)
            return &hdr->tensors[i];
    return NULL;
}

static void gguf_free_header(gguf_header *hdr) {
    free(hdr->tensors);
    hdr->tensors = NULL;
}

// ─── High-level: dequant a GGUF tensor to fp16 ──────────────────────────────

static inline const void *gguf_tensor_data(const gguf_mmap *g,
                                            const gguf_header *hdr,
                                            const gguf_tensor_info *t) {
    return (const uint8_t*)g->data + hdr->data_offset + t->offset;
}

static inline uint64_t gguf_tensor_nelements(const gguf_tensor_info *t) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n *= t->ne[d];
    return n;
}

// Dequantize any supported tensor to fp16.
// Returns malloc'd fp16 buffer (caller frees). Sets *out_rows, *out_cols.
static _Float16 *gguf_dequant_tensor(const gguf_mmap *g,
                                      const gguf_header *hdr,
                                      const gguf_tensor_info *t,
                                      int *out_rows, int *out_cols) {
    uint64_t nel = gguf_tensor_nelements(t);
    const void *src = gguf_tensor_data(g, hdr, t);

    int cols = (int)t->ne[0];
    int rows = (t->n_dims >= 2) ? (int)t->ne[1] : 1;
    if (out_rows) *out_rows = rows;
    if (out_cols) *out_cols = cols;

    _Float16 *dst = (_Float16*)malloc(nel * sizeof(_Float16));
    if (!dst) return NULL;

    switch (t->type) {
        case GGML_TYPE_Q4_0:
            dequant_q4_0_matrix(src, dst, rows, cols);
            break;
        case GGML_TYPE_Q4_K:
            dequant_q4_K_matrix(src, dst, rows, cols);
            break;
        case GGML_TYPE_F16:
            memcpy(dst, src, nel * sizeof(_Float16));
            break;
        case GGML_TYPE_F32: {
            const float *f = (const float *)src;
            int i = 0, n = (int)nel;
            for (; i + 7 < n; i += 8) {
                float16x8_t h = vcombine_f16(
                    vcvt_f16_f32(vld1q_f32(f + i)),
                    vcvt_f16_f32(vld1q_f32(f + i + 4)));
                vst1q_f16((__fp16*)(dst + i), h);
            }
            for (; i < n; i++) dst[i] = (_Float16)f[i];
            break;
        }
        default:
            fprintf(stderr, "Unsupported quant type %d for tensor %s\n", t->type, t->name);
            free(dst);
            return NULL;
    }
    return dst;
}

// ─── ANE weight blob from fp16 (matches existing blob format) ────────────────

static inline void *build_weight_blob_fp16(const _Float16 *weights, int rows, int cols, size_t *out_size) {
    size_t wsize = (size_t)rows * cols * 2;
    size_t total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    if (!buf) return NULL;

    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128;

    memcpy(buf + 128, weights, wsize);
    if (out_size) *out_size = total;
    return buf;
}

// Full pipeline: GGUF tensor name → ANE weight blob (malloc'd, caller frees)
static void *gguf_tensor_to_ane_blob(const gguf_mmap *g,
                                      const gguf_header *hdr,
                                      const char *tensor_name,
                                      int *out_rows, int *out_cols,
                                      size_t *out_blob_size) {
    const gguf_tensor_info *t = gguf_find_tensor(hdr, tensor_name);
    if (!t) {
        fprintf(stderr, "Tensor '%s' not found in GGUF\n", tensor_name);
        return NULL;
    }

    int rows, cols;
    _Float16 *fp16 = gguf_dequant_tensor(g, hdr, t, &rows, &cols);
    if (!fp16) return NULL;

    if (out_rows) *out_rows = rows;
    if (out_cols) *out_cols = cols;

    void *blob = build_weight_blob_fp16(fp16, rows, cols, out_blob_size);
    free(fp16);
    return blob;
}

// ─── Benchmark ───────────────────────────────────────────────────────────────

static double bench_dequant_q4_0(int rows, int cols, int iterations) {
    int bpr = cols / QK4_0;
    int total_blocks = rows * bpr;
    block_q4_0 *src = (block_q4_0*)calloc(total_blocks, sizeof(block_q4_0));
    _Float16 *dst = (_Float16*)malloc((size_t)rows * cols * sizeof(_Float16));

    for (int i = 0; i < total_blocks; i++) {
        src[i].d = (_Float16)0.01f;
        for (int j = 0; j < QK4_0/2; j++) src[i].qs[j] = 0x48;
    }

    mach_timebase_info_data_t tbi;
    mach_timebase_info(&tbi);

    uint64_t start = mach_absolute_time();
    for (int it = 0; it < iterations; it++)
        dequant_q4_0_matrix(src, dst, rows, cols);
    uint64_t end = mach_absolute_time();

    double ns = (double)(end - start) * tbi.numer / tbi.denom;
    double ms_per_iter = ns / 1e6 / iterations;
    free(src); free(dst);
    return ms_per_iter;
}

static double bench_dequant_q4_K(int rows, int cols, int iterations) {
    int bpr = cols / QK_K;
    int total_blocks = rows * bpr;
    block_q4_K *src = (block_q4_K*)calloc(total_blocks, sizeof(block_q4_K));
    _Float16 *dst = (_Float16*)malloc((size_t)rows * cols * sizeof(_Float16));

    for (int i = 0; i < total_blocks; i++) {
        src[i].d = (_Float16)0.01f;
        src[i].dmin = (_Float16)0.001f;
        for (int j = 0; j < 12; j++) src[i].scales[j] = 0x11;
        for (int j = 0; j < QK_K/2; j++) src[i].qs[j] = 0x48;
    }

    mach_timebase_info_data_t tbi;
    mach_timebase_info(&tbi);

    uint64_t start = mach_absolute_time();
    for (int it = 0; it < iterations; it++)
        dequant_q4_K_matrix(src, dst, rows, cols);
    uint64_t end = mach_absolute_time();

    double ns = (double)(end - start) * tbi.numer / tbi.denom;
    double ms_per_iter = ns / 1e6 / iterations;
    free(src); free(dst);
    return ms_per_iter;
}
