// dequant.h — Q4_0/Q4_K dequantization + fused Q4_0 matvec for CPU decode
// Self-contained header: block structs + NEON dequant from gguf_dequant.h,
// plus a fused Q4_0 matrix-vector multiply that avoids materializing fp16 weights.
#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

// ─── Block layouts (match ggml exactly) ──────────────────────────────────────

#define QK4_0  32
#define QK_K  256

typedef struct {
    _Float16 d;
    uint8_t  qs[QK4_0/2];
} block_q4_0;
_Static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size");

typedef struct {
    _Float16 d;
    _Float16 dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K/2];
} block_q4_K;
_Static_assert(sizeof(block_q4_K) == 144, "block_q4_K size");

// ─── Q4_0 block dequant to fp16 (NEON) ──────────────────────────────────────

static void dequant_q4_0_block_neon(const block_q4_0 *block, _Float16 *out) {
    float16x8_t vscale = vdupq_n_f16(block->d);
    int8x16_t v8 = vdupq_n_s8(8);

    uint8x16_t raw = vld1q_u8(block->qs);

    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), v8);
    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8);

    {
        int16x8_t a = vmovl_s8(vget_low_s8(lo));
        int16x8_t b = vmovl_s8(vget_high_s8(lo));
        vst1q_f16((__fp16*)(out + 0), vmulq_f16(vcvtq_f16_s16(a), vscale));
        vst1q_f16((__fp16*)(out + 8), vmulq_f16(vcvtq_f16_s16(b), vscale));
    }
    {
        int16x8_t a = vmovl_s8(vget_low_s8(hi));
        int16x8_t b = vmovl_s8(vget_high_s8(hi));
        vst1q_f16((__fp16*)(out + 16), vmulq_f16(vcvtq_f16_s16(a), vscale));
        vst1q_f16((__fp16*)(out + 24), vmulq_f16(vcvtq_f16_s16(b), vscale));
    }
}

// ─── Q4_K block dequant to fp16 (NEON) ──────────────────────────────────────

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

        float16x8_t vd1 = vdupq_n_f16((_Float16)(d * sc1));
        float16x8_t vm1 = vdupq_n_f16((_Float16)(dmin * m1));
        float16x8_t vd2 = vdupq_n_f16((_Float16)(d * sc2));
        float16x8_t vm2 = vdupq_n_f16((_Float16)(dmin * m2));

        uint8x16_t raw0 = vld1q_u8(qs);
        uint8x16_t raw1 = vld1q_u8(qs + 16);

        uint8x16_t lo0 = vandq_u8(raw0, vdupq_n_u8(0x0F));
        uint8x16_t lo1 = vandq_u8(raw1, vdupq_n_u8(0x0F));
        uint8x16_t hi0 = vshrq_n_u8(raw0, 4);
        uint8x16_t hi1 = vshrq_n_u8(raw1, 4);

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

// ─── Bulk dequant to fp16 (for ANE prefill path) ────────────────────────────

static void dequant_q4_0_to_fp16(const void *src, _Float16 *dst, int rows, int cols) {
    int bpr = cols / QK4_0;
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < bpr; i++)
            dequant_q4_0_block_neon(&blocks[r * bpr + i], dst + r * cols + i * QK4_0);
}

static void dequant_q4_K_to_fp16(const void *src, _Float16 *dst, int rows, int cols) {
    int bpr = cols / QK_K;
    const block_q4_K *blocks = (const block_q4_K *)src;
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < bpr; i++)
            dequant_q4_K_block_neon(&blocks[r * bpr + i], dst + r * cols + i * QK_K);
}

// ─── Fused Q4_0 matrix-vector multiply (fp32 accumulation, 4-row unroll) ────
//
// y[out_dim] += W_q4[out_dim, in_dim] @ x[in_dim]
// Reads Q4_0 blocks directly from mmap'd data, dequants and accumulates in
// one pass without materializing the full fp16 weight matrix.
// 4-row unrolled for NEON throughput. For decode (S=1).

static void q4_0_matvec_f32(const void *W_q4, const float *x, float *y,
                             int out_dim, int in_dim) {
    const int bpr = in_dim / QK4_0;  // blocks per row
    const block_q4_0 *blocks = (const block_q4_0 *)W_q4;

    int row = 0;

    // 4-row unrolled main loop
    for (; row + 3 < out_dim; row += 4) {
        const block_q4_0 *row0 = blocks + (row + 0) * bpr;
        const block_q4_0 *row1 = blocks + (row + 1) * bpr;
        const block_q4_0 *row2 = blocks + (row + 2) * bpr;
        const block_q4_0 *row3 = blocks + (row + 3) * bpr;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (int b = 0; b < bpr; b++) {
            int j = b * QK4_0;

            // Load x[j..j+31] as 8 float32x4_t
            float32x4_t x0 = vld1q_f32(x + j + 0);
            float32x4_t x1 = vld1q_f32(x + j + 4);
            float32x4_t x2 = vld1q_f32(x + j + 8);
            float32x4_t x3 = vld1q_f32(x + j + 12);
            float32x4_t x4 = vld1q_f32(x + j + 16);
            float32x4_t x5 = vld1q_f32(x + j + 20);
            float32x4_t x6 = vld1q_f32(x + j + 24);
            float32x4_t x7 = vld1q_f32(x + j + 28);

            // Process each of the 4 rows
            #define PROCESS_ROW(ROW_PTR, ACC) do { \
                float scale = (float)(ROW_PTR)[b].d; \
                float32x4_t vscale = vdupq_n_f32(scale); \
                int8x16_t v8 = vdupq_n_s8(8); \
                uint8x16_t raw = vld1q_u8((ROW_PTR)[b].qs); \
                \
                /* Low nibbles -> values 0..15 */ \
                int8x16_t lo = vsubq_s8( \
                    vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), v8); \
                /* High nibbles -> values 16..31 */ \
                int8x16_t hi = vsubq_s8( \
                    vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8); \
                \
                /* Low nibbles: convert s8 -> s16 -> s32 -> f32, dot with x */ \
                int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo)); \
                int16x8_t lo16_1 = vmovl_s8(vget_high_s8(lo)); \
                \
                float32x4_t w0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))), vscale); \
                float32x4_t w1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0))), vscale); \
                float32x4_t w2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))), vscale); \
                float32x4_t w3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1))), vscale); \
                \
                ACC = vfmaq_f32(ACC, w0, x0); \
                ACC = vfmaq_f32(ACC, w1, x1); \
                ACC = vfmaq_f32(ACC, w2, x2); \
                ACC = vfmaq_f32(ACC, w3, x3); \
                \
                /* High nibbles */ \
                int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi)); \
                int16x8_t hi16_1 = vmovl_s8(vget_high_s8(hi)); \
                \
                float32x4_t w4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))), vscale); \
                float32x4_t w5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_0))), vscale); \
                float32x4_t w6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))), vscale); \
                float32x4_t w7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_1))), vscale); \
                \
                ACC = vfmaq_f32(ACC, w4, x4); \
                ACC = vfmaq_f32(ACC, w5, x5); \
                ACC = vfmaq_f32(ACC, w6, x6); \
                ACC = vfmaq_f32(ACC, w7, x7); \
            } while(0)

            PROCESS_ROW(row0, acc0);
            PROCESS_ROW(row1, acc1);
            PROCESS_ROW(row2, acc2);
            PROCESS_ROW(row3, acc3);

            #undef PROCESS_ROW
        }

        // Horizontal reduce each accumulator and add to y
        y[row + 0] += vaddvq_f32(acc0);
        y[row + 1] += vaddvq_f32(acc1);
        y[row + 2] += vaddvq_f32(acc2);
        y[row + 3] += vaddvq_f32(acc3);
    }

    // Tail: remaining rows (0-3)
    for (; row < out_dim; row++) {
        const block_q4_0 *rblk = blocks + row * bpr;
        float32x4_t acc = vdupq_n_f32(0.0f);

        for (int b = 0; b < bpr; b++) {
            int j = b * QK4_0;
            float scale = (float)rblk[b].d;
            float32x4_t vscale = vdupq_n_f32(scale);
            int8x16_t v8 = vdupq_n_s8(8);
            uint8x16_t raw = vld1q_u8(rblk[b].qs);

            int8x16_t lo = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), v8);
            int8x16_t hi = vsubq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8);

            int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo));
            int16x8_t lo16_1 = vmovl_s8(vget_high_s8(lo));

            float32x4_t w0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))), vscale);
            float32x4_t w1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0))), vscale);
            float32x4_t w2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))), vscale);
            float32x4_t w3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1))), vscale);

            acc = vfmaq_f32(acc, w0, vld1q_f32(x + j + 0));
            acc = vfmaq_f32(acc, w1, vld1q_f32(x + j + 4));
            acc = vfmaq_f32(acc, w2, vld1q_f32(x + j + 8));
            acc = vfmaq_f32(acc, w3, vld1q_f32(x + j + 12));

            int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi));
            int16x8_t hi16_1 = vmovl_s8(vget_high_s8(hi));

            float32x4_t w4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))), vscale);
            float32x4_t w5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_0))), vscale);
            float32x4_t w6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))), vscale);
            float32x4_t w7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_1))), vscale);

            acc = vfmaq_f32(acc, w4, vld1q_f32(x + j + 16));
            acc = vfmaq_f32(acc, w5, vld1q_f32(x + j + 20));
            acc = vfmaq_f32(acc, w6, vld1q_f32(x + j + 24));
            acc = vfmaq_f32(acc, w7, vld1q_f32(x + j + 28));
        }

        y[row] += vaddvq_f32(acc);
    }
}

// ─── Q6_K dequantization (scalar, for output.weight only) ────────────────────
// Q6_K: 256 values per super-block, 210 bytes each
// Layout: ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] (int8), d (fp16)
typedef struct {
    uint8_t ql[128];  // lower 4 bits of quantized values
    uint8_t qh[64];   // upper 2 bits of quantized values
    int8_t  scales[16]; // scales for 16 sub-blocks of 16 values
    _Float16 d;        // super-block scale
} block_q6_K;
_Static_assert(sizeof(block_q6_K) == 210, "block_q6_K size");

static void dequant_q6_K_block(const block_q6_K *block, float *out) {
    float d = (float)block->d;
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;

    for (int n = 0; n < 256; n++) {
        int il = n % 128;     // index into ql
        int is = n / 16;      // scale index
        int ih = n / 128;     // 0 or 1 (which half for qh bits)

        uint8_t q_lo = (n < 128) ? (ql[il] & 0xF) : (ql[il] >> 4);
        uint8_t q_hi = (qh[il % 64] >> (2 * ih + (il >= 64 ? 4 : 0))) & 3;
        int8_t q = (int8_t)((q_lo | (q_hi << 4)) - 32);
        out[n] = d * sc[is] * q;
    }
}

// Dequant Q6_K matrix to fp32 — scalar reference (kept for testing)
static void dequant_q6_K_matvec_f32_scalar(const void *W, const float *x, float *y, int out_dim, int in_dim) {
    const block_q6_K *blocks = (const block_q6_K *)W;
    int blocks_per_row = in_dim / 256;
    float *tmp = (float *)malloc(256 * sizeof(float));

    for (int row = 0; row < out_dim; row++) {
        float sum = 0;
        for (int b = 0; b < blocks_per_row; b++) {
            dequant_q6_K_block(&blocks[row * blocks_per_row + b], tmp);
            int base = b * 256;
            for (int i = 0; i < 256; i++) {
                sum += tmp[i] * x[base + i];
            }
        }
        y[row] = sum;
    }
    free(tmp);
}

// ─── Fused Q6_K matrix-vector multiply (NEON, fp32 accum, 2-row unroll) ─────
//
// y[out_dim] = W_q6k[out_dim, in_dim] @ x[in_dim]
// Reads Q6_K blocks directly, dequants + dot-products in one pass.
// Processes 16 values per sub-block (matching scale granularity).
//
// Q6_K block layout (256 values, 210 bytes):
//   ql[128]: packed low 4 bits (two nibbles per byte)
//   qh[64]:  packed high 2 bits (four 2-bit fields per byte)
//   scales[16]: int8 per-sub-block scales
//   d: fp16 super-block scale
//
// Value n mapping (0..255):
//   n=0..63:    ql[n]    & 0xF, qh[n]    >> 0 & 3
//   n=64..127:  ql[n]    & 0xF, qh[n-64] >> 4 & 3
//   n=128..191: ql[n-128]>> 4,  qh[n-128]>> 2 & 3
//   n=192..255: ql[n-128]>> 4,  qh[n-192]>> 6 & 3
//   q6 = (lo | (hi << 4)) - 32;  weight = d * scales[n/16] * q6

// Process 16 values: dequant Q6_K sub-block and fma with x vector.
// ql16: 16 bytes of ql data, qh16: 16 bytes of qh data (pre-shifted to bits [0:1]),
// vds: float32x4_t broadcast of d*scale for this sub-block.
#define Q6K_SUBBLOCK_FMA(ACC, ql16, qh16, vds, xptr) do { \
    /* Combine: q6 = (lo4 | (hi2 << 4)) - 32, range [-32, 31] */ \
    int8x16_t q6 = vsubq_s8( \
        vreinterpretq_s8_u8(vorrq_u8(ql16, vshlq_n_u8(vandq_u8(qh16, vdup_mask2), 4))), \
        vdup_32); \
    /* Widen to s16 then s32, convert to f32, fma with x */ \
    int16x8_t q16_lo = vmovl_s8(vget_low_s8(q6)); \
    int16x8_t q16_hi = vmovl_s8(vget_high_s8(q6)); \
    ACC = vfmaq_f32(ACC, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_lo))),  vds), vld1q_f32(xptr + 0)); \
    ACC = vfmaq_f32(ACC, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_lo))), vds), vld1q_f32(xptr + 4)); \
    ACC = vfmaq_f32(ACC, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_hi))),  vds), vld1q_f32(xptr + 8)); \
    ACC = vfmaq_f32(ACC, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_hi))), vds), vld1q_f32(xptr + 12)); \
} while(0)

static void q6_K_matvec_f32_neon(const void *W, const float *x, float *y,
                                  int out_dim, int in_dim) {
    const block_q6_K *blocks = (const block_q6_K *)W;
    const int bpr = in_dim / 256;

    const uint8x16_t vdup_mask2 = vdupq_n_u8(3);
    const int8x16_t  vdup_32    = vdupq_n_s8(32);
    const uint8x16_t vdup_0xF   = vdupq_n_u8(0x0F);

    int row = 0;

    // 2-row unrolled main loop
    for (; row + 1 < out_dim; row += 2) {
        const block_q6_K *r0 = blocks + (row + 0) * bpr;
        const block_q6_K *r1 = blocks + (row + 1) * bpr;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        for (int b = 0; b < bpr; b++) {
            const float *xb = x + b * 256;

            // Prefetch next block's weight data (block_q6_K is 208B = 2 cachelines)
            if (b + 1 < bpr) {
                __builtin_prefetch(&r0[b+1], 0, 1);
                __builtin_prefetch(&r1[b+1], 0, 1);
            }

            // Process both rows for this block
            #define PROCESS_Q6K_BLOCK(BLK, ACC) do { \
                float d = (float)(BLK)->d; \
                const uint8_t *ql = (BLK)->ql; \
                const uint8_t *qh = (BLK)->qh; \
                const int8_t  *sc = (BLK)->scales; \
                \
                /* --- n = 0..63: ql[0..63] & 0xF, qh[0..63] >> 0 --- */ \
                { \
                    uint8x16_t qh0_raw = vld1q_u8(qh + 0);   /* qh[0..15]  */ \
                    uint8x16_t qh1_raw = vld1q_u8(qh + 16);  /* qh[16..31] */ \
                    uint8x16_t qh2_raw = vld1q_u8(qh + 32);  /* qh[32..47] */ \
                    uint8x16_t qh3_raw = vld1q_u8(qh + 48);  /* qh[48..63] */ \
                    /* Sub-block 0: n=0..15, ql[0..15] & 0xF, qh[0..15] >> 0 */ \
                    float32x4_t vds0 = vdupq_n_f32(d * sc[0]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 0), vdup_0xF), qh0_raw, vds0, xb + 0); \
                    /* Sub-block 1: n=16..31, ql[16..31] & 0xF, qh[16..31] >> 0 */ \
                    float32x4_t vds1 = vdupq_n_f32(d * sc[1]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 16), vdup_0xF), qh1_raw, vds1, xb + 16); \
                    /* Sub-block 2: n=32..47 */ \
                    float32x4_t vds2 = vdupq_n_f32(d * sc[2]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 32), vdup_0xF), qh2_raw, vds2, xb + 32); \
                    /* Sub-block 3: n=48..63 */ \
                    float32x4_t vds3 = vdupq_n_f32(d * sc[3]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 48), vdup_0xF), qh3_raw, vds3, xb + 48); \
                    \
                    /* --- n = 64..127: ql[64..127] & 0xF, qh[0..63] >> 4 --- */ \
                    uint8x16_t qh0_s4 = vshrq_n_u8(qh0_raw, 4); \
                    uint8x16_t qh1_s4 = vshrq_n_u8(qh1_raw, 4); \
                    uint8x16_t qh2_s4 = vshrq_n_u8(qh2_raw, 4); \
                    uint8x16_t qh3_s4 = vshrq_n_u8(qh3_raw, 4); \
                    float32x4_t vds4 = vdupq_n_f32(d * sc[4]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 64), vdup_0xF), qh0_s4, vds4, xb + 64); \
                    float32x4_t vds5 = vdupq_n_f32(d * sc[5]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 80), vdup_0xF), qh1_s4, vds5, xb + 80); \
                    float32x4_t vds6 = vdupq_n_f32(d * sc[6]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 96), vdup_0xF), qh2_s4, vds6, xb + 96); \
                    float32x4_t vds7 = vdupq_n_f32(d * sc[7]); \
                    Q6K_SUBBLOCK_FMA(ACC, vandq_u8(vld1q_u8(ql + 112), vdup_0xF), qh3_s4, vds7, xb + 112); \
                    \
                    /* --- n = 128..191: ql[0..63] >> 4, qh[0..63] >> 2 --- */ \
                    uint8x16_t qh0_s2 = vshrq_n_u8(qh0_raw, 2); \
                    uint8x16_t qh1_s2 = vshrq_n_u8(qh1_raw, 2); \
                    uint8x16_t qh2_s2 = vshrq_n_u8(qh2_raw, 2); \
                    uint8x16_t qh3_s2 = vshrq_n_u8(qh3_raw, 2); \
                    float32x4_t vds8 = vdupq_n_f32(d * sc[8]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 0), 4), qh0_s2, vds8, xb + 128); \
                    float32x4_t vds9 = vdupq_n_f32(d * sc[9]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 16), 4), qh1_s2, vds9, xb + 144); \
                    float32x4_t vds10 = vdupq_n_f32(d * sc[10]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 32), 4), qh2_s2, vds10, xb + 160); \
                    float32x4_t vds11 = vdupq_n_f32(d * sc[11]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 48), 4), qh3_s2, vds11, xb + 176); \
                    \
                    /* --- n = 192..255: ql[64..127] >> 4, qh[0..63] >> 6 --- */ \
                    uint8x16_t qh0_s6 = vshrq_n_u8(qh0_raw, 6); \
                    uint8x16_t qh1_s6 = vshrq_n_u8(qh1_raw, 6); \
                    uint8x16_t qh2_s6 = vshrq_n_u8(qh2_raw, 6); \
                    uint8x16_t qh3_s6 = vshrq_n_u8(qh3_raw, 6); \
                    float32x4_t vds12 = vdupq_n_f32(d * sc[12]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 64), 4), qh0_s6, vds12, xb + 192); \
                    float32x4_t vds13 = vdupq_n_f32(d * sc[13]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 80), 4), qh1_s6, vds13, xb + 208); \
                    float32x4_t vds14 = vdupq_n_f32(d * sc[14]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 96), 4), qh2_s6, vds14, xb + 224); \
                    float32x4_t vds15 = vdupq_n_f32(d * sc[15]); \
                    Q6K_SUBBLOCK_FMA(ACC, vshrq_n_u8(vld1q_u8(ql + 112), 4), qh3_s6, vds15, xb + 240); \
                } \
            } while(0)

            PROCESS_Q6K_BLOCK(&r0[b], acc0);
            PROCESS_Q6K_BLOCK(&r1[b], acc1);

            #undef PROCESS_Q6K_BLOCK
        }

        y[row + 0] = vaddvq_f32(acc0);
        y[row + 1] = vaddvq_f32(acc1);
    }

    // Tail: single remaining row
    for (; row < out_dim; row++) {
        const block_q6_K *r0 = blocks + row * bpr;
        float32x4_t acc0 = vdupq_n_f32(0.0f);

        for (int b = 0; b < bpr; b++) {
            if (b + 1 < bpr)
                __builtin_prefetch(&r0[b+1], 0, 1);

            const float *xb = x + b * 256;
            const block_q6_K *blk = &r0[b];
            float d = (float)blk->d;
            const uint8_t *ql = blk->ql;
            const uint8_t *qh = blk->qh;
            const int8_t  *sc = blk->scales;

            uint8x16_t qh0_raw = vld1q_u8(qh + 0);
            uint8x16_t qh1_raw = vld1q_u8(qh + 16);
            uint8x16_t qh2_raw = vld1q_u8(qh + 32);
            uint8x16_t qh3_raw = vld1q_u8(qh + 48);

            // n=0..63
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+0),  vdup_0xF), qh0_raw, vdupq_n_f32(d*sc[0]),  xb+0);
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+16), vdup_0xF), qh1_raw, vdupq_n_f32(d*sc[1]),  xb+16);
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+32), vdup_0xF), qh2_raw, vdupq_n_f32(d*sc[2]),  xb+32);
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+48), vdup_0xF), qh3_raw, vdupq_n_f32(d*sc[3]),  xb+48);
            // n=64..127
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+64),  vdup_0xF), vshrq_n_u8(qh0_raw,4), vdupq_n_f32(d*sc[4]),  xb+64);
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+80),  vdup_0xF), vshrq_n_u8(qh1_raw,4), vdupq_n_f32(d*sc[5]),  xb+80);
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+96),  vdup_0xF), vshrq_n_u8(qh2_raw,4), vdupq_n_f32(d*sc[6]),  xb+96);
            Q6K_SUBBLOCK_FMA(acc0, vandq_u8(vld1q_u8(ql+112), vdup_0xF), vshrq_n_u8(qh3_raw,4), vdupq_n_f32(d*sc[7]),  xb+112);
            // n=128..191
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+0),  4), vshrq_n_u8(qh0_raw,2), vdupq_n_f32(d*sc[8]),  xb+128);
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+16), 4), vshrq_n_u8(qh1_raw,2), vdupq_n_f32(d*sc[9]),  xb+144);
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+32), 4), vshrq_n_u8(qh2_raw,2), vdupq_n_f32(d*sc[10]), xb+160);
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+48), 4), vshrq_n_u8(qh3_raw,2), vdupq_n_f32(d*sc[11]), xb+176);
            // n=192..255
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+64),  4), vshrq_n_u8(qh0_raw,6), vdupq_n_f32(d*sc[12]), xb+192);
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+80),  4), vshrq_n_u8(qh1_raw,6), vdupq_n_f32(d*sc[13]), xb+208);
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+96),  4), vshrq_n_u8(qh2_raw,6), vdupq_n_f32(d*sc[14]), xb+224);
            Q6K_SUBBLOCK_FMA(acc0, vshrq_n_u8(vld1q_u8(ql+112), 4), vshrq_n_u8(qh3_raw,6), vdupq_n_f32(d*sc[15]), xb+240);
        }

        y[row] = vaddvq_f32(acc0);
    }
}

#undef Q6K_SUBBLOCK_FMA

// Drop-in replacement name
#define dequant_q6_K_matvec_f32 q6_K_matvec_f32_neon

// ─── W4A8 SDOT: Q8_0 activation block + fused Q4_0×Q8_0 matvec ─────────────
//
// Instead of dequanting W to float and doing FMA with float x, we:
//   1. Pre-quantize fp32 x → Q8_0 (int8 + scale per 32 values) once per layer
//   2. Use SDOT (vdotq_s32) to compute int8×int8 dot products directly
//   3. Convert to float and apply scale correction only at block boundaries
//
// Inner loop per block: ~9 NEON ops vs ~43 in the float path.
// SDOT computes 4 lanes of (4 int8 muls + accumulate) per instruction.

#define QK8_0 32

typedef struct {
    float d;           // scale: max(|x|)/127 for this block
    int8_t qs[QK8_0];  // quantized values: round(x[i] / d), clamped [-127, 127]
} block_q8_0;
_Static_assert(sizeof(block_q8_0) == 36, "block_q8_0 size");

// Quantize fp32 vector to Q8_0 format. n must be a multiple of 32.
// Called once per layer before all matvecs sharing the same input vector.
static void quantize_f32_to_q8_0(const float *x, block_q8_0 *out, int n) {
    const int nb = n / QK8_0;

    for (int i = 0; i < nb; i++) {
        const float *src = x + i * QK8_0;

        // Find max absolute value across 32 floats using NEON
        float32x4_t v0 = vabsq_f32(vld1q_f32(src + 0));
        float32x4_t v1 = vabsq_f32(vld1q_f32(src + 4));
        float32x4_t v2 = vabsq_f32(vld1q_f32(src + 8));
        float32x4_t v3 = vabsq_f32(vld1q_f32(src + 12));
        float32x4_t v4 = vabsq_f32(vld1q_f32(src + 16));
        float32x4_t v5 = vabsq_f32(vld1q_f32(src + 20));
        float32x4_t v6 = vabsq_f32(vld1q_f32(src + 24));
        float32x4_t v7 = vabsq_f32(vld1q_f32(src + 28));

        float32x4_t m01 = vmaxq_f32(v0, v1);
        float32x4_t m23 = vmaxq_f32(v2, v3);
        float32x4_t m45 = vmaxq_f32(v4, v5);
        float32x4_t m67 = vmaxq_f32(v6, v7);
        float32x4_t m0123 = vmaxq_f32(m01, m23);
        float32x4_t m4567 = vmaxq_f32(m45, m67);
        float amax = vmaxvq_f32(vmaxq_f32(m0123, m4567));

        float d = amax / 127.0f;
        out[i].d = d;

        if (amax == 0.0f) {
            memset(out[i].qs, 0, QK8_0);
            continue;
        }

        float id = 1.0f / d;  // inverse scale
        float32x4_t vid = vdupq_n_f32(id);

        // Quantize: round(x[j] * id), clamp to [-127, 127]
        // Process 8 floats at a time → 8 int8 values
        for (int j = 0; j < QK8_0; j += 8) {
            float32x4_t f0 = vmulq_f32(vld1q_f32(src + j + 0), vid);
            float32x4_t f1 = vmulq_f32(vld1q_f32(src + j + 4), vid);

            // Round to nearest: vcvtnq_s32_f32 (ties to even)
            int32x4_t i0 = vcvtnq_s32_f32(f0);
            int32x4_t i1 = vcvtnq_s32_f32(f1);

            // Narrow s32 → s16 → s8 (saturating, clamps to [-128,127])
            int16x4_t n0 = vqmovn_s32(i0);
            int16x4_t n1 = vqmovn_s32(i1);
            int16x8_t n01 = vcombine_s16(n0, n1);
            int8x8_t q = vqmovn_s16(n01);

            vst1_s8(out[i].qs + j, q);
        }
    }
}

// Fused W4A8 matvec: Q4_0 weights × Q8_0 activations using SDOT.
// y[out_dim] = W_q4[out_dim, in_dim] @ x_q8[in_dim/32 blocks]
// 4-row unrolled, int32 accumulation within blocks, fp32 scale at boundaries.
//
// Per-block inner loop:
//   Load 16 bytes Q4_0 → extract 32 int8 values (lo nibbles, hi nibbles)
//   Load 32 int8 Q8_0 activation values
//   2x vdotq_s32 for lo half (16 values), 2x vdotq_s32 for hi half (16 values)
//   Horizontal sum → float, multiply by w_scale * x_scale, accumulate
//
// Q4_0 nibble layout: qs[i] packs values [2i] (low nibble) and [2i+16] (high nibble)
// Low nibble values are indices 0..15, high nibble values are indices 16..31.

static void q4_0_q8_0_matvec_sdot(const void *W_q4, const block_q8_0 *x_q8,
                                    float *y, int out_dim, int in_dim) {
    const int bpr = in_dim / QK4_0;
    const block_q4_0 *blocks = (const block_q4_0 *)W_q4;

    memset(y, 0, out_dim * sizeof(float));

    // Constant for nibble extraction: subtract 8 to center [-8, 7]
    const int8x16_t v8 = vdupq_n_s8(8);
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    int row = 0;

    // 4-row unrolled main loop
    for (; row + 3 < out_dim; row += 4) {
        const block_q4_0 *r0 = blocks + (row + 0) * bpr;
        const block_q4_0 *r1 = blocks + (row + 1) * bpr;
        const block_q4_0 *r2 = blocks + (row + 2) * bpr;
        const block_q4_0 *r3 = blocks + (row + 3) * bpr;

        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        for (int b = 0; b < bpr; b++) {
            // Prefetch next block's weight data for all 4 rows
            if (b + 1 < bpr) {
                __builtin_prefetch(&r0[b+1].qs, 0, 1);
                __builtin_prefetch(&r1[b+1].qs, 0, 1);
                __builtin_prefetch(&r2[b+1].qs, 0, 1);
                __builtin_prefetch(&r3[b+1].qs, 0, 1);
            }

            // Activation block: 32 int8 values + scale
            const int8x16_t xv_lo = vld1q_s8(x_q8[b].qs + 0);   // x[0..15]
            const int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);  // x[16..31]
            const float x_scale = x_q8[b].d;

            // Process 4 rows against same activation block
            #define SDOT_ROW(RPTR, SUM) do { \
                float w_scale = (float)(RPTR)[b].d; \
                uint8x16_t raw = vld1q_u8((RPTR)[b].qs); \
                \
                /* Extract low nibbles → int8, values 0..15 of the block */ \
                int8x16_t w_lo = vsubq_s8( \
                    vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), v8); \
                /* Extract high nibbles → int8, values 16..31 of the block */ \
                int8x16_t w_hi = vsubq_s8( \
                    vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8); \
                \
                /* SDOT: each vdotq_s32 does 4 lanes × 4 int8 muls + accumulate */ \
                /* w_lo (16 int8) dot xv_lo (16 int8) → 4 int32 partial sums */ \
                int32x4_t dot = vdupq_n_s32(0); \
                dot = vdotq_s32(dot, w_lo, xv_lo); \
                dot = vdotq_s32(dot, w_hi, xv_hi); \
                \
                /* Horizontal sum of 4 int32 lanes → scalar int32 */ \
                int32_t isum = vaddvq_s32(dot); \
                \
                /* Apply both scales at block boundary */ \
                SUM += (float)isum * w_scale * x_scale; \
            } while(0)

            SDOT_ROW(r0, sum0);
            SDOT_ROW(r1, sum1);
            SDOT_ROW(r2, sum2);
            SDOT_ROW(r3, sum3);

            #undef SDOT_ROW
        }

        y[row + 0] = sum0;
        y[row + 1] = sum1;
        y[row + 2] = sum2;
        y[row + 3] = sum3;
    }

    // Tail: remaining rows (0-3)
    for (; row < out_dim; row++) {
        const block_q4_0 *rblk = blocks + row * bpr;
        float sum = 0.0f;

        for (int b = 0; b < bpr; b++) {
            if (b + 1 < bpr)
                __builtin_prefetch(&rblk[b+1].qs, 0, 1);

            const int8x16_t xv_lo = vld1q_s8(x_q8[b].qs + 0);
            const int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);

            float w_scale = (float)rblk[b].d;
            float x_scale = x_q8[b].d;
            uint8x16_t raw = vld1q_u8(rblk[b].qs);

            int8x16_t w_lo = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), v8);
            int8x16_t w_hi = vsubq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8);

            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, w_lo, xv_lo);
            dot = vdotq_s32(dot, w_hi, xv_hi);

            sum += (float)vaddvq_s32(dot) * w_scale * x_scale;
        }

        y[row] = sum;
    }
}

// ─── I8MM path: 2 rows at a time via vmmlaq_s32 (SMMLA) ─────────────────────
//
// vmmlaq_s32(acc, a, b): acc[2x2] += a[2x8] @ b[2x8]^T
//   a = int8x16_t viewed as { row0[8], row1[8] }
//   b = int8x16_t viewed as { col0[8], col1[8] }
//   result[0] = row0 . col0, result[1] = row0 . col1
//   result[2] = row1 . col0, result[3] = row1 . col1
//
// For matvec we duplicate x into both b-rows:
//   b = { x_chunk[8], x_chunk[8] }
// Then result lanes 0,2 give the partial dot products for rows 0,1.
// 4 vmmlaq calls cover all 32 weight values for 2 rows:
//   lo[0..7], lo[8..15], hi[0..7], hi[8..15]
// That's 4 instructions vs 8 vdotq for the same 2 rows -- half the
// instruction count means less decode/issue pressure and better IPC.

#ifdef __ARM_FEATURE_MATMUL_INT8

static void q4_0_q8_0_matvec_i8mm(const void *W_q4, const block_q8_0 *x_q8,
                                    float *y, int out_dim, int in_dim) {
    const int bpr = in_dim / QK4_0;
    const block_q4_0 *blocks = (const block_q4_0 *)W_q4;
    const int8x16_t v8 = vdupq_n_s8(8);
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    memset(y, 0, out_dim * sizeof(float));

    int row = 0;

    // Main loop: 2 rows via SMMLA
    for (; row + 1 < out_dim; row += 2) {
        const block_q4_0 *r0 = blocks + (row + 0) * bpr;
        const block_q4_0 *r1 = blocks + (row + 1) * bpr;

        float sumf0 = 0, sumf1 = 0;

        for (int b = 0; b < bpr; b++) {
            const float x_scale = x_q8[b].d;
            const float w_scale0 = (float)r0[b].d;
            const float w_scale1 = (float)r1[b].d;

            // Extract weight int8 from Q4_0 nibbles
            uint8x16_t raw0 = vld1q_u8(r0[b].qs);
            uint8x16_t raw1 = vld1q_u8(r1[b].qs);

            int8x16_t w0_lo = vsubq_s8(vreinterpretq_s8_u8(
                vandq_u8(raw0, mask_lo)), v8);
            int8x16_t w0_hi = vsubq_s8(vreinterpretq_s8_u8(
                vshrq_n_u8(raw0, 4)), v8);
            int8x16_t w1_lo = vsubq_s8(vreinterpretq_s8_u8(
                vandq_u8(raw1, mask_lo)), v8);
            int8x16_t w1_hi = vsubq_s8(vreinterpretq_s8_u8(
                vshrq_n_u8(raw1, 4)), v8);

            // Activation int8 values
            int8x16_t xv0 = vld1q_s8(x_q8[b].qs + 0);   // x[0..15]
            int8x16_t xv1 = vld1q_s8(x_q8[b].qs + 16);  // x[16..31]

            // Pack weight rows: a = { r0_chunk[8], r1_chunk[8] }
            int8x16_t a_lo_0 = vcombine_s8(vget_low_s8(w0_lo),  vget_low_s8(w1_lo));
            int8x16_t a_lo_1 = vcombine_s8(vget_high_s8(w0_lo), vget_high_s8(w1_lo));
            int8x16_t a_hi_0 = vcombine_s8(vget_low_s8(w0_hi),  vget_low_s8(w1_hi));
            int8x16_t a_hi_1 = vcombine_s8(vget_high_s8(w0_hi), vget_high_s8(w1_hi));

            // Duplicate x chunks: b = { x_chunk[8], x_chunk[8] }
            int8x16_t bx0 = vcombine_s8(vget_low_s8(xv0),  vget_low_s8(xv0));
            int8x16_t bx1 = vcombine_s8(vget_high_s8(xv0), vget_high_s8(xv0));
            int8x16_t bx2 = vcombine_s8(vget_low_s8(xv1),  vget_low_s8(xv1));
            int8x16_t bx3 = vcombine_s8(vget_high_s8(xv1), vget_high_s8(xv1));

            // Low nibbles: w[0..15] . x[0..15]
            int32x4_t isum_lo = vdupq_n_s32(0);
            isum_lo = vmmlaq_s32(isum_lo, a_lo_0, bx0);  // w[0..7] . x[0..7]
            isum_lo = vmmlaq_s32(isum_lo, a_lo_1, bx1);  // w[8..15] . x[8..15]

            // High nibbles: w[16..31] . x[16..31]
            int32x4_t isum_hi = vdupq_n_s32(0);
            isum_hi = vmmlaq_s32(isum_hi, a_hi_0, bx2);  // w[16..23] . x[16..23]
            isum_hi = vmmlaq_s32(isum_hi, a_hi_1, bx3);  // w[24..31] . x[24..31]

            // vmmlaq result layout: { r0.col0, r0.col1, r1.col0, r1.col1 }
            // With duplicated x in b, col0==col1, so lane 0 = lane 1 = row0,
            // lane 2 = lane 3 = row1. Use lanes 0 and 2.
            int32_t idot0 = vgetq_lane_s32(isum_lo, 0) + vgetq_lane_s32(isum_hi, 0);
            int32_t idot1 = vgetq_lane_s32(isum_lo, 2) + vgetq_lane_s32(isum_hi, 2);

            sumf0 += w_scale0 * x_scale * (float)idot0;
            sumf1 += w_scale1 * x_scale * (float)idot1;
        }

        y[row + 0] = sumf0;
        y[row + 1] = sumf1;
    }

    // Tail: single remaining row via SDOT
    for (; row < out_dim; row++) {
        const block_q4_0 *rblk = blocks + row * bpr;
        float sumf = 0;

        for (int b = 0; b < bpr; b++) {
            const int8x16_t xv_lo = vld1q_s8(x_q8[b].qs + 0);
            const int8x16_t xv_hi = vld1q_s8(x_q8[b].qs + 16);
            float w_scale = (float)rblk[b].d;
            float x_scale = x_q8[b].d;

            uint8x16_t raw = vld1q_u8(rblk[b].qs);
            int8x16_t w_lo = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(raw, mask_lo)), v8);
            int8x16_t w_hi = vsubq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8);

            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, w_lo, xv_lo);
            dot = vdotq_s32(dot, w_hi, xv_hi);

            sumf += (float)vaddvq_s32(dot) * w_scale * x_scale;
        }

        y[row] = sumf;
    }
}

#endif // __ARM_FEATURE_MATMUL_INT8

// ─── Dispatch: i8mm when available, SDOT otherwise ───────────────────────────

#if defined(__ARM_FEATURE_MATMUL_INT8)
  #define q4_0_q8_0_matvec q4_0_q8_0_matvec_i8mm
#else
  #define q4_0_q8_0_matvec q4_0_q8_0_matvec_sdot
#endif

// ─── Row-parallel Q4_0 matvec with atomic work-stealing ─────────────────────
//
// Splits output rows across GCD threads. Each thread grabs MATVEC_CHUNK_ROWS
// rows at a time via atomic counter. Faster cores (P-cores) naturally steal
// more chunks. Input x is shared read-only (fits L1), each thread writes a
// disjoint slice of y.

#include <stdatomic.h>
#include <dispatch/dispatch.h>

// Adaptive chunk sizing: smaller chunks for large matrices (hidden dim 14336)
// to reduce per-chunk L2 footprint. For in_dim > 8192:
//   32 rows × 448 blocks/row × 18B = 258KB per chunk (vs 516KB at 64 rows)
// For dim-sized matrices (4096): 64 rows × 128 blocks/row × 18B = 147KB per chunk.
#define MATVEC_CHUNK_ROWS 64
#define MATVEC_CHUNK_ROWS_LARGE 32

static dispatch_queue_t _matvec_concurrent_q = NULL;

static inline dispatch_queue_t matvec_get_concurrent_queue(void) {
    if (!_matvec_concurrent_q) {
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INTERACTIVE, 0);
        _matvec_concurrent_q = dispatch_queue_create(
            "com.mistral.matvec_parallel", attr);
    }
    return _matvec_concurrent_q;
}

static void q4_0_matvec_f32_parallel(const void *W_q4, const float *x, float *y,
                                      int out_dim, int in_dim) {
    const int bpr = in_dim / QK4_0;
    const block_q4_0 *blocks = (const block_q4_0 *)W_q4;
    const int n_chunks = (out_dim + MATVEC_CHUNK_ROWS - 1) / MATVEC_CHUNK_ROWS;

    memset(y, 0, out_dim * sizeof(float));

    __block _Atomic int chunk_counter = 0;

    dispatch_queue_t q = matvec_get_concurrent_queue();

    dispatch_apply((size_t)n_chunks, q, ^(size_t _iter __attribute__((unused))) {
        int chunk;
        while ((chunk = atomic_fetch_add(&chunk_counter, 1)) < n_chunks) {
            int row_start = chunk * MATVEC_CHUNK_ROWS;
            int row_end = row_start + MATVEC_CHUNK_ROWS;
            if (row_end > out_dim) row_end = out_dim;

            int row = row_start;

            for (; row + 3 < row_end; row += 4) {
                const block_q4_0 *row0 = blocks + (row + 0) * bpr;
                const block_q4_0 *row1 = blocks + (row + 1) * bpr;
                const block_q4_0 *row2 = blocks + (row + 2) * bpr;
                const block_q4_0 *row3 = blocks + (row + 3) * bpr;

                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);

                for (int b = 0; b < bpr; b++) {
                    int j = b * QK4_0;

                    float32x4_t x0 = vld1q_f32(x + j + 0);
                    float32x4_t x1 = vld1q_f32(x + j + 4);
                    float32x4_t x2 = vld1q_f32(x + j + 8);
                    float32x4_t x3 = vld1q_f32(x + j + 12);
                    float32x4_t x4 = vld1q_f32(x + j + 16);
                    float32x4_t x5 = vld1q_f32(x + j + 20);
                    float32x4_t x6 = vld1q_f32(x + j + 24);
                    float32x4_t x7 = vld1q_f32(x + j + 28);

                    #define PAR_PROCESS_ROW(ROW_PTR, ACC) do { \
                        float scale = (float)(ROW_PTR)[b].d; \
                        float32x4_t vscale = vdupq_n_f32(scale); \
                        int8x16_t v8_ = vdupq_n_s8(8); \
                        uint8x16_t raw = vld1q_u8((ROW_PTR)[b].qs); \
                        \
                        int8x16_t lo = vsubq_s8( \
                            vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), v8_); \
                        int8x16_t hi = vsubq_s8( \
                            vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8_); \
                        \
                        int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo)); \
                        int16x8_t lo16_1 = vmovl_s8(vget_high_s8(lo)); \
                        \
                        float32x4_t w0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))), vscale); \
                        float32x4_t w1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0))), vscale); \
                        float32x4_t w2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))), vscale); \
                        float32x4_t w3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1))), vscale); \
                        \
                        ACC = vfmaq_f32(ACC, w0, x0); \
                        ACC = vfmaq_f32(ACC, w1, x1); \
                        ACC = vfmaq_f32(ACC, w2, x2); \
                        ACC = vfmaq_f32(ACC, w3, x3); \
                        \
                        int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi)); \
                        int16x8_t hi16_1 = vmovl_s8(vget_high_s8(hi)); \
                        \
                        float32x4_t w4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))), vscale); \
                        float32x4_t w5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_0))), vscale); \
                        float32x4_t w6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))), vscale); \
                        float32x4_t w7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_1))), vscale); \
                        \
                        ACC = vfmaq_f32(ACC, w4, x4); \
                        ACC = vfmaq_f32(ACC, w5, x5); \
                        ACC = vfmaq_f32(ACC, w6, x6); \
                        ACC = vfmaq_f32(ACC, w7, x7); \
                    } while(0)

                    PAR_PROCESS_ROW(row0, acc0);
                    PAR_PROCESS_ROW(row1, acc1);
                    PAR_PROCESS_ROW(row2, acc2);
                    PAR_PROCESS_ROW(row3, acc3);

                    #undef PAR_PROCESS_ROW
                }

                y[row + 0] = vaddvq_f32(acc0);
                y[row + 1] = vaddvq_f32(acc1);
                y[row + 2] = vaddvq_f32(acc2);
                y[row + 3] = vaddvq_f32(acc3);
            }

            for (; row < row_end; row++) {
                const block_q4_0 *rblk = blocks + row * bpr;
                float32x4_t acc = vdupq_n_f32(0.0f);

                for (int b = 0; b < bpr; b++) {
                    int j = b * QK4_0;
                    float scale = (float)rblk[b].d;
                    float32x4_t vscale = vdupq_n_f32(scale);
                    int8x16_t v8_ = vdupq_n_s8(8);
                    uint8x16_t raw = vld1q_u8(rblk[b].qs);

                    int8x16_t lo = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), v8_);
                    int8x16_t hi = vsubq_s8(
                        vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), v8_);

                    int16x8_t lo16_0 = vmovl_s8(vget_low_s8(lo));
                    int16x8_t lo16_1 = vmovl_s8(vget_high_s8(lo));

                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_0))), vscale), vld1q_f32(x + j + 0));
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_0))), vscale), vld1q_f32(x + j + 4));
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_1))), vscale), vld1q_f32(x + j + 8));
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_1))), vscale), vld1q_f32(x + j + 12));

                    int16x8_t hi16_0 = vmovl_s8(vget_low_s8(hi));
                    int16x8_t hi16_1 = vmovl_s8(vget_high_s8(hi));

                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_0))), vscale), vld1q_f32(x + j + 16));
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_0))), vscale), vld1q_f32(x + j + 20));
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_1))), vscale), vld1q_f32(x + j + 24));
                    acc = vfmaq_f32(acc, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_1))), vscale), vld1q_f32(x + j + 28));
                }

                y[row] = vaddvq_f32(acc);
            }
        }
    });
}

// ─── Row-parallel Q4_0/Q8_0 SDOT matvec with atomic work-stealing ───────────
//
// Same work-stealing pattern, but uses vdotq_s32 (SDOT) for int8 x int8 dot
// product. Input x is pre-quantized to Q8_0 blocks.

static void q4_0_q8_0_matvec_sdot_parallel(const void *W_q4, const block_q8_0 *x_q8,
                                             float *y, int out_dim, int in_dim) {
    const int bpr = in_dim / QK4_0;
    const block_q4_0 *blocks = (const block_q4_0 *)W_q4;
    // Use smaller chunks for large matrices (hidden dim 14336, bpr=448) to
    // reduce per-chunk L2 footprint: 32 rows × 448 × 18B = 258KB vs 516KB.
    const int chunk_rows = (in_dim > 8192) ? MATVEC_CHUNK_ROWS_LARGE : MATVEC_CHUNK_ROWS;
    const int n_chunks = (out_dim + chunk_rows - 1) / chunk_rows;

    memset(y, 0, out_dim * sizeof(float));

    __block _Atomic int chunk_counter = 0;

    dispatch_queue_t q = matvec_get_concurrent_queue();

    dispatch_apply((size_t)n_chunks, q, ^(size_t _iter __attribute__((unused))) {
        int chunk;
        while ((chunk = atomic_fetch_add(&chunk_counter, 1)) < n_chunks) {
            int row_start = chunk * chunk_rows;
            int row_end = row_start + chunk_rows;
            if (row_end > out_dim) row_end = out_dim;

            // Prefetch first few blocks of this chunk to warm L1
            __builtin_prefetch(&blocks[row_start * bpr].qs, 0, 1);
            if (row_start + 1 < row_end)
                __builtin_prefetch(&blocks[(row_start + 1) * bpr].qs, 0, 1);

            int row = row_start;

            for (; row + 3 < row_end; row += 4) {
                const block_q4_0 *r0 = blocks + (row + 0) * bpr;
                const block_q4_0 *r1 = blocks + (row + 1) * bpr;
                const block_q4_0 *r2 = blocks + (row + 2) * bpr;
                const block_q4_0 *r3 = blocks + (row + 3) * bpr;

                float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;

                for (int b = 0; b < bpr; b++) {
                    // Prefetch next block's weight data for all 4 rows
                    if (b + 1 < bpr) {
                        __builtin_prefetch(&r0[b+1].qs, 0, 1);
                        __builtin_prefetch(&r1[b+1].qs, 0, 1);
                        __builtin_prefetch(&r2[b+1].qs, 0, 1);
                        __builtin_prefetch(&r3[b+1].qs, 0, 1);
                    }

                    const int8x16_t xv0 = vld1q_s8(x_q8[b].qs);
                    const int8x16_t xv1 = vld1q_s8(x_q8[b].qs + 16);
                    const float xd = x_q8[b].d;

                    #define PAR_SDOT_ROW(RPTR, SUMF) do { \
                        float wd = (float)(RPTR)[b].d; \
                        uint8x16_t raw = vld1q_u8((RPTR)[b].qs); \
                        \
                        int8x16_t wlo = vsubq_s8( \
                            vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), \
                            vdupq_n_s8(8)); \
                        int8x16_t whi = vsubq_s8( \
                            vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), \
                            vdupq_n_s8(8)); \
                        \
                        int32x4_t isum = vdupq_n_s32(0); \
                        isum = vdotq_s32(isum, wlo, xv0); \
                        isum = vdotq_s32(isum, whi, xv1); \
                        \
                        SUMF += (wd * xd) * (float)vaddvq_s32(isum); \
                    } while(0)

                    PAR_SDOT_ROW(r0, sumf0);
                    PAR_SDOT_ROW(r1, sumf1);
                    PAR_SDOT_ROW(r2, sumf2);
                    PAR_SDOT_ROW(r3, sumf3);

                    #undef PAR_SDOT_ROW
                }

                y[row + 0] = sumf0;
                y[row + 1] = sumf1;
                y[row + 2] = sumf2;
                y[row + 3] = sumf3;
            }

            for (; row < row_end; row++) {
                const block_q4_0 *rblk = blocks + row * bpr;
                float sumf = 0.0f;

                for (int b = 0; b < bpr; b++) {
                    if (b + 1 < bpr)
                        __builtin_prefetch(&rblk[b+1].qs, 0, 1);

                    float wd = (float)rblk[b].d;
                    float xd = x_q8[b].d;

                    uint8x16_t raw = vld1q_u8(rblk[b].qs);
                    int8x16_t wlo = vsubq_s8(
                        vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))),
                        vdupq_n_s8(8));
                    int8x16_t whi = vsubq_s8(
                        vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)),
                        vdupq_n_s8(8));

                    int8x16_t xv0 = vld1q_s8(x_q8[b].qs);
                    int8x16_t xv1 = vld1q_s8(x_q8[b].qs + 16);

                    int32x4_t isum = vdupq_n_s32(0);
                    isum = vdotq_s32(isum, wlo, xv0);
                    isum = vdotq_s32(isum, whi, xv1);

                    sumf += (wd * xd) * (float)vaddvq_s32(isum);
                }

                y[row] = sumf;
            }
        }
    });
}
