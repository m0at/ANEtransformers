// mistral_ane_prefill.h — ANE baked-weight prefill + BLAS prefill for Mistral 7B
//
// ANE path: 128 compiled programs (4 per layer x 32 layers) with fp16 baked weights.
//   - Q: single conv1x1 [dim->dim]
//   - Fused K+V: 2 parallel conv1x1 [dim->kv_dim], 2 outputs (ANE 3-output broken)
//   - Wo: single conv1x1 [dim->dim]
//   - Fused FFN: W1->sigmoid->silu->W3->mul->W2 in one program (intermediates in ANE SRAM)
//   Weights dequanted Q4->fp16 at init, compiled once, dispatched during prefill.
//   RMSNorm, RoPE, attention stay on CPU (ANE lacks reduce_mean/rsqrt ops).
//
// 2-phase loading strategy:
//   Phase 1 (cold): compile all 128 programs, save hexIds to manifest (~80s for 7B)
//   Phase 2 (warm): forged-load from daemon cache using saved hexIds (~200ms)
//   All 128 kernels pre-loaded before any inference starts.
//
// BLAS path: tiled dequant + cblas_sgemm (AMX-accelerated). Fallback when ANE unavailable.
#pragma once

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "mistral_model.h"
#include "ane_mil_gen_mistral.h"
#include "../training/ane_runtime.h"
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>

static double ane_time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

// --- Shared helpers ----------------------------------------------------------

// Dequant any weight to fp16 buffer
static void dequant_weight_to_fp16_buf(const void *src, uint32_t type,
                                        _Float16 *dst, int rows, int cols) {
    if (type == GGML_TYPE_F16) {
        memcpy(dst, src, (size_t)rows * cols * sizeof(_Float16));
    } else if (type == GGML_TYPE_Q4_0) {
        dequant_q4_0_to_fp16(src, dst, rows, cols);
    } else if (type == GGML_TYPE_Q4_K) {
        dequant_q4_K_to_fp16(src, dst, rows, cols);
    } else if (type == GGML_TYPE_Q6_K) {
        const block_q6_K *blocks = (const block_q6_K *)src;
        int bpr = cols / 256;
        float tmp[256];
        for (int r = 0; r < rows; r++) {
            for (int b = 0; b < bpr; b++) {
                dequant_q6_K_block(&blocks[r * bpr + b], tmp);
                for (int i = 0; i < 256; i++)
                    dst[r * cols + b * 256 + i] = (_Float16)tmp[i];
            }
        }
    } else if (type == GGML_TYPE_F32) {
        const float *s = (const float *)src;
        for (int i = 0; i < rows * cols; i++) dst[i] = (_Float16)s[i];
    }
}

// fp32 -> fp16 conversion (NEON)
static void cvt_f32_to_f16(const float *src, _Float16 *dst, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float32x4_t a = vld1q_f32(src + i);
        float32x4_t b = vld1q_f32(src + i + 4);
        float16x4_t ha = vcvt_f16_f32(a);
        float16x4_t hb = vcvt_f16_f32(b);
        vst1_f16((float16_t *)(dst + i), ha);
        vst1_f16((float16_t *)(dst + i + 4), hb);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// fp16 -> fp32 conversion (NEON)
static void cvt_f16_to_f32(const _Float16 *src, float *dst, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float16x4_t ha = vld1_f16((const float16_t *)(src + i));
        float16x4_t hb = vld1_f16((const float16_t *)(src + i + 4));
        vst1q_f32(dst + i, vcvt_f32_f16(ha));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(hb));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}

// --- Transpose helpers (token-first <-> channel-first) -----------------------
// ANE uses channel-first layout: [1, C, 1, S] stored as data[c * S + s].
// Transformer code uses token-first: data[t * dim + d].

// Token-first [S, dim] -> Channel-first [dim, S] for ANE input
static void transpose_to_ane(const float *token_first, float *channel_first, int dim, int S) {
    // 4x4 block transpose for better cache behavior
    int c = 0;
    for (; c + 3 < dim; c += 4) {
        int s = 0;
        for (; s + 3 < S; s += 4) {
            // Load 4x4 block from token-first (row = token, col = channel)
            float32x4_t r0 = vld1q_f32(token_first + (s + 0) * dim + c);
            float32x4_t r1 = vld1q_f32(token_first + (s + 1) * dim + c);
            float32x4_t r2 = vld1q_f32(token_first + (s + 2) * dim + c);
            float32x4_t r3 = vld1q_f32(token_first + (s + 3) * dim + c);
            // Transpose 4x4
            float32x4x2_t t01 = vtrnq_f32(r0, r1);
            float32x4x2_t t23 = vtrnq_f32(r2, r3);
            float32x4_t o0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            float32x4_t o1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            float32x4_t o2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            float32x4_t o3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
            // Store to channel-first (row = channel, col = seq)
            vst1q_f32(channel_first + (c + 0) * S + s, o0);
            vst1q_f32(channel_first + (c + 1) * S + s, o1);
            vst1q_f32(channel_first + (c + 2) * S + s, o2);
            vst1q_f32(channel_first + (c + 3) * S + s, o3);
        }
        for (; s < S; s++) {
            channel_first[(c + 0) * S + s] = token_first[s * dim + c + 0];
            channel_first[(c + 1) * S + s] = token_first[s * dim + c + 1];
            channel_first[(c + 2) * S + s] = token_first[s * dim + c + 2];
            channel_first[(c + 3) * S + s] = token_first[s * dim + c + 3];
        }
    }
    for (; c < dim; c++)
        for (int s = 0; s < S; s++)
            channel_first[c * S + s] = token_first[s * dim + c];
}

// Channel-first [dim, S] -> Token-first [S, dim] for ANE output
static void transpose_from_ane(const float *channel_first, float *token_first, int dim, int S) {
    int c = 0;
    for (; c + 3 < dim; c += 4) {
        int s = 0;
        for (; s + 3 < S; s += 4) {
            float32x4_t r0 = vld1q_f32(channel_first + (c + 0) * S + s);
            float32x4_t r1 = vld1q_f32(channel_first + (c + 1) * S + s);
            float32x4_t r2 = vld1q_f32(channel_first + (c + 2) * S + s);
            float32x4_t r3 = vld1q_f32(channel_first + (c + 3) * S + s);
            float32x4x2_t t01 = vtrnq_f32(r0, r1);
            float32x4x2_t t23 = vtrnq_f32(r2, r3);
            float32x4_t o0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            float32x4_t o1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            float32x4_t o2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            float32x4_t o3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
            vst1q_f32(token_first + (s + 0) * dim + c, o0);
            vst1q_f32(token_first + (s + 1) * dim + c, o1);
            vst1q_f32(token_first + (s + 2) * dim + c, o2);
            vst1q_f32(token_first + (s + 3) * dim + c, o3);
        }
        for (; s < S; s++) {
            token_first[s * dim + c + 0] = channel_first[(c + 0) * S + s];
            token_first[s * dim + c + 1] = channel_first[(c + 1) * S + s];
            token_first[s * dim + c + 2] = channel_first[(c + 2) * S + s];
            token_first[s * dim + c + 3] = channel_first[(c + 3) * S + s];
        }
    }
    for (; c < dim; c++)
        for (int s = 0; s < S; s++)
            token_first[s * dim + c] = channel_first[c * S + s];
}

// RMSNorm for S tokens
static void rmsnorm_batch(float *out, const float *in, const float *w,
                           int dim, int S, float eps) {
    for (int t = 0; t < S; t++) {
        const float *xi = in + t * dim;
        float *xo = out + t * dim;
        float ss = 0;
        vDSP_dotpr(xi, 1, xi, 1, &ss, (vDSP_Length)dim);
        ss = 1.0f / sqrtf(ss / dim + eps);
        vDSP_vsmul(xi, 1, &ss, xo, 1, (vDSP_Length)dim);
        vDSP_vmul(xo, 1, w, 1, xo, 1, (vDSP_Length)dim);
    }
}

// RoPE for batch
static void apply_rope_batch(float *Q, float *K, int start_pos,
                              const float *theta_inv,
                              int n_heads, int n_kv_heads, int head_dim, int S) {
    for (int t = 0; t < S; t++) {
        apply_rope(Q + t * (n_heads * head_dim),
                   K + t * (n_kv_heads * head_dim),
                   start_pos + t, theta_inv,
                   n_heads, n_kv_heads, head_dim);
    }
}

// Write K,V batch to cache
static void kv_write_batch(KVCache *kv, int layer, int start_pos,
                            const float *K_buf, const float *V_buf,
                            int kv_dim, int S) {
    for (int t = 0; t < S; t++) {
        int pos = start_pos + t;
        const float *kt = K_buf + t * kv_dim;
        const float *vt = V_buf + t * kv_dim;
        _Float16 k16[kv_dim], v16[kv_dim];
        cvt_f32_to_f16(kt, k16, kv_dim);
        cvt_f32_to_f16(vt, v16, kv_dim);
        kv_write(kv, layer, pos % kv->max_seq, k16, v16);
    }
    int end_pos = start_pos + S;
    if (end_pos > kv->len && end_pos <= kv->max_seq) kv->len = end_pos;
    else if (end_pos > kv->max_seq) kv->len = kv->max_seq;
}

// Causal attention for S query tokens against KV cache
static void attention_batch(const float *Q, float *attn_out,
                             const KVCache *kv, int layer, int start_pos,
                             const MistralConfig *c, int S, float *att_scratch) {
    int dim = c->dim;
    int hd = c->head_dim;
    int n_heads = c->n_heads;
    int n_kv = c->n_kv_heads;
    int heads_per_kv = n_heads / n_kv;
    int max_seq = kv->max_seq;

    memset(attn_out, 0, (size_t)dim * S * sizeof(float));

    for (int t = 0; t < S; t++) {
        int pos = start_pos + t;
        int seq_len = pos + 1;
        if (seq_len > max_seq) seq_len = max_seq;

        const float *qt = Q + t * dim;
        float *out_t = attn_out + t * dim;

        int kv_dim_ab = n_kv * hd;
        _Float16 *kcache_base_ab = kv_k(kv, layer);
        _Float16 *vcache_base_ab = kv_v(kv, layer);

        for (int h = 0; h < n_heads; h++) {
            const float *qh = qt + h * hd;
            int kvh = h / heads_per_kv;

            float *att_h = att_scratch + (size_t)h * max_seq;
            float scale = 1.0f / sqrtf((float)hd);

            for (int s = 0; s < seq_len; s++) {
                int cache_s = s;
                if (pos + 1 > max_seq)
                    cache_s = (pos + 1 - seq_len + s) % max_seq;
                const _Float16 *ks = kcache_base_ab + (size_t)cache_s * kv_dim_ab + kvh * hd;
                float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
                for (int d = 0; d < hd; d += 8) {
                    float16x4_t k0 = vld1_f16((const float16_t *)(ks + d));
                    float16x4_t k1 = vld1_f16((const float16_t *)(ks + d + 4));
                    acc0 = vfmaq_f32(acc0, vld1q_f32(qh + d),     vcvt_f32_f16(k0));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(qh + d + 4), vcvt_f32_f16(k1));
                }
                att_h[s] = vaddvq_f32(vaddq_f32(acc0, acc1)) * scale;
            }

            softmax_vec(att_h, seq_len);

            float *oh = out_t + h * hd;
            for (int s = 0; s < seq_len; s++) {
                float a = att_h[s];
                if (a == 0) continue;
                int cache_s = s;
                if (pos + 1 > max_seq)
                    cache_s = (pos + 1 - seq_len + s) % max_seq;
                const _Float16 *vs = vcache_base_ab + (size_t)cache_s * kv_dim_ab + kvh * hd;
                float32x4_t av = vdupq_n_f32(a);
                for (int d = 0; d < hd; d += 8) {
                    float16x4_t v0 = vld1_f16((const float16_t *)(vs + d));
                    float16x4_t v1 = vld1_f16((const float16_t *)(vs + d + 4));
                    float32x4_t o0 = vld1q_f32(oh + d);
                    float32x4_t o1 = vld1q_f32(oh + d + 4);
                    vst1q_f32(oh + d,     vfmaq_f32(o0, av, vcvt_f32_f16(v0)));
                    vst1q_f32(oh + d + 4, vfmaq_f32(o1, av, vcvt_f32_f16(v1)));
                }
            }
        }
    }
}

// =============================================================================
// ANE BAKED-WEIGHT PREFILL — 2-Phase Loading
//
// Phase 1 (cold start): compile all 128 programs, save hexIds to manifest
//   - Dequant Q4->fp16, build DEADBEEF blobs, compile via ane daemon
//   - ~80s for 7B (32 layers x 4 programs), ~25s for 3B
//   - hexIds saved to ~/.cache/ane_mistral/manifest.plist
//
// Phase 2 (warm start): forged-load from daemon cache
//   - Dummy model + KVC forge + loadWithQoS = ~200ms for 128 programs
//   - No dequant, no blob building, no compilation
//
// All 128 kernels pre-loaded before any inference starts.
// Layer loop: just dispatch (no per-layer compile/free).
//
// 4 programs per layer (Q, fused K+V, Wo, fused FFN).
// 32x4 = 128 total programs. K+V fused into single 2-output program.
// =============================================================================

#define ANE_PREFILL_TILE 64  // tokens per ANE dispatch

typedef enum {
    ANE_LK_Q = 0,    // Wq conv
    ANE_LK_KV,       // Fused Wk+Wv conv (2 outputs)
    ANE_LK_WO,       // Wo conv
    ANE_LK_FFN,      // Fused W1+W3+W2+sigmoid+mul
    ANE_LK_COUNT      // = 4 programs per layer, 32x4 = 128 total
} ANELayerKernelType;

typedef struct {
    int tile_size;

    // fp32 scratch (allocated once, reused per layer)
    float *X;         // [dim * S]
    float *Q;         // [dim * S]
    float *K;         // [kv_dim * S]
    float *V;         // [kv_dim * S]
    float *Wo_out;    // [dim * S]
    float *down_out;  // [dim * S]
    float *att;       // [n_heads * max_seq]

    // fp16 scratch for weight dequant (used during phase 1 compile)
    _Float16 *w16[2]; // [max(hidden*dim, dim*dim)] x 2

    // Transpose scratch buffers (token-first <-> channel-first)
    float *ane_in;     // [dim * S] for transpose before ANE input
    float *ane_out;    // [dim * S] for transpose after ANE output
    float *ane_out_kv; // [kv_dim * S] for K/V output transpose

    // Pre-generated MIL text (shared across layers)
    NSData *milData_q;
    NSData *milData_kv;
    NSData *milData_wo;
    NSData *milData_ffn;

    // I/O sizes
    size_t in_dim_S;
    size_t out_dim_S;
    size_t out_kv_S;

    // Pre-loaded kernels for all layers (2-phase)
    ANEKernel *kernels[32][ANE_LK_COUNT];
    bool kernels_loaded;

    bool initialized;
} ANEBakedPrefillState;

static ANEBakedPrefillState g_ane_baked = {0};

static void ane_free_layer(ANEKernel *kernels[ANE_LK_COUNT]) {
    for (int k = 0; k < ANE_LK_COUNT; k++) {
        if (kernels[k]) { ane_free_ex(kernels[k], true); kernels[k] = NULL; }
    }
}

// Compile a single layer's 4 programs (Q, KV, Wo, FFN) from model weights.
// Used by tests. For production use ane_compile_all_programs() instead.
static bool ane_compile_layer(MistralModel *m, int layer, ANEKernel *out[ANE_LK_COUNT]) {
    @autoreleasepool {
        MistralConfig *c = &m->cfg;
        int dim = c->dim, kv_dim = c->n_kv_heads * c->head_dim, hidden = c->hidden_dim;
        int S = ANE_PREFILL_TILE;
        LayerWeights *lw = &m->layers[layer];
        memset(out, 0, ANE_LK_COUNT * sizeof(ANEKernel*));

        size_t in_sz = (size_t)dim * S * sizeof(float);
        size_t out_dim_sz = in_sz;
        size_t out_kv_sz = (size_t)kv_dim * S * sizeof(float);
        size_t kv_outSizes[] = {out_kv_sz, out_kv_sz};

        _Float16 *w16 = (_Float16 *)malloc((size_t)hidden * dim * sizeof(_Float16));

        // Q
        dequant_weight_to_fp16_buf(lw->wq, lw->wq_type, w16, dim, dim);
        NSData *blob_q = mil_build_single_weight_blob(w16, dim, dim);
        NSString *mil_q = mil_gen_conv_baked(dim, dim, S);
        NSData *milData_q = [mil_q dataUsingEncoding:NSUTF8StringEncoding];
        out[ANE_LK_Q] = ane_compile(milData_q, blob_q, 1, &in_sz, 1, &out_dim_sz);

        // KV
        _Float16 *wk16 = (_Float16 *)malloc((size_t)kv_dim * dim * sizeof(_Float16));
        _Float16 *wv16 = (_Float16 *)malloc((size_t)kv_dim * dim * sizeof(_Float16));
        dequant_weight_to_fp16_buf(lw->wk, lw->wk_type, wk16, kv_dim, dim);
        dequant_weight_to_fp16_buf(lw->wv, lw->wv_type, wv16, kv_dim, dim);
        NSData *blob_kv = mil_build_kv_fused_blob(wk16, wv16, kv_dim, dim);
        NSString *mil_kv = mil_gen_kv_fused(dim, kv_dim, S);
        NSData *milData_kv = [mil_kv dataUsingEncoding:NSUTF8StringEncoding];
        out[ANE_LK_KV] = ane_compile(milData_kv, blob_kv, 1, &in_sz, 2, kv_outSizes);
        free(wk16); free(wv16);

        // Wo
        dequant_weight_to_fp16_buf(lw->wo, lw->wo_type, w16, dim, dim);
        NSData *blob_wo = mil_build_single_weight_blob(w16, dim, dim);
        NSString *mil_wo = mil_gen_conv_baked(dim, dim, S);
        NSData *milData_wo = [mil_wo dataUsingEncoding:NSUTF8StringEncoding];
        out[ANE_LK_WO] = ane_compile(milData_wo, blob_wo, 1, &in_sz, 1, &out_dim_sz);

        // FFN
        _Float16 *w1 = (_Float16 *)malloc((size_t)hidden * dim * sizeof(_Float16));
        _Float16 *w3 = (_Float16 *)malloc((size_t)hidden * dim * sizeof(_Float16));
        _Float16 *w2 = (_Float16 *)malloc((size_t)dim * hidden * sizeof(_Float16));
        dequant_weight_to_fp16_buf(lw->w1, lw->w1_type, w1, hidden, dim);
        dequant_weight_to_fp16_buf(lw->w3, lw->w3_type, w3, hidden, dim);
        dequant_weight_to_fp16_buf(lw->w2, lw->w2_type, w2, dim, hidden);
        NSData *blob_ffn = mil_build_ffn_fused_blob(w1, w3, hidden, w2, dim);
        NSString *mil_ffn = mil_gen_ffn_fused(dim, hidden, S);
        NSData *milData_ffn = [mil_ffn dataUsingEncoding:NSUTF8StringEncoding];
        out[ANE_LK_FFN] = ane_compile(milData_ffn, blob_ffn, 1, &in_sz, 1, &out_dim_sz);
        free(w1); free(w3); free(w2);

        free(w16);

        for (int k = 0; k < ANE_LK_COUNT; k++) {
            if (!out[k]) { ane_free_layer(out); return false; }
        }
        return true;
    }
}

static void ane_baked_prefill_cleanup(void) {
    ANEBakedPrefillState *st = &g_ane_baked;
    if (!st->initialized) return;
    // Free all pre-loaded kernels
    if (st->kernels_loaded) {
        for (int l = 0; l < 32; l++)
            ane_free_layer(st->kernels[l]);
        st->kernels_loaded = false;
    }
    free(st->X); free(st->Q); free(st->K); free(st->V);
    free(st->Wo_out); free(st->down_out); free(st->att);
    free(st->w16[0]); free(st->w16[1]);
    free(st->ane_in); free(st->ane_out); free(st->ane_out_kv);
    st->milData_q = nil; st->milData_kv = nil;
    st->milData_wo = nil; st->milData_ffn = nil;
    memset((void*)st, 0, sizeof(ANEBakedPrefillState));
}

static bool ane_baked_prefill_init(MistralModel *m, int tile_size, int max_seq) {
    @autoreleasepool {
        ANEBakedPrefillState *st = &g_ane_baked;
        if (st->initialized) {
            if (st->tile_size == tile_size) return true;
            ane_baked_prefill_cleanup();
        }

        MistralConfig *c = &m->cfg;
        int dim = c->dim;
        int kv_dim = c->n_kv_heads * c->head_dim;
        int hidden = c->hidden_dim;
        int S = tile_size;

        // Generate MIL text for each kernel shape
        NSString *mil_q   = mil_gen_conv_baked(dim, dim, S);
        NSString *mil_kv  = mil_gen_kv_fused(dim, kv_dim, S);
        NSString *mil_wo  = mil_gen_conv_baked(dim, dim, S);
        NSString *mil_ffn = mil_gen_ffn_fused(dim, hidden, S);

        st->milData_q   = [mil_q dataUsingEncoding:NSUTF8StringEncoding];
        st->milData_kv  = [mil_kv dataUsingEncoding:NSUTF8StringEncoding];
        st->milData_wo  = [mil_wo dataUsingEncoding:NSUTF8StringEncoding];
        st->milData_ffn = [mil_ffn dataUsingEncoding:NSUTF8StringEncoding];

        st->in_dim_S  = (size_t)dim * S * sizeof(float);
        st->out_dim_S = (size_t)dim * S * sizeof(float);
        st->out_kv_S  = (size_t)kv_dim * S * sizeof(float);

        // Scratch for weight dequant (phase 1)
        size_t max_w16 = (size_t)hidden * dim;
        st->w16[0]   = (_Float16 *)malloc(max_w16 * sizeof(_Float16));
        st->w16[1]   = (_Float16 *)malloc(max_w16 * sizeof(_Float16));

        // Main scratch buffers
        st->X        = (float *)calloc((size_t)dim * S, sizeof(float));
        st->Q        = (float *)calloc((size_t)dim * S, sizeof(float));
        st->K        = (float *)calloc((size_t)kv_dim * S, sizeof(float));
        st->V        = (float *)calloc((size_t)kv_dim * S, sizeof(float));
        st->Wo_out   = (float *)calloc((size_t)dim * S, sizeof(float));
        st->down_out = (float *)calloc((size_t)dim * S, sizeof(float));
        st->att      = (float *)calloc((size_t)c->n_heads * max_seq, sizeof(float));

        // Transpose scratch buffers
        st->ane_in     = (float *)calloc((size_t)dim * S, sizeof(float));
        st->ane_out    = (float *)calloc((size_t)dim * S, sizeof(float));
        st->ane_out_kv = (float *)calloc((size_t)kv_dim * S, sizeof(float));

        memset(st->kernels, 0, sizeof(st->kernels));
        st->kernels_loaded = false;

        st->tile_size = S;
        st->initialized = true;

        fprintf(stderr, "[ANE] 2-phase prefill initialized (S=%d)\n", S);
        return true;
    }
}

// --- Blob disk cache ---------------------------------------------------------
// Pre-materialized fp16 DEADBEEF blobs saved to ~/.cache/ane_mistral/
// On cache hit: mmap from disk (~0.1ms) instead of dequant+build (~150ms)

static NSString *g_ane_blob_cache_dir = nil;

static NSString *ane_blob_cache_dir(void) {
    if (!g_ane_blob_cache_dir) {
        g_ane_blob_cache_dir = [NSHomeDirectory() stringByAppendingPathComponent:@".cache/ane_mistral"];
    }
    return g_ane_blob_cache_dir;
}

static NSString *ane_blob_path(int layer, const char *name) {
    return [NSString stringWithFormat:@"%@/L%02d_%s.bin", ane_blob_cache_dir(), layer, name];
}

static bool ane_blob_cache_exists(int layer) {
    return [[NSFileManager defaultManager] fileExistsAtPath:ane_blob_path(layer, "kv")];
}

static void ane_blob_cache_save(NSData *blob, int layer, const char *name) {
    NSString *path = ane_blob_path(layer, name);
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[path stringByDeletingLastPathComponent]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [blob writeToFile:path atomically:YES];
}

static NSData *ane_blob_cache_load(int layer, const char *name) {
    return [NSData dataWithContentsOfFile:ane_blob_path(layer, name)
                                 options:NSDataReadingMappedIfSafe
                                   error:nil];
}

// --- HexId Manifest (save/load) ----------------------------------------------
// Phase 1 saves hexIds; Phase 2 loads them for forged loading.

typedef struct {
    NSString *hexIds[32][ANE_LK_COUNT]; // [layer][program_type]
    int n_layers;
    bool valid;
} ANEPrefillManifest;

static NSString *ane_manifest_path(void) {
    return [ane_blob_cache_dir() stringByAppendingPathComponent:@"manifest.plist"];
}

static bool ane_manifest_save(ANEPrefillManifest *mf) {
    @autoreleasepool {
        NSMutableDictionary *dict = [NSMutableDictionary new];
        dict[@"n_layers"] = @(mf->n_layers);
        NSMutableArray *layers = [NSMutableArray new];
        for (int l = 0; l < mf->n_layers; l++) {
            NSMutableArray *progs = [NSMutableArray new];
            for (int p = 0; p < ANE_LK_COUNT; p++)
                [progs addObject:mf->hexIds[l][p] ?: @""];
            [layers addObject:progs];
        }
        dict[@"hexIds"] = layers;
        NSString *path = ane_manifest_path();
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[path stringByDeletingLastPathComponent]
            withIntermediateDirectories:YES attributes:nil error:nil];
        return [dict writeToFile:path atomically:YES];
    }
}

static bool ane_manifest_load(ANEPrefillManifest *mf) {
    @autoreleasepool {
        NSDictionary *dict = [NSDictionary dictionaryWithContentsOfFile:ane_manifest_path()];
        if (!dict) return false;
        mf->n_layers = [dict[@"n_layers"] intValue];
        NSArray *layers = dict[@"hexIds"];
        if (!layers || layers.count != (NSUInteger)mf->n_layers) return false;
        for (int l = 0; l < mf->n_layers; l++) {
            NSArray *progs = layers[l];
            if (progs.count != ANE_LK_COUNT) return false;
            for (int p = 0; p < ANE_LK_COUNT; p++) {
                NSString *h = progs[p];
                if (h.length == 0) return false;
                mf->hexIds[l][p] = h;
            }
        }
        mf->valid = true;
        return true;
    }
}

// --- Phase 1: Compile all programs -------------------------------------------
// Stream-compile all 128 programs, save hexIds to manifest.
// Uses ane_compile_and_get_hexid() from ane_runtime.h.

static bool ane_compile_all_programs(MistralModel *m, int tile_size, ANEPrefillManifest *mf) {
    ANEBakedPrefillState *st = &g_ane_baked;
    MistralConfig *c = &m->cfg;
    int dim = c->dim, kv_dim = c->n_kv_heads * c->head_dim, hidden = c->hidden_dim;
    (void)tile_size; // tile_size already baked into MIL text at init

    _Float16 *w16 = st->w16[0]; // reuse existing scratch
    mf->n_layers = c->n_layers;

    double t_total = ane_time_ms();
    for (int l = 0; l < c->n_layers; l++) {
        @autoreleasepool {
            LayerWeights *lw = &m->layers[l];
            double t0 = ane_time_ms();

            NSData *blob_q, *blob_kv, *blob_wo, *blob_ffn;

            // Fast path: load pre-materialized blobs from disk cache
            if (ane_blob_cache_exists(l)) {
                blob_q   = ane_blob_cache_load(l, "q");
                blob_kv  = ane_blob_cache_load(l, "kv");
                blob_wo  = ane_blob_cache_load(l, "wo");
                blob_ffn = ane_blob_cache_load(l, "ffn");
            } else {
                // Slow path: dequant + build + save to cache

                // Q blob
                dequant_weight_to_fp16_buf(lw->wq, lw->wq_type, w16, dim, dim);
                blob_q = mil_build_single_weight_blob(w16, dim, dim);

                // KV blob
                _Float16 *wk16 = (_Float16 *)malloc((size_t)kv_dim * dim * sizeof(_Float16));
                _Float16 *wv16 = (_Float16 *)malloc((size_t)kv_dim * dim * sizeof(_Float16));
                dequant_weight_to_fp16_buf(lw->wk, lw->wk_type, wk16, kv_dim, dim);
                dequant_weight_to_fp16_buf(lw->wv, lw->wv_type, wv16, kv_dim, dim);
                blob_kv = mil_build_kv_fused_blob(wk16, wv16, kv_dim, dim);
                free(wk16); free(wv16);

                // Wo blob
                dequant_weight_to_fp16_buf(lw->wo, lw->wo_type, w16, dim, dim);
                blob_wo = mil_build_single_weight_blob(w16, dim, dim);

                // FFN blob
                _Float16 *w1_16 = (_Float16 *)malloc((size_t)hidden * dim * sizeof(_Float16));
                _Float16 *w3_16 = (_Float16 *)malloc((size_t)hidden * dim * sizeof(_Float16));
                _Float16 *w2_16 = (_Float16 *)malloc((size_t)dim * hidden * sizeof(_Float16));
                dequant_weight_to_fp16_buf(lw->w1, lw->w1_type, w1_16, hidden, dim);
                dequant_weight_to_fp16_buf(lw->w3, lw->w3_type, w3_16, hidden, dim);
                dequant_weight_to_fp16_buf(lw->w2, lw->w2_type, w2_16, dim, hidden);
                blob_ffn = mil_build_ffn_fused_blob(w1_16, w3_16, hidden, w2_16, dim);
                free(w1_16); free(w3_16); free(w2_16);

                // Save blobs to disk cache
                ane_blob_cache_save(blob_q, l, "q");
                ane_blob_cache_save(blob_kv, l, "kv");
                ane_blob_cache_save(blob_wo, l, "wo");
                ane_blob_cache_save(blob_ffn, l, "ffn");
            }

            // Compile and get hexIds
            mf->hexIds[l][ANE_LK_Q]   = ane_compile_and_get_hexid(st->milData_q, blob_q);
            mf->hexIds[l][ANE_LK_KV]  = ane_compile_and_get_hexid(st->milData_kv, blob_kv);
            mf->hexIds[l][ANE_LK_WO]  = ane_compile_and_get_hexid(st->milData_wo, blob_wo);
            mf->hexIds[l][ANE_LK_FFN] = ane_compile_and_get_hexid(st->milData_ffn, blob_ffn);

            double dt = ane_time_ms() - t0;
            fprintf(stderr, "  [L%02d] compiled 4 programs (%.0fms)\n", l, dt);

            for (int p = 0; p < ANE_LK_COUNT; p++) {
                if (!mf->hexIds[l][p]) {
                    fprintf(stderr, "[ANE] Layer %d program %d compile failed\n", l, p);
                    return false;
                }
            }
        }
    }

    mf->valid = true;
    ane_manifest_save(mf);

    fprintf(stderr, "[ANE] Compiled all %d programs in %.1fs\n",
            c->n_layers * ANE_LK_COUNT, (ane_time_ms() - t_total) / 1000.0);
    return true;
}

// --- Phase 2: Load all forged ------------------------------------------------
// Load all 128 programs from daemon cache via forged hexIds.

static bool ane_load_all_forged(ANEBakedPrefillState *st, ANEPrefillManifest *mf) {
    double t0 = ane_time_ms();
    size_t kv_outSizes[] = {st->out_kv_S, st->out_kv_S};

    for (int l = 0; l < mf->n_layers; l++) {
        @autoreleasepool {
            st->kernels[l][ANE_LK_Q] = ane_load_forged(
                st->milData_q, mf->hexIds[l][ANE_LK_Q],
                1, &st->in_dim_S, 1, &st->out_dim_S);

            st->kernels[l][ANE_LK_KV] = ane_load_forged(
                st->milData_kv, mf->hexIds[l][ANE_LK_KV],
                1, &st->in_dim_S, 2, kv_outSizes);

            st->kernels[l][ANE_LK_WO] = ane_load_forged(
                st->milData_wo, mf->hexIds[l][ANE_LK_WO],
                1, &st->in_dim_S, 1, &st->out_dim_S);

            st->kernels[l][ANE_LK_FFN] = ane_load_forged(
                st->milData_ffn, mf->hexIds[l][ANE_LK_FFN],
                1, &st->in_dim_S, 1, &st->out_dim_S);

            for (int p = 0; p < ANE_LK_COUNT; p++) {
                if (!st->kernels[l][p]) {
                    fprintf(stderr, "[ANE] Forged load failed L%d P%d -- cache miss, need recompile\n", l, p);
                    return false;
                }
            }
        }
    }

    fprintf(stderr, "[ANE] Forged load: %d programs in %.1fms\n",
            mf->n_layers * ANE_LK_COUNT, ane_time_ms() - t0);
    return true;
}

// Process one layer with pre-loaded ANE kernels (with transpose)
// S_ane: ANE compile-time tile size (always ANE_PREFILL_TILE, e.g. 64)
// n_tok: actual number of valid tokens in X (may be < S_ane, rest is zero-padded)
static void ane_baked_prefill_layer(MistralModel *m, KVCache *kv, float *X,
                                     ANEKernel *lk[ANE_LK_COUNT],
                                     int layer, int start_pos, int S_ane, int n_tok) {
    ANEBakedPrefillState *st = &g_ane_baked;
    MistralConfig *c = &m->cfg;
    LayerWeights *lw = &m->layers[layer];
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int S = S_ane; // ANE always operates on full tile

    size_t dim_S_bytes = (size_t)dim * S * sizeof(float);
    size_t kv_S_bytes = (size_t)kv_dim * S * sizeof(float);

    // ANE ops use full tile S (compile-time). CPU ops use n_tok (actual tokens).

    // 1. Attention RMSNorm (CPU) — only n_tok tokens need normalization
    rmsnorm_batch(st->X, X, lw->rms_att, dim, S, c->rms_eps);

    // 2. Q projection (ANE) — transpose full tile, dispatch, transpose back
    transpose_to_ane(st->X, st->ane_in, dim, S);
    ane_write_input(lk[ANE_LK_Q], 0, st->ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_Q]);
    ane_read_output(lk[ANE_LK_Q], 0, st->ane_out, dim_S_bytes);
    transpose_from_ane(st->ane_out, st->Q, dim, S);

    // 3. Fused K+V projection (ANE) — same transposed input, 2 outputs
    ane_write_input(lk[ANE_LK_KV], 0, st->ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_KV]);
    ane_read_output(lk[ANE_LK_KV], 0, st->ane_out_kv, kv_S_bytes);
    transpose_from_ane(st->ane_out_kv, st->K, kv_dim, S);
    ane_read_output(lk[ANE_LK_KV], 1, st->ane_out_kv, kv_S_bytes);
    transpose_from_ane(st->ane_out_kv, st->V, kv_dim, S);

    // 4. RoPE (CPU) — only n_tok actual tokens
    apply_rope_batch(st->Q, st->K, start_pos,
                     m->rope_theta_inv,
                     c->n_heads, c->n_kv_heads, c->head_dim, n_tok);

    // 5. Write K,V to cache (CPU) — only n_tok tokens
    kv_write_batch(kv, layer, start_pos, st->K, st->V, kv_dim, n_tok);

    // 6. Attention (CPU) — only n_tok query tokens
    attention_batch(st->Q, st->Wo_out, kv, layer, start_pos, c, n_tok, st->att);

    // 7. Wo projection (ANE) — need to zero-pad Wo_out for positions n_tok..S-1
    if (n_tok < S)
        memset(st->Wo_out + n_tok * dim, 0, (size_t)(S - n_tok) * dim * sizeof(float));
    transpose_to_ane(st->Wo_out, st->ane_in, dim, S);
    ane_write_input(lk[ANE_LK_WO], 0, st->ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_WO]);
    ane_read_output(lk[ANE_LK_WO], 0, st->ane_out, dim_S_bytes);
    transpose_from_ane(st->ane_out, st->X, dim, S);

    // 8. Residual (CPU)
    vDSP_vadd(X, 1, st->X, 1, X, 1, (vDSP_Length)(dim * S));

    // 9. FFN RMSNorm (CPU)
    rmsnorm_batch(st->X, X, lw->rms_ffn, dim, S, c->rms_eps);

    // 10. Fused FFN (ANE) — W1->sigmoid->silu->W3->mul->W2 in one dispatch
    transpose_to_ane(st->X, st->ane_in, dim, S);
    ane_write_input(lk[ANE_LK_FFN], 0, st->ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_FFN]);
    ane_read_output(lk[ANE_LK_FFN], 0, st->ane_out, dim_S_bytes);
    transpose_from_ane(st->ane_out, st->down_out, dim, S);

    // 11. Residual (CPU)
    vDSP_vadd(X, 1, st->down_out, 1, X, 1, (vDSP_Length)(dim * S));
}

static bool g_ane_baked_tried_and_failed = false;

// 2-phase ANE baked-weight prefill
static bool ane_baked_prefill_forward(MistralModel *m, KVCache *kv,
                                       const int *tokens, int n_tokens, float *x_out) {
    if (g_ane_baked_tried_and_failed) return false;
    @autoreleasepool {
        MistralConfig *c = &m->cfg;
        int dim = c->dim;
        int S = ANE_PREFILL_TILE;

        if (!ane_baked_prefill_init(m, S, kv->max_seq)) {
            g_ane_baked_tried_and_failed = true;
            return false;
        }

        ANEBakedPrefillState *st = &g_ane_baked;

        // -- 2-phase kernel loading --
        if (!st->kernels_loaded) {
            ANEPrefillManifest mf = {0};
            bool loaded = false;

            // Phase 2 attempt: try forged load from manifest
            if (ane_manifest_load(&mf)) {
                fprintf(stderr, "[ANE] Manifest found (%d layers), attempting forged load...\n", mf.n_layers);
                if (mf.n_layers == c->n_layers) {
                    loaded = ane_load_all_forged(st, &mf);
                    if (!loaded) {
                        fprintf(stderr, "[ANE] Forged load failed (cache miss), recompiling...\n");
                        // Clean up any partially loaded kernels
                        for (int l = 0; l < 32; l++)
                            ane_free_layer(st->kernels[l]);
                    }
                } else {
                    fprintf(stderr, "[ANE] Manifest layer count mismatch (%d vs %d), recompiling...\n",
                            mf.n_layers, c->n_layers);
                }
            }

            // Phase 1: compile all programs (cold start or cache miss)
            if (!loaded) {
                fprintf(stderr, "[ANE] Compiling all %d programs (first run)...\n",
                        c->n_layers * ANE_LK_COUNT);
                memset((void *)&mf, 0, sizeof(mf));
                if (!ane_compile_all_programs(m, S, &mf)) {
                    g_ane_baked_tried_and_failed = true;
                    return false;
                }
                loaded = ane_load_all_forged(st, &mf);
                if (!loaded) {
                    fprintf(stderr, "[ANE] Forged load after compile failed\n");
                    g_ane_baked_tried_and_failed = true;
                    return false;
                }
            }
            st->kernels_loaded = true;
        }

        // -- Prefill with pre-loaded kernels --
        float *X = (float *)calloc((size_t)dim * S, sizeof(float));
        double t_total = ane_time_ms();

        for (int pos = 0; pos < n_tokens; pos += S) {
            int tile = n_tokens - pos;
            if (tile > S) tile = S;

            for (int t = 0; t < tile; t++)
                embed_token(m, tokens[pos + t], X + t * dim);
            if (tile < S)
                memset(X + tile * dim, 0, (size_t)(S - tile) * dim * sizeof(float));

            for (int l = 0; l < c->n_layers; l++)
                ane_baked_prefill_layer(m, kv, X, st->kernels[l], l, pos, S, tile);
        }

        double dt = ane_time_ms() - t_total;
        fprintf(stderr, "[ANE] Prefill %d tokens in %.1fms (%.1f tok/s)\n",
                n_tokens, dt, n_tokens / (dt / 1000.0));

        int last_tile = n_tokens % S;
        if (last_tile == 0) last_tile = S;
        memcpy(x_out, X + (last_tile - 1) * dim, dim * sizeof(float));
        free(X);
        return true;
    }
}

// =============================================================================
// BLAS PREFILL PATH (unchanged)
// Falls back from ANE to AMX-accelerated cblas_sgemm.
// Uses tiled dequant+GEMM to avoid materializing the full fp32 weight matrix.
// =============================================================================

typedef struct {
    float *w32;       // [GEMM_TILE_ROWS x max(hidden, dim)] tile dequant scratch (~7 MB)
    float *X;         // [dim * S]
    float *Q;         // [dim * S]
    float *K;         // [kv_dim * S]
    float *V;         // [kv_dim * S]
    float *Wo_out;    // [dim * S]
    float *gate;      // [hidden * S]
    float *up;        // [hidden * S]
    float *down_in;   // [hidden * S]
    float *down_out;  // [dim * S]
    float *att;       // [n_heads * max_seq]
    int seq_len;
    bool initialized;
} BLASPrefillState;

static BLASPrefillState g_blas_prefill = {0};

static void blas_prefill_cleanup(void) {
    BLASPrefillState *st = &g_blas_prefill;
    if (!st->initialized) return;
    free(st->w32);     free(st->X);      free(st->Q);
    free(st->K);       free(st->V);      free(st->Wo_out);
    free(st->gate);    free(st->up);
    free(st->down_in); free(st->down_out);
    free(st->att);
    memset(st, 0, sizeof(BLASPrefillState));
}

// Dequant any weight to fp32 buffer -- row-parallel for large matrices
static void dequant_weight_to_fp32_buf(const void *src, uint32_t type,
                                        float *dst, int rows, int cols) {
    int n = rows * cols;
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t)n * sizeof(float));
        return;
    }
    if (type == GGML_TYPE_F16) {
        cvt_f16_to_f32((const _Float16 *)src, dst, n);
        return;
    }
    if (type == GGML_TYPE_Q4_0) {
        const block_q4_0 *blocks = (const block_q4_0 *)src;
        int bpr = cols / 32;
        if (rows >= 128) {
            dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
                float *row = dst + r * cols;
                for (int b = 0; b < bpr; b++) {
                    const block_q4_0 *blk = &blocks[r * bpr + b];
                    float d = (float)blk->d;
                    float *out = row + b * 32;
                    for (int i = 0; i < 16; i++) {
                        uint8_t v = blk->qs[i];
                        out[i]      = d * ((float)(v & 0x0F) - 8.0f);
                        out[i + 16] = d * ((float)(v >> 4) - 8.0f);
                    }
                }
            });
        } else {
            for (int r = 0; r < rows; r++) {
                float *row = dst + r * cols;
                for (int b = 0; b < bpr; b++) {
                    const block_q4_0 *blk = &blocks[r * bpr + b];
                    float d = (float)blk->d;
                    float *out = row + b * 32;
                    for (int i = 0; i < 16; i++) {
                        uint8_t v = blk->qs[i];
                        out[i]      = d * ((float)(v & 0x0F) - 8.0f);
                        out[i + 16] = d * ((float)(v >> 4) - 8.0f);
                    }
                }
            }
        }
        return;
    }
    if (type == GGML_TYPE_Q4_K) {
        const block_q4_K *blocks = (const block_q4_K *)src;
        int bpr = cols / QK_K;
        if (rows >= 128) {
            dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
                _Float16 tmp16[QK_K];
                for (int b = 0; b < bpr; b++) {
                    dequant_q4_K_block_neon(&blocks[r * bpr + b], tmp16);
                    cvt_f16_to_f32(tmp16, dst + r * cols + b * QK_K, QK_K);
                }
            });
        } else {
            _Float16 tmp16[QK_K];
            for (int r = 0; r < rows; r++) {
                for (int b = 0; b < bpr; b++) {
                    dequant_q4_K_block_neon(&blocks[r * bpr + b], tmp16);
                    cvt_f16_to_f32(tmp16, dst + r * cols + b * QK_K, QK_K);
                }
            }
        }
        return;
    }
    if (type == GGML_TYPE_Q6_K) {
        const block_q6_K *blocks = (const block_q6_K *)src;
        int bpr = cols / 256;
        if (rows >= 128) {
            dispatch_apply((size_t)rows, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t r) {
                float tmp[256];
                for (int b = 0; b < bpr; b++) {
                    dequant_q6_K_block(&blocks[r * bpr + b], tmp);
                    memcpy(dst + r * cols + b * 256, tmp, 256 * sizeof(float));
                }
            });
        } else {
            float tmp[256];
            for (int r = 0; r < rows; r++) {
                for (int b = 0; b < bpr; b++) {
                    dequant_q6_K_block(&blocks[r * bpr + b], tmp);
                    memcpy(dst + r * cols + b * 256, tmp, 256 * sizeof(float));
                }
            }
        }
        return;
    }
}

// Batched matmul: Y = X @ W^T using cblas_sgemm (AMX-accelerated)
static void blas_matmul(const float *W, int out_dim, int in_dim,
                        const float *X, int S, float *Y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                S, out_dim, in_dim,
                1.0f, X, in_dim, W, in_dim,
                0.0f, Y, out_dim);
}

#define GEMM_TILE_ROWS 128

static inline size_t q_row_bytes(uint32_t type, int cols) {
    if (type == GGML_TYPE_Q4_0) return (size_t)(cols / 32) * sizeof(block_q4_0);
    if (type == GGML_TYPE_Q4_K) return (size_t)(cols / QK_K) * sizeof(block_q4_K);
    if (type == GGML_TYPE_Q6_K) return (size_t)(cols / QK_K) * sizeof(block_q6_K);
    if (type == GGML_TYPE_F32)  return (size_t)cols * sizeof(float);
    if (type == GGML_TYPE_F16)  return (size_t)cols * sizeof(_Float16);
    return 0;
}

static void tiled_q4_gemm(const void *W, uint32_t type, int out_dim, int in_dim,
                           const float *X, int S, float *Y, float *tile_buf) {
    size_t row_bytes = q_row_bytes(type, in_dim);
    if (row_bytes == 0) {
        dequant_weight_to_fp32_buf(W, type, tile_buf, out_dim, in_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    S, out_dim, in_dim,
                    1.0f, X, in_dim, tile_buf, in_dim,
                    0.0f, Y, out_dim);
        return;
    }

    const char *W_bytes = (const char *)W;
    for (int row_start = 0; row_start < out_dim; row_start += GEMM_TILE_ROWS) {
        int tile_rows = out_dim - row_start;
        if (tile_rows > GEMM_TILE_ROWS) tile_rows = GEMM_TILE_ROWS;

        const void *tile_src = W_bytes + (size_t)row_start * row_bytes;
        dequant_weight_to_fp32_buf(tile_src, type, tile_buf, tile_rows, in_dim);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    S, tile_rows, in_dim,
                    1.0f, X, in_dim, tile_buf, in_dim,
                    0.0f, Y + row_start, out_dim);
    }
}

static bool blas_prefill_init(MistralModel *m, int seq_len, int max_seq) {
    BLASPrefillState *st = &g_blas_prefill;
    if (st->initialized && st->seq_len == seq_len) return true;
    if (st->initialized) blas_prefill_cleanup();

    MistralConfig *c = &m->cfg;
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden = c->hidden_dim;
    int S = seq_len;

    size_t max_dim = (size_t)(hidden > dim ? hidden : dim);
    size_t max_w = (size_t)GEMM_TILE_ROWS * max_dim;

    st->w32      = (float *)malloc(max_w * sizeof(float));
    st->X        = (float *)calloc((size_t)dim * S, sizeof(float));
    st->Q        = (float *)calloc((size_t)dim * S, sizeof(float));
    st->K        = (float *)calloc((size_t)kv_dim * S, sizeof(float));
    st->V        = (float *)calloc((size_t)kv_dim * S, sizeof(float));
    st->Wo_out   = (float *)calloc((size_t)dim * S, sizeof(float));
    st->gate     = (float *)calloc((size_t)hidden * S, sizeof(float));
    st->up       = (float *)calloc((size_t)hidden * S, sizeof(float));
    st->down_in  = (float *)calloc((size_t)hidden * S, sizeof(float));
    st->down_out = (float *)calloc((size_t)dim * S, sizeof(float));
    st->att      = (float *)calloc((size_t)c->n_heads * max_seq, sizeof(float));

    st->seq_len = S;
    st->initialized = true;

    size_t total_mb = (max_w + (size_t)dim*S*4 + (size_t)kv_dim*S*2 +
                       (size_t)hidden*S*3 + (size_t)c->n_heads*max_seq) * 4 / (1024*1024);
    fprintf(stderr, "[BLAS] Prefill initialized for S=%d, ctx=%d (~%zu MB scratch, tile_w32=%zu MB)\n",
            S, max_seq, total_mb, max_w * sizeof(float) / (1024*1024));
    return true;
}

static void blas_prefill_layer(MistralModel *m, KVCache *kv, float *X,
                                int layer, int start_pos, int seq_len) {
    BLASPrefillState *st = &g_blas_prefill;
    MistralConfig *c = &m->cfg;
    LayerWeights *lw = &m->layers[layer];
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden = c->hidden_dim;
    int S = seq_len;

    rmsnorm_batch(st->X, X, lw->rms_att, dim, S, c->rms_eps);
    tiled_q4_gemm(lw->wq, lw->wq_type, dim, dim, st->X, S, st->Q, st->w32);
    tiled_q4_gemm(lw->wk, lw->wk_type, kv_dim, dim, st->X, S, st->K, st->w32);
    tiled_q4_gemm(lw->wv, lw->wv_type, kv_dim, dim, st->X, S, st->V, st->w32);
    apply_rope_batch(st->Q, st->K, start_pos,
                     m->rope_theta_inv,
                     c->n_heads, c->n_kv_heads, c->head_dim, S);
    kv_write_batch(kv, layer, start_pos, st->K, st->V, kv_dim, S);
    attention_batch(st->Q, st->Wo_out, kv, layer, start_pos, c, S, st->att);
    tiled_q4_gemm(lw->wo, lw->wo_type, dim, dim, st->Wo_out, S, st->X, st->w32);
    vDSP_vadd(X, 1, st->X, 1, X, 1, (vDSP_Length)(dim * S));
    rmsnorm_batch(st->X, X, lw->rms_ffn, dim, S, c->rms_eps);
    tiled_q4_gemm(lw->w1, lw->w1_type, hidden, dim, st->X, S, st->gate, st->w32);
    tiled_q4_gemm(lw->w3, lw->w3_type, hidden, dim, st->X, S, st->up, st->w32);
    int n_ffn = hidden * S;
    silu_mul_neon(st->gate, st->up, n_ffn);
    memcpy(st->down_in, st->gate, n_ffn * sizeof(float));
    tiled_q4_gemm(lw->w2, lw->w2_type, dim, hidden, st->down_in, S, st->down_out, st->w32);
    vDSP_vadd(X, 1, st->down_out, 1, X, 1, (vDSP_Length)(dim * S));
}

static bool blas_prefill_forward(MistralModel *m, KVCache *kv,
                                  const int *tokens, int n_tokens, float *x_out) {
    MistralConfig *c = &m->cfg;
    int dim = c->dim;
    int S = n_tokens;

    if (!blas_prefill_init(m, S, kv->max_seq)) return false;

    float *X = (float *)calloc((size_t)dim * S, sizeof(float));
    for (int t = 0; t < S; t++)
        embed_token(m, tokens[t], X + t * dim);

    for (int l = 0; l < c->n_layers; l++)
        blas_prefill_layer(m, kv, X, l, 0, S);

    memcpy(x_out, X + (S - 1) * dim, dim * sizeof(float));
    free(X);
    return true;
}
