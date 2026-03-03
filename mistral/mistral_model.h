// mistral_model.h — Mistral 7B model: weight management, CPU decode, ANE prefill
#pragma once

#include "gguf_loader.h"
#include "dequant.h"
#include "kv_cache.h"
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>

// ─── Mistral constants ───────────────────────────────────────────────────────
#define MISTRAL_MAX_LAYERS 32

// ─── Per-layer weight pointers (into mmap'd GGUF Q4 data) ────────────────────
typedef struct {
    const void *wq;        // [dim, dim] Q4
    const void *wk;        // [kv_dim, dim] Q4
    const void *wv;        // [kv_dim, dim] Q4
    const void *wo;        // [dim, dim] Q4
    const void *w1;        // [hidden, dim] Q4  (gate)
    const void *w3;        // [hidden, dim] Q4  (up)
    const void *w2;        // [dim, hidden] Q4  (down)
    uint32_t wq_type, wk_type, wv_type, wo_type;
    uint32_t w1_type, w3_type, w2_type;
    float *rms_att;        // [dim] fp32
    float *rms_ffn;        // [dim] fp32
} LayerWeights;

// ─── Model ───────────────────────────────────────────────────────────────────
typedef struct {
    MistralConfig cfg;
    GGUFFile *gguf;

    // Per-layer weights
    LayerWeights layers[MISTRAL_MAX_LAYERS];

    // Embedding table: on-demand dequant from mmap'd GGUF
    const void *token_embed_raw;  // pointer into mmap'd GGUF data
    uint32_t token_embed_type;    // GGML_TYPE_* of the raw data
    float *token_embed;           // [dim] scratch buffer (one row)
    float *rms_final;      // [dim] fp32
    const void *lm_head;   // Q4 pointer to output.weight
    uint32_t lm_head_type;

    // RoPE: precomputed theta inverses only (compute cos/sin on demand)
    float *rope_theta_inv; // [head_dim/2] — 1/theta^(2i/head_dim)

    // Scratch buffers for decode (S=1, fp32)
    float *xb;             // [dim]
    float *xb2;            // [dim]
    float *q;              // [dim]  (n_heads * head_dim)
    float *k;              // [kv_dim]
    float *v;              // [kv_dim]
    float *att;            // [n_heads * max_seq]  attention scores
    float *hb;             // [hidden]
    float *hb2;            // [hidden]
    float *logits;         // [vocab_size]

    // Q8_0 scratch buffers for SDOT matvec (quantized activations)
    block_q8_0 *xb_q8;    // [dim/32 blocks] — reused for QKV and FFN gate/up input
    block_q8_0 *hb_q8;    // [hidden/32 blocks] — for FFN down projection input

    // Metal GPU context (NULL if CPU-only, cast to MetalContext* when used)
    void *metal;
} MistralModel;

// ─── RoPE theta precomputation (256 bytes vs 64 MB) ─────────────────────────
static void precompute_rope_theta(MistralModel *m) {
    int hd2 = m->cfg.head_dim / 2;
    m->rope_theta_inv = (float *)malloc(hd2 * sizeof(float));
    for (int i = 0; i < hd2; i++)
        m->rope_theta_inv[i] = 1.0f / powf(m->cfg.rope_theta, (2.0f * i) / m->cfg.head_dim);
}

// ─── Apply RoPE to q and k vectors for one position ──────────────────────────
// q: [n_heads * head_dim], k: [n_kv_heads * head_dim]
// Computes cos/sin on the fly from theta_inv — saves 64 MB vs precomputed tables.
static void apply_rope(float *q, float *k, int pos,
                       const float *theta_inv,
                       int n_heads, int n_kv_heads, int head_dim) {
    int hd2 = head_dim / 2;

    // Precompute cos/sin once for this position (64 pairs vs 2560 trig calls)
    float cos_tab[128], sin_tab[128]; // head_dim/2 max = 64, but stack-safe up to 128
    float angles[128];
    float fpos = (float)pos;
    vDSP_vsmul(theta_inv, 1, &fpos, angles, 1, (vDSP_Length)hd2);
    int nn = hd2;
    vvsincosf(sin_tab, cos_tab, angles, &nn);

    // Apply to Q heads (NEON interleaved: q[2i] = q0*cos - q1*sin, q[2i+1] = q0*sin + q1*cos)
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < hd2; i += 4) {
            // Process 4 pairs at a time (8 floats)
            float32x4_t c = vld1q_f32(cos_tab + i);
            float32x4_t s = vld1q_f32(sin_tab + i);
            // Load interleaved pairs: [q0,q1, q2,q3, q4,q5, q6,q7]
            float32x4x2_t qp = vld2q_f32(qh + 2*i);
            float32x4_t even = qp.val[0]; // q0, q2, q4, q6
            float32x4_t odd  = qp.val[1]; // q1, q3, q5, q7
            qp.val[0] = vfmsq_f32(vmulq_f32(even, c), odd, s);  // even*cos - odd*sin
            qp.val[1] = vfmaq_f32(vmulq_f32(odd, c), even, s);  // odd*cos + even*sin
            vst2q_f32(qh + 2*i, qp);
        }
    }
    // Apply to K heads
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < hd2; i += 4) {
            float32x4_t c = vld1q_f32(cos_tab + i);
            float32x4_t s = vld1q_f32(sin_tab + i);
            float32x4x2_t kp = vld2q_f32(kh + 2*i);
            float32x4_t even = kp.val[0];
            float32x4_t odd  = kp.val[1];
            kp.val[0] = vfmsq_f32(vmulq_f32(even, c), odd, s);
            kp.val[1] = vfmaq_f32(vmulq_f32(odd, c), even, s);
            vst2q_f32(kh + 2*i, kp);
        }
    }
}

// ─── On-demand embedding dequant (saves ~496 MB vs full fp32 table) ──────────
static void embed_token(MistralModel *m, int token_id, float *out) {
    int dim = m->cfg.dim;
    if (m->token_embed_type == GGML_TYPE_F32) {
        const float *src = (const float *)m->token_embed_raw;
        memcpy(out, src + (size_t)token_id * dim, dim * sizeof(float));
    } else if (m->token_embed_type == GGML_TYPE_F16) {
        const _Float16 *src = (const _Float16 *)m->token_embed_raw;
        src += (size_t)token_id * dim;
        for (int i = 0; i < dim; i++) out[i] = (float)src[i];
    } else if (m->token_embed_type == GGML_TYPE_Q4_0) {
        int bpr = dim / QK4_0;
        const block_q4_0 *blocks = (const block_q4_0 *)m->token_embed_raw;
        const block_q4_0 *row = blocks + (size_t)token_id * bpr;
        for (int b = 0; b < bpr; b++) {
            float d = (float)row[b].d;
            for (int i = 0; i < 16; i++) {
                uint8_t v = row[b].qs[i];
                out[b * 32 + i]      = d * ((float)(v & 0x0F) - 8.0f);
                out[b * 32 + i + 16] = d * ((float)(v >> 4)   - 8.0f);
            }
        }
    } else if (m->token_embed_type == GGML_TYPE_Q4_K) {
        int bpr = dim / QK_K;
        const block_q4_K *blocks = (const block_q4_K *)m->token_embed_raw;
        const block_q4_K *row = blocks + (size_t)token_id * bpr;
        _Float16 tmp16[QK_K];
        for (int b = 0; b < bpr; b++) {
            dequant_q4_K_block_neon(&row[b], tmp16);
            for (int i = 0; i < QK_K; i++)
                out[b * QK_K + i] = (float)tmp16[i];
        }
    }
}

// ─── NEON fp32→fp16 conversion ───────────────────────────────────────────────
static inline void cvt_f32_to_f16_neon(const float *src, _Float16 *dst, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float32x4_t a = vld1q_f32(src + i);
        float32x4_t b = vld1q_f32(src + i + 4);
        vst1_f16((float16_t *)(dst + i),     vcvt_f16_f32(a));
        vst1_f16((float16_t *)(dst + i + 4), vcvt_f16_f32(b));
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// ─── NEON SiLU(gate)*up — exact sigmoid via vDSP + NEON fma ──────────────────
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Uses vvexpf (Accelerate) for vectorized exp, then NEON for 1/(1+exp) * g * up.
static inline void silu_mul_neon(float *gate, const float *up, int n) {
    // Negate gate into temp buffer, compute exp(-gate), then silu = gate / (1 + exp(-gate)) * up
    size_t alloc_n = ((n + 3) & ~3);
    float *neg = (alloc_n <= 16384)
        ? (float *)__builtin_alloca(alloc_n * sizeof(float))
        : (float *)malloc(alloc_n * sizeof(float));
    float minus_one = -1.0f;
    vDSP_vsmul(gate, 1, &minus_one, neg, 1, (vDSP_Length)n);
    int nn = n;
    vvexpf(neg, neg, &nn);  // neg[i] = exp(-gate[i])

    int i = 0;
    float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t g   = vld1q_f32(gate + i);
        float32x4_t u   = vld1q_f32(up + i);
        float32x4_t e   = vld1q_f32(neg + i);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, e));  // sigmoid(g)
        vst1q_f32(gate + i, vmulq_f32(vmulq_f32(g, sig), u)); // g * sigmoid(g) * up
    }
    for (; i < n; i++) {
        float sig = 1.0f / (1.0f + neg[i]);
        gate[i] = gate[i] * sig * up[i];
    }
    if (alloc_n > 16384) free(neg);
}

// ─── RMSNorm (single vector, S=1) ───────────────────────────────────────────
static void rmsnorm_vec(float *out, const float *x, const float *w, int dim, float eps) {
    float ss = 0;
    vDSP_dotpr(x, 1, x, 1, &ss, (vDSP_Length)dim);
    ss = 1.0f / sqrtf(ss / dim + eps);
    vDSP_vsmul(x, 1, &ss, out, 1, (vDSP_Length)dim);
    vDSP_vmul(out, 1, w, 1, out, 1, (vDSP_Length)dim);
}

// ─── Q4 matvec dispatch (handles Q4_0 and Q4_K) ─────────────────────────────
static void q4_matvec(const void *W, uint32_t type, const float *x, float *y, int out_dim, int in_dim) {
    memset(y, 0, out_dim * sizeof(float));
    if (type == GGML_TYPE_Q4_0) {
        q4_0_matvec_f32(W, x, y, out_dim, in_dim);
    } else if (type == GGML_TYPE_Q4_K) {
        // Fallback: dequant then BLAS for Q4_K (TODO: fused Q4_K matvec)
        _Float16 *tmp = (_Float16 *)malloc(out_dim * in_dim * sizeof(_Float16));
        dequant_q4_K_to_fp16(W, tmp, out_dim, in_dim);
        // Convert fp16 to fp32 for sgemv
        float *Wf = (float *)malloc(out_dim * in_dim * sizeof(float));
        for (int i = 0; i < out_dim * in_dim; i++) Wf[i] = (float)tmp[i];
        free(tmp);
        // y = W @ x (W is [out_dim, in_dim] row-major)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim,
                    1.0f, Wf, in_dim, x, 1, 0.0f, y, 1);
        free(Wf);
    } else if (type == GGML_TYPE_F16) {
        // fp16 weights: convert and sgemv
        const _Float16 *Wh = (const _Float16 *)W;
        float *Wf = (float *)malloc(out_dim * in_dim * sizeof(float));
        for (int i = 0; i < out_dim * in_dim; i++) Wf[i] = (float)Wh[i];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim,
                    1.0f, Wf, in_dim, x, 1, 0.0f, y, 1);
        free(Wf);
    } else if (type == GGML_TYPE_F32) {
        const float *Wf = (const float *)W;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim,
                    1.0f, Wf, in_dim, x, 1, 0.0f, y, 1);
    } else if (type == GGML_TYPE_Q6_K) {
        dequant_q6_K_matvec_f32(W, x, y, out_dim, in_dim);
    }
}

// ─── Softmax (in-place, single vector) ───────────────────────────────────────
static void softmax_vec(float *x, int n) {
    float max_val;
    vDSP_maxv(x, 1, &max_val, (vDSP_Length)n);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, (vDSP_Length)n);
    int nn = n;
    vvexpf(x, x, &nn);
    float sum;
    vDSP_sve(x, 1, &sum, (vDSP_Length)n);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv_sum, x, 1, (vDSP_Length)n);
}

// Forward declaration for row-parallel init (defined after q4_matvec)
static void matvec_parallel_init(void);

// ─── Load model from GGUF ────────────────────────────────────────────────────
static MistralModel *mistral_load(const char *gguf_path) {
    GGUFFile *gguf = gguf_open(gguf_path);
    if (!gguf) return NULL;
    gguf_print_info(gguf);

    MistralModel *m = (MistralModel *)calloc(1, sizeof(MistralModel));
    m->cfg = gguf->cfg;
    m->gguf = gguf;

    int dim = m->cfg.dim;
    int kv_dim = m->cfg.n_kv_heads * m->cfg.head_dim;
    int hidden = m->cfg.hidden_dim;
    int vocab = m->cfg.vocab_size;
    int n_layers = m->cfg.n_layers;
    int max_seq = m->cfg.sliding_window > 0 ? m->cfg.sliding_window : 4096;

    // Load per-layer weight pointers
    for (int l = 0; l < n_layers; l++) {
        LayerWeights *lw = &m->layers[l];
        char name[128];

        #define LOAD_WEIGHT(field, fmt, type_field) do { \
            snprintf(name, sizeof(name), fmt, l); \
            GGUFTensor *t = gguf_find(gguf, name); \
            if (t) { lw->field = gguf_data(gguf, t); lw->type_field = t->type; } \
            else fprintf(stderr, "WARNING: tensor '%s' not found\n", name); \
        } while(0)

        LOAD_WEIGHT(wq, "blk.%d.attn_q.weight", wq_type);
        LOAD_WEIGHT(wk, "blk.%d.attn_k.weight", wk_type);
        LOAD_WEIGHT(wv, "blk.%d.attn_v.weight", wv_type);
        LOAD_WEIGHT(wo, "blk.%d.attn_output.weight", wo_type);
        LOAD_WEIGHT(w1, "blk.%d.ffn_gate.weight", w1_type);
        LOAD_WEIGHT(w3, "blk.%d.ffn_up.weight", w3_type);
        LOAD_WEIGHT(w2, "blk.%d.ffn_down.weight", w2_type);
        #undef LOAD_WEIGHT

        // RMSNorm weights (always F32, small — copy to owned buffer)
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        GGUFTensor *t = gguf_find(gguf, name);
        if (t) {
            lw->rms_att = (float *)malloc(dim * sizeof(float));
            memcpy(lw->rms_att, gguf_data(gguf, t), dim * sizeof(float));
        }
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        t = gguf_find(gguf, name);
        if (t) {
            lw->rms_ffn = (float *)malloc(dim * sizeof(float));
            memcpy(lw->rms_ffn, gguf_data(gguf, t), dim * sizeof(float));
        }
    }

    // Token embedding — store raw pointer for on-demand dequant
    GGUFTensor *emb_t = gguf_find(gguf, "token_embd.weight");
    if (emb_t) {
        m->token_embed_raw  = gguf_data(gguf, emb_t);
        m->token_embed_type = emb_t->type;
        // Single-row scratch buffer: only [dim] floats instead of [vocab * dim]
        m->token_embed = (float *)malloc(dim * sizeof(float));
    }

    // Final RMSNorm
    GGUFTensor *fn_t = gguf_find(gguf, "output_norm.weight");
    if (fn_t) {
        m->rms_final = (float *)malloc(dim * sizeof(float));
        memcpy(m->rms_final, gguf_data(gguf, fn_t), dim * sizeof(float));
    }

    // LM head
    GGUFTensor *lm_t = gguf_find(gguf, "output.weight");
    if (lm_t) {
        m->lm_head = gguf_data(gguf, lm_t);
        m->lm_head_type = lm_t->type;
    }

    // Precompute RoPE theta inverses only (256 bytes vs 64 MB)
    precompute_rope_theta(m);

    // Init row-parallel GCD queue (P-core affinity via USER_INTERACTIVE QoS)
    matvec_parallel_init();

    // Allocate scratch buffers
    m->xb     = (float *)calloc(dim, sizeof(float));
    m->xb2    = (float *)calloc(dim, sizeof(float));
    m->q      = (float *)calloc(dim, sizeof(float));
    m->k      = (float *)calloc(kv_dim, sizeof(float));
    m->v      = (float *)calloc(kv_dim, sizeof(float));
    m->att    = (float *)calloc(m->cfg.n_heads * max_seq, sizeof(float));
    m->hb     = (float *)calloc(hidden, sizeof(float));
    m->hb2    = (float *)calloc(hidden, sizeof(float));
    m->logits = (float *)calloc(vocab, sizeof(float));
    m->xb_q8  = (block_q8_0 *)calloc(dim / QK8_0, sizeof(block_q8_0));
    m->hb_q8  = (block_q8_0 *)calloc(hidden / QK8_0, sizeof(block_q8_0));

    return m;
}

// ─── NEON GQA attention (decode, S=1) ────────────────────────────────────────
// Computes multi-head attention with grouped-query attention using NEON intrinsics.
// head_dim must be 128 (16 iterations of 8-wide NEON).
static inline void gqa_attention_neon(
    float *q, float *att, float *out,
    _Float16 *kcache_base, _Float16 *vcache_base,
    int n_heads, int heads_per_kv, int hd, int max_seq, int pos)
{
    int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
    int ring = (pos + 1 > max_seq);
    int ring_off = ring ? (pos + 1 - seq_len) : 0;

    float inv_sqrt_hd = 1.0f / sqrtf((float)hd);

    // Pre-scale query vectors: q[h*hd + d] *= inv_sqrt_hd
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * hd;
        float32x4_t scale = vdupq_n_f32(inv_sqrt_hd);
        for (int d = 0; d < hd; d += 4) {
            float32x4_t v = vld1q_f32(qh + d);
            vst1q_f32(qh + d, vmulq_f32(v, scale));
        }
    }

    int n_kv_heads = n_heads / heads_per_kv;
    int kv_dim = n_kv_heads * hd;

    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * hd;
        float *att_h = att + h * max_seq;
        int kvh = h / heads_per_kv;

        // ── Q @ K^T ─────────────────────────────────────────────────────
        // Layout: kcache_base[ct * kv_dim + kvh * hd] = K[ct, kvh, :]
        // 128 contiguous fp16 values — no gather.

        for (int t = 0; t < seq_len; t++) {
            int ct = ring ? ((ring_off + t) % max_seq) : t;
            const _Float16 *kt = kcache_base + (size_t)ct * kv_dim + kvh * hd;

            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            for (int d = 0; d < 128; d += 32) {
                float16x8_t k0 = vld1q_f16((const float16_t *)(kt + d));
                float16x8_t k1 = vld1q_f16((const float16_t *)(kt + d + 8));
                float16x8_t k2 = vld1q_f16((const float16_t *)(kt + d + 16));
                float16x8_t k3 = vld1q_f16((const float16_t *)(kt + d + 24));

                float32x4_t klo0 = vcvt_f32_f16(vget_low_f16(k0));
                float32x4_t khi0 = vcvt_f32_f16(vget_high_f16(k0));
                float32x4_t klo1 = vcvt_f32_f16(vget_low_f16(k1));
                float32x4_t khi1 = vcvt_f32_f16(vget_high_f16(k1));
                float32x4_t klo2 = vcvt_f32_f16(vget_low_f16(k2));
                float32x4_t khi2 = vcvt_f32_f16(vget_high_f16(k2));
                float32x4_t klo3 = vcvt_f32_f16(vget_low_f16(k3));
                float32x4_t khi3 = vcvt_f32_f16(vget_high_f16(k3));

                float32x4_t q0 = vld1q_f32(qh + d);
                float32x4_t q1 = vld1q_f32(qh + d + 4);
                float32x4_t q2 = vld1q_f32(qh + d + 8);
                float32x4_t q3 = vld1q_f32(qh + d + 12);
                float32x4_t q4 = vld1q_f32(qh + d + 16);
                float32x4_t q5 = vld1q_f32(qh + d + 20);
                float32x4_t q6 = vld1q_f32(qh + d + 24);
                float32x4_t q7 = vld1q_f32(qh + d + 28);

                acc0 = vfmaq_f32(acc0, q0, klo0);
                acc1 = vfmaq_f32(acc1, q1, khi0);
                acc2 = vfmaq_f32(acc2, q2, klo1);
                acc3 = vfmaq_f32(acc3, q3, khi1);
                acc0 = vfmaq_f32(acc0, q4, klo2);
                acc1 = vfmaq_f32(acc1, q5, khi2);
                acc2 = vfmaq_f32(acc2, q6, klo3);
                acc3 = vfmaq_f32(acc3, q7, khi3);
            }

            acc0 = vaddq_f32(acc0, acc1);
            acc2 = vaddq_f32(acc2, acc3);
            acc0 = vaddq_f32(acc0, acc2);
            att_h[t] = vaddvq_f32(acc0);
        }

        // ── Softmax ─────────────────────────────────────────────────────
        softmax_vec(att_h, seq_len);

        // ── Att @ V ─────────────────────────────────────────────────────
        // Layout: vcache_base[ct * kv_dim + kvh * hd] = V[ct, kvh, :]
        // 128 contiguous fp16 values — no gather.
        float *oh = out + h * hd;
        memset(oh, 0, hd * sizeof(float));

        for (int t = 0; t < seq_len; t++) {
            float a = att_h[t];
            if (a == 0.0f) continue;

            int ct = ring ? ((ring_off + t) % max_seq) : t;
            const _Float16 *vt = vcache_base + (size_t)ct * kv_dim + kvh * hd;
            float32x4_t av = vdupq_n_f32(a);

            for (int d = 0; d < 128; d += 16) {
                float16x8_t v0 = vld1q_f16((const float16_t *)(vt + d));
                float16x8_t v1 = vld1q_f16((const float16_t *)(vt + d + 8));

                float32x4_t vlo0 = vcvt_f32_f16(vget_low_f16(v0));
                float32x4_t vhi0 = vcvt_f32_f16(vget_high_f16(v0));
                float32x4_t vlo1 = vcvt_f32_f16(vget_low_f16(v1));
                float32x4_t vhi1 = vcvt_f32_f16(vget_high_f16(v1));

                float32x4_t o0 = vld1q_f32(oh + d);
                float32x4_t o1 = vld1q_f32(oh + d + 4);
                float32x4_t o2 = vld1q_f32(oh + d + 8);
                float32x4_t o3 = vld1q_f32(oh + d + 12);

                o0 = vfmaq_f32(o0, av, vlo0);
                o1 = vfmaq_f32(o1, av, vhi0);
                o2 = vfmaq_f32(o2, av, vlo1);
                o3 = vfmaq_f32(o3, av, vhi1);

                vst1q_f32(oh + d, o0);
                vst1q_f32(oh + d + 4, o1);
                vst1q_f32(oh + d + 8, o2);
                vst1q_f32(oh + d + 12, o3);
            }
        }
    }
}

// ─── Single-layer forward (decode, S=1) ──────────────────────────────────────
// x: [dim] fp32, modified in-place. pos: absolute position.
static void mistral_layer_decode(MistralModel *m, KVCache *kv, float *x, int layer, int pos) {
    MistralConfig *c = &m->cfg;
    LayerWeights *lw = &m->layers[layer];
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden = c->hidden_dim;
    int hd = c->head_dim;
    int hd2 = hd / 2;
    int n_heads = c->n_heads;
    int n_kv = c->n_kv_heads;
    int heads_per_kv = n_heads / n_kv;

    // 1. Attention RMSNorm
    rmsnorm_vec(m->xb, x, lw->rms_att, dim, c->rms_eps);

    // 2. QKV projections (fused Q4 matvec)
    q4_matvec(lw->wq, lw->wq_type, m->xb, m->q, dim, dim);
    q4_matvec(lw->wk, lw->wk_type, m->xb, m->k, kv_dim, dim);
    q4_matvec(lw->wv, lw->wv_type, m->xb, m->v, kv_dim, dim);

    // 3. RoPE
    apply_rope(m->q, m->k, pos, m->rope_theta_inv, n_heads, n_kv, hd);

    // 4. Write K, V to cache
    // Convert fp32 k,v to fp16 for cache (NEON vectorized)
    _Float16 k16[kv_dim], v16[kv_dim];
    cvt_f32_to_f16_neon(m->k, k16, kv_dim);
    cvt_f32_to_f16_neon(m->v, v16, kv_dim);
    int cache_pos = pos % kv->max_seq;
    kv_write(kv, layer, cache_pos, k16, v16);
    if (kv->len < kv->max_seq && pos + 1 > kv->len) kv->len = pos + 1;

    // 5. Multi-head attention with GQA (NEON)
    gqa_attention_neon(m->q, m->att, m->xb2,
                       kv_k(kv, layer), kv_v(kv, layer),
                       n_heads, heads_per_kv, hd, kv->max_seq, pos);

    // 6. Output projection: Wo @ attn_out
    q4_matvec(lw->wo, lw->wo_type, m->xb2, m->xb, dim, dim);

    // 7. Residual connection
    vDSP_vadd(x, 1, m->xb, 1, x, 1, (vDSP_Length)dim);

    // 8. FFN RMSNorm
    rmsnorm_vec(m->xb, x, lw->rms_ffn, dim, c->rms_eps);

    // 9. SwiGLU FFN
    // gate = W1 @ xb, up = W3 @ xb
    q4_matvec(lw->w1, lw->w1_type, m->xb, m->hb, hidden, dim);
    q4_matvec(lw->w3, lw->w3_type, m->xb, m->hb2, hidden, dim);

    // SiLU(gate) * up (NEON polynomial sigmoid)
    silu_mul_neon(m->hb, m->hb2, hidden);

    // down = W2 @ (silu_gate * up)
    q4_matvec(lw->w2, lw->w2_type, m->hb, m->xb, dim, hidden);

    // 10. Residual connection
    vDSP_vadd(x, 1, m->xb, 1, x, 1, (vDSP_Length)dim);
}

// ─── Row-parallel matvec infrastructure ──────────────────────────────────────
// Strategy: instead of running 3 matvecs on 3 cores (inter-op parallelism),
// run each matvec on ALL 4 P-cores (intra-op parallelism via dispatch_apply).
// For bandwidth-bound decode this is better because:
//   - 4 cores share L2 on P-cluster, so weight data gets better cache reuse
//   - Input vector x stays hot in all 4 cores' L1
//   - Less memory controller contention vs 3 simultaneous weight streams
//
// QoS_CLASS_USER_INTERACTIVE hints the scheduler to use P-cores.

static dispatch_queue_t _matvec_parallel_q = NULL;

static void matvec_parallel_init(void) {
    if (_matvec_parallel_q) return;
    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_CONCURRENT,
        QOS_CLASS_USER_INTERACTIVE, 0);
    _matvec_parallel_q = dispatch_queue_create(
        "com.mistral.row_parallel_matvec", attr);
}

// Row-parallel Q4_0 matvec: splits output rows across P-cores via dispatch_apply.
// Each chunk processes a contiguous range of output rows — no synchronization needed
// since each row writes to a unique y[row].
// Adaptive chunk sizing: MATVEC_CHUNK_ROWS_LARGE (32) for in_dim > 8192 (hidden=14336),
// MATVEC_CHUNK_ROWS (64) for smaller matrices (dim=4096).
//   dim=4096:    64 rows x 128 blocks x 18B = 147KB per chunk (fits L1)
//   hidden=14336: 32 rows x 448 blocks x 18B = 258KB per chunk (vs 516KB at 64 rows)

static void q4_matvec_parallel(const void *W, uint32_t type, const float *x,
                                float *y, int out_dim, int in_dim) {
    // For small matrices or non-Q4_0 types, fall back to serial.
    // Row-parallel only helps when there are enough rows to split.
    if (out_dim < MATVEC_CHUNK_ROWS * 2 || !_matvec_parallel_q) {
        q4_matvec(W, type, x, y, out_dim, in_dim);
        return;
    }

    // Q4_0: row-parallel fused dequant+matvec
    if (type == GGML_TYPE_Q4_0) {
        int chunk = (in_dim > 8192) ? MATVEC_CHUNK_ROWS_LARGE : MATVEC_CHUNK_ROWS;
        int n_chunks = (out_dim + chunk - 1) / chunk;
        int bpr = in_dim / QK4_0;  // blocks per row
        size_t row_stride = bpr * sizeof(block_q4_0);

        dispatch_apply((size_t)n_chunks, _matvec_parallel_q, ^(size_t ci) {
            int row_start = (int)ci * chunk;
            int row_end = row_start + chunk;
            if (row_end > out_dim) row_end = out_dim;
            int chunk_rows = row_end - row_start;

            // Point into the correct offset within the Q4_0 block array
            const void *W_chunk = (const char *)W + row_start * row_stride;

            // Prefetch start of this chunk before computing
            __builtin_prefetch(W_chunk, 0, 1);

            // Zero this chunk's output
            memset(y + row_start, 0, chunk_rows * sizeof(float));

            // Reuse the existing fused NEON kernel on the row slice
            q4_0_matvec_f32(W_chunk, x, y + row_start, chunk_rows, in_dim);
        });
        return;
    }

    // Q6_K: row-parallel fused dequant+matvec
    if (type == GGML_TYPE_Q6_K) {
        int chunk = (in_dim > 8192) ? MATVEC_CHUNK_ROWS_LARGE : MATVEC_CHUNK_ROWS;
        int n_chunks = (out_dim + chunk - 1) / chunk;
        int bpr = in_dim / QK_K;  // blocks per row (QK_K = 256)
        size_t row_stride = bpr * sizeof(block_q6_K);

        dispatch_apply((size_t)n_chunks, _matvec_parallel_q, ^(size_t ci) {
            int row_start = (int)ci * chunk;
            int row_end = row_start + chunk;
            if (row_end > out_dim) row_end = out_dim;
            int chunk_rows = row_end - row_start;

            const void *W_chunk = (const char *)W + row_start * row_stride;

            // Prefetch start of this chunk
            __builtin_prefetch(W_chunk, 0, 1);

            memset(y + row_start, 0, chunk_rows * sizeof(float));
            dequant_q6_K_matvec_f32(W_chunk, x, y + row_start, chunk_rows, in_dim);
        });
        return;
    }

    // All other types: serial fallback
    q4_matvec(W, type, x, y, out_dim, in_dim);
}

// ─── SDOT + row-parallel dispatch: quantize x to Q8_0, use SDOT inner loop ──
// For Q4_0 weights: quantize x once, then row-parallel SDOT matvec.
// For other types: fall back to fp32 row-parallel path.
static void q4_matvec_sdot_parallel(const void *W, uint32_t type,
                                     const block_q8_0 *x_q8,
                                     float *y, int out_dim, int in_dim) {
    if (type == GGML_TYPE_Q4_0 && out_dim >= 128 && _matvec_parallel_q) {
        q4_0_q8_0_matvec_sdot_parallel(W, x_q8, y, out_dim, in_dim);
        return;
    }
    if (type == GGML_TYPE_Q4_0) {
        memset(y, 0, out_dim * sizeof(float));
        q4_0_q8_0_matvec(W, x_q8, y, out_dim, in_dim);
        return;
    }
    // Non-Q4_0: dequant x_q8 back to fp32, use fp32 path
    float *x_f32 = (float *)malloc(in_dim * sizeof(float));
    int nb = in_dim / QK8_0;
    for (int i = 0; i < nb; i++) {
        float d = x_q8[i].d;
        for (int j = 0; j < QK8_0; j++)
            x_f32[i * QK8_0 + j] = d * x_q8[i].qs[j];
    }
    q4_matvec_parallel(W, type, x_f32, y, out_dim, in_dim);
    free(x_f32);
}

// ─── Fused QKV matvec: single dispatch_apply for Q+K+V projections ───────────
// All three weight matrices share the same quantized input xb_q8.
// Maps chunks to Q (0..dim-1), K (dim..dim+kv_dim-1), V (dim+kv_dim..end).
static void q4_matvec_sdot_parallel_fused_qkv(
    const void *Wq, const void *Wk, const void *Wv,
    const block_q8_0 *x_q8,
    float *yq, float *yk, float *yv,
    int dim, int kv_dim, int in_dim)
{
    const int bpr = in_dim / QK4_0;
    const block_q4_0 *Bq = (const block_q4_0 *)Wq;
    const block_q4_0 *Bk = (const block_q4_0 *)Wk;
    const block_q4_0 *Bv = (const block_q4_0 *)Wv;

    const int total_rows = dim + kv_dim + kv_dim;
    const int n_chunks = (total_rows + MATVEC_CHUNK_ROWS - 1) / MATVEC_CHUNK_ROWS;

    memset(yq, 0, dim    * sizeof(float));
    memset(yk, 0, kv_dim * sizeof(float));
    memset(yv, 0, kv_dim * sizeof(float));

    __block _Atomic int chunk_counter = 0;
    dispatch_queue_t q = matvec_get_concurrent_queue();

    dispatch_apply((size_t)n_chunks, q, ^(size_t _iter __attribute__((unused))) {
        int chunk;
        while ((chunk = atomic_fetch_add(&chunk_counter, 1)) < n_chunks) {
            int virt_start = chunk * MATVEC_CHUNK_ROWS;
            int virt_end   = virt_start + MATVEC_CHUNK_ROWS;
            if (virt_end > total_rows) virt_end = total_rows;

            for (int vrow = virt_start; vrow < virt_end; vrow++) {
                const block_q4_0 *rblk;
                float *yout;

                if (vrow < dim) {
                    rblk = Bq + vrow * bpr;
                    yout = yq + vrow;
                } else if (vrow < dim + kv_dim) {
                    rblk = Bk + (vrow - dim) * bpr;
                    yout = yk + (vrow - dim);
                } else {
                    rblk = Bv + (vrow - dim - kv_dim) * bpr;
                    yout = yv + (vrow - dim - kv_dim);
                }

                float sumf = 0.0f;
                for (int b = 0; b < bpr; b++) {
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
                *yout = sumf;
            }
        }
    });
}

// ─── Fused gate+up matvec: single dispatch_apply for W1+W3 projections ────────
// W1 (gate) and W3 (up) share the same quantized input xb_q8.
// Maps chunks to rows of W1 (0..hidden-1) then W3 (hidden..2*hidden-1).
static void q4_matvec_sdot_parallel_fused_pair(
    const void *W1, const void *W3,
    const block_q8_0 *x_q8,
    float *y1, float *y3,
    int out_dim, int in_dim)
{
    const int bpr = in_dim / QK4_0;
    const block_q4_0 *B1 = (const block_q4_0 *)W1;
    const block_q4_0 *B3 = (const block_q4_0 *)W3;

    const int total_rows = out_dim * 2;
    const int n_chunks = (total_rows + MATVEC_CHUNK_ROWS - 1) / MATVEC_CHUNK_ROWS;

    memset(y1, 0, out_dim * sizeof(float));
    memset(y3, 0, out_dim * sizeof(float));

    __block _Atomic int chunk_counter = 0;
    dispatch_queue_t q = matvec_get_concurrent_queue();

    dispatch_apply((size_t)n_chunks, q, ^(size_t _iter __attribute__((unused))) {
        int chunk;
        while ((chunk = atomic_fetch_add(&chunk_counter, 1)) < n_chunks) {
            int virt_start = chunk * MATVEC_CHUNK_ROWS;
            int virt_end   = virt_start + MATVEC_CHUNK_ROWS;
            if (virt_end > total_rows) virt_end = total_rows;

            for (int vrow = virt_start; vrow < virt_end; vrow++) {
                const block_q4_0 *rblk;
                float *yout;

                if (vrow < out_dim) {
                    rblk = B1 + vrow * bpr;
                    yout = y1 + vrow;
                } else {
                    rblk = B3 + (vrow - out_dim) * bpr;
                    yout = y3 + (vrow - out_dim);
                }

                float sumf = 0.0f;
                for (int b = 0; b < bpr; b++) {
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
                *yout = sumf;
            }
        }
    });
}

// ─── Row-parallel SDOT decode: sequential matvecs, each using all P-cores ────
// Quantizes activations to Q8_0 once per shared-input group, then uses SDOT.
static void mistral_layer_decode_parallel(MistralModel *m, KVCache *kv, float *x, int layer, int pos) {
    MistralConfig *c = &m->cfg;
    LayerWeights *lw = &m->layers[layer];
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden = c->hidden_dim;
    int hd = c->head_dim;
    int n_heads = c->n_heads;
    int n_kv = c->n_kv_heads;
    int heads_per_kv = n_heads / n_kv;

    // 1. Attention RMSNorm
    rmsnorm_vec(m->xb, x, lw->rms_att, dim, c->rms_eps);

    // 2. Quantize xb to Q8_0 once, reuse for Q/K/V projections
    quantize_f32_to_q8_0(m->xb, m->xb_q8, dim);

    // 3. QKV projections — fused single dispatch if all Q4_0, else sequential fallback
    if (lw->wq_type == GGML_TYPE_Q4_0 &&
        lw->wk_type == GGML_TYPE_Q4_0 &&
        lw->wv_type == GGML_TYPE_Q4_0 &&
        _matvec_parallel_q) {
        q4_matvec_sdot_parallel_fused_qkv(
            lw->wq, lw->wk, lw->wv, m->xb_q8,
            m->q, m->k, m->v, dim, kv_dim, dim);
    } else {
        q4_matvec_sdot_parallel(lw->wq, lw->wq_type, m->xb_q8, m->q, dim, dim);
        q4_matvec_sdot_parallel(lw->wk, lw->wk_type, m->xb_q8, m->k, kv_dim, dim);
        q4_matvec_sdot_parallel(lw->wv, lw->wv_type, m->xb_q8, m->v, kv_dim, dim);
    }

    // 4. RoPE
    apply_rope(m->q, m->k, pos, m->rope_theta_inv, n_heads, n_kv, hd);

    // 5. Write K, V to cache
    _Float16 k16[kv_dim], v16[kv_dim];
    cvt_f32_to_f16_neon(m->k, k16, kv_dim);
    cvt_f32_to_f16_neon(m->v, v16, kv_dim);
    int cache_pos = pos % kv->max_seq;
    kv_write(kv, layer, cache_pos, k16, v16);
    if (kv->len < kv->max_seq && pos + 1 > kv->len) kv->len = pos + 1;

    // 6. Multi-head attention with GQA (NEON)
    gqa_attention_neon(m->q, m->att, m->xb2,
                       kv_k(kv, layer), kv_v(kv, layer),
                       n_heads, heads_per_kv, hd, kv->max_seq, pos);

    // 7. Output projection — quantize attn output, SDOT + row-parallel
    quantize_f32_to_q8_0(m->xb2, m->xb_q8, dim);
    q4_matvec_sdot_parallel(lw->wo, lw->wo_type, m->xb_q8, m->xb, dim, dim);

    // 8. Residual connection
    vDSP_vadd(x, 1, m->xb, 1, x, 1, (vDSP_Length)dim);

    // 9. FFN RMSNorm
    rmsnorm_vec(m->xb, x, lw->rms_ffn, dim, c->rms_eps);

    // 10. Quantize xb to Q8_0 once, reuse for gate/up projections
    quantize_f32_to_q8_0(m->xb, m->xb_q8, dim);

    // 11. SwiGLU FFN — fused gate+up if both Q4_0
    if (lw->w1_type == GGML_TYPE_Q4_0 &&
        lw->w3_type == GGML_TYPE_Q4_0 &&
        _matvec_parallel_q) {
        q4_matvec_sdot_parallel_fused_pair(
            lw->w1, lw->w3, m->xb_q8,
            m->hb, m->hb2, hidden, dim);
    } else {
        q4_matvec_sdot_parallel(lw->w1, lw->w1_type, m->xb_q8, m->hb, hidden, dim);
        q4_matvec_sdot_parallel(lw->w3, lw->w3_type, m->xb_q8, m->hb2, hidden, dim);
    }

    // SiLU(gate) * up (NEON polynomial sigmoid)
    silu_mul_neon(m->hb, m->hb2, hidden);

    // down projection — quantize SiLU output to Q8_0, then SDOT row-parallel
    quantize_f32_to_q8_0(m->hb, m->hb_q8, hidden);
    q4_matvec_sdot_parallel(lw->w2, lw->w2_type, m->hb_q8, m->xb, dim, hidden);

    // 12. Residual connection
    vDSP_vadd(x, 1, m->xb, 1, x, 1, (vDSP_Length)dim);
}

// ─── Chunked prefill (process multiple tokens per layer) ─────────────────────
#define PREFILL_CHUNK_SIZE 32

// Process chunk_size tokens through one transformer layer.
// X: [chunk_size][dim] fp32 row-major — each row is one token's hidden state.
// positions: [chunk_size] absolute positions for each token.
// Modifies X in-place (residual connections applied per token).
static void mistral_layer_prefill_chunk(MistralModel *m, KVCache *kv, float *X,
                                         int *positions, int chunk_size, int layer) {
    MistralConfig *c = &m->cfg;
    LayerWeights *lw = &m->layers[layer];
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden = c->hidden_dim;
    int hd = c->head_dim;
    int n_heads = c->n_heads;
    int n_kv = c->n_kv_heads;
    int heads_per_kv = n_heads / n_kv;
    int max_seq = kv->max_seq;

    // Per-token scratch buffers
    float *xb      = (float *)malloc(dim * sizeof(float));
    float *q_buf   = (float *)malloc(dim * sizeof(float));
    float *k_buf   = (float *)malloc(kv_dim * sizeof(float));
    float *v_buf   = (float *)malloc(kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(dim * sizeof(float));
    float *hb      = (float *)malloc(hidden * sizeof(float));
    float *hb2     = (float *)malloc(hidden * sizeof(float));
    float *att     = (float *)malloc(n_heads * max_seq * sizeof(float));

    // Store all Q vectors for the attention phase
    float *Q_all = (float *)malloc(chunk_size * dim * sizeof(float));

    // ── Phase 1: RMSNorm + QKV + RoPE + KV write (per token) ────────────
    for (int ti = 0; ti < chunk_size; ti++) {
        float *xi = X + ti * dim;
        int pos = positions[ti];

        rmsnorm_vec(xb, xi, lw->rms_att, dim, c->rms_eps);

        q4_matvec(lw->wq, lw->wq_type, xb, q_buf, dim, dim);
        q4_matvec(lw->wk, lw->wk_type, xb, k_buf, kv_dim, dim);
        q4_matvec(lw->wv, lw->wv_type, xb, v_buf, kv_dim, dim);

        apply_rope(q_buf, k_buf, pos, m->rope_theta_inv, n_heads, n_kv, hd);

        // Write K,V to cache (fp32 -> fp16, NEON vectorized)
        _Float16 k16[1024], v16[1024]; // kv_dim = 1024
        cvt_f32_to_f16_neon(k_buf, k16, kv_dim);
        cvt_f32_to_f16_neon(v_buf, v16, kv_dim);
        kv_write(kv, layer, pos % max_seq, k16, v16);
        if (kv->len < max_seq && pos + 1 > kv->len) kv->len = pos + 1;

        memcpy(Q_all + ti * dim, q_buf, dim * sizeof(float));
    }

    // ── Phase 2: Causal attention + output proj + FFN (per token) ────────
    _Float16 *kcache_base = kv_k(kv, layer);
    _Float16 *vcache_base = kv_v(kv, layer);

    for (int ti = 0; ti < chunk_size; ti++) {
        float *qi = Q_all + ti * dim;
        float *xi = X + ti * dim;
        int pos = positions[ti];
        int seq_len = pos + 1;
        if (seq_len > max_seq) seq_len = max_seq;
        int ring = (pos + 1 > max_seq);
        int ring_off = ring ? (pos + 1 - seq_len) : 0;

        // Pre-scale Q by 1/sqrt(head_dim)
        float inv_sqrt_hd = 1.0f / sqrtf((float)hd);
        vDSP_vsmul(qi, 1, &inv_sqrt_hd, qi, 1, (vDSP_Length)dim);

        memset(attn_out, 0, dim * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            float *qh = qi + h * hd;
            float *att_h = att + h * max_seq;
            int kvh = h / heads_per_kv;
            int kv_dim_chunk = (n_heads / heads_per_kv) * hd;

            // Q @ K^T — contiguous load per position (sequence-major layout)
            for (int t = 0; t < seq_len; t++) {
                int ct = ring ? ((ring_off + t) % max_seq) : t;
                const _Float16 *kt = kcache_base + (size_t)ct * kv_dim_chunk + kvh * hd;

                float32x4_t acc0 = vdupq_n_f32(0);
                float32x4_t acc1 = vdupq_n_f32(0);
                float32x4_t acc2 = vdupq_n_f32(0);
                float32x4_t acc3 = vdupq_n_f32(0);

                for (int d = 0; d < 128; d += 32) {
                    float16x8_t kv0 = vld1q_f16((const float16_t *)(kt + d));
                    float16x8_t kv1 = vld1q_f16((const float16_t *)(kt + d + 8));
                    float16x8_t kv2 = vld1q_f16((const float16_t *)(kt + d + 16));
                    float16x8_t kv3 = vld1q_f16((const float16_t *)(kt + d + 24));

                    acc0 = vfmaq_f32(acc0, vld1q_f32(qh + d),      vcvt_f32_f16(vget_low_f16(kv0)));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(qh + d + 4),  vcvt_f32_f16(vget_high_f16(kv0)));
                    acc2 = vfmaq_f32(acc2, vld1q_f32(qh + d + 8),  vcvt_f32_f16(vget_low_f16(kv1)));
                    acc3 = vfmaq_f32(acc3, vld1q_f32(qh + d + 12), vcvt_f32_f16(vget_high_f16(kv1)));
                    acc0 = vfmaq_f32(acc0, vld1q_f32(qh + d + 16), vcvt_f32_f16(vget_low_f16(kv2)));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(qh + d + 20), vcvt_f32_f16(vget_high_f16(kv2)));
                    acc2 = vfmaq_f32(acc2, vld1q_f32(qh + d + 24), vcvt_f32_f16(vget_low_f16(kv3)));
                    acc3 = vfmaq_f32(acc3, vld1q_f32(qh + d + 28), vcvt_f32_f16(vget_high_f16(kv3)));
                }

                acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
                att_h[t] = vaddvq_f32(acc0);
            }

            softmax_vec(att_h, seq_len);

            // Att @ V — contiguous load per position (sequence-major layout)
            float *oh = attn_out + h * hd;

            for (int t = 0; t < seq_len; t++) {
                float a = att_h[t];
                if (a == 0.0f) continue;
                int ct = ring ? ((ring_off + t) % max_seq) : t;
                const _Float16 *vt = vcache_base + (size_t)ct * kv_dim_chunk + kvh * hd;
                float32x4_t av = vdupq_n_f32(a);

                for (int d = 0; d < 128; d += 16) {
                    float16x8_t vv0 = vld1q_f16((const float16_t *)(vt + d));
                    float16x8_t vv1 = vld1q_f16((const float16_t *)(vt + d + 8));

                    float32x4_t o0 = vld1q_f32(oh + d);
                    float32x4_t o1 = vld1q_f32(oh + d + 4);
                    float32x4_t o2 = vld1q_f32(oh + d + 8);
                    float32x4_t o3 = vld1q_f32(oh + d + 12);

                    o0 = vfmaq_f32(o0, av, vcvt_f32_f16(vget_low_f16(vv0)));
                    o1 = vfmaq_f32(o1, av, vcvt_f32_f16(vget_high_f16(vv0)));
                    o2 = vfmaq_f32(o2, av, vcvt_f32_f16(vget_low_f16(vv1)));
                    o3 = vfmaq_f32(o3, av, vcvt_f32_f16(vget_high_f16(vv1)));

                    vst1q_f32(oh + d, o0);
                    vst1q_f32(oh + d + 4, o1);
                    vst1q_f32(oh + d + 8, o2);
                    vst1q_f32(oh + d + 12, o3);
                }
            }
        }

        // Output projection: Wo @ attn_out -> xb
        q4_matvec(lw->wo, lw->wo_type, attn_out, xb, dim, dim);

        // Residual
        vDSP_vadd(xi, 1, xb, 1, xi, 1, (vDSP_Length)dim);

        // FFN RMSNorm
        rmsnorm_vec(xb, xi, lw->rms_ffn, dim, c->rms_eps);

        // SwiGLU FFN
        q4_matvec(lw->w1, lw->w1_type, xb, hb, hidden, dim);
        q4_matvec(lw->w3, lw->w3_type, xb, hb2, hidden, dim);
        // SiLU(gate) * up (NEON polynomial sigmoid)
        silu_mul_neon(hb, hb2, hidden);
        q4_matvec(lw->w2, lw->w2_type, hb, xb, dim, hidden);

        // Residual
        vDSP_vadd(xi, 1, xb, 1, xi, 1, (vDSP_Length)dim);
    }

    free(Q_all);
    free(att);
    free(hb2);
    free(hb);
    free(attn_out);
    free(v_buf);
    free(k_buf);
    free(q_buf);
    free(xb);
}

// Process entire prompt in chunks, returning last token's hidden state in x_out.
static void mistral_prefill(MistralModel *m, KVCache *kv, int *tokens, int n_tokens, float *x_out) {
    int dim = m->cfg.dim;
    int n_layers = m->cfg.n_layers;
    int chunk = PREFILL_CHUNK_SIZE;

    float *X = (float *)malloc(chunk * dim * sizeof(float));
    int positions[PREFILL_CHUNK_SIZE];

    for (int start = 0; start < n_tokens; start += chunk) {
        int cs = n_tokens - start;
        if (cs > chunk) cs = chunk;

        // Embed all tokens in this chunk (on-demand dequant)
        for (int i = 0; i < cs; i++) {
            embed_token(m, tokens[start + i], X + i * dim);
            positions[i] = start + i;
        }

        // Process chunk through all layers
        for (int l = 0; l < n_layers; l++) {
            mistral_layer_prefill_chunk(m, kv, X, positions, cs, l);
        }
    }

    // Copy last token's hidden state to output
    int last_in_chunk = (n_tokens - 1) % chunk;
    memcpy(x_out, X + last_in_chunk * dim, dim * sizeof(float));

    free(X);
}

// ─── Compute logits from hidden state ────────────────────────────────────────
static void mistral_logits(MistralModel *m, const float *x) {
    // Final RMSNorm
    rmsnorm_vec(m->xb, x, m->rms_final, m->cfg.dim, m->cfg.rms_eps);
    // LM head: [vocab, dim] @ xb → logits — use SDOT for Q4_0 (32000 rows, biggest matvec)
    if (m->lm_head_type == GGML_TYPE_Q4_0) {
        quantize_f32_to_q8_0(m->xb, m->xb_q8, m->cfg.dim);
        q4_matvec_sdot_parallel(m->lm_head, m->lm_head_type, m->xb_q8,
                                 m->logits, m->cfg.vocab_size, m->cfg.dim);
    } else {
        q4_matvec_parallel(m->lm_head, m->lm_head_type, m->xb, m->logits,
                           m->cfg.vocab_size, m->cfg.dim);
    }
}

// ─── Token sampling ──────────────────────────────────────────────────────────

// Sampling parameters
typedef struct {
    float temp;         // temperature (0 = greedy)
    int top_k;          // top-k filtering (0 = disabled)
    float top_p;        // nucleus sampling threshold (1.0 = disabled)
    float rep_penalty;  // repetition penalty multiplier (1.0 = disabled)
    const int *prev_tokens;  // previous token IDs for rep penalty
    int n_prev;              // number of previous tokens
} SampleParams;

static int sample_argmax(const float *logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

static int sample_token(float *logits, int n, SampleParams *p) {
    // Apply repetition penalty
    if (p->rep_penalty != 1.0f && p->prev_tokens && p->n_prev > 0) {
        for (int i = 0; i < p->n_prev; i++) {
            int tok = p->prev_tokens[i];
            if (tok < 0 || tok >= n) continue;
            if (logits[tok] > 0)
                logits[tok] /= p->rep_penalty;
            else
                logits[tok] *= p->rep_penalty;
        }
    }

    if (p->temp <= 0) return sample_argmax(logits, n);

    int top_k = p->top_k > 0 ? p->top_k : 40;
    if (top_k > n) top_k = n;

    // Find top-k indices by partial sort
    int *topk_idx = (int *)malloc(top_k * sizeof(int));
    float *topk_val = (float *)malloc(top_k * sizeof(float));
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (k < top_k) {
            topk_idx[k] = i;
            topk_val[k] = logits[i];
            k++;
            if (k == top_k) {
                for (int a = 0; a < top_k; a++)
                    for (int b = a + 1; b < top_k; b++)
                        if (topk_val[a] > topk_val[b]) {
                            float tv = topk_val[a]; topk_val[a] = topk_val[b]; topk_val[b] = tv;
                            int ti = topk_idx[a]; topk_idx[a] = topk_idx[b]; topk_idx[b] = ti;
                        }
            }
        } else if (logits[i] > topk_val[0]) {
            topk_val[0] = logits[i];
            topk_idx[0] = i;
            for (int a = 0; a < top_k - 1; a++) {
                if (topk_val[a] > topk_val[a + 1]) {
                    float tv = topk_val[a]; topk_val[a] = topk_val[a + 1]; topk_val[a + 1] = tv;
                    int ti = topk_idx[a]; topk_idx[a] = topk_idx[a + 1]; topk_idx[a + 1] = ti;
                } else break;
            }
        }
    }

    // Sort top-k descending for top-p truncation
    for (int a = 0; a < top_k - 1; a++)
        for (int b = a + 1; b < top_k; b++)
            if (topk_val[a] < topk_val[b]) {
                float tv = topk_val[a]; topk_val[a] = topk_val[b]; topk_val[b] = tv;
                int ti = topk_idx[a]; topk_idx[a] = topk_idx[b]; topk_idx[b] = ti;
            }

    // Softmax over top-k with temperature
    float max_val = topk_val[0];
    float *probs = (float *)malloc(top_k * sizeof(float));
    float sum = 0;
    for (int i = 0; i < top_k; i++) {
        probs[i] = expf((topk_val[i] - max_val) / p->temp);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) probs[i] /= sum;

    // Top-p (nucleus) truncation
    int effective_k = top_k;
    if (p->top_p > 0 && p->top_p < 1.0f) {
        float cumsum = 0;
        for (int i = 0; i < top_k; i++) {
            cumsum += probs[i];
            if (cumsum >= p->top_p) {
                effective_k = i + 1;
                break;
            }
        }
        // Renormalize
        sum = 0;
        for (int i = 0; i < effective_k; i++) sum += probs[i];
        for (int i = 0; i < effective_k; i++) probs[i] /= sum;
    }

    // Sample from distribution
    float r = (float)arc4random() / (float)UINT32_MAX;
    float cdf = 0;
    int result = topk_idx[0];
    for (int i = 0; i < effective_k; i++) {
        cdf += probs[i];
        if (r < cdf) { result = topk_idx[i]; break; }
    }

    free(topk_idx);
    free(topk_val);
    free(probs);
    return result;
}

// Legacy wrapper
static int sample_temperature(const float *logits_in, int n, float temp) {
    float *logits = (float *)malloc(n * sizeof(float));
    memcpy(logits, logits_in, n * sizeof(float));
    SampleParams p = { .temp = temp, .top_k = 40, .top_p = 0.9f, .rep_penalty = 1.0f };
    int tok = sample_token(logits, n, &p);
    free(logits);
    return tok;
}

// ─── Free model ──────────────────────────────────────────────────────────────
static void mistral_free(MistralModel *m) {
    if (!m) return;
    for (int l = 0; l < m->cfg.n_layers; l++) {
        free(m->layers[l].rms_att);
        free(m->layers[l].rms_ffn);
    }
    free(m->token_embed);
    free(m->rms_final);
    free(m->rope_theta_inv);
    free(m->xb); free(m->xb2);
    free(m->q); free(m->k); free(m->v);
    free(m->att);
    free(m->hb); free(m->hb2);
    free(m->logits);
    gguf_close(m->gguf);
    free(m);
}
