// test_cpu_ref.h -- CPU reference implementations for all Mistral 7B ops
// These are ground-truth implementations using Accelerate where possible.
// All tensors are row-major [S, D] unless noted otherwise.
#pragma once
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------- Mistral 7B config ----------
typedef struct {
    int dim;            // 4096
    int hidden_dim;     // 14336
    int n_heads;        // 32
    int n_kv_heads;     // 8  (GQA: 4 Q heads per KV head)
    int head_dim;       // 128
    int vocab_size;     // 32000
    int n_layers;       // 32
    int max_seq;        // 32768 (sliding window 4096)
    float rope_theta;   // 10000.0  (Mistral uses 1000000.0 for v0.3+)
} MistralConfig;

static MistralConfig mistral_7b_config(void) {
    return (MistralConfig){
        .dim = 4096, .hidden_dim = 14336,
        .n_heads = 32, .n_kv_heads = 8, .head_dim = 128,
        .vocab_size = 32000, .n_layers = 32, .max_seq = 32768,
        .rope_theta = 1000000.0f
    };
}

// Smaller config for fast unit tests
static MistralConfig mistral_test_config(void) {
    return (MistralConfig){
        .dim = 256, .hidden_dim = 512,
        .n_heads = 8, .n_kv_heads = 2, .head_dim = 32,
        .vocab_size = 256, .n_layers = 2, .max_seq = 128,
        .rope_theta = 10000.0f
    };
}

// ---------- RMSNorm ----------
// x: [S, D], w: [D], out: [S, D]
static void ref_rmsnorm(float *out, const float *x, const float *w, int S, int D) {
    for (int t = 0; t < S; t++) {
        const float *row = x + t * D;
        float *orow = out + t * D;
        float ss = 0;
        vDSP_dotpr(row, 1, row, 1, &ss, (vDSP_Length)D);
        ss = 1.0f / sqrtf(ss / D + 1e-5f);
        for (int i = 0; i < D; i++)
            orow[i] = row[i] * ss * w[i];
    }
}

// ---------- RoPE (Mistral-style) ----------
// q: [S, n_heads * head_dim], k: [S, n_kv_heads * head_dim]
// pos_offset: starting position (for decode with KV cache)
static void ref_rope(float *q, float *k, int S, int n_heads, int n_kv_heads,
                     int head_dim, float theta, int pos_offset) {
    for (int t = 0; t < S; t++) {
        int pos = t + pos_offset;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, 2.0f * i / head_dim);
            float angle = pos * freq;
            float cos_v = cosf(angle), sin_v = sinf(angle);
            // Apply to all Q heads
            for (int h = 0; h < n_heads; h++) {
                int off = t * n_heads * head_dim + h * head_dim + 2 * i;
                float q0 = q[off], q1 = q[off + 1];
                q[off]     = q0 * cos_v - q1 * sin_v;
                q[off + 1] = q0 * sin_v + q1 * cos_v;
            }
            // Apply to all KV heads
            for (int h = 0; h < n_kv_heads; h++) {
                int off = t * n_kv_heads * head_dim + h * head_dim + 2 * i;
                float k0 = k[off], k1 = k[off + 1];
                k[off]     = k0 * cos_v - k1 * sin_v;
                k[off + 1] = k0 * sin_v + k1 * cos_v;
            }
        }
    }
}

// ---------- SiLU ----------
static void ref_silu(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] / (1.0f + expf(-x[i]));
}

// ---------- Softmax (numerically stable) ----------
// x: [rows, cols], out: [rows, cols], softmax along cols (last dim)
static void ref_softmax(float *out, const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = x + r * cols;
        float *orow = out + r * cols;
        float mx;
        vDSP_maxv(row, 1, &mx, (vDSP_Length)cols);
        float sum = 0;
        for (int i = 0; i < cols; i++) {
            orow[i] = expf(row[i] - mx);
            sum += orow[i];
        }
        float inv = 1.0f / sum;
        vDSP_vsmul(orow, 1, &inv, orow, 1, (vDSP_Length)cols);
    }
}

// ---------- GQA Attention ----------
// q: [S, n_heads, head_dim], k: [S, n_kv_heads, head_dim], v: same
// out: [S, n_heads, head_dim]
// Causal masking. GQA: each KV head serves (n_heads/n_kv_heads) Q heads.
static void ref_gqa_attention(float *out, const float *q, const float *k, const float *v,
                               int S, int n_heads, int n_kv_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int gqa_ratio = n_heads / n_kv_heads;
    float *scores = (float *)malloc(S * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        for (int t = 0; t < S; t++) {
            // Compute scores for position t, head h
            float mx = -1e30f;
            for (int s = 0; s <= t; s++) {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += q[t * n_heads * head_dim + h * head_dim + d]
                         * k[s * n_kv_heads * head_dim + kv_h * head_dim + d];
                scores[s] = dot * scale;
                if (scores[s] > mx) mx = scores[s];
            }
            // Stable softmax
            float sm = 0;
            for (int s = 0; s <= t; s++) { scores[s] = expf(scores[s] - mx); sm += scores[s]; }
            float inv = 1.0f / sm;
            for (int s = 0; s <= t; s++) scores[s] *= inv;
            // Weighted sum of V
            for (int d = 0; d < head_dim; d++) {
                float val = 0;
                for (int s = 0; s <= t; s++)
                    val += scores[s] * v[s * n_kv_heads * head_dim + kv_h * head_dim + d];
                out[t * n_heads * head_dim + h * head_dim + d] = val;
            }
        }
    }
    free(scores);
}

// ---------- Matmul via Accelerate BLAS ----------
// C = A @ B^T, A: [M, K], B: [N, K], C: [M, N]
static void ref_matmul(float *C, const float *A, const float *B, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
}

// C = A @ B (no transpose), A: [M, K], B: [K, N], C: [M, N]
static void ref_matmul_nn(float *C, const float *A, const float *B, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

// ---------- SwiGLU FFN ----------
// x: [S, dim], w1: [hidden, dim], w2: [dim, hidden], w3: [hidden, dim]
// out: [S, dim]
// out = w2 @ (silu(w1 @ x) * (w3 @ x))
static void ref_ffn(float *out, const float *x, const float *w1, const float *w2,
                    const float *w3, int S, int dim, int hidden) {
    float *h1 = (float *)malloc(S * hidden * sizeof(float));
    float *h3 = (float *)malloc(S * hidden * sizeof(float));
    float *gate = (float *)malloc(S * hidden * sizeof(float));

    // h1 = x @ w1^T, h3 = x @ w3^T
    ref_matmul(h1, x, w1, S, hidden, dim);
    ref_matmul(h3, x, w3, S, hidden, dim);

    // gate = silu(h1) * h3
    ref_silu(gate, h1, S * hidden);
    for (int i = 0; i < S * hidden; i++) gate[i] *= h3[i];

    // out = gate @ w2^T
    ref_matmul(out, gate, w2, S, dim, hidden);

    free(h1); free(h3); free(gate);
}

// ---------- Full transformer layer ----------
// x: [S, dim] (in-place residual)
// Returns modified x with residual connections applied.
static void ref_transformer_layer(float *x, const float *rms_att_w, const float *rms_ffn_w,
                                    const float *wq, const float *wk, const float *wv,
                                    const float *wo, const float *w1, const float *w2,
                                    const float *w3, int S, int n_heads, int n_kv_heads,
                                    int head_dim, int dim, int hidden, float rope_theta,
                                    int pos_offset) {
    // Pre-attention RMSNorm
    float *xnorm = (float *)malloc(S * dim * sizeof(float));
    ref_rmsnorm(xnorm, x, rms_att_w, S, dim);

    // QKV projections: q=[S, n_heads*hd], k=[S, n_kv_heads*hd], v=[S, n_kv_heads*hd]
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    float *q = (float *)malloc(S * q_dim * sizeof(float));
    float *k = (float *)malloc(S * kv_dim * sizeof(float));
    float *v = (float *)malloc(S * kv_dim * sizeof(float));

    ref_matmul(q, xnorm, wq, S, q_dim, dim);
    ref_matmul(k, xnorm, wk, S, kv_dim, dim);
    ref_matmul(v, xnorm, wv, S, kv_dim, dim);

    // RoPE
    ref_rope(q, k, S, n_heads, n_kv_heads, head_dim, rope_theta, pos_offset);

    // GQA attention
    float *attn_out = (float *)malloc(S * q_dim * sizeof(float));
    ref_gqa_attention(attn_out, q, k, v, S, n_heads, n_kv_heads, head_dim);

    // Output projection + residual
    float *o_out = (float *)malloc(S * dim * sizeof(float));
    ref_matmul(o_out, attn_out, wo, S, dim, q_dim);
    for (int i = 0; i < S * dim; i++) x[i] += o_out[i];

    // Pre-FFN RMSNorm
    ref_rmsnorm(xnorm, x, rms_ffn_w, S, dim);

    // FFN + residual
    float *ffn_out = (float *)malloc(S * dim * sizeof(float));
    ref_ffn(ffn_out, xnorm, w1, w2, w3, S, dim, hidden);
    for (int i = 0; i < S * dim; i++) x[i] += ffn_out[i];

    free(xnorm); free(q); free(k); free(v);
    free(attn_out); free(o_out); free(ffn_out);
}

// ---------- Q4_0 dequantization (llama.cpp compatible) ----------
// Q4_0 block: 2 bytes scale (fp16) + 16 bytes data (32 nibbles) = 18 bytes per block of 32
#define Q4_0_BLOCK_SIZE 32
typedef struct { _Float16 d; uint8_t qs[Q4_0_BLOCK_SIZE / 2]; } block_q4_0;

static void ref_dequant_q4_0(float *out, const void *data, int n) {
    int nb = n / Q4_0_BLOCK_SIZE;
    const block_q4_0 *blocks = (const block_q4_0 *)data;
    for (int b = 0; b < nb; b++) {
        float d = (float)blocks[b].d;
        for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++) {
            uint8_t byte = blocks[b].qs[j];
            out[b * Q4_0_BLOCK_SIZE + j]                    = d * ((int)(byte & 0xF) - 8);
            out[b * Q4_0_BLOCK_SIZE + j + Q4_0_BLOCK_SIZE/2] = d * ((int)(byte >> 4) - 8);
        }
    }
}

// ---------- Q4_K_M dequantization (simplified, matches llama.cpp) ----------
// Q4_K block: super-blocks of 256 with sub-blocks of 32
// For test purposes, we validate against known byte patterns
#define QK_K 256
typedef struct {
    _Float16 d;
    _Float16 dmin;
    uint8_t scales[12]; // packed 6-bit scales + mins
    uint8_t qs[QK_K / 2];
} block_q4_K;

static void ref_dequant_q4_K(float *out, const void *data, int n) {
    int nb = n / QK_K;
    const block_q4_K *blocks = (const block_q4_K *)data;
    for (int b = 0; b < nb; b++) {
        float d = (float)blocks[b].d;
        float dmin = (float)blocks[b].dmin;
        // Unpack scales and mins from the 12-byte packed format
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 4; j++) {
            sc[j] = blocks[b].scales[j] & 0x3F;
            mn[j] = blocks[b].scales[j + 4] & 0x3F;
        }
        for (int j = 0; j < 4; j++) {
            sc[j + 4] = ((blocks[b].scales[j + 8] & 0xF) << 2) | (blocks[b].scales[j] >> 6);
            mn[j + 4] = ((blocks[b].scales[j + 8] >> 4) << 2) | (blocks[b].scales[j + 4] >> 6);
        }
        // Dequantize each sub-block of 32
        for (int sb = 0; sb < QK_K / 32; sb++) {
            float scale = d * sc[sb];
            float min_val = dmin * mn[sb];
            for (int j = 0; j < 16; j++) {
                uint8_t byte = blocks[b].qs[sb * 16 + j];
                out[b * QK_K + sb * 32 + j]      = scale * (byte & 0xF) - min_val;
                out[b * QK_K + sb * 32 + j + 16]  = scale * (byte >> 4) - min_val;
            }
        }
    }
}
