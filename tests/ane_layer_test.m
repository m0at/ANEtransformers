// ane_layer_test.m — Single transformer layer: ANE vs BLAS correctness test
//
// Loads a Mistral 7B GGUF, runs layer 0 through both BLAS and ANE paths,
// compares output element-wise. Tests the transpose approach independently
// (does NOT rely on the buggy ane_baked_prefill_layer code).
//
// Build:
//   cd mistral && xcrun clang -O2 -ffast-math -fobjc-arc -Wall -Wno-unused-function \
//     -Wno-unused-variable -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -I. \
//     -o ane_layer_test ../tests/ane_layer_test.m \
//     -framework Foundation -framework IOSurface -framework Accelerate \
//     -framework Metal -framework MetalPerformanceShaders -ldl
//
// Usage:
//   ./ane_layer_test --model ~/models/mistral-7b-instruct-v0.2.Q4_0.gguf

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Pull in the project headers (order matters: gguf_loader -> dequant -> kv_cache -> model -> ane stuff)
#include "mistral_model.h"
#include "mistral_ane_prefill.h"

// Transpose helpers are now in mistral_ane_prefill.h (transpose_to_ane / transpose_from_ane)

// ─── Timing helper ──────────────────────────────────────────────────────────
static double test_time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

// ─── Error metrics ──────────────────────────────────────────────────────────
static void compute_errors(const float *ref, const float *test, int n,
                           float *out_max_err, float *out_mean_err) {
    float max_e = 0, sum_e = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(ref[i] - test[i]);
        if (e > max_e) max_e = e;
        sum_e += e;
    }
    *out_max_err = max_e;
    *out_mean_err = sum_e / (float)n;
}

// ─── ANE layer with proper transpose (independent of mistral_ane_prefill.h) ─
// This implements the correct transposed path:
//   1. RMSNorm (CPU, token-first)
//   2. Transpose token-first -> channel-first for ANE I/O
//   3. Q projection (ANE) -> transpose output back
//   4. K+V projection (ANE) -> transpose outputs back
//   5. RoPE (CPU, token-first)
//   6. KV cache write (CPU)
//   7. Attention (CPU)
//   8. Wo projection (ANE, with transposes)
//   9. Residual add
//  10. FFN RMSNorm (CPU)
//  11. FFN (ANE, with transposes)
//  12. Residual add
static void ane_layer_with_transpose(MistralModel *m, KVCache *kv, float *X,
                                      ANEKernel *lk[ANE_LK_COUNT],
                                      int layer, int start_pos, int S) {
    MistralConfig *c = &m->cfg;
    LayerWeights *lw = &m->layers[layer];
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;

    size_t dim_S_bytes = (size_t)dim * S * sizeof(float);
    size_t kv_S_bytes = (size_t)kv_dim * S * sizeof(float);

    // Scratch buffers
    float *normed   = (float *)calloc((size_t)dim * S, sizeof(float));
    float *ane_in   = (float *)calloc((size_t)dim * S, sizeof(float));   // channel-first
    float *ane_out  = (float *)calloc((size_t)dim * S, sizeof(float));   // channel-first
    float *Q_buf    = (float *)calloc((size_t)dim * S, sizeof(float));   // token-first
    float *K_buf    = (float *)calloc((size_t)kv_dim * S, sizeof(float));
    float *V_buf    = (float *)calloc((size_t)kv_dim * S, sizeof(float));
    float *ane_kv_out = (float *)calloc((size_t)kv_dim * S, sizeof(float)); // channel-first
    float *Wo_in    = (float *)calloc((size_t)dim * S, sizeof(float));   // token-first (attn output)
    float *Wo_out   = (float *)calloc((size_t)dim * S, sizeof(float));   // token-first
    float *ffn_norm = (float *)calloc((size_t)dim * S, sizeof(float));   // token-first
    float *ffn_out  = (float *)calloc((size_t)dim * S, sizeof(float));   // token-first
    float *att      = (float *)calloc((size_t)c->n_heads * kv->max_seq, sizeof(float));

    // 1. Attention RMSNorm (CPU, token-first)
    rmsnorm_batch(normed, X, lw->rms_att, dim, S, c->rms_eps);

    // 2. Transpose to channel-first for ANE
    transpose_to_ane(normed, ane_in, dim, S);

    // 3. Q projection (ANE)
    ane_write_input(lk[ANE_LK_Q], 0, ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_Q]);
    ane_read_output(lk[ANE_LK_Q], 0, ane_out, dim_S_bytes);
    transpose_from_ane(ane_out, Q_buf, dim, S);

    // 4. Fused K+V projection (ANE, 2 outputs)
    ane_write_input(lk[ANE_LK_KV], 0, ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_KV]);
    ane_read_output(lk[ANE_LK_KV], 0, ane_kv_out, kv_S_bytes);
    transpose_from_ane(ane_kv_out, K_buf, kv_dim, S);
    ane_read_output(lk[ANE_LK_KV], 1, ane_kv_out, kv_S_bytes);
    transpose_from_ane(ane_kv_out, V_buf, kv_dim, S);

    // 5. RoPE (CPU, token-first)
    apply_rope_batch(Q_buf, K_buf, start_pos,
                     m->rope_theta_inv,
                     c->n_heads, c->n_kv_heads, c->head_dim, S);

    // 6. Write K,V to cache
    kv_write_batch(kv, layer, start_pos, K_buf, V_buf, kv_dim, S);

    // 7. Attention (CPU)
    attention_batch(Q_buf, Wo_in, kv, layer, start_pos, c, S, att);

    // 8. Wo projection (ANE, with transposes)
    transpose_to_ane(Wo_in, ane_in, dim, S);
    ane_write_input(lk[ANE_LK_WO], 0, ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_WO]);
    ane_read_output(lk[ANE_LK_WO], 0, ane_out, dim_S_bytes);
    transpose_from_ane(ane_out, Wo_out, dim, S);

    // 9. Residual
    vDSP_vadd(X, 1, Wo_out, 1, X, 1, (vDSP_Length)(dim * S));

    // 10. FFN RMSNorm (CPU, token-first)
    rmsnorm_batch(ffn_norm, X, lw->rms_ffn, dim, S, c->rms_eps);

    // 11. Fused FFN (ANE, with transposes)
    transpose_to_ane(ffn_norm, ane_in, dim, S);
    ane_write_input(lk[ANE_LK_FFN], 0, ane_in, dim_S_bytes);
    ane_eval(lk[ANE_LK_FFN]);
    ane_read_output(lk[ANE_LK_FFN], 0, ane_out, dim_S_bytes);
    transpose_from_ane(ane_out, ffn_out, dim, S);

    // 12. Residual
    vDSP_vadd(X, 1, ffn_out, 1, X, 1, (vDSP_Length)(dim * S));

    free(normed); free(ane_in); free(ane_out);
    free(Q_buf); free(K_buf); free(V_buf); free(ane_kv_out);
    free(Wo_in); free(Wo_out); free(ffn_norm); free(ffn_out); free(att);
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = NULL;
        int S = 64;  // sequence length (must be >= 16 for ANE)

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
                model_path = argv[++i];
            else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc)
                S = atoi(argv[++i]);
        }

        if (!model_path) {
            fprintf(stderr, "Usage: %s --model <path.gguf> [--seq <S>]\n", argv[0]);
            return 1;
        }

        if (S < 16) {
            fprintf(stderr, "ERROR: S=%d too small for ANE (minimum 16)\n", S);
            return 1;
        }

        // ── Load model ──────────────────────────────────────────────────
        fprintf(stderr, "Loading model: %s\n", model_path);
        MistralModel *m = mistral_load(model_path);
        if (!m) { fprintf(stderr, "Failed to load model\n"); return 1; }

        MistralConfig *c = &m->cfg;
        int dim = c->dim;
        int kv_dim = c->n_kv_heads * c->head_dim;
        int max_seq = c->sliding_window > 0 ? c->sliding_window : 4096;

        fprintf(stderr, "Model: dim=%d kv_dim=%d hidden=%d layers=%d\n",
                dim, kv_dim, c->hidden_dim, c->n_layers);
        fprintf(stderr, "Test:  layer=0 S=%d max_seq=%d\n", S, max_seq);

        // ── Create test input tokens ────────────────────────────────────
        // Use BOS + some common tokens to get realistic embeddings
        int *tokens = (int *)malloc(S * sizeof(int));
        tokens[0] = 1;  // BOS
        for (int i = 1; i < S; i++)
            tokens[i] = 1000 + (i % 500);  // arbitrary but deterministic

        // ── Embed tokens ────────────────────────────────────────────────
        float *X_blas = (float *)calloc((size_t)dim * S, sizeof(float));
        float *X_ane  = (float *)calloc((size_t)dim * S, sizeof(float));
        for (int t = 0; t < S; t++) {
            embed_token(m, tokens[t], X_blas + t * dim);
        }
        memcpy(X_ane, X_blas, (size_t)dim * S * sizeof(float));

        // ── BLAS reference path ─────────────────────────────────────────
        fprintf(stderr, "\n=== BLAS reference (layer 0) ===\n");
        KVCache kv_blas = kv_alloc(c->n_layers, c->n_kv_heads, c->head_dim, max_seq);

        if (!blas_prefill_init(m, S, max_seq)) {
            fprintf(stderr, "BLAS init failed\n");
            return 1;
        }

        double t0 = test_time_ms();
        blas_prefill_layer(m, &kv_blas, X_blas, 0, 0, S);
        double t_blas = test_time_ms() - t0;
        fprintf(stderr, "BLAS layer 0: %.1f ms\n", t_blas);

        // ── ANE path with correct transpose ─────────────────────────────
        fprintf(stderr, "\n=== ANE + transpose (layer 0) ===\n");
        KVCache kv_ane = kv_alloc(c->n_layers, c->n_kv_heads, c->head_dim, max_seq);

        // Initialize ANE baked prefill state (generates MIL, allocates scratch)
        if (!ane_baked_prefill_init(m, S, max_seq)) {
            fprintf(stderr, "ANE init failed\n");
            return 1;
        }

        // Compile 4 ANE programs for layer 0
        ANEKernel *lk[ANE_LK_COUNT];
        fprintf(stderr, "Compiling ANE kernels for layer 0...\n");
        t0 = test_time_ms();
        if (!ane_compile_layer(m, 0, lk)) {
            fprintf(stderr, "ANE compile failed for layer 0\n");
            return 1;
        }
        double t_compile = test_time_ms() - t0;
        fprintf(stderr, "ANE compile: %.1f ms (4 programs)\n", t_compile);

        // Run ANE layer with proper transpose
        t0 = test_time_ms();
        ane_layer_with_transpose(m, &kv_ane, X_ane, lk, 0, 0, S);
        double t_ane = test_time_ms() - t0;
        fprintf(stderr, "ANE layer 0: %.1f ms\n", t_ane);

        // ── Compare outputs ─────────────────────────────────────────────
        fprintf(stderr, "\n=== Comparison ===\n");

        int n = dim * S;
        float max_err, mean_err;
        compute_errors(X_blas, X_ane, n, &max_err, &mean_err);

        fprintf(stderr, "Output shape: [%d, %d] = %d elements\n", S, dim, n);
        fprintf(stderr, "Max  error:  %.6f\n", max_err);
        fprintf(stderr, "Mean error:  %.6f\n", mean_err);

        // Print a few sample values for sanity
        fprintf(stderr, "\nSample values (first token, first 8 dims):\n");
        fprintf(stderr, "  BLAS: ");
        for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%8.4f ", X_blas[i]);
        fprintf(stderr, "\n  ANE:  ");
        for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%8.4f ", X_ane[i]);
        fprintf(stderr, "\n");

        fprintf(stderr, "\nSample values (last token, first 8 dims):\n");
        fprintf(stderr, "  BLAS: ");
        for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%8.4f ", X_blas[(S-1)*dim + i]);
        fprintf(stderr, "\n  ANE:  ");
        for (int i = 0; i < 8 && i < dim; i++) fprintf(stderr, "%8.4f ", X_ane[(S-1)*dim + i]);
        fprintf(stderr, "\n");

        // Find worst element
        float worst = 0;
        int worst_idx = 0;
        for (int i = 0; i < n; i++) {
            float e = fabsf(X_blas[i] - X_ane[i]);
            if (e > worst) { worst = e; worst_idx = i; }
        }
        int worst_t = worst_idx / dim, worst_d = worst_idx % dim;
        fprintf(stderr, "\nWorst element: [t=%d, d=%d] blas=%.6f ane=%.6f err=%.6f\n",
                worst_t, worst_d, X_blas[worst_idx], X_ane[worst_idx], worst);

        // ── Pass/fail ───────────────────────────────────────────────────
        bool pass = (max_err < 0.1f) && (mean_err < 0.01f);
        fprintf(stderr, "\n%s  max_err=%.6f (<0.1) mean_err=%.6f (<0.01)\n",
                pass ? "PASS" : "FAIL", max_err, mean_err);

        // ── Cleanup ─────────────────────────────────────────────────────
        ane_free_layer(lk);
        ane_baked_prefill_cleanup();
        blas_prefill_cleanup();
        kv_free(&kv_blas);
        kv_free(&kv_ane);
        free(X_blas);
        free(X_ane);
        free(tokens);

        return pass ? 0 : 1;
    }
}
