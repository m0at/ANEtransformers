// ane_prefill_compare.m — Compare ANE baked prefill vs BLAS prefill output
//
// Loads actual Mistral 7B, runs both prefill paths on the same prompt,
// compares hidden states and KV cache contents.
//
// Build (from mistral/ dir):
//   make ane_prefill_compare
// Usage:
//   ./ane_prefill_compare --model ~/models/mistral-7b-instruct-v0.2.Q4_0.gguf

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>

#include "gguf_loader.h"
#include "dequant.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "mistral_model.h"
#include "mistral_ane_prefill.h"

static double time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

// Compute max absolute error between two float arrays
static float max_abs_err(const float *a, const float *b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// Compute mean absolute error
static float mean_abs_err(const float *a, const float *b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += fabsf(a[i] - b[i]);
    return sum / n;
}

// Cosine similarity between two float arrays
static float cosine_sim(const float *a, const float *b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

// Cosine similarity between two fp16 arrays
static float cosine_sim_f16(const _Float16 *a, const _Float16 *b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        float fa = (float)a[i], fb = (float)b[i];
        dot += fa * fb;
        na += fa * fa;
        nb += fb * fb;
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

// Max absolute error between two fp16 arrays
static float max_abs_err_f16(const _Float16 *a, const _Float16 *b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf((float)a[i] - (float)b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// Check if array is all zeros
static bool is_all_zeros(const float *a, int n) {
    for (int i = 0; i < n; i++)
        if (a[i] != 0.0f) return false;
    return true;
}

// Print first N non-zero elements for debugging
static void print_sample(const char *label, const float *a, int n, int max_print) {
    fprintf(stderr, "  %s [first %d]: ", label, max_print);
    int printed = 0;
    for (int i = 0; i < n && printed < max_print; i++) {
        fprintf(stderr, "%.4f ", a[i]);
        printed++;
    }
    fprintf(stderr, "\n");
}

static void print_sample_f16(const char *label, const _Float16 *a, int n, int max_print) {
    fprintf(stderr, "  %s [first %d]: ", label, max_print);
    int printed = 0;
    for (int i = 0; i < n && printed < max_print; i++) {
        fprintf(stderr, "%.4f ", (float)a[i]);
        printed++;
    }
    fprintf(stderr, "\n");
}

// Compute L2 norm
static float l2_norm(const float *a, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * a[i];
    return sqrtf(s);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = NULL;
        const char *prompt = "The capital of France is";

        static struct option long_opts[] = {
            {"model",  required_argument, 0, 'm'},
            {"prompt", required_argument, 0, 'p'},
            {0, 0, 0, 0}
        };
        int opt;
        while ((opt = getopt_long(argc, argv, "m:p:", long_opts, NULL)) != -1) {
            switch (opt) {
                case 'm': model_path = optarg; break;
                case 'p': prompt = optarg; break;
            }
        }

        if (!model_path) {
            fprintf(stderr, "Usage: %s --model <gguf> [--prompt \"...\"]\n", argv[0]);
            return 1;
        }

        // ── Load model ──────────────────────────────────────────────────
        fprintf(stderr, "Loading model: %s\n", model_path);
        double t0 = time_ms();
        MistralModel *model = mistral_load(model_path);
        if (!model) {
            fprintf(stderr, "FAIL: could not load model\n");
            return 1;
        }
        fprintf(stderr, "Model loaded in %.1f ms\n", time_ms() - t0);

        MistralConfig *cfg = &model->cfg;
        int dim = cfg->dim;
        int kv_dim = cfg->n_kv_heads * cfg->head_dim;

        // ── Init tokenizer ──────────────────────────────────────────────
        Tokenizer *tok = tokenizer_init(model->gguf);
        if (!tok) {
            fprintf(stderr, "FAIL: tokenizer init\n");
            mistral_free(model);
            return 1;
        }

        // ── Tokenize prompt ─────────────────────────────────────────────
        int tokens[512];
        int n_tokens = tokenizer_encode(tok, prompt, tokens, 512, 1);
        fprintf(stderr, "Prompt: \"%s\" -> %d tokens: [", prompt, n_tokens);
        for (int i = 0; i < n_tokens; i++)
            fprintf(stderr, "%d%s", tokens[i], i < n_tokens - 1 ? ", " : "");
        fprintf(stderr, "]\n");

        if (n_tokens < 2) {
            fprintf(stderr, "FAIL: need at least 2 tokens (got %d)\n", n_tokens);
            tokenizer_free(tok);
            mistral_free(model);
            return 1;
        }

        int max_seq = 4096;

        // ── Create TWO separate KV caches ───────────────────────────────
        KVCache kv_blas = kv_alloc(cfg->n_layers, cfg->n_kv_heads, cfg->head_dim, max_seq);
        KVCache kv_ane  = kv_alloc(cfg->n_layers, cfg->n_kv_heads, cfg->head_dim, max_seq);

        float *x_blas = (float *)calloc(dim, sizeof(float));
        float *x_ane  = (float *)calloc(dim, sizeof(float));

        // ── Run BLAS prefill ────────────────────────────────────────────
        fprintf(stderr, "\n=== BLAS PREFILL ===\n");
        t0 = time_ms();
        bool blas_ok = blas_prefill_forward(model, &kv_blas, tokens, n_tokens, x_blas);
        double t_blas = time_ms() - t0;
        if (!blas_ok) {
            fprintf(stderr, "FAIL: BLAS prefill failed\n");
            goto cleanup;
        }
        fprintf(stderr, "BLAS prefill: %.1f ms (%.1f tok/s)\n",
                t_blas, n_tokens / (t_blas / 1000.0));

        // Reset global state so ANE prefill can init cleanly
        blas_prefill_cleanup();

        // ── Run ANE baked prefill ───────────────────────────────────────
        fprintf(stderr, "\n=== ANE BAKED PREFILL ===\n");
        // Reset the tried-and-failed flag in case a previous run set it
        g_ane_baked_tried_and_failed = false;

        t0 = time_ms();
        bool ane_ok = ane_baked_prefill_forward(model, &kv_ane, tokens, n_tokens, x_ane);
        double t_ane = time_ms() - t0;
        if (!ane_ok) {
            fprintf(stderr, "FAIL: ANE baked prefill failed (compile error or ANE unavailable)\n");
            fprintf(stderr, "  This is expected if ANE private API is unavailable.\n");
            goto cleanup;
        }
        fprintf(stderr, "ANE baked prefill: %.1f ms (%.1f tok/s)\n",
                t_ane, n_tokens / (t_ane / 1000.0));

        ane_baked_prefill_cleanup();

        // ══════════════════════════════════════════════════════════════════
        // COMPARISON
        // ══════════════════════════════════════════════════════════════════
        fprintf(stderr, "\n========================================\n");
        fprintf(stderr, "   COMPARISON: BLAS vs ANE PREFILL\n");
        fprintf(stderr, "========================================\n\n");

        // ── Hidden state comparison ─────────────────────────────────────
        float h_max_err   = max_abs_err(x_blas, x_ane, dim);
        float h_mean_err  = mean_abs_err(x_blas, x_ane, dim);
        float h_cos_sim   = cosine_sim(x_blas, x_ane, dim);
        float blas_norm   = l2_norm(x_blas, dim);
        float ane_norm    = l2_norm(x_ane, dim);
        bool ane_zeros    = is_all_zeros(x_ane, dim);

        fprintf(stderr, "--- Hidden state (x_out, dim=%d) ---\n", dim);
        fprintf(stderr, "  BLAS L2 norm:    %.4f\n", blas_norm);
        fprintf(stderr, "  ANE  L2 norm:    %.4f\n", ane_norm);
        fprintf(stderr, "  Max abs error:   %.6f\n", h_max_err);
        fprintf(stderr, "  Mean abs error:  %.6f\n", h_mean_err);
        fprintf(stderr, "  Cosine sim:      %.6f\n", h_cos_sim);
        print_sample("BLAS x_out", x_blas, dim, 8);
        print_sample("ANE  x_out", x_ane, dim, 8);

        // Verdict on hidden state
        bool h_pass = (h_max_err < 1.0f && h_cos_sim > 0.995f);
        if (ane_zeros) {
            fprintf(stderr, "\n  ** ANE prefill output is ALL ZEROS **\n");
            fprintf(stderr, "  ANE prefill output is WRONG -- likely transpose bug (channel-first vs token-first)\n\n");
            h_pass = false;
        } else if (h_cos_sim < 0.5f) {
            fprintf(stderr, "\n  ** ANE prefill output is WRONG -- likely transpose bug (channel-first vs token-first) **\n");
            fprintf(stderr, "  cosine_sim=%.4f is far below 0.99 threshold\n\n", h_cos_sim);
            h_pass = false;
        } else if (!h_pass) {
            fprintf(stderr, "\n  ** Hidden state mismatch exceeds tolerance **\n");
            fprintf(stderr, "  max_err=%.4f (threshold 0.5), cosine_sim=%.6f (threshold 0.99)\n\n",
                    h_max_err, h_cos_sim);
        }

        fprintf(stderr, "  Hidden state: %s\n\n", h_pass ? "PASS" : "FAIL");

        // ── KV cache comparison (layer 0) ───────────────────────────────
        fprintf(stderr, "--- KV cache layer 0 (first %d positions, kv_dim=%d) ---\n",
                n_tokens, kv_dim);

        _Float16 *k_blas_L0 = kv_k(&kv_blas, 0);
        _Float16 *k_ane_L0  = kv_k(&kv_ane, 0);
        _Float16 *v_blas_L0 = kv_v(&kv_blas, 0);
        _Float16 *v_ane_L0  = kv_v(&kv_ane, 0);

        int kv_total = n_tokens * kv_dim;
        float k_max_err  = max_abs_err_f16(k_blas_L0, k_ane_L0, kv_total);
        float k_cos_sim  = cosine_sim_f16(k_blas_L0, k_ane_L0, kv_total);
        float v_max_err  = max_abs_err_f16(v_blas_L0, v_ane_L0, kv_total);
        float v_cos_sim  = cosine_sim_f16(v_blas_L0, v_ane_L0, kv_total);

        fprintf(stderr, "  K cache: max_err=%.6f  cosine_sim=%.6f\n", k_max_err, k_cos_sim);
        fprintf(stderr, "  V cache: max_err=%.6f  cosine_sim=%.6f\n", v_max_err, v_cos_sim);
        print_sample_f16("BLAS K[0,0:8]", k_blas_L0, kv_dim, 8);
        print_sample_f16("ANE  K[0,0:8]", k_ane_L0, kv_dim, 8);
        print_sample_f16("BLAS V[0,0:8]", v_blas_L0, kv_dim, 8);
        print_sample_f16("ANE  V[0,0:8]", v_ane_L0, kv_dim, 8);

        bool kv_pass = (k_max_err < 0.1f && v_max_err < 0.1f &&
                        k_cos_sim > 0.99f && v_cos_sim > 0.99f);
        if (k_cos_sim < 0.5f || v_cos_sim < 0.5f) {
            fprintf(stderr, "\n  ** KV cache is WRONG -- likely transpose bug (channel-first vs token-first) **\n\n");
            kv_pass = false;
        }
        fprintf(stderr, "  KV cache layer 0: %s\n\n", kv_pass ? "PASS" : "FAIL");

        // ── Overall verdict ─────────────────────────────────────────────
        fprintf(stderr, "========================================\n");
        if (h_pass && kv_pass) {
            fprintf(stderr, "  OVERALL: PASS -- ANE and BLAS prefill match\n");
        } else {
            fprintf(stderr, "  OVERALL: FAIL -- ANE and BLAS prefill diverge\n");
            if (!h_pass)
                fprintf(stderr, "    Hidden state: max_err=%.4f cosine_sim=%.6f\n", h_max_err, h_cos_sim);
            if (!kv_pass)
                fprintf(stderr, "    KV cache L0:  K max_err=%.4f V max_err=%.4f\n", k_max_err, v_max_err);
        }
        fprintf(stderr, "========================================\n");

        // ── Speed comparison ────────────────────────────────────────────
        fprintf(stderr, "\n--- Speed ---\n");
        fprintf(stderr, "  BLAS: %.1f ms (%.1f tok/s)\n", t_blas, n_tokens / (t_blas / 1000.0));
        fprintf(stderr, "  ANE:  %.1f ms (%.1f tok/s)\n", t_ane, n_tokens / (t_ane / 1000.0));
        fprintf(stderr, "  Ratio: ANE is %.2fx %s than BLAS\n",
                t_blas > t_ane ? t_blas / t_ane : t_ane / t_blas,
                t_blas > t_ane ? "faster" : "slower");

cleanup:
        free(x_blas);
        free(x_ane);
        kv_free(&kv_blas);
        kv_free(&kv_ane);
        tokenizer_free(tok);
        mistral_free(model);
    }
    return 0;
}
