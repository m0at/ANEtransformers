// test_blas_prefill.m — Verify BLAS tiled Q4 GEMM prefill matches sequential decode
// Usage: ./test_prefill --model path/to/mistral-7b-q4_0.gguf [--tokens "prompt text"]
//
// Runs both BLAS prefill and sequential SDOT decode on the same prompt,
// then compares the resulting hidden states and logits.

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include "gguf_loader.h"
#include "dequant.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "mistral_model.h"
#include "mistral_ane_prefill.h"

static void compare_vectors(const char *name, const float *ref, const float *test,
                            int n, float atol, float rtol) {
    float max_abs = 0, max_rel = 0, sum_sq = 0;
    int max_abs_idx = 0, max_rel_idx = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        float rel = (fabsf(ref[i]) > 1e-6f) ? diff / fabsf(ref[i]) : diff;
        sum_sq += diff * diff;
        if (diff > max_abs) { max_abs = diff; max_abs_idx = i; }
        if (rel > max_rel) { max_rel = rel; max_rel_idx = i; }
    }
    float rmse = sqrtf(sum_sq / n);

    int pass = (max_abs < atol) && (max_rel < rtol);
    fprintf(stderr, "  %-20s: max_abs=%.6e [%d] max_rel=%.6e [%d] rmse=%.6e  %s\n",
            name, max_abs, max_abs_idx, max_rel, max_rel_idx, rmse,
            pass ? "PASS" : "FAIL");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = NULL;
        const char *prompt = "The quick brown fox jumps over the lazy dog and then runs away";

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

        // Load model
        fprintf(stderr, "Loading model: %s\n", model_path);
        MistralModel *model = mistral_load(model_path);
        if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

        MistralConfig *cfg = &model->cfg;
        int dim = cfg->dim;
        int vocab = cfg->vocab_size;

        Tokenizer *tok = tokenizer_init(model->gguf);
        if (!tok) { fprintf(stderr, "Failed to init tokenizer\n"); return 1; }

        int max_seq = 4096;
        KVCache kv_ref = kv_alloc(cfg->n_layers, cfg->n_kv_heads, cfg->head_dim, max_seq);
        KVCache kv_blas = kv_alloc(cfg->n_layers, cfg->n_kv_heads, cfg->head_dim, max_seq);

        // Tokenize
        int *tokens = (int *)malloc(max_seq * sizeof(int));
        int n = tokenizer_encode(tok, prompt, tokens, max_seq, 1);
        fprintf(stderr, "Prompt: %d tokens\n", n);

        if (n < 16) {
            fprintf(stderr, "Need at least 16 tokens for BLAS prefill test (got %d)\n", n);
            return 1;
        }

        // ── Reference: sequential SDOT decode ────────────────────────────
        fprintf(stderr, "\n--- Sequential SDOT decode (reference) ---\n");
        float *x_ref = (float *)calloc(dim, sizeof(float));
        for (int t = 0; t < n; t++) {
            embed_token(model, tokens[t], x_ref);
            for (int l = 0; l < cfg->n_layers; l++)
                mistral_layer_decode_parallel(model, &kv_ref, x_ref, l, t);
        }
        // Get logits
        mistral_logits(model, x_ref);
        float *logits_ref = (float *)malloc(vocab * sizeof(float));
        memcpy(logits_ref, model->logits, vocab * sizeof(float));

        // Find top-1 token
        int top_ref = 0;
        for (int i = 1; i < vocab; i++)
            if (logits_ref[i] > logits_ref[top_ref]) top_ref = i;
        fprintf(stderr, "  Reference top-1: token %d (%.4f)\n", top_ref, logits_ref[top_ref]);

        // ── Test: BLAS tiled prefill ─────────────────────────────────────
        fprintf(stderr, "\n--- BLAS tiled prefill ---\n");
        float *x_blas = (float *)calloc(dim, sizeof(float));
        bool ok = blas_prefill_forward(model, &kv_blas, tokens, n, x_blas);
        if (!ok) {
            fprintf(stderr, "BLAS prefill failed!\n");
            return 1;
        }
        // Get logits
        mistral_logits(model, x_blas);
        float *logits_blas = (float *)malloc(vocab * sizeof(float));
        memcpy(logits_blas, model->logits, vocab * sizeof(float));

        int top_blas = 0;
        for (int i = 1; i < vocab; i++)
            if (logits_blas[i] > logits_blas[top_blas]) top_blas = i;
        fprintf(stderr, "  BLAS top-1: token %d (%.4f)\n", top_blas, logits_blas[top_blas]);

        // ── Compare ──────────────────────────────────────────────────────
        fprintf(stderr, "\n--- Comparison ---\n");
        compare_vectors("hidden_state", x_ref, x_blas, dim, 0.01f, 0.05f);
        compare_vectors("logits", logits_ref, logits_blas, vocab, 0.5f, 0.1f);

        fprintf(stderr, "\n  Top-1 match: %s\n", top_ref == top_blas ? "YES" : "NO");

        // Check top-5 overlap
        int top5_ref[5], top5_blas[5];
        for (int k = 0; k < 5; k++) { top5_ref[k] = 0; top5_blas[k] = 0; }
        // Simple O(n*k) selection
        float *lr = (float *)malloc(vocab * sizeof(float));
        float *lb = (float *)malloc(vocab * sizeof(float));
        memcpy(lr, logits_ref, vocab * sizeof(float));
        memcpy(lb, logits_blas, vocab * sizeof(float));
        for (int k = 0; k < 5; k++) {
            int br = 0, bb = 0;
            for (int i = 1; i < vocab; i++) {
                if (lr[i] > lr[br]) br = i;
                if (lb[i] > lb[bb]) bb = i;
            }
            top5_ref[k] = br;  lr[br] = -1e30f;
            top5_blas[k] = bb; lb[bb] = -1e30f;
        }
        int overlap = 0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (top5_ref[i] == top5_blas[j]) overlap++;
        fprintf(stderr, "  Top-5 overlap: %d/5\n", overlap);

        // Cleanup
        free(lr); free(lb);
        free(logits_ref); free(logits_blas);
        free(x_ref); free(x_blas);
        free(tokens);
        kv_free(&kv_ref); kv_free(&kv_blas);
        blas_prefill_cleanup();
        tokenizer_free(tok);
        mistral_free(model);

        fprintf(stderr, "\nDone.\n");
    }
    return 0;
}
