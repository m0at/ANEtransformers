// mistral_infer.m — Mistral 7B inference on Apple M5
// CPU decode with fused Q4 NEON matvec, optional Metal GPU backend (--metal)
// Usage: ./mistral --model path/to/mistral-7b-q4_0.gguf --prompt "Hello" [--tokens 128] [--temp 0.7]

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "gguf_loader.h"
#include "dequant.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "mistral_model.h"
#include "metal_matvec.h"
#include "mistral_ane_prefill.h"
#include "speculative.h"

// ─── Timing helper ───────────────────────────────────────────────────────────
static double time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

// ─── Metal GPU full forward pass (all 32 layers, 1 command buffer) ──────────
// ALL operations on GPU: RMSNorm, GEMV, RoPE, KV cache, GQA attention, residuals.
// Single command buffer submission per decode step = minimal overhead.
static void metal_forward(MetalContext *ctx, MistralModel *m, KVCache *kv, int pos) {
    MistralConfig *c = &m->cfg;
    uint32_t dim = c->dim;
    uint32_t kv_dim = c->n_kv_heads * c->head_dim;
    uint32_t hidden = c->hidden_dim;
    uint32_t n_heads = c->n_heads;
    uint32_t n_kv = c->n_kv_heads;
    uint32_t hd = c->head_dim;
    float eps = c->rms_eps;
    uint32_t cache_pos = pos % kv->max_seq;
    uint32_t seq_len = (pos + 1 < kv->max_seq) ? pos + 1 : kv->max_seq;
    uint32_t ring_off = (pos + 1 > kv->max_seq) ? pos + 1 - seq_len : 0;
    uint32_t max_s = (uint32_t)kv->max_seq;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (int l = 0; l < c->n_layers; l++) {
            LayerWeights *lw = &m->layers[l];

            // 1. RMSNorm_att: x → xb
            metal_encode_rmsnorm(ctx, enc, ctx->x_buf, ctx->xb_buf,
                                 ctx->rms_att_bufs[l], dim, eps);
            // 2. QKV projections: xb → q, k, v
            metal_encode_gemv(ctx, enc, lw->wq, ctx->xb_buf, ctx->q_buf, dim, dim);
            metal_encode_gemv(ctx, enc, lw->wk, ctx->xb_buf, ctx->k_buf, kv_dim, dim);
            metal_encode_gemv(ctx, enc, lw->wv, ctx->xb_buf, ctx->v_buf, kv_dim, dim);
            // 3. RoPE: rotate q, k in-place
            metal_encode_rope(ctx, enc, ctx->q_buf, ctx->k_buf,
                              (uint32_t)pos, n_heads, n_kv, hd);
            // 4. KV cache write: fp32 k,v → fp16 cache
            metal_encode_kv_write(ctx, enc, ctx->k_buf, ctx->v_buf,
                                   l, cache_pos, kv_dim);
            // 5. GQA attention: q + k_cache + v_cache → xb2
            metal_encode_gqa_attention(ctx, enc, ctx->q_buf, ctx->xb2_buf,
                                        l, n_heads, n_kv, hd, kv_dim,
                                        seq_len, max_s, ring_off);
            // 6. Wo projection: xb2 → xb
            metal_encode_gemv(ctx, enc, lw->wo, ctx->xb2_buf, ctx->xb_buf, dim, dim);
            // 7. Attention residual: x += xb
            metal_encode_vadd(ctx, enc, ctx->x_buf, ctx->xb_buf, ctx->x_buf, dim);
            // 8. RMSNorm_ffn: x → xb
            metal_encode_rmsnorm(ctx, enc, ctx->x_buf, ctx->xb_buf,
                                 ctx->rms_ffn_bufs[l], dim, eps);
            // 9-12. FFN: gate, up, SiLU*mul, down
            metal_encode_gemv(ctx, enc, lw->w1, ctx->xb_buf, ctx->hb_buf, hidden, dim);
            metal_encode_gemv(ctx, enc, lw->w3, ctx->xb_buf, ctx->hb2_buf, hidden, dim);
            metal_encode_silu_mul(ctx, enc, ctx->hb_buf, ctx->hb2_buf, ctx->hb_buf, hidden);
            metal_encode_gemv(ctx, enc, lw->w2, ctx->hb_buf, ctx->xb_buf, dim, hidden);
            // 13. FFN residual: x += xb
            metal_encode_vadd(ctx, enc, ctx->x_buf, ctx->xb_buf, ctx->x_buf, dim);
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Update KV cache metadata (CPU-side bookkeeping)
    if (kv->len < kv->max_seq && pos + 1 > kv->len) kv->len = pos + 1;
}

// ─── Metal GPU logits ────────────────────────────────────────────────────────
static void metal_logits(MetalContext *ctx, MistralModel *m) {
    float *gpu_x = (float *)[ctx->x_buf contents];
    mistral_logits(m, gpu_x);
}

// ─── Metal GPU batch forward pass (S tokens, 1 command buffer) ──────────────
// All S tokens through all 32 layers on GPU. For speculative verify / prefill.
// x_batch_buf must contain [S * dim] fp32 on entry, result written back there.
static void metal_forward_batch(MetalContext *ctx, MistralModel *m, KVCache *kv,
                                 int base_pos, int S) {
    MistralConfig *c = &m->cfg;
    uint32_t dim = c->dim;
    uint32_t kv_dim = c->n_kv_heads * c->head_dim;
    uint32_t hidden = c->hidden_dim;
    uint32_t n_heads = c->n_heads;
    uint32_t n_kv = c->n_kv_heads;
    uint32_t hd = c->head_dim;
    float eps = c->rms_eps;
    uint32_t max_s = (uint32_t)kv->max_seq;
    uint32_t uS = (uint32_t)S;

    // Ring buffer positions for the batch
    uint32_t base_cache_pos = (uint32_t)(base_pos % kv->max_seq);
    // For causal attention: tokens at base_pos..base_pos+S-1 attend to
    // all previous tokens + causally masked within the batch
    uint32_t base_seq_len = (uint32_t)((base_pos < kv->max_seq) ? base_pos : kv->max_seq);
    uint32_t ring_off = (base_pos > kv->max_seq) ? (uint32_t)(base_pos - kv->max_seq) : 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (int l = 0; l < c->n_layers; l++) {
            LayerWeights *lw = &m->layers[l];

            // 1. RMSNorm_att: x_batch → xb_batch
            metal_encode_rmsnorm_batch(ctx, enc, ctx->x_batch_buf, ctx->xb_batch_buf,
                                        ctx->rms_att_bufs[l], dim, eps, uS);
            // 2-4. QKV projections via GEMM
            metal_encode_gemm(ctx, enc, lw->wq, ctx->xb_batch_buf, ctx->q_batch_buf, dim, dim, uS);
            metal_encode_gemm(ctx, enc, lw->wk, ctx->xb_batch_buf, ctx->k_batch_buf, kv_dim, dim, uS);
            metal_encode_gemm(ctx, enc, lw->wv, ctx->xb_batch_buf, ctx->v_batch_buf, kv_dim, dim, uS);
            // 5. RoPE batch
            metal_encode_rope_batch(ctx, enc, ctx->q_batch_buf, ctx->k_batch_buf,
                                     (uint32_t)base_pos, n_heads, n_kv, hd, uS);
            // 6. KV cache write batch
            metal_encode_kv_write_batch(ctx, enc, ctx->k_batch_buf, ctx->v_batch_buf,
                                         l, base_cache_pos, kv_dim, max_s, uS);
            // 7. GQA attention (causal within batch)
            uint32_t att_stride_val = base_seq_len + uS;
            metal_encode_gqa_attention_causal(ctx, enc, ctx->q_batch_buf, ctx->xb2_batch_buf,
                                               l, n_heads, n_kv, hd, kv_dim,
                                               base_seq_len, max_s, ring_off, uS,
                                               att_stride_val);
            // 8. Wo projection
            metal_encode_gemm(ctx, enc, lw->wo, ctx->xb2_batch_buf, ctx->xb_batch_buf, dim, dim, uS);
            // 9. Attention residual: x_batch += xb_batch
            metal_encode_vadd(ctx, enc, ctx->x_batch_buf, ctx->xb_batch_buf,
                               ctx->x_batch_buf, dim * uS);
            // 10. RMSNorm_ffn: x_batch → xb_batch
            metal_encode_rmsnorm_batch(ctx, enc, ctx->x_batch_buf, ctx->xb_batch_buf,
                                        ctx->rms_ffn_bufs[l], dim, eps, uS);
            // 11-12. Gate + Up projections
            metal_encode_gemm(ctx, enc, lw->w1, ctx->xb_batch_buf, ctx->hb_batch_buf, hidden, dim, uS);
            metal_encode_gemm(ctx, enc, lw->w3, ctx->xb_batch_buf, ctx->hb2_batch_buf, hidden, dim, uS);
            // 13. SiLU * mul
            metal_encode_silu_mul(ctx, enc, ctx->hb_batch_buf, ctx->hb2_batch_buf,
                                   ctx->hb_batch_buf, hidden * uS);
            // 14. Down projection
            metal_encode_gemm(ctx, enc, lw->w2, ctx->hb_batch_buf, ctx->xb_batch_buf, dim, hidden, uS);
            // 15. FFN residual: x_batch += xb_batch
            metal_encode_vadd(ctx, enc, ctx->x_batch_buf, ctx->xb_batch_buf,
                               ctx->x_batch_buf, dim * uS);
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // Update KV cache metadata
    int new_len = base_pos + S;
    if (new_len > kv->max_seq) new_len = kv->max_seq;
    if (new_len > kv->len) kv->len = new_len;
}

// ─── Metal GPU prefill (tiled GEMM) ─────────────────────────────────────────
// Process n_prompt tokens through all layers using GEMM, tiled into chunks.
// Returns hidden state of last token in x_out.
static bool metal_prefill(MetalContext *ctx, MistralModel *m, KVCache *kv,
                            const int *tokens, int n_prompt, float *x_out) {
    int dim = m->cfg.dim;
    int kv_dim_val = m->cfg.n_kv_heads * m->cfg.head_dim;
    int hidden = m->cfg.hidden_dim;
    int n_heads = m->cfg.n_heads;

    // Tile size: balance GEMM efficiency vs att_scratch memory
    int tile_size = 64;
    if (n_prompt <= tile_size) tile_size = n_prompt;

    // max_att_len: last tile's last token sees all n_prompt tokens
    metal_alloc_batch_buffers(ctx, tile_size, dim, kv_dim_val, hidden, n_heads, n_prompt);

    float *x_batch = (float *)[ctx->x_batch_buf contents];

    int pos = 0;
    while (pos < n_prompt) {
        int chunk = n_prompt - pos;
        if (chunk > tile_size) chunk = tile_size;

        // Embed chunk tokens on CPU → x_batch_buf
        for (int i = 0; i < chunk; i++)
            embed_token(m, tokens[pos + i], x_batch + (size_t)i * dim);

        metal_forward_batch(ctx, m, kv, pos, chunk);
        pos += chunk;
    }

    // Copy last token's hidden state to output
    x_batch = (float *)[ctx->x_batch_buf contents];
    int last_chunk = n_prompt - ((n_prompt - 1) / tile_size) * tile_size;
    if (last_chunk <= 0) last_chunk = tile_size;
    memcpy(x_out, x_batch + (size_t)(last_chunk - 1) * dim, dim * sizeof(float));
    return true;
}

// ─── Metal speculative verify ────────────────────────────────────────────────
// Embeds S tokens on CPU, runs batch forward on GPU, computes logits on CPU.
// target_logits: output [S * vocab], caller-allocated.
static int metal_spec_verify(MetalContext *ctx, MistralModel *m, KVCache *kv,
                              const int *tokens, int S, int start_pos,
                              float *target_logits) {
    int dim   = m->cfg.dim;
    int vocab = m->cfg.vocab_size;

    // Embed all S tokens on CPU → x_batch [S, dim] row-major
    float *x_batch = (float *)[ctx->x_batch_buf contents];
    for (int i = 0; i < S; i++)
        embed_token(m, tokens[i], x_batch + (size_t)i * dim);

    // Batch forward on GPU
    metal_forward_batch(ctx, m, kv, start_pos, S);

    // Read back and compute logits per token on CPU (lm_head is Q6_K)
    x_batch = (float *)[ctx->x_batch_buf contents];  // re-read after GPU
    float *x_tmp = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < S; i++) {
        memcpy(x_tmp, x_batch + (size_t)i * dim, dim * sizeof(float));
        mistral_logits(m, x_tmp);
        memcpy(target_logits + (size_t)i * vocab, m->logits, vocab * sizeof(float));
    }
    free(x_tmp);
    return 0;
}

// ─── Metal speculative decode loop ──────────────────────────────────────────
// Draft-verify-accept loop using n-gram drafting + Metal batch verify.
static void metal_speculative_decode(MetalContext *ctx, MistralModel *m, KVCache *kv,
                                      Tokenizer *tok, int first_token, int start_pos,
                                      int max_tokens, float temperature,
                                      int *token_history, int *n_history,
                                      int rep_window, float rep_penalty,
                                      int top_k, float top_p) {
    int vocab = m->cfg.vocab_size;
    (void)m->cfg.dim;  // dim used indirectly via metal_spec_verify

    // Init n-gram drafter from token history
    NGramDraft *drafter = ngram_draft_init(token_history, *n_history, 2, SPEC_MAX_DRAFT);
    SpeculativeState spec;
    spec_state_init(&spec);

    int accepted[SPEC_MAX_DRAFT + 1];
    int current_token = first_token;
    int pos = start_pos;
    int tokens_generated = 0;

    double t_start = time_ms();
    double t_draft_total = 0, t_verify_total = 0;

    while (tokens_generated < max_tokens) {
        if (current_token == tok->eos_id) break;

        // ---- Draft phase ----
        double t0 = time_ms();
        int n_draft = ngram_draft_fn(drafter, m, kv, current_token, pos, temperature, &spec);
        if (n_draft > SPEC_MAX_DRAFT) n_draft = SPEC_MAX_DRAFT;
        double t1 = time_ms();
        t_draft_total += t1 - t0;

        if (n_draft <= 0) {
            // No draft: single-token Metal forward
            float *gpu_x = (float *)[ctx->x_buf contents];
            embed_token(m, current_token, gpu_x);
            metal_forward(ctx, m, kv, pos);
            metal_logits(ctx, m);

            float *logits_buf = (float *)malloc(vocab * sizeof(float));
            memcpy(logits_buf, m->logits, vocab * sizeof(float));
            int hist_start = *n_history > rep_window ? *n_history - rep_window : 0;
            SampleParams sp = {
                .temp = temperature, .top_k = top_k, .top_p = top_p,
                .rep_penalty = rep_penalty,
                .prev_tokens = token_history + hist_start,
                .n_prev = *n_history - hist_start
            };
            current_token = sample_token(logits_buf, vocab, &sp);
            free(logits_buf);

            ngram_draft_push(drafter, current_token);
            token_history[(*n_history)++] = current_token;

            const char *piece = tokenizer_decode_token(tok, current_token);
            if (piece) { printf("%s", piece); fflush(stdout); }

            pos++;
            tokens_generated++;
            spec.total_iterations++;
            continue;
        }

        // ---- Verify phase: batch of n_draft+1 tokens on GPU ----
        int n_verify = n_draft + 1;
        int verify_tokens[SPEC_MAX_DRAFT + 1];
        verify_tokens[0] = current_token;
        memcpy(verify_tokens + 1, spec.draft_tokens, n_draft * sizeof(int));

        float *all_logits = (float *)malloc((size_t)n_verify * vocab * sizeof(float));
        double tv0 = time_ms();
        metal_spec_verify(ctx, m, kv, verify_tokens, n_verify, pos, all_logits);
        double tv1 = time_ms();
        t_verify_total += tv1 - tv0;

        // ---- Accept/reject ----
        int n_accepted = spec_accept(spec.draft_probs, all_logits, spec.draft_tokens,
                                      n_draft, vocab, temperature, accepted);

        // Fix bonus token: should come from all_logits[n_draft], not [n_draft-1]
        if (n_accepted == n_draft + 1) {
            float *bonus_p = (float *)malloc(vocab * sizeof(float));
            spec_softmax(all_logits + (size_t)n_draft * vocab, bonus_p, vocab, temperature);
            accepted[n_draft] = spec_sample_probs(bonus_p, vocab);
            free(bonus_p);
        }

        // ---- Output accepted tokens ----
        for (int i = 0; i < n_accepted; i++) {
            ngram_draft_push(drafter, accepted[i]);
            token_history[(*n_history)++] = accepted[i];

            const char *piece = tokenizer_decode_token(tok, accepted[i]);
            if (piece) { printf("%s", piece); fflush(stdout); }
            if (accepted[i] == tok->eos_id) {
                tokens_generated += i + 1;
                free(all_logits);
                goto metal_spec_done;
            }
        }

        // ---- Update stats and advance ----
        spec.total_draft += n_draft;
        spec.total_accepted += n_accepted;
        spec.total_iterations++;

        pos += n_accepted;
        current_token = accepted[n_accepted - 1];
        tokens_generated += n_accepted;

        free(all_logits);
    }

metal_spec_done:;
    double t_total = time_ms() - t_start;

    fprintf(stderr, "\n--- Metal Speculative Decoding Stats ---\n");
    fprintf(stderr, "Generated tokens:     %d\n", tokens_generated);
    fprintf(stderr, "Draft iterations:     %lld\n", (long long)spec.total_iterations);
    if (spec.total_iterations > 0) {
        fprintf(stderr, "Avg draft/iter:       %.1f\n",
                (double)spec.total_draft / spec.total_iterations);
        fprintf(stderr, "Avg accepted/iter:    %.1f\n",
                (double)spec.total_accepted / spec.total_iterations);
        int64_t pure_accepted = spec.total_accepted - spec.total_iterations;
        fprintf(stderr, "Acceptance rate:      %.1f%%\n",
                (spec.total_draft > 0) ? 100.0 * pure_accepted / spec.total_draft : 0.0);
    }
    fprintf(stderr, "Draft time:           %.1f ms (%.1f%%)\n",
            t_draft_total, (t_total > 0) ? 100.0 * t_draft_total / t_total : 0.0);
    fprintf(stderr, "Verify time:          %.1f ms (%.1f%%)\n",
            t_verify_total, (t_total > 0) ? 100.0 * t_verify_total / t_total : 0.0);
    fprintf(stderr, "Total decode:         %.1f ms\n", t_total);
    if (tokens_generated > 0) {
        fprintf(stderr, "Effective tok/s:      %.2f\n",
                tokens_generated / (t_total / 1000.0));
    }

    ngram_draft_free(drafter);
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = NULL;
        const char *prompt = "Hello, I am";
        int max_tokens = 128;
        float temperature = 0.7f;
        float top_p = 0.9f;
        int top_k = 40;
        float rep_penalty = 1.1f;
        int rep_window = 64;
        int chat_mode = 0;
        int context_len = 0;  // 0 = use model default (sliding_window)
        int use_speculative  = 0;
        int spec_K           = 4;   // speculation depth
        int spec_draft_layers = 8;  // first N layers as draft (of 32 total)
        int use_metal = 0;
        int use_ane = 0;

        // Parse args
        static struct option long_opts[] = {
            {"model",          required_argument, 0, 'm'},
            {"prompt",         required_argument, 0, 'p'},
            {"tokens",         required_argument, 0, 'n'},
            {"temp",           required_argument, 0, 't'},
            {"top-p",          required_argument, 0, 'P'},
            {"top-k",          required_argument, 0, 'K'},
            {"rep-penalty",    required_argument, 0, 'r'},
            {"rep-window",     required_argument, 0, 'w'},
            {"chat",           no_argument,       0, 'c'},
            {"context",        required_argument, 0, 'C'},
            {"speculative",    no_argument,       0, 'S'},
            {"spec-k",         required_argument, 0, 'k'},
            {"spec-layers",    required_argument, 0, 'L'},
            {"metal",          no_argument,       0, 'G'},
            {"ane",            no_argument,       0, 'A'},
            {0, 0, 0, 0}
        };
        int opt;
        while ((opt = getopt_long(argc, argv, "m:p:n:t:P:K:r:w:c:C:Sk:L:GA", long_opts, NULL)) != -1) {
            switch (opt) {
                case 'm': model_path = optarg; break;
                case 'p': prompt = optarg; break;
                case 'n': max_tokens = atoi(optarg); break;
                case 't': temperature = atof(optarg); break;
                case 'P': top_p = atof(optarg); break;
                case 'K': top_k = atoi(optarg); break;
                case 'r': rep_penalty = atof(optarg); break;
                case 'w': rep_window = atoi(optarg); break;
                case 'c': chat_mode = 1; break;
                case 'C': context_len = atoi(optarg); break;
                case 'S': use_speculative = 1; break;
                case 'k': spec_K = atoi(optarg); break;
                case 'L': spec_draft_layers = atoi(optarg); break;
                case 'G': use_metal = 1; break;
                case 'A': use_ane = 1; break;
            }
        }

        if (!model_path) {
            fprintf(stderr, "Usage: %s --model <gguf> [--prompt \"...\"] [--tokens N] [--temp T]\n", argv[0]);
            fprintf(stderr, "       [--top-p P] [--top-k K] [--rep-penalty R] [--rep-window W]\n");
            fprintf(stderr, "       [--context N] [--chat] [--metal] [--ane]\n");
            fprintf(stderr, "       [--speculative] [--spec-k K] [--spec-layers N]\n");
            fprintf(stderr, "  --metal          Use Metal GPU for matvec (default: CPU SDOT)\n");
            fprintf(stderr, "  --ane            Use ANE baked-weight prefill (96 compiled programs)\n");
            fprintf(stderr, "  --speculative    Enable self-speculative decode (default: off)\n");
            fprintf(stderr, "  --spec-k K       Speculation depth (default: 4, max: %d)\n", SPEC_MAX_DRAFT);
            fprintf(stderr, "  --spec-layers N  Draft layers (default: 8, target total: 32)\n");
            return 1;
        }

        // ── Load model ───────────────────────────────────────────────────
        fprintf(stderr, "Loading model: %s\n", model_path);
        double t0 = time_ms();
        MistralModel *model = mistral_load(model_path);
        if (!model) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }
        double t_load = time_ms() - t0;
        fprintf(stderr, "Model loaded in %.1f ms\n", t_load);

        MistralConfig *cfg = &model->cfg;
        int dim = cfg->dim;
        int vocab = cfg->vocab_size;

        // ── Init Metal GPU backend (optional) ────────────────────────────
        MetalContext *metal_ctx = NULL;
        if (use_metal) {
            int kv_dim = cfg->n_kv_heads * cfg->head_dim;
            // Metal init needs max_seq — compute it now (same logic as below)
            int metal_max_seq;
            if (context_len > 0) metal_max_seq = context_len;
            else metal_max_seq = 65536;
            if (metal_max_seq > 131072) metal_max_seq = 131072;

            metal_ctx = metal_context_init(model->gguf->mmap_base, model->gguf->mmap_len,
                                           dim, kv_dim, cfg->hidden_dim, vocab,
                                           cfg->n_heads, cfg->n_layers, metal_max_seq);
            if (metal_ctx) {
                model->metal = metal_ctx;
                for (int l = 0; l < cfg->n_layers; l++)
                    metal_set_rmsnorm_bufs(metal_ctx, l,
                                           model->layers[l].rms_att,
                                           model->layers[l].rms_ffn, dim);
                metal_set_theta_inv(metal_ctx, model->rope_theta_inv, cfg->head_dim / 2);
                fprintf(stderr, "Metal GPU backend enabled (1 CB/step, all-GPU decode)\n");
            } else {
                fprintf(stderr, "Metal init failed, falling back to CPU SDOT\n");
                use_metal = 0;
            }
        }

        // ── Init tokenizer ───────────────────────────────────────────────
        Tokenizer *tok = tokenizer_init(model->gguf);
        if (!tok) {
            fprintf(stderr, "Failed to init tokenizer\n");
            if (metal_ctx) metal_context_free(metal_ctx);
            mistral_free(model);
            return 1;
        }
        fprintf(stderr, "Tokenizer: %d tokens\n", tok->vocab_size);

        // ── Init KV cache ────────────────────────────────────────────────
        int max_seq;
        if (context_len > 0)
            max_seq = context_len;
        else
            max_seq = 65536;
        if (max_seq > 131072) max_seq = 131072;
        KVCache kv_store = kv_alloc(cfg->n_layers, cfg->n_kv_heads, cfg->head_dim, max_seq);
        KVCache *kv = &kv_store;
        // Wire KV cache to Metal shared buffers (both CPU and GPU access same memory)
        if (metal_ctx) metal_wire_kv_cache(metal_ctx, kv);
        fprintf(stderr, "KV cache: %d layers x %d slots x %d kv_dim = %.1f MB\n",
                cfg->n_layers, max_seq, kv->kv_dim,
                (double)cfg->n_layers * kv->kv_dim * max_seq * 2 * 2 / 1e6);

        // Reallocate attention scratch for potentially larger context
        if (max_seq > (cfg->sliding_window > 0 ? cfg->sliding_window : 4096)) {
            free(model->att);
            model->att = (float *)calloc((size_t)cfg->n_heads * max_seq, sizeof(float));
        }

        // ── Format prompt ────────────────────────────────────────────────
        const char *final_prompt = prompt;
        char *chat_buf = NULL;
        if (chat_mode) {
            size_t plen = strlen(prompt);
            chat_buf = (char *)malloc(plen + 32);
            snprintf(chat_buf, plen + 32, "[INST] %s [/INST]", prompt);
            final_prompt = chat_buf;
        }

        // ── Tokenize ─────────────────────────────────────────────────────
        int *prompt_tokens = (int *)malloc(max_seq * sizeof(int));
        int n_prompt = tokenizer_encode(tok, final_prompt, prompt_tokens, max_seq, 1);
        fprintf(stderr, "Prompt: %d tokens\n", n_prompt);
        fprintf(stderr, "Sampling: temp=%.2f top_k=%d top_p=%.2f rep_penalty=%.2f rep_window=%d\n",
                temperature, top_k, top_p, rep_penalty, rep_window);

        if (n_prompt == 0) {
            fprintf(stderr, "Empty prompt after tokenization\n");
            if (chat_buf) free(chat_buf);
            kv_free(kv);
            tokenizer_free(tok);
            if (metal_ctx) metal_context_free(metal_ctx);
            mistral_free(model);
            return 1;
        }

        // ── Token history for repetition penalty ─────────────────────────
        int *token_history = (int *)calloc(max_tokens + n_prompt, sizeof(int));
        memcpy(token_history, prompt_tokens, n_prompt * sizeof(int));
        int n_history = n_prompt;

        // ── Prefill ──────────────────────────────────────────────────────
        float *x = (float *)calloc(dim, sizeof(float));
        double t_prefill_start = time_ms();

        // Prefill strategy cascade:
        // - ANE baked (--ane, ≥16 tok): fp16 baked conv, fused FFN
        // - Metal GPU (--metal, ≥16 tok): direct Q4_0 GEMM, no dequant
        // - CPU BLAS (≥16 tok): tiled dequant + cblas_sgemm (AMX)
        // - CPU SDOT (<16 tok): sequential decode
        const char *prefill_method = "CPU-SDOT";
        bool used_fast = false;
        if (use_ane && n_prompt >= 16) {
            used_fast = ane_baked_prefill_forward(model, kv, prompt_tokens, n_prompt, x);
            if (used_fast) prefill_method = "ANE-baked";
        }
        if (!used_fast && metal_ctx && n_prompt >= 16) {
            used_fast = metal_prefill(metal_ctx, model, kv, prompt_tokens, n_prompt, x);
            if (used_fast) prefill_method = "Metal-GEMM";
        }
        if (!used_fast && n_prompt >= 16) {
            used_fast = blas_prefill_forward(model, kv, prompt_tokens, n_prompt, x);
            if (used_fast) prefill_method = "BLAS";
        }
        if (!used_fast) {
            for (int t = 0; t < n_prompt; t++) {
                embed_token(model, prompt_tokens[t], x);
                for (int l = 0; l < cfg->n_layers; l++)
                    mistral_layer_decode_parallel(model, kv, x, l, t);
            }
        }
        fprintf(stderr, "Prefilling %d tokens (%s)...\n", n_prompt, prefill_method);

        // Get logits for last prompt token
        if (metal_ctx) {
            memcpy([metal_ctx->x_buf contents], x, dim * sizeof(float));
            metal_logits(metal_ctx, model);
        } else {
            mistral_logits(model, x);
        }

        // Sample first token with rep penalty over prompt tokens
        float *logits_buf = (float *)malloc(vocab * sizeof(float));
        memcpy(logits_buf, model->logits, vocab * sizeof(float));
        int hist_start = n_history > rep_window ? n_history - rep_window : 0;
        SampleParams sp = {
            .temp = temperature, .top_k = top_k, .top_p = top_p,
            .rep_penalty = rep_penalty,
            .prev_tokens = token_history + hist_start,
            .n_prev = n_history - hist_start
        };
        int next_token = sample_token(logits_buf, vocab, &sp);
        token_history[n_history++] = next_token;

        double t_prefill = time_ms() - t_prefill_start;
        fprintf(stderr, "Prefill: %.1f ms (%.1f tok/s, %s), TTFT: %.1f ms\n",
                t_prefill, n_prompt / (t_prefill / 1000.0),
                prefill_method, t_prefill);

        // Print first generated token
        fprintf(stderr, "\n--- Generation ---\n");
        const char *piece = tokenizer_decode_token(tok, next_token);
        if (piece) printf("%s", piece);
        fflush(stdout);

        // ── Decode loop ──────────────────────────────────────────────────
        double t_decode_start = time_ms();
        int generated = 0;

        if (use_speculative && metal_ctx) {
            // Metal speculative decode: n-gram draft + GPU batch verify
            if (spec_K < 1) spec_K = 1;
            if (spec_K > SPEC_MAX_DRAFT) spec_K = SPEC_MAX_DRAFT;

            int kv_dim_s = cfg->n_kv_heads * cfg->head_dim;
            int batch_S = spec_K + 1;  // current token + K draft tokens
            // att_stride: worst case is full context + batch size
            int spec_att_len = n_prompt + max_tokens + batch_S;
            if (spec_att_len > max_seq) spec_att_len = max_seq;
            metal_alloc_batch_buffers(metal_ctx, batch_S, dim, kv_dim_s,
                                       cfg->hidden_dim, cfg->n_heads, spec_att_len);

            fprintf(stderr, "Metal speculative: K=%d (n-gram draft + GPU batch verify)\n", spec_K);

            metal_speculative_decode(metal_ctx, model, kv, tok,
                                      next_token, n_prompt,
                                      max_tokens - 1, temperature,
                                      token_history, &n_history,
                                      rep_window, rep_penalty, top_k, top_p);
            // generated count comes from the spec decode stats printed internally
            generated = n_history - n_prompt;  // total tokens added to history
        } else if (use_speculative) {
            // CPU speculative decode: self-speculative with first spec_draft_layers as draft.
            if (spec_K < 1) spec_K = 1;
            if (spec_K > SPEC_MAX_DRAFT) spec_K = SPEC_MAX_DRAFT;
            if (spec_draft_layers < 1) spec_draft_layers = 1;
            if (spec_draft_layers >= cfg->n_layers) spec_draft_layers = cfg->n_layers - 1;

            fprintf(stderr, "Speculative: K=%d draft_layers=%d/%d\n",
                    spec_K, spec_draft_layers, cfg->n_layers);

            SelfSpecDraft *draft = self_spec_draft_init(model, spec_draft_layers,
                                                         spec_K, max_seq);
            SpeculativeState spec_st;
            spec_state_init(&spec_st);

            for (int t = 0; t < n_prompt; t++) {
                float *dx = draft->x;
                embed_token(model, prompt_tokens[t], dx);
                for (int l = 0; l < spec_draft_layers; l++)
                    mistral_layer_decode_parallel(model, &draft->kv, dx, l, t);
            }
            draft->kv.len = n_prompt;

            spec_decode_loop(model, kv, tok, &spec_st,
                             self_spec_draft_fn, draft,
                             next_token, n_prompt,
                             max_tokens - 1, temperature);

            generated = (int)spec_st.total_accepted;
            self_spec_draft_free(draft);
        } else {
            for (int t = 0; t < max_tokens; t++) {
                if (next_token == tok->eos_id) break;

                int pos = n_prompt + t;

                // Embed
                embed_token(model, next_token, x);

                // Forward through all layers (CPU SDOT — faster than GPU GEMV for S=1)
                for (int l = 0; l < cfg->n_layers; l++)
                    mistral_layer_decode_parallel(model, kv, x, l, pos);
                mistral_logits(model, x);
                memcpy(logits_buf, model->logits, vocab * sizeof(float));
                hist_start = n_history > rep_window ? n_history - rep_window : 0;
                sp.prev_tokens = token_history + hist_start;
                sp.n_prev = n_history - hist_start;
                next_token = sample_token(logits_buf, vocab, &sp);
                token_history[n_history++] = next_token;
                generated++;

                // Print token
                piece = tokenizer_decode_token(tok, next_token);
                if (piece) printf("%s", piece);
                fflush(stdout);
            }
        }

        double t_decode = time_ms() - t_decode_start;
        printf("\n");

        // ── Stats ────────────────────────────────────────────────────────
        fprintf(stderr, "\n--- Stats ---\n");
        fprintf(stderr, "Prompt tokens:    %d\n", n_prompt);
        fprintf(stderr, "Generated tokens: %d\n", generated);
        fprintf(stderr, "Backend:          %s\n", metal_ctx ? "Metal GEMM prefill + CPU SDOT decode" : "CPU SDOT");
        fprintf(stderr, "Prefill:          %.1f ms (%.2f tok/s)\n",
                t_prefill, n_prompt / (t_prefill / 1000.0));
        if (generated > 0) {
            fprintf(stderr, "Decode:           %.1f ms (%.2f tok/s)\n",
                    t_decode, generated / (t_decode / 1000.0));
            fprintf(stderr, "Per-token:        %.1f ms\n", t_decode / generated);
        }
        fprintf(stderr, "Total:            %.1f ms\n", t_prefill + t_decode);

        // ── Cleanup ──────────────────────────────────────────────────────
        ane_baked_prefill_cleanup();
        blas_prefill_cleanup();
        if (metal_ctx) {
            // KV cache memory owned by Metal buffers, not malloc — zero pointers before kv_free
            kv->k_cache = NULL;
            kv->v_cache = NULL;
            metal_context_free(metal_ctx);
        }
        free(x);
        free(logits_buf);
        free(token_history);
        free(prompt_tokens);
        if (chat_buf) free(chat_buf);
        kv_free(kv);
        tokenizer_free(tok);
        mistral_free(model);
    }
    return 0;
}
