// speculative.h -- Speculative decoding for Mistral 7B
// Verify K draft tokens against full 32-layer target model, accept/reject per
// Leviathan et al. "Fast Inference from Transformers via Speculative Decoding"
#pragma once

#include "mistral_model.h"
#include "tokenizer.h"
#include <mach/mach_time.h>

// ---- Constants ----
#define SPEC_MAX_DRAFT 8

// ---- Speculative state (owned by caller, holds draft buffers + stats) --------
typedef struct {
    // Draft tokens and their probabilities from the draft model
    int   draft_tokens[SPEC_MAX_DRAFT];
    float draft_probs[SPEC_MAX_DRAFT];  // p_q(draft_token[i]) under draft model

    // Stats
    int64_t total_draft;       // total draft tokens proposed
    int64_t total_accepted;    // total tokens accepted (including bonus)
    int64_t total_iterations;  // number of draft-verify rounds
} SpeculativeState;

static void spec_state_init(SpeculativeState *s) {
    memset(s, 0, sizeof(SpeculativeState));
}

// ---- Timing helper (local) --------------------------------------------------
static double spec_time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

// ---- softmax into caller-supplied buffer (vectorized via vvexpf + vDSP) -----
static void spec_softmax(const float *logits, float *probs, int n, float temperature) {
    // Find max (vDSP)
    float max_val;
    vDSP_maxv(logits, 1, &max_val, (vDSP_Length)n);

    // probs[i] = (logits[i] - max_val) * inv_t
    float neg_max = -max_val;
    float inv_t = (temperature > 0) ? 1.0f / temperature : 0.0f;
    vDSP_vsadd(logits, 1, &neg_max, probs, 1, (vDSP_Length)n);
    vDSP_vsmul(probs, 1, &inv_t, probs, 1, (vDSP_Length)n);

    // Batch exp (32000 elements — vvexpf vs scalar: ~10x faster)
    int nn = n;
    vvexpf(probs, probs, &nn);

    // Sum + normalize
    float sum;
    vDSP_sve(probs, 1, &sum, (vDSP_Length)n);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(probs, 1, &inv_sum, probs, 1, (vDSP_Length)n);
}

// ---- Sample from a probability distribution ---------------------------------
static int spec_sample_probs(const float *probs, int n) {
    float r = (float)arc4random() / (float)UINT32_MAX;
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    return n - 1;
}

// =============================================================================
// spec_verify -- Run n_draft tokens through the full 32-layer target model
//
// Sequential: for each draft token, embed -> 32 layers -> logits.
// This populates the target KV cache at positions start_pos..start_pos+n_draft-1,
// which is correct for all accepted tokens. Rejected positions get overwritten
// on subsequent iterations (ring buffer).
//
// target_logits: output [n_draft][vocab_size], caller-allocated.
// Returns 0 on success.
// =============================================================================
static int spec_verify(MistralModel *model, KVCache *kv,
                       const int *draft_tokens, int n_draft, int start_pos,
                       float *target_logits) {
    int dim   = model->cfg.dim;
    int vocab = model->cfg.vocab_size;
    int n_layers = model->cfg.n_layers;

    // For small n_draft, sequential decode is fine (no BLAS setup overhead)
    float *x = (float *)malloc(dim * sizeof(float));
    if (!x) return -1;

    for (int i = 0; i < n_draft; i++) {
        int token = draft_tokens[i];
        int pos   = start_pos + i;

        embed_token(model, token, x);
        for (int l = 0; l < n_layers; l++)
            mistral_layer_decode_parallel(model, kv, x, l, pos);
        mistral_logits(model, x);
        memcpy(target_logits + (size_t)i * vocab, model->logits, vocab * sizeof(float));
    }

    free(x);
    return 0;
}

// =============================================================================
// spec_accept -- Standard speculative decoding acceptance
//
// For each draft position i:
//   p = softmax(target_logits[i] / T)  -- target distribution
//   q = draft_probs[i]                 -- scalar prob of draft token under draft model
//
//   Accept if rand() < min(1, p[draft_token] / q)
//   Otherwise resample from max(0, p - q_adjusted) normalized.
//   q_adjusted is zero everywhere except at draft_tokens[i] where it equals draft_probs[i].
//
// Returns n_accepted (1..n_draft+1). accepted_tokens has that many tokens.
// The last token is always a "bonus" -- either resampled or from target at last pos.
// =============================================================================
static int spec_accept(const float *draft_probs,      // [n_draft] scalar prob per token
                       const float *target_logits,     // [n_draft][vocab_size]
                       const int *draft_tokens,        // [n_draft]
                       int n_draft, int vocab_size,
                       float temperature,
                       int *accepted_tokens) {
    // Greedy mode (temp ≤ 0): accept if argmax matches, otherwise reject with argmax
    if (temperature <= 0.0f) {
        for (int i = 0; i < n_draft; i++) {
            const float *logits_i = target_logits + (size_t)i * vocab_size;
            int target_tok = 0;
            float best = logits_i[0];
            for (int v = 1; v < vocab_size; v++) {
                if (logits_i[v] > best) { best = logits_i[v]; target_tok = v; }
            }
            if (target_tok == draft_tokens[i]) {
                accepted_tokens[i] = target_tok;
            } else {
                accepted_tokens[i] = target_tok;
                return i + 1;
            }
        }
        // All accepted — bonus from last logits
        const float *bonus_logits = target_logits + (size_t)(n_draft - 1) * vocab_size;
        int bonus = 0;
        float best = bonus_logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (bonus_logits[v] > best) { best = bonus_logits[v]; bonus = v; }
        }
        accepted_tokens[n_draft] = bonus;
        return n_draft + 1;
    }

    // Stochastic mode: standard Leviathan et al. acceptance
    float *p   = (float *)malloc(vocab_size * sizeof(float));
    float *adj = (float *)malloc(vocab_size * sizeof(float));

    for (int i = 0; i < n_draft; i++) {
        spec_softmax(target_logits + (size_t)i * vocab_size, p, vocab_size, temperature);

        int tok    = draft_tokens[i];
        float p_tok = p[tok];
        float q_tok = draft_probs[i];

        float accept_prob = (q_tok > 0) ? fminf(1.0f, p_tok / q_tok) : 0.0f;
        float r = (float)arc4random() / (float)UINT32_MAX;

        if (r < accept_prob) {
            accepted_tokens[i] = tok;
        } else {
            float sum_adj = 0;
            for (int v = 0; v < vocab_size; v++) {
                float q_v = (v == tok) ? q_tok : 0.0f;
                adj[v] = fmaxf(0.0f, p[v] - q_v);
                sum_adj += adj[v];
            }
            if (sum_adj > 0) {
                float inv = 1.0f / sum_adj;
                for (int v = 0; v < vocab_size; v++) adj[v] *= inv;
            } else {
                memcpy(adj, p, vocab_size * sizeof(float));
            }

            accepted_tokens[i] = spec_sample_probs(adj, vocab_size);

            free(p);
            free(adj);
            return i + 1;
        }
    }

    spec_softmax(target_logits + (size_t)(n_draft - 1) * vocab_size, p, vocab_size, temperature);
    accepted_tokens[n_draft] = spec_sample_probs(p, vocab_size);

    free(p);
    free(adj);
    return n_draft + 1;
}

// =============================================================================
// Draft function pointer type
//
// Fills spec->draft_tokens and spec->draft_probs with up to SPEC_MAX_DRAFT
// candidates. Returns number of draft tokens produced.
// =============================================================================
typedef int (*SpecDraftFn)(void *draft_ctx, MistralModel *model, KVCache *kv,
                           int current_token, int pos, float temperature,
                           SpeculativeState *spec);

// =============================================================================
// spec_decode_loop -- Main decode loop with speculative decoding
//
// Replaces the sequential decode loop in mistral_infer.m.
// The verify phase runs current_token + all draft tokens through the target
// model so that target_logits[0] predicts draft_tokens[0], etc.
// =============================================================================
static void spec_decode_loop(MistralModel *model, KVCache *kv, Tokenizer *tok,
                             SpeculativeState *spec,
                             SpecDraftFn draft_fn, void *draft_ctx,
                             int first_token, int start_pos,
                             int max_tokens, float temperature) {
    int vocab = model->cfg.vocab_size;
    int dim   = model->cfg.dim;

    int accepted[SPEC_MAX_DRAFT + 1];

    int current_token = first_token;
    int pos = start_pos;
    int tokens_generated = 0;

    double t_start = spec_time_ms();
    double t_draft_total = 0;
    double t_verify_total = 0;

    while (tokens_generated < max_tokens) {
        if (current_token == tok->eos_id) break;

        // ---- Draft phase ----
        double t0 = spec_time_ms();
        int n_draft = draft_fn(draft_ctx, model, kv, current_token, pos, temperature, spec);
        if (n_draft <= 0) {
            // No draft available: single target forward pass
            float *x = (float *)malloc(dim * sizeof(float));
            embed_token(model, current_token, x);
            for (int l = 0; l < model->cfg.n_layers; l++)
                mistral_layer_decode_parallel(model, kv, x, l, pos);
            mistral_logits(model, x);
            current_token = sample_temperature(model->logits, vocab, temperature);
            free(x);

            const char *piece = tokenizer_decode_token(tok, current_token);
            if (piece) { printf("%s", piece); fflush(stdout); }

            pos++;
            tokens_generated++;
            spec->total_iterations++;
            continue;
        }
        if (n_draft > SPEC_MAX_DRAFT) n_draft = SPEC_MAX_DRAFT;
        double t_draft_end = spec_time_ms();
        t_draft_total += t_draft_end - t0;

        // ---- Verify phase ----
        // Build verify sequence: [current_token, draft_tokens[0..n_draft-1]]
        // Run all n_draft+1 tokens through target model.
        // target_logits[0] (from current_token) predicts draft_tokens[0]
        // target_logits[i] (from draft_tokens[i-1]) predicts draft_tokens[i]
        // target_logits[n_draft] (from draft_tokens[n_draft-1]) predicts bonus

        int n_verify = n_draft + 1;
        int verify_tokens[SPEC_MAX_DRAFT + 1];
        verify_tokens[0] = current_token;
        memcpy(verify_tokens + 1, spec->draft_tokens, n_draft * sizeof(int));

        float *all_logits = (float *)malloc((size_t)n_verify * vocab * sizeof(float));
        spec_verify(model, kv, verify_tokens, n_verify, pos, all_logits);

        double t_verify_end = spec_time_ms();
        t_verify_total += t_verify_end - t_draft_end;

        // ---- Accept/reject ----
        // all_logits[0] = logits from current_token, predicts draft_tokens[0]
        // all_logits[i] = logits from draft_tokens[i-1], predicts draft_tokens[i]
        // We pass all_logits (starting from index 0) and draft_tokens (n_draft of them).
        // spec_accept checks: target_logits[i] against draft_tokens[i].
        // If all accepted, bonus sampled from all_logits[n_draft].
        //
        // But spec_accept expects target_logits[n_draft-1] for the bonus.
        // We pass n_verify logits starting at all_logits, but spec_accept only
        // indexes [0..n_draft-1] for acceptance and [n_draft-1] for bonus.
        // Wait -- spec_accept uses target_logits[n_draft-1] for bonus, but
        // the correct bonus logits are all_logits[n_draft] (from the last draft token).
        //
        // Fix: pass all_logits+0 as target_logits to spec_accept, but the bonus
        // case in spec_accept reads index n_draft-1. We need it to read index n_draft.
        // Solution: pass n_draft+1 positions worth of logits and adjust spec_accept,
        // OR just handle the bonus here.
        //
        // Simplest: use spec_accept for the first n_draft positions,
        // then handle bonus separately if all accepted.

        int n_accepted = spec_accept(spec->draft_probs,
                                     all_logits,  // [0]=current_token logits -> predicts draft[0]
                                     spec->draft_tokens,
                                     n_draft, vocab, temperature,
                                     accepted);

        // If all were accepted, spec_accept sampled bonus from all_logits[n_draft-1].
        // But the correct bonus should come from all_logits[n_draft] (the logits
        // produced by running the last draft token through the target model).
        // Override the bonus token:
        if (n_accepted == n_draft + 1) {
            float *bonus_p = (float *)malloc(vocab * sizeof(float));
            spec_softmax(all_logits + (size_t)n_draft * vocab, bonus_p, vocab, temperature);
            accepted[n_draft] = spec_sample_probs(bonus_p, vocab);
            free(bonus_p);
        }

        // ---- Output accepted tokens ----
        for (int i = 0; i < n_accepted; i++) {
            const char *piece = tokenizer_decode_token(tok, accepted[i]);
            if (piece) { printf("%s", piece); fflush(stdout); }
            if (accepted[i] == tok->eos_id) {
                tokens_generated += i + 1;
                free(all_logits);
                goto done;
            }
        }

        // ---- Update stats ----
        spec->total_draft += n_draft;
        spec->total_accepted += n_accepted;
        spec->total_iterations++;

        // ---- Advance ----
        // The target KV cache has entries at pos..pos+n_verify-1.
        // Only pos..pos+n_accepted-1 are valid (from verify_tokens[0..n_accepted-1]).
        // pos+n_accepted corresponds to the first token that needs to be processed
        // next iteration. The stale KV entries at pos+n_accepted..pos+n_verify-1
        // will be overwritten when those ring buffer positions are reused.
        pos += n_accepted;
        current_token = accepted[n_accepted - 1];
        tokens_generated += n_accepted;

        free(all_logits);
    }

done:;
    double t_total = spec_time_ms() - t_start;

    // ---- Print stats ----
    fprintf(stderr, "\n--- Speculative Decoding Stats ---\n");
    fprintf(stderr, "Generated tokens:     %d\n", tokens_generated);
    fprintf(stderr, "Draft iterations:     %lld\n", (long long)spec->total_iterations);
    if (spec->total_iterations > 0) {
        fprintf(stderr, "Avg draft/iter:       %.1f\n",
                (double)spec->total_draft / spec->total_iterations);
        fprintf(stderr, "Avg accepted/iter:    %.1f\n",
                (double)spec->total_accepted / spec->total_iterations);
        // Acceptance rate: (accepted - bonus_tokens) / total_draft
        // Each iteration produces exactly 1 bonus, so bonus_count = total_iterations
        int64_t pure_accepted = spec->total_accepted - spec->total_iterations;
        fprintf(stderr, "Acceptance rate:      %.1f%%\n",
                (spec->total_draft > 0)
                    ? 100.0 * pure_accepted / spec->total_draft
                    : 0.0);
    }
    fprintf(stderr, "Draft time:           %.1f ms (%.1f%%)\n",
            t_draft_total, (t_total > 0) ? 100.0 * t_draft_total / t_total : 0.0);
    fprintf(stderr, "Verify time:          %.1f ms (%.1f%%)\n",
            t_verify_total, (t_total > 0) ? 100.0 * t_verify_total / t_total : 0.0);
    fprintf(stderr, "Total decode:         %.1f ms\n", t_total);
    if (tokens_generated > 0) {
        fprintf(stderr, "Effective tok/s:      %.2f\n",
                tokens_generated / (t_total / 1000.0));
        fprintf(stderr, "Per-token (effective): %.1f ms\n",
                t_total / tokens_generated);
    }
}

// =============================================================================
// N-gram draft model -- zero-cost baseline for testing the speculative framework
//
// Scans token history for n-gram matches and predicts continuation.
// No neural network needed. Plug in via SpecDraftFn.
// =============================================================================
typedef struct {
    int *history;       // all tokens seen (prompt + generated)
    int  history_len;
    int  history_cap;
    int  ngram_n;       // n-gram order (default 3)
    int  max_draft;     // max tokens to draft per iteration
} NGramDraft;

static NGramDraft *ngram_draft_init(const int *prompt_tokens, int n_prompt,
                                     int ngram_n, int max_draft) {
    NGramDraft *d = (NGramDraft *)calloc(1, sizeof(NGramDraft));
    d->ngram_n = (ngram_n > 0) ? ngram_n : 3;
    d->max_draft = (max_draft > 0) ? max_draft : 4;
    if (d->max_draft > SPEC_MAX_DRAFT) d->max_draft = SPEC_MAX_DRAFT;
    d->history_cap = n_prompt + 4096;
    d->history = (int *)malloc(d->history_cap * sizeof(int));
    memcpy(d->history, prompt_tokens, n_prompt * sizeof(int));
    d->history_len = n_prompt;
    return d;
}

static void ngram_draft_push(NGramDraft *d, int token) {
    if (d->history_len >= d->history_cap) {
        d->history_cap *= 2;
        d->history = (int *)realloc(d->history, d->history_cap * sizeof(int));
    }
    d->history[d->history_len++] = token;
}

static void ngram_draft_push_n(NGramDraft *d, const int *tokens, int n) {
    for (int i = 0; i < n; i++) ngram_draft_push(d, tokens[i]);
}

static void ngram_draft_free(NGramDraft *d) {
    if (d) { free(d->history); free(d); }
}

// N-gram draft function conforming to SpecDraftFn
static int ngram_draft_fn(void *draft_ctx, MistralModel *model, KVCache *kv,
                          int current_token, int pos, float temperature,
                          SpeculativeState *spec) {
    (void)model; (void)kv; (void)temperature;
    NGramDraft *d = (NGramDraft *)draft_ctx;
    int n = d->ngram_n;
    int hlen = d->history_len;

    if (hlen < n) return 0;

    // Build pattern: last (n-1) history tokens + current_token
    int plen = n;
    if (plen > 16) plen = 16;
    int pattern[16];
    for (int i = 0; i < plen - 1; i++)
        pattern[i] = d->history[hlen - plen + 1 + i];
    pattern[plen - 1] = current_token;

    // Scan history for first matching n-gram (skip tail to avoid self-match)
    int match_pos = -1;
    for (int i = 0; i <= hlen - plen - 1; i++) {
        int ok = 1;
        for (int j = 0; j < plen; j++) {
            if (d->history[i + j] != pattern[j]) { ok = 0; break; }
        }
        if (ok) { match_pos = i + plen; break; }
    }

    if (match_pos < 0) return 0;

    // Draft: copy continuation tokens after the match
    int n_draft = 0;
    for (int i = match_pos; i < hlen && n_draft < d->max_draft; i++, n_draft++) {
        spec->draft_tokens[n_draft] = d->history[i];
        // N-gram "probability" -- use 0.9 to let the target model override easily
        spec->draft_probs[n_draft] = 0.9f;
    }

    return n_draft;
}

// =============================================================================
// Self-speculative draft -- use the first `draft_layers` of the target model
//
// No separate model needed. The draft runs a truncated forward pass (e.g. 8
// of 32 layers) on its own KV cache, then greedily samples. Because the draft
// shares weights with the target, the output distribution is aligned by
// construction; acceptance rates of 40-60% are typical.
//
// The draft KV cache is allocated once and persists across iterations. On
// rejection at position k, the draft KV simply gets overwritten on the next
// call because writes are indexed by `pos` (ring buffer). No explicit rollback
// needed beyond truncating kv->len.
// =============================================================================
typedef struct {
    MistralModel *model;
    KVCache       kv;           // draft-depth KV cache (draft_layers deep)
    float        *x;            // [dim] scratch buffer
    int           draft_layers; // number of layers to run (< model->cfg.n_layers)
    int           max_draft;    // tokens to generate per iteration (== K)
} SelfSpecDraft;

static SelfSpecDraft *self_spec_draft_init(MistralModel *model, int draft_layers,
                                            int max_draft, int max_seq) {
    SelfSpecDraft *d = (SelfSpecDraft *)calloc(1, sizeof(SelfSpecDraft));
    d->model        = model;
    d->draft_layers = draft_layers;
    d->max_draft    = (max_draft > SPEC_MAX_DRAFT) ? SPEC_MAX_DRAFT : max_draft;
    d->kv = kv_alloc(draft_layers, model->cfg.n_kv_heads,
                     model->cfg.head_dim, max_seq);
    d->x = (float *)calloc(model->cfg.dim, sizeof(float));
    return d;
}

static void self_spec_draft_free(SelfSpecDraft *d) {
    if (!d) return;
    kv_free(&d->kv);
    free(d->x);
    free(d);
}

// Conform to SpecDraftFn. Runs draft_layers of the target model greedily
// to produce up to max_draft token predictions. draft_probs are set to 1.0
// (greedy), which causes spec_accept to always accept matching tokens (since
// min(1, p_target/1.0) = p_target < 1, actually stochastic). For a fully
// greedy comparison use temperature=0 in the outer loop.
static int self_spec_draft_fn(void *draft_ctx, MistralModel *model, KVCache *kv,
                               int current_token, int pos, float temperature,
                               SpeculativeState *spec) {
    (void)kv;  // target KV -- we use the draft's own KV
    SelfSpecDraft *d = (SelfSpecDraft *)draft_ctx;
    int dim   = model->cfg.dim;
    int vocab = model->cfg.vocab_size;
    int n_layers = d->draft_layers;

    // Roll back draft KV length to `pos` so attention only covers valid range.
    if (d->kv.len > pos) d->kv.len = pos;

    int tok = current_token;
    for (int k = 0; k < d->max_draft; k++) {
        embed_token(model, tok, d->x);
        for (int l = 0; l < n_layers; l++)
            mistral_layer_decode_parallel(model, &d->kv, d->x, l, pos + k);
        mistral_logits(model, d->x);

        // Greedy argmax (draft_probs = 1.0 signals greedy to spec_accept)
        int next = sample_argmax(model->logits, vocab);
        spec->draft_tokens[k] = next;
        spec->draft_probs[k]  = 1.0f;
        tok = next;
    }

    return d->max_draft;
}
