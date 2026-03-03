// ane_benchmark.m — Detailed ANE prefill pipeline benchmark
// Measures each phase: transpose, dequant, blob build, compile, forged load, eval
// Build: cd mistral && xcrun clang -O2 -ffast-math -fobjc-arc -Wall -Wno-unused-function
//   -Wno-unused-variable -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -I. -o ane_benchmark
//   ../tests/ane_benchmark.m -framework Foundation -framework IOSurface -framework Accelerate -ldl

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#include "dequant.h"
#include "../training/ane_runtime.h"
#include "ane_mil_gen_mistral.h"

// ─── Config ─────────────────────────────────────────────────────────────────
#define DIM     4096
#define KV_DIM  1024
#define HIDDEN  14336
#define S       64
#define NLAYERS 32

// ─── Timer ──────────────────────────────────────────────────────────────────
static double timer_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)(end - start) * tb.numer / tb.denom / 1e6;
}

// ─── Transpose ──────────────────────────────────────────────────────────────
static void transpose_to_ane(const float *token_first, float *channel_first, int dim, int seq) {
    for (int c = 0; c < dim; c++)
        for (int s = 0; s < seq; s++)
            channel_first[c * seq + s] = token_first[s * dim + c];
}

static void transpose_from_ane(const float *channel_first, float *token_first, int dim, int seq) {
    for (int c = 0; c < dim; c++)
        for (int s = 0; s < seq; s++)
            token_first[s * dim + c] = channel_first[c * seq + s];
}

// ─── Fake Q4_0 data ─────────────────────────────────────────────────────────
// Returns calloc'd Q4_0 block array sized for rows×cols
static void *make_fake_q4(int rows, int cols) {
    int bpr = cols / QK4_0;
    size_t nbytes = (size_t)rows * bpr * sizeof(block_q4_0);
    void *buf = calloc(1, nbytes);
    // Fill with non-zero scales so dequant has real work
    block_q4_0 *b = (block_q4_0 *)buf;
    for (int i = 0; i < rows * bpr; i++) {
        b[i].d = (_Float16)0.01f;
        memset(b[i].qs, 0x55, QK4_0/2);
    }
    return buf;
}

static size_t q4_bytes(int rows, int cols) {
    return (size_t)rows * (cols / QK4_0) * sizeof(block_q4_0);
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main(void) {
    @autoreleasepool {
        printf("=== ANE Prefill Benchmark (dim=%d kv=%d hidden=%d S=%d) ===\n\n",
               DIM, KV_DIM, HIDDEN, S);

        // ── 1. Transpose overhead ──────────────────────────────────────
        {
            printf("1. Transpose:\n");
            int dims[] = {DIM, KV_DIM};
            const char *labels[] = {"4096", "1024"};
            for (int d = 0; d < 2; d++) {
                int dim = dims[d];
                float *tok = (float *)calloc(dim * S, sizeof(float));
                float *ane = (float *)calloc(dim * S, sizeof(float));
                for (int i = 0; i < dim * S; i++) tok[i] = (float)i * 0.001f;

                // Warmup
                transpose_to_ane(tok, ane, dim, S);
                transpose_from_ane(ane, tok, dim, S);

                int iters = 1000;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++)
                    transpose_to_ane(tok, ane, dim, S);
                uint64_t t1 = mach_absolute_time();
                for (int i = 0; i < iters; i++)
                    transpose_from_ane(ane, tok, dim, S);
                uint64_t t2 = mach_absolute_time();

                printf("   to_ane  (%s×%d):  %.3fms\n", labels[d], S, timer_ms(t0, t1) / iters);
                printf("   from_ane(%s×%d):  %.3fms\n", labels[d], S, timer_ms(t1, t2) / iters);
                free(tok); free(ane);
            }
            printf("\n");
        }

        // ── 2. Dequant Q4→fp16 ────────────────────────────────────────
        {
            printf("2. Dequant Q4→fp16:\n");
            struct { int rows; int cols; const char *label; } sizes[] = {
                {DIM,    DIM,    "Q/Wo (4096×4096)"},
                {KV_DIM, DIM,    "KV   (1024×4096)"},
                {HIDDEN, DIM,    "FFN  (14336×4096)"},
                {DIM,    HIDDEN, "W2   (4096×14336)"},
            };

            for (int i = 0; i < 4; i++) {
                int rows = sizes[i].rows, cols = sizes[i].cols;
                void *q4 = make_fake_q4(rows, cols);
                _Float16 *fp16 = (_Float16 *)malloc((size_t)rows * cols * 2);

                // Warmup
                dequant_q4_0_to_fp16(q4, fp16, rows, cols);

                int iters = 10;
                uint64_t t0 = mach_absolute_time();
                for (int j = 0; j < iters; j++)
                    dequant_q4_0_to_fp16(q4, fp16, rows, cols);
                uint64_t t1 = mach_absolute_time();

                double ms = timer_ms(t0, t1) / iters;
                double mb = (double)rows * cols * 2 / 1e6;
                printf("   %s:  %.2fms (%.1f MB fp16)\n", sizes[i].label, ms, mb);
                free(q4); free(fp16);
            }
            printf("\n");
        }

        // ── 3. Blob build ──────────────────────────────────────────────
        {
            printf("3. Blob build:\n");

            // Allocate fp16 weight buffers
            _Float16 *wq  = (_Float16 *)calloc((size_t)DIM * DIM, 2);
            _Float16 *wk  = (_Float16 *)calloc((size_t)KV_DIM * DIM, 2);
            _Float16 *wv  = (_Float16 *)calloc((size_t)KV_DIM * DIM, 2);
            _Float16 *wo  = (_Float16 *)calloc((size_t)DIM * DIM, 2);
            _Float16 *w1  = (_Float16 *)calloc((size_t)HIDDEN * DIM, 2);
            _Float16 *w3  = (_Float16 *)calloc((size_t)HIDDEN * DIM, 2);
            _Float16 *w2  = (_Float16 *)calloc((size_t)DIM * HIDDEN, 2);

            int iters = 10;

            // Q blob (single weight)
            uint64_t t0 = mach_absolute_time();
            NSData *q_blob = nil;
            for (int i = 0; i < iters; i++) {
                q_blob = mil_build_single_weight_blob(wq, DIM, DIM);
            }
            uint64_t t1 = mach_absolute_time();
            printf("   Q  blob (4096×4096):   %.2fms  (%lu bytes)\n",
                   timer_ms(t0, t1) / iters, (unsigned long)[q_blob length]);

            // KV blob (fused)
            t0 = mach_absolute_time();
            NSData *kv_blob = nil;
            for (int i = 0; i < iters; i++) {
                kv_blob = mil_build_kv_fused_blob(wk, wv, KV_DIM, DIM);
            }
            t1 = mach_absolute_time();
            printf("   KV blob (1024×4096×2): %.2fms  (%lu bytes)\n",
                   timer_ms(t0, t1) / iters, (unsigned long)[kv_blob length]);

            // Wo blob (same as Q)
            t0 = mach_absolute_time();
            NSData *wo_blob = nil;
            for (int i = 0; i < iters; i++) {
                wo_blob = mil_build_single_weight_blob(wo, DIM, DIM);
            }
            t1 = mach_absolute_time();
            printf("   Wo blob (4096×4096):   %.2fms  (%lu bytes)\n",
                   timer_ms(t0, t1) / iters, (unsigned long)[wo_blob length]);

            // FFN blob (fused W1+W3+W2)
            t0 = mach_absolute_time();
            NSData *ffn_blob = nil;
            for (int i = 0; i < iters; i++) {
                ffn_blob = mil_build_ffn_fused_blob(w1, w3, HIDDEN, w2, DIM);
            }
            t1 = mach_absolute_time();
            printf("   FFN blob (14336×4096×3): %.2fms  (%lu bytes)\n",
                   timer_ms(t0, t1) / iters, (unsigned long)[ffn_blob length]);
            printf("\n");

            // ── 4. Compile (ane_compile_and_get_hexid) ─────────────────
            printf("4. Compile (ane_compile_and_get_hexid):\n");

            NSString *mil_q   = mil_gen_conv_baked(DIM, DIM, S);
            NSString *mil_kv  = mil_gen_kv_fused(DIM, KV_DIM, S);
            NSString *mil_wo  = mil_gen_conv_baked(DIM, DIM, S);
            NSString *mil_ffn = mil_gen_ffn_fused(DIM, HIDDEN, S);

            // Use different weight data for Wo so hexId differs from Q
            for (int i = 0; i < DIM * DIM; i++) ((_Float16*)wo)[i] = (_Float16)0.002f;
            NSData *wo_blob2 = mil_build_single_weight_blob(wo, DIM, DIM);

            NSData *mil_q_data   = [mil_q dataUsingEncoding:NSUTF8StringEncoding];
            NSData *mil_kv_data  = [mil_kv dataUsingEncoding:NSUTF8StringEncoding];
            NSData *mil_wo_data  = [mil_wo dataUsingEncoding:NSUTF8StringEncoding];
            NSData *mil_ffn_data = [mil_ffn dataUsingEncoding:NSUTF8StringEncoding];

            NSString *hex_q = nil, *hex_kv = nil, *hex_wo = nil, *hex_ffn = nil;

            t0 = mach_absolute_time();
            hex_q = ane_compile_and_get_hexid(mil_q_data, q_blob);
            t1 = mach_absolute_time();
            double ms_q = timer_ms(t0, t1);
            printf("   Q   (4096→4096):    %.1fms %s\n", ms_q, hex_q ? "OK" : "FAIL");

            t0 = mach_absolute_time();
            hex_kv = ane_compile_and_get_hexid(mil_kv_data, kv_blob);
            t1 = mach_absolute_time();
            double ms_kv = timer_ms(t0, t1);
            printf("   KV  (4096→1024×2):  %.1fms %s\n", ms_kv, hex_kv ? "OK" : "FAIL");

            t0 = mach_absolute_time();
            hex_wo = ane_compile_and_get_hexid(mil_wo_data, wo_blob2);
            t1 = mach_absolute_time();
            double ms_wo = timer_ms(t0, t1);
            printf("   Wo  (4096→4096):    %.1fms %s\n", ms_wo, hex_wo ? "OK" : "FAIL");

            t0 = mach_absolute_time();
            hex_ffn = ane_compile_and_get_hexid(mil_ffn_data, ffn_blob);
            t1 = mach_absolute_time();
            double ms_ffn = timer_ms(t0, t1);
            printf("   FFN (4096→14336→4096): %.1fms %s\n", ms_ffn, hex_ffn ? "OK" : "FAIL");

            double compile_per_layer = ms_q + ms_kv + ms_wo + ms_ffn;
            printf("   Total per layer: %.1fms\n\n", compile_per_layer);

            // ── 5. Forged load ─────────────────────────────────────────
            printf("5. Forged load (ane_load_forged):\n");

            if (!hex_q || !hex_kv || !hex_wo || !hex_ffn) {
                printf("   SKIP — compile failed\n\n");
            } else {
                size_t in_dim  = DIM * S * 4;    // fp32
                size_t out_dim = DIM * S * 4;
                size_t out_kv  = KV_DIM * S * 4;

                ANEKernel *k_q = nil, *k_kv = nil, *k_wo = nil, *k_ffn = nil;

                t0 = mach_absolute_time();
                k_q = ane_load_forged(mil_q_data, hex_q, 1, &in_dim, 1, &out_dim);
                t1 = mach_absolute_time();
                double ld_q = timer_ms(t0, t1);
                printf("   Q:   %.2fms %s\n", ld_q, k_q ? "OK" : "FAIL");

                size_t out_kvs[] = {out_kv, out_kv};
                t0 = mach_absolute_time();
                k_kv = ane_load_forged(mil_kv_data, hex_kv, 1, &in_dim, 2, out_kvs);
                t1 = mach_absolute_time();
                double ld_kv = timer_ms(t0, t1);
                printf("   KV:  %.2fms %s\n", ld_kv, k_kv ? "OK" : "FAIL");

                t0 = mach_absolute_time();
                k_wo = ane_load_forged(mil_wo_data, hex_wo, 1, &in_dim, 1, &out_dim);
                t1 = mach_absolute_time();
                double ld_wo = timer_ms(t0, t1);
                printf("   Wo:  %.2fms %s\n", ld_wo, k_wo ? "OK" : "FAIL");

                t0 = mach_absolute_time();
                k_ffn = ane_load_forged(mil_ffn_data, hex_ffn, 1, &in_dim, 1, &out_dim);
                t1 = mach_absolute_time();
                double ld_ffn = timer_ms(t0, t1);
                printf("   FFN: %.2fms %s\n", ld_ffn, k_ffn ? "OK" : "FAIL");

                double load_per_layer = ld_q + ld_kv + ld_wo + ld_ffn;
                printf("   Total per layer: %.2fms\n\n", load_per_layer);

                // ── 6. Eval ────────────────────────────────────────────
                printf("6. Eval (10 iterations avg):\n");

                // Prepare input
                float *inp = (float *)calloc(DIM * S, sizeof(float));
                for (int i = 0; i < DIM * S; i++) inp[i] = (float)(i % 1000) * 0.001f;

                struct {
                    ANEKernel *k;
                    const char *label;
                    int in_ch, out_ch;
                    int nouts;
                    double flops; // per eval
                } progs[] = {
                    {k_q,   "Q  ", DIM, DIM,    1, 2.0 * DIM * DIM * S},
                    {k_kv,  "KV ", DIM, KV_DIM, 2, 2.0 * 2 * KV_DIM * DIM * S},
                    {k_wo,  "Wo ", DIM, DIM,    1, 2.0 * DIM * DIM * S},
                    {k_ffn, "FFN", DIM, DIM,    1, 2.0 * (2.0 * HIDDEN * DIM + DIM * HIDDEN) * S},
                };

                int eval_iters = 10;
                double eval_times[4] = {0};

                for (int p = 0; p < 4; p++) {
                    if (!progs[p].k) {
                        printf("   %s: SKIP (load failed)\n", progs[p].label);
                        continue;
                    }

                    ane_write_input(progs[p].k, 0, inp, DIM * S * sizeof(float));

                    // Warmup
                    ane_eval(progs[p].k);
                    ane_eval(progs[p].k);

                    t0 = mach_absolute_time();
                    for (int i = 0; i < eval_iters; i++)
                        ane_eval(progs[p].k);
                    t1 = mach_absolute_time();

                    double ms = timer_ms(t0, t1) / eval_iters;
                    double tflops = progs[p].flops / (ms * 1e9);
                    eval_times[p] = ms;
                    printf("   %s: %.3fms (%.1f TFLOPS)\n", progs[p].label, ms, tflops);
                }
                printf("\n");

                // ── 7. Full layer simulation ───────────────────────────
                printf("7. Full layer (3 dispatches + transpose):\n");

                float *ane_buf = (float *)calloc(DIM * S, sizeof(float));

                // Measure one full layer: transpose_in → Q eval → KV eval → Wo eval → FFN eval → transpose_out
                // Warmup
                transpose_to_ane(inp, ane_buf, DIM, S);
                ane_write_input(k_q, 0, ane_buf, DIM * S * 4);
                ane_eval(k_q);

                int layer_iters = 10;
                t0 = mach_absolute_time();
                for (int i = 0; i < layer_iters; i++) {
                    // Transpose in
                    transpose_to_ane(inp, ane_buf, DIM, S);
                    // Q
                    ane_write_input(k_q, 0, ane_buf, DIM * S * 4);
                    ane_eval(k_q);
                    // KV
                    ane_write_input(k_kv, 0, ane_buf, DIM * S * 4);
                    ane_eval(k_kv);
                    // Wo (read Q output, write as Wo input)
                    ane_read_output(k_q, 0, ane_buf, DIM * S * 4);
                    ane_write_input(k_wo, 0, ane_buf, DIM * S * 4);
                    ane_eval(k_wo);
                    // FFN
                    ane_read_output(k_wo, 0, ane_buf, DIM * S * 4);
                    ane_write_input(k_ffn, 0, ane_buf, DIM * S * 4);
                    ane_eval(k_ffn);
                    // Transpose out
                    ane_read_output(k_ffn, 0, ane_buf, DIM * S * 4);
                    transpose_from_ane(ane_buf, inp, DIM, S);
                }
                t1 = mach_absolute_time();
                double layer_ms = timer_ms(t0, t1) / layer_iters;
                double total_ms = layer_ms * NLAYERS;
                double tok_per_s = S / (total_ms / 1000.0);

                printf("   %.2fms/layer × %d = %.1fms total\n", layer_ms, NLAYERS, total_ms);
                printf("   → %.1f tok/s at S=%d\n\n", tok_per_s, S);

                free(inp);
                free(ane_buf);

                // ── Summary ────────────────────────────────────────────
                printf("Summary:\n");

                // Cold start estimate: per layer = dequant all weights + blob build + compile
                // Dequant: Q(4096²) + K(1024×4096) + V(1024×4096) + Wo(4096²) + W1(14336×4096) + W3(14336×4096) + W2(4096×14336)
                // Re-run dequant to get timing
                void *fq = make_fake_q4(DIM, DIM);
                _Float16 *ftmp = (_Float16 *)malloc((size_t)HIDDEN * DIM * 2);
                uint64_t td0 = mach_absolute_time();
                dequant_q4_0_to_fp16(fq, ftmp, DIM, DIM);      // Q
                dequant_q4_0_to_fp16(fq, ftmp, KV_DIM, DIM);   // K
                dequant_q4_0_to_fp16(fq, ftmp, KV_DIM, DIM);   // V
                dequant_q4_0_to_fp16(fq, ftmp, DIM, DIM);      // Wo
                void *fq2 = make_fake_q4(HIDDEN, DIM);
                dequant_q4_0_to_fp16(fq2, ftmp, HIDDEN, DIM);  // W1
                dequant_q4_0_to_fp16(fq2, ftmp, HIDDEN, DIM);  // W3
                dequant_q4_0_to_fp16(fq2, ftmp, DIM, HIDDEN);  // W2
                uint64_t td1 = mach_absolute_time();
                double dequant_per_layer = timer_ms(td0, td1);
                free(fq); free(fq2); free(ftmp);

                double cold_per_layer = dequant_per_layer + compile_per_layer;
                printf("  Cold start (Phase 1): %.1fs (%d layers × %.1fms dequant + %.1fms compile)\n",
                       cold_per_layer * NLAYERS / 1000.0, NLAYERS, dequant_per_layer, compile_per_layer);
                printf("  Warm start (Phase 2): %.1fms (%d layers × %.2fms forged load)\n",
                       load_per_layer * NLAYERS, NLAYERS, load_per_layer);
                printf("  Inference:            %.1fms/tile (%d layers × %.2fms dispatch)\n",
                       total_ms, NLAYERS, layer_ms);

                // Cleanup
                if (k_q)   ane_free(k_q);
                if (k_kv)  ane_free(k_kv);
                if (k_wo)  ane_free(k_wo);
                if (k_ffn) ane_free(k_ffn);
            }

            free(wq); free(wk); free(wv); free(wo);
            free(w1); free(w3); free(w2);
        }

        printf("\nDone.\n");
    }
    return 0;
}
