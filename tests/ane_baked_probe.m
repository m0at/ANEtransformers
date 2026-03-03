// ane_baked_probe.m — Phase 1: Verify baked conv ops at Mistral 7B weight sizes
// Tests: single conv, fused QKV, fused FFN. Measures compile/load/eval/throughput.
// Build: cd mistral && make ane_probe
// Run:   ./ane_probe [--model <gguf>]
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>

#include "../training/ane_runtime.h"
#include "../mistral/ane_mil_gen_mistral.h"

// ─── Timing ─────────────────────────────────────────────────────────────────
static double time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

// ─── Random fp16 weights ────────────────────────────────────────────────────
static _Float16 *rand_fp16(int n) {
    _Float16 *buf = (_Float16 *)malloc((size_t)n * sizeof(_Float16));
    for (int i = 0; i < n; i++)
        buf[i] = (_Float16)((float)arc4random() / (float)UINT32_MAX * 0.02f - 0.01f);
    return buf;
}

static float *rand_fp32(int n) {
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++)
        buf[i] = (float)arc4random() / (float)UINT32_MAX * 2.0f - 1.0f;
    return buf;
}

// ─── Test: Single conv ──────────────────────────────────────────────────────
static bool test_single_conv(int out_ch, int in_ch, int S, const char *label) {
    fprintf(stderr, "\n=== %s: conv [%d, %d] S=%d ===\n", label, out_ch, in_ch, S);

    _Float16 *W = rand_fp16(out_ch * in_ch);
    NSData *blob = mil_build_single_weight_blob(W, out_ch, in_ch);
    fprintf(stderr, "  Weight blob: %.1f MB\n", (double)blob.length / 1e6);

    NSString *mil = mil_gen_conv_baked(in_ch, out_ch, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    size_t inSz = (size_t)in_ch * S * sizeof(float);
    size_t outSz = (size_t)out_ch * S * sizeof(float);

    double t0 = time_ms();
    ANEKernel *k = ane_compile(milData, blob, 1, &inSz, 1, &outSz);
    double t_compile = time_ms() - t0;

    if (!k) {
        fprintf(stderr, "  COMPILE FAILED\n");
        free(W);
        return false;
    }
    fprintf(stderr, "  Compile+load: %.1f ms\n", t_compile);

    // Eval
    float *x = rand_fp32(in_ch * S);
    float *y = (float *)calloc((size_t)out_ch * S, sizeof(float));

    ane_write_input(k, 0, x, inSz);
    ane_eval(k);
    ane_read_output(k, 0, y, outSz);

    // Verify non-zero
    float sum = 0;
    for (int i = 0; i < out_ch * S && i < 1000; i++) sum += fabsf(y[i]);
    fprintf(stderr, "  Output check: sum(|y|)=%.6f %s\n", sum, sum > 0 ? "OK" : "ZERO!");

    // Throughput (10 iterations)
    int iters = 10;
    t0 = time_ms();
    for (int i = 0; i < iters; i++) {
        ane_write_input(k, 0, x, inSz);
        ane_eval(k);
        ane_read_output(k, 0, y, outSz);
    }
    double dt = (time_ms() - t0) / iters;
    double flops = 2.0 * out_ch * in_ch * S;
    double tflops = flops / dt / 1e9;
    fprintf(stderr, "  Eval: %.2f ms/iter, %.2f TFLOPS (%.0f GFLOP)\n", dt, tflops, flops/1e9);

    ane_free(k);
    free(W); free(x); free(y);
    return true;
}

// ─── Test: Fused QKV ────────────────────────────────────────────────────────
static bool test_fused_qkv(int dim, int kv_dim, int S) {
    fprintf(stderr, "\n=== Fused QKV: Wq[%d,%d] Wk[%d,%d] Wv[%d,%d] S=%d ===\n",
            dim, dim, kv_dim, dim, kv_dim, dim, S);

    _Float16 *wq = rand_fp16(dim * dim);
    _Float16 *wk = rand_fp16(kv_dim * dim);
    _Float16 *wv = rand_fp16(kv_dim * dim);

    NSData *blob = mil_build_qkv_baked_blob(wq, dim, wk, wv, kv_dim, dim);
    fprintf(stderr, "  Weight blob: %.1f MB\n", (double)blob.length / 1e6);

    NSString *mil = mil_gen_qkv_baked(dim, dim, kv_dim, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    size_t inSz = (size_t)dim * S * sizeof(float);
    size_t outSzQ = (size_t)dim * S * sizeof(float);
    size_t outSzKV = (size_t)kv_dim * S * sizeof(float);
    size_t outSizes[3] = {outSzQ, outSzKV, outSzKV};

    double t0 = time_ms();
    ANEKernel *k = ane_compile(milData, blob, 1, &inSz, 3, outSizes);
    double t_compile = time_ms() - t0;

    if (!k) {
        fprintf(stderr, "  COMPILE FAILED\n");
        free(wq); free(wk); free(wv);
        return false;
    }
    fprintf(stderr, "  Compile+load: %.1f ms\n", t_compile);

    float *x = rand_fp32(dim * S);
    float *q = (float *)calloc(outSzQ / sizeof(float), sizeof(float));
    float *kk = (float *)calloc(outSzKV / sizeof(float), sizeof(float));
    float *v = (float *)calloc(outSzKV / sizeof(float), sizeof(float));

    ane_write_input(k, 0, x, inSz);
    ane_eval(k);
    ane_read_output(k, 0, q, outSzQ);
    ane_read_output(k, 1, kk, outSzKV);
    ane_read_output(k, 2, v, outSzKV);

    float sq = 0, sk = 0, sv = 0;
    for (int i = 0; i < 1000 && i < dim*S; i++) sq += fabsf(q[i]);
    for (int i = 0; i < 1000 && i < kv_dim*S; i++) { sk += fabsf(kk[i]); sv += fabsf(v[i]); }
    fprintf(stderr, "  Output: |Q|=%.4f |K|=%.4f |V|=%.4f %s\n",
            sq, sk, sv, (sq > 0 && sk > 0 && sv > 0) ? "OK" : "ZERO!");

    // Throughput
    int iters = 10;
    t0 = time_ms();
    for (int i = 0; i < iters; i++) {
        ane_write_input(k, 0, x, inSz);
        ane_eval(k);
        ane_read_output(k, 0, q, outSzQ);
        ane_read_output(k, 1, kk, outSzKV);
        ane_read_output(k, 2, v, outSzKV);
    }
    double dt = (time_ms() - t0) / iters;
    double flops = 2.0 * ((double)dim*dim + 2.0*kv_dim*dim) * S;
    fprintf(stderr, "  Eval: %.2f ms/iter, %.2f TFLOPS\n", dt, flops / dt / 1e9);

    ane_free(k);
    free(wq); free(wk); free(wv); free(x); free(q); free(kk); free(v);
    return true;
}

// ─── Test: Fused FFN ────────────────────────────────────────────────────────
static bool test_fused_ffn(int dim, int hidden, int S) {
    fprintf(stderr, "\n=== Fused FFN: W1[%d,%d] W3[%d,%d] W2[%d,%d] S=%d ===\n",
            hidden, dim, hidden, dim, dim, hidden, S);

    _Float16 *w1 = rand_fp16(hidden * dim);
    _Float16 *w3 = rand_fp16(hidden * dim);
    _Float16 *w2 = rand_fp16(dim * hidden);

    NSData *blob = mil_build_ffn_fused_blob(w1, w3, hidden, w2, dim);
    fprintf(stderr, "  Weight blob: %.1f MB\n", (double)blob.length / 1e6);

    NSString *mil = mil_gen_ffn_fused(dim, hidden, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    size_t inSz = (size_t)dim * S * sizeof(float);
    size_t outSz = (size_t)dim * S * sizeof(float);

    double t0 = time_ms();
    ANEKernel *k = ane_compile(milData, blob, 1, &inSz, 1, &outSz);
    double t_compile = time_ms() - t0;

    if (!k) {
        fprintf(stderr, "  COMPILE FAILED\n");
        free(w1); free(w3); free(w2);
        return false;
    }
    fprintf(stderr, "  Compile+load: %.1f ms\n", t_compile);

    float *x = rand_fp32(dim * S);
    float *y = (float *)calloc((size_t)dim * S, sizeof(float));

    ane_write_input(k, 0, x, inSz);
    ane_eval(k);
    ane_read_output(k, 0, y, outSz);

    float sum = 0;
    for (int i = 0; i < dim * S && i < 1000; i++) sum += fabsf(y[i]);
    fprintf(stderr, "  Output check: sum(|y|)=%.6f %s\n", sum, sum > 0 ? "OK" : "ZERO!");

    // Throughput
    int iters = 10;
    t0 = time_ms();
    for (int i = 0; i < iters; i++) {
        ane_write_input(k, 0, x, inSz);
        ane_eval(k);
        ane_read_output(k, 0, y, outSz);
    }
    double dt = (time_ms() - t0) / iters;
    // FFN FLOPs: 2*hidden*dim*S (W1) + 2*hidden*dim*S (W3) + 2*dim*hidden*S (W2) + 3*hidden*S (sigmoid+mul+mul)
    double flops = (4.0*hidden*dim + 2.0*dim*hidden) * S;
    fprintf(stderr, "  Eval: %.2f ms/iter, %.2f TFLOPS\n", dt, flops / dt / 1e9);

    ane_free(k);
    free(w1); free(w3); free(w2); free(x); free(y);
    return true;
}

// ─── Test: Compile limit (many programs) ────────────────────────────────────
static void test_compile_limit(int dim, int S) {
    fprintf(stderr, "\n=== Compile limit test: multiple conv [%d,%d] S=%d ===\n", dim, dim, S);

    _Float16 *W = rand_fp16(dim * dim);
    NSData *blob = mil_build_single_weight_blob(W, dim, dim);
    NSString *mil = mil_gen_conv_baked(dim, dim, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    size_t inSz = (size_t)dim * S * sizeof(float);
    size_t outSz = (size_t)dim * S * sizeof(float);

    ANEKernel *kernels[128] = {0};
    int compiled = 0;
    double t0 = time_ms();

    for (int i = 0; i < 128; i++) {
        ANEKernel *k = ane_compile(milData, blob, 1, &inSz, 1, &outSz);
        if (!k) {
            fprintf(stderr, "  Compile #%d FAILED — limit reached\n", i);
            break;
        }
        kernels[i] = k;
        compiled++;
    }
    double dt = time_ms() - t0;
    fprintf(stderr, "  Compiled %d kernels in %.0f ms (%.1f ms/kernel)\n",
            compiled, dt, dt / compiled);

    // Memory estimate: check process memory
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
    fprintf(stderr, "  Process resident memory: %.1f MB\n", (double)info.resident_size / 1e6);

    // Cleanup
    for (int i = 0; i < compiled; i++)
        ane_free(kernels[i]);
    free(W);
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    @autoreleasepool {
        // Mistral 7B dimensions
        int dim = 4096;
        int kv_dim = 1024;    // 8 KV heads × 128 head_dim
        int hidden = 14336;

        fprintf(stderr, "ANE Baked Weight Probe — Mistral 7B dimensions\n");
        fprintf(stderr, "dim=%d, kv_dim=%d, hidden=%d\n\n", dim, kv_dim, hidden);

        int pass = 0, fail = 0;

        // ── Single convs at increasing sizes ──
        int seq_lens[] = {16, 32, 64, 128};
        int n_seq = sizeof(seq_lens) / sizeof(seq_lens[0]);

        // Small conv: KV projection (1024×4096)
        for (int si = 0; si < n_seq; si++) {
            if (test_single_conv(kv_dim, dim, seq_lens[si], "KV proj")) pass++; else fail++;
        }

        // Medium conv: Q/Wo projection (4096×4096)
        for (int si = 0; si < n_seq; si++) {
            if (test_single_conv(dim, dim, seq_lens[si], "Q/Wo proj")) pass++; else fail++;
        }

        // Large conv: FFN up/gate (14336×4096)
        for (int si = 0; si < n_seq; si++) {
            if (test_single_conv(hidden, dim, seq_lens[si], "FFN up")) pass++; else fail++;
        }

        // Large conv: FFN down (4096×14336)
        for (int si = 0; si < n_seq; si++) {
            if (test_single_conv(dim, hidden, seq_lens[si], "FFN down")) pass++; else fail++;
        }

        // ── Fused QKV ──
        for (int si = 0; si < n_seq; si++) {
            if (test_fused_qkv(dim, kv_dim, seq_lens[si])) pass++; else fail++;
        }

        // ── Fused FFN (the big one) ──
        for (int si = 0; si < n_seq; si++) {
            if (test_fused_ffn(dim, hidden, seq_lens[si])) pass++; else fail++;
        }

        // ── Compile limit test ──
        // Use small dim to avoid OOM
        test_compile_limit(512, 16);

        fprintf(stderr, "\n══════════════════════════════════════\n");
        fprintf(stderr, "Results: %d passed, %d failed\n", pass, fail);

        return fail > 0 ? 1 : 0;
    }
}
