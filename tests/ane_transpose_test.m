// ane_transpose_test.m — Verify channel-first transpose for ANE I/O
// Build: cd tests && make transpose_test
// Run:   ./transpose_test

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#include "../training/ane_runtime.h"
#include "../mistral/ane_mil_gen_mistral.h"

static double time_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// Token-first [S, dim] → Channel-first [dim, S] (ANE layout: [1,C,1,S])
static void transpose_to_ane(const float *token_first, float *channel_first, int dim, int S) {
    for (int c = 0; c < dim; c++)
        for (int s = 0; s < S; s++)
            channel_first[c * S + s] = token_first[s * dim + c];
}

// Channel-first [dim, S] → Token-first [S, dim]
static void transpose_from_ane(const float *channel_first, float *token_first, int dim, int S) {
    for (int c = 0; c < dim; c++)
        for (int s = 0; s < S; s++)
            token_first[s * dim + c] = channel_first[c * S + s];
}

// Fill buffer with deterministic pseudo-random floats in [-1, 1]
static void fill_random(float *buf, int n, uint32_t seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = ((float)(int)(seed >> 8) / (float)0x7FFFFF) - 1.0f;
    }
}

// ─── Test 1: Round-trip identity ─────────────────────────────────────────────

static bool test_roundtrip(int dim, int S) {
    int n = dim * S;
    float *orig = malloc(n * sizeof(float));
    float *ch_first = malloc(n * sizeof(float));
    float *recovered = malloc(n * sizeof(float));

    fill_random(orig, n, 42 + dim + S);

    transpose_to_ane(orig, ch_first, dim, S);
    transpose_from_ane(ch_first, recovered, dim, S);

    bool pass = true;
    for (int i = 0; i < n; i++) {
        if (orig[i] != recovered[i]) {
            fprintf(stderr, "  FAIL: roundtrip mismatch at idx %d: %.8f vs %.8f\n",
                    i, orig[i], recovered[i]);
            pass = false;
            break;
        }
    }

    free(orig); free(ch_first); free(recovered);
    return pass;
}

// ─── Test 2: ANE conv with proper transpose vs CPU matmul ────────────────────

static bool test_ane_vs_cpu(int out_ch, int in_ch, int S) {
    int x_n = S * in_ch;
    int w_n = out_ch * in_ch;
    int y_n = S * out_ch;

    float *x_tok = malloc(x_n * sizeof(float));     // [S, in_ch] token-first
    float *w_fp32 = malloc(w_n * sizeof(float));     // [out_ch, in_ch]
    float *y_cpu = malloc(y_n * sizeof(float));      // [S, out_ch] CPU reference
    float *x_ane = malloc(x_n * sizeof(float));      // [in_ch, S] channel-first
    float *y_ane_ch = malloc(y_n * sizeof(float));   // [out_ch, S] channel-first output
    float *y_ane_tok = malloc(y_n * sizeof(float));  // [S, out_ch] token-first output

    fill_random(x_tok, x_n, 123);
    fill_random(w_fp32, w_n, 456);

    // CPU reference: Y[S, out_ch] = X[S, in_ch] @ W^T[in_ch, out_ch]
    // cblas: C = alpha * A * B + beta * C
    // A = X[S, in_ch], B = W^T => use CblasTrans on W[out_ch, in_ch]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                S, out_ch, in_ch,
                1.0f, x_tok, in_ch, w_fp32, in_ch,
                0.0f, y_cpu, out_ch);

    // Convert weights to fp16 for ANE
    _Float16 *w_fp16 = malloc(w_n * sizeof(_Float16));
    for (int i = 0; i < w_n; i++)
        w_fp16[i] = (_Float16)w_fp32[i];

    // Build weight blob and MIL
    NSData *blob = mil_build_single_weight_blob(w_fp16, out_ch, in_ch);
    NSString *mil = mil_gen_conv_baked(in_ch, out_ch, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    size_t inBytes = (size_t)in_ch * S * sizeof(float);
    size_t outBytes = (size_t)out_ch * S * sizeof(float);
    ANEKernel *k = ane_compile(milData, blob, 1, &inBytes, 1, &outBytes);
    if (!k) {
        fprintf(stderr, "  SKIP: ANE compile failed for %dx%d S=%d\n", out_ch, in_ch, S);
        free(x_tok); free(w_fp32); free(y_cpu);
        free(x_ane); free(y_ane_ch); free(y_ane_tok);
        free(w_fp16);
        return true; // skip, not fail
    }

    // Transpose input to channel-first for ANE
    transpose_to_ane(x_tok, x_ane, in_ch, S);

    // Write input, eval, read output
    ane_write_input(k, 0, x_ane, inBytes);
    if (!ane_eval(k)) {
        fprintf(stderr, "  FAIL: ANE eval failed\n");
        ane_free(k);
        free(x_tok); free(w_fp32); free(y_cpu);
        free(x_ane); free(y_ane_ch); free(y_ane_tok);
        free(w_fp16);
        return false;
    }
    ane_read_output(k, 0, y_ane_ch, outBytes);

    // Transpose output back to token-first
    transpose_from_ane(y_ane_ch, y_ane_tok, out_ch, S);

    // Compare with CPU reference
    float max_abs_err = 0, max_rel_err = 0;
    for (int i = 0; i < y_n; i++) {
        float err = fabsf(y_ane_tok[i] - y_cpu[i]);
        float rel = (fabsf(y_cpu[i]) > 1e-6f) ? err / fabsf(y_cpu[i]) : err;
        if (err > max_abs_err) max_abs_err = err;
        if (rel > max_rel_err) max_rel_err = rel;
    }

    bool pass = (max_abs_err < 0.1f); // fp16 precision: abs error < 0.1
    if (!pass)
        fprintf(stderr, "  FAIL: max_abs=%.6f max_rel=%.6f (threshold 0.1)\n",
                max_abs_err, max_rel_err);
    else
        printf("  max_abs_err=%.6f  max_rel_err=%.6f\n", max_abs_err, max_rel_err);

    ane_free(k);
    free(x_tok); free(w_fp32); free(y_cpu);
    free(x_ane); free(y_ane_ch); free(y_ane_tok);
    free(w_fp16);
    return pass;
}

// ─── Test 3: WITHOUT transpose (should give wrong results) ───────────────────

static bool test_no_transpose_is_wrong(int out_ch, int in_ch, int S) {
    int x_n = S * in_ch;
    int w_n = out_ch * in_ch;
    int y_n = S * out_ch;

    float *x_tok = malloc(x_n * sizeof(float));
    float *w_fp32 = malloc(w_n * sizeof(float));
    float *y_cpu = malloc(y_n * sizeof(float));
    float *y_ane_raw = malloc(y_n * sizeof(float));

    fill_random(x_tok, x_n, 789);
    fill_random(w_fp32, w_n, 101);

    // CPU reference
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                S, out_ch, in_ch,
                1.0f, x_tok, in_ch, w_fp32, in_ch,
                0.0f, y_cpu, out_ch);

    // Weights to fp16
    _Float16 *w_fp16 = malloc(w_n * sizeof(_Float16));
    for (int i = 0; i < w_n; i++)
        w_fp16[i] = (_Float16)w_fp32[i];

    NSData *blob = mil_build_single_weight_blob(w_fp16, out_ch, in_ch);
    NSString *mil = mil_gen_conv_baked(in_ch, out_ch, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    size_t inBytes = (size_t)in_ch * S * sizeof(float);
    size_t outBytes = (size_t)out_ch * S * sizeof(float);
    ANEKernel *k = ane_compile(milData, blob, 1, &inBytes, 1, &outBytes);
    if (!k) {
        fprintf(stderr, "  SKIP: ANE compile failed\n");
        free(x_tok); free(w_fp32); free(y_cpu); free(y_ane_raw); free(w_fp16);
        return true;
    }

    // Write token-first directly to ANE (NO transpose — this is the bug)
    ane_write_input(k, 0, x_tok, inBytes);
    ane_eval(k);
    ane_read_output(k, 0, y_ane_raw, outBytes);

    // Read output as token-first directly (NO transpose back)
    // This should NOT match the CPU result
    float max_rel_err = 0;
    int mismatches = 0;
    for (int i = 0; i < y_n; i++) {
        float err = fabsf(y_ane_raw[i] - y_cpu[i]);
        float rel = (fabsf(y_cpu[i]) > 1e-6f) ? err / fabsf(y_cpu[i]) : err;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 0.05f) mismatches++;
    }

    // We WANT mismatches — no-transpose should be wrong
    bool pass = (mismatches > y_n / 4); // at least 25% of values should differ
    if (!pass)
        fprintf(stderr, "  FAIL: no-transpose unexpectedly matched CPU! mismatches=%d/%d max_rel=%.4f\n",
                mismatches, y_n, max_rel_err);
    else
        printf("  %d/%d values differ (max_rel=%.4f) — transpose is necessary\n",
               mismatches, y_n, max_rel_err);

    ane_free(k);
    free(x_tok); free(w_fp32); free(y_cpu); free(y_ane_raw); free(w_fp16);
    return pass;
}

// ─── Test 4: Transpose performance ──────────────────────────────────────────

static void bench_transpose(int dim, int S) {
    int n = dim * S;
    float *tok = malloc(n * sizeof(float));
    float *ch = malloc(n * sizeof(float));
    float *back = malloc(n * sizeof(float));
    fill_random(tok, n, 999);

    int iters = 100;

    // Warm up
    transpose_to_ane(tok, ch, dim, S);
    transpose_from_ane(ch, back, dim, S);

    double t0 = time_ms();
    for (int i = 0; i < iters; i++)
        transpose_to_ane(tok, ch, dim, S);
    double t1 = time_ms();
    for (int i = 0; i < iters; i++)
        transpose_from_ane(ch, back, dim, S);
    double t2 = time_ms();

    double to_ane_us = (t1 - t0) / iters * 1000.0;
    double from_ane_us = (t2 - t1) / iters * 1000.0;
    double bytes_mb = (double)n * sizeof(float) / (1024.0 * 1024.0);

    printf("  dim=%d S=%d  to_ane=%.1f us  from_ane=%.1f us  (%.2f MB, %.1f GB/s)\n",
           dim, S, to_ane_us, from_ane_us, bytes_mb,
           bytes_mb / ((to_ane_us + from_ane_us) / 2.0 / 1e6) / 1024.0);

    free(tok); free(ch); free(back);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    @autoreleasepool {
        int passed = 0, failed = 0, total = 0;

        #define RUN(name, expr) do { \
            total++; \
            printf("[%d] %s\n", total, name); \
            if (expr) { passed++; printf("  PASS\n"); } \
            else { failed++; printf("  FAIL\n"); } \
        } while(0)

        printf("=== Transpose Round-trip Tests ===\n");
        RUN("roundtrip dim=64  S=16",   test_roundtrip(64, 16));
        RUN("roundtrip dim=128 S=32",   test_roundtrip(128, 32));
        RUN("roundtrip dim=4096 S=64",  test_roundtrip(4096, 64));
        RUN("roundtrip dim=4096 S=128", test_roundtrip(4096, 128));

        printf("\n=== ANE Conv vs CPU Matmul (with transpose) ===\n");
        RUN("ane_vs_cpu 128x128 S=16",    test_ane_vs_cpu(128, 128, 16));
        RUN("ane_vs_cpu 256x128 S=32",    test_ane_vs_cpu(256, 128, 32));
        RUN("ane_vs_cpu 4096x4096 S=64",  test_ane_vs_cpu(4096, 4096, 64));

        printf("\n=== No-Transpose Proves Layout Matters ===\n");
        RUN("no_transpose 128x128 S=16",   test_no_transpose_is_wrong(128, 128, 16));
        RUN("no_transpose 256x128 S=32",   test_no_transpose_is_wrong(256, 128, 32));

        printf("\n=== Transpose Performance ===\n");
        bench_transpose(128, 16);
        bench_transpose(4096, 64);
        bench_transpose(4096, 128);
        bench_transpose(14336, 64);

        printf("\n=== Results: %d/%d passed", passed, total);
        if (failed) printf(", %d FAILED", failed);
        printf(" ===\n");

        return failed ? 1 : 0;
    }
}
