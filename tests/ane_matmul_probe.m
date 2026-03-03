// ane_matmul_probe.m — Test runtime-weight matmul at Mistral 7B dimensions
// If matmul accepts runtime weight IOSurfaces, we only need 5 programs total
// and can swap weights per-layer via IOSurface writes.
// Build: cd mistral && make matmul_probe
// Run:   ./matmul_probe
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <math.h>

#include "../training/ane_runtime.h"
#include "../training/ane_mil_gen.h"

static double time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

static float *rand_fp32(int n) {
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++)
        buf[i] = (float)arc4random() / (float)UINT32_MAX * 2.0f - 1.0f;
    return buf;
}

// Test matmul with runtime weight input at given dimensions
static bool test_matmul_runtime(int out_ch, int in_ch, int S, const char *label) {
    fprintf(stderr, "\n=== %s: matmul [%d, %d] S=%d ===\n", label, out_ch, in_ch, S);

    // Generate MIL with 2 inputs: x[1, in_ch, S] and W[1, out_ch, in_ch]
    NSString *mil = mil_gen_matmul(in_ch, out_ch, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    // I/O sizes: 2 inputs (x + W), 1 output (y)
    size_t x_bytes = (size_t)in_ch * S * sizeof(float);
    size_t w_bytes = (size_t)out_ch * in_ch * sizeof(float);
    size_t y_bytes = (size_t)out_ch * S * sizeof(float);
    size_t inputSizes[2] = {x_bytes, w_bytes};

    double t0 = time_ms();
    ANEKernel *k = ane_compile(milData, nil, 2, inputSizes, 1, &y_bytes);
    double t_compile = time_ms() - t0;

    if (!k) {
        fprintf(stderr, "  COMPILE FAILED — matmul with runtime weights doesn't work at [%d,%d]\n", out_ch, in_ch);
        return false;
    }
    fprintf(stderr, "  Compile+load: %.1f ms\n", t_compile);

    // Create test data
    float *x = rand_fp32(in_ch * S);
    float *W = rand_fp32(out_ch * in_ch);
    float *y = (float *)calloc((size_t)out_ch * S, sizeof(float));

    // Write inputs and eval
    ane_write_input(k, 0, x, x_bytes);
    ane_write_input(k, 1, W, w_bytes);
    ane_eval(k);
    ane_read_output(k, 0, y, y_bytes);

    // Verify non-zero output
    float sum = 0;
    for (int i = 0; i < out_ch * S && i < 1000; i++) sum += fabsf(y[i]);
    fprintf(stderr, "  Output check: sum(|y|)=%.6f %s\n", sum, sum > 0 ? "OK" : "ZERO!");

    if (sum > 0) {
        // Test weight swapping: write new weights, eval again
        float *W2 = rand_fp32(out_ch * in_ch);
        float *y2 = (float *)calloc((size_t)out_ch * S, sizeof(float));
        ane_write_input(k, 0, x, x_bytes);
        ane_write_input(k, 1, W2, w_bytes);
        ane_eval(k);
        ane_read_output(k, 0, y2, y_bytes);

        float sum2 = 0;
        for (int i = 0; i < out_ch * S && i < 1000; i++) sum2 += fabsf(y2[i]);
        // Check that results differ (different weights should give different output)
        float diff = 0;
        for (int i = 0; i < out_ch * S && i < 1000; i++) diff += fabsf(y[i] - y2[i]);
        fprintf(stderr, "  Weight swap: sum(|y2|)=%.6f diff=%.6f %s\n",
                sum2, diff, diff > 0.01f ? "DIFFERENT (good)" : "SAME (bad)");

        // Throughput (weight already loaded, just eval)
        int iters = 10;
        t0 = time_ms();
        for (int i = 0; i < iters; i++) {
            ane_write_input(k, 0, x, x_bytes);
            ane_write_input(k, 1, W2, w_bytes);
            ane_eval(k);
            ane_read_output(k, 0, y2, y_bytes);
        }
        double dt = (time_ms() - t0) / iters;
        double flops = 2.0 * out_ch * in_ch * S;
        double tflops = flops / dt / 1e9;
        fprintf(stderr, "  Eval: %.2f ms/iter, %.2f TFLOPS (%.0f GFLOP)\n", dt, tflops, flops/1e9);

        free(W2); free(y2);
    }

    ane_free(k);
    free(x); free(W); free(y);
    return sum > 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        fprintf(stderr, "ANE Runtime Matmul Probe — Testing runtime weight inputs\n");
        fprintf(stderr, "If matmul works at scale, we only need 5 compiled programs total!\n\n");

        int pass = 0, fail = 0;

        // Small: should work (proven for attention Q@K^T)
        if (test_matmul_runtime(128, 128, 16, "Small 128x128")) pass++; else fail++;

        // Medium: head_dim scale
        if (test_matmul_runtime(512, 512, 16, "Medium 512x512")) pass++; else fail++;

        // KV projection scale
        if (test_matmul_runtime(1024, 4096, 16, "KV proj 1024x4096")) pass++; else fail++;

        // Q/Wo projection scale
        if (test_matmul_runtime(4096, 4096, 16, "Q/Wo proj 4096x4096")) pass++; else fail++;

        // FFN up/gate scale
        if (test_matmul_runtime(14336, 4096, 16, "FFN up 14336x4096")) pass++; else fail++;

        // FFN down scale
        if (test_matmul_runtime(4096, 14336, 16, "FFN down 4096x14336")) pass++; else fail++;

        // If large works, test with bigger S
        if (pass >= 4) {
            fprintf(stderr, "\n--- Large dims work! Testing S=32,64 ---\n");
            test_matmul_runtime(4096, 4096, 32, "Q/Wo 4096x4096 S=32");
            test_matmul_runtime(4096, 4096, 64, "Q/Wo 4096x4096 S=64");
            test_matmul_runtime(14336, 4096, 32, "FFN up S=32");
            test_matmul_runtime(14336, 4096, 64, "FFN up S=64");
        }

        fprintf(stderr, "\n══════════════════════════════════════\n");
        fprintf(stderr, "Results: %d passed, %d failed\n", pass, fail);

        return fail > 0 ? 1 : 0;
    }
}
