#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include "../training/ane_runtime.h"

// Build weight blob with DEADBEEF header
static NSData *build_weight_blob(const _Float16 *fp16, int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128;
    memcpy(buf + 128, fp16, wsize);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSString *make_mil(int ic, int oc, int s) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
        ic, s, ic, s, oc, ic, oc, ic, oc, s, oc, s];
}

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

typedef struct { int ic, oc, s; const char *name; } TestCase;

int main(void) {
    @autoreleasepool {
        ane_init();

        // Mistral-relevant sizes
        TestCase cases[] = {
            {4096, 4096,  1, "4096x4096 S=1  (Q/K/O proj decode)"},
            {4096, 4096, 16, "4096x4096 S=16 (Q/K/O proj prefill)"},
            {4096, 4096, 64, "4096x4096 S=64"},
            {4096, 14336, 1, "4096x14336 S=1 (FFN up/gate decode)"},
            {14336, 4096, 1, "14336x4096 S=1 (FFN down decode)"},
            {4096, 14336,16, "4096x14336 S=16 (FFN up/gate prefill)"},
            {14336, 4096,16, "14336x4096 S=16 (FFN down prefill)"},
            {4096, 1024,  1, "4096x1024 S=1  (GQA KV proj)"},
        };
        int ncases = sizeof(cases) / sizeof(cases[0]);

        fprintf(stderr, "=== ANE MIL Validation + Performance ===\n\n");

        for (int t = 0; t < ncases; t++) {
            int IC = cases[t].ic, OC = cases[t].oc, S = cases[t].s;
            fprintf(stderr, "--- %s ---\n", cases[t].name);

            // Random weights + input
            _Float16 *wfp16 = malloc(OC * IC * sizeof(_Float16));
            float *xf = malloc(IC * S * sizeof(float));
            srand(42 + t);
            for (int i = 0; i < OC * IC; i++)
                wfp16[i] = (_Float16)(0.01f * ((rand() % 200) - 100));
            for (int i = 0; i < IC * S; i++)
                xf[i] = 0.01f * ((rand() % 200) - 100);

            NSData *wdata = build_weight_blob(wfp16, OC, IC);
            NSString *mil = make_mil(IC, OC, S);
            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            size_t inSz[1]  = { (size_t)IC * S * sizeof(float) };
            size_t outSz[1] = { (size_t)OC * S * sizeof(float) };

            // Compile
            double t0 = now_ms();
            ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
            double compile_ms = now_ms() - t0;

            if (!k) {
                fprintf(stderr, "  COMPILE FAIL\n\n");
                free(wfp16); free(xf);
                continue;
            }
            fprintf(stderr, "  compile: %.1f ms\n", compile_ms);

            // CPU reference: fp16 matmul (cast x to fp16, matmul, cast back)
            _Float16 *xfp16 = malloc(IC * S * sizeof(_Float16));
            for (int i = 0; i < IC * S; i++) xfp16[i] = (_Float16)xf[i];
            float *cpu_out = calloc(OC * S, sizeof(float));
            // W is [OC, IC], x is [IC, S] in column-major → y = W * x
            // But ANE conv layout: x is [1, IC, 1, S], W is [OC, IC, 1, 1]
            // This is y[oc][s] = sum_ic W[oc][ic] * x[ic][s]
            // In BLAS: y = W * x where W is OC×IC, x is IC×S
            _Float16 *yfp16 = calloc(OC * S, sizeof(_Float16));
            // Use vDSP for reference since we want fp16 precision match
            for (int oc = 0; oc < OC; oc++) {
                for (int s = 0; s < S; s++) {
                    float acc = 0;
                    for (int ic = 0; ic < IC; ic++)
                        acc += (float)wfp16[oc * IC + ic] * (float)xfp16[ic * S + s];
                    cpu_out[oc * S + s] = acc;
                }
            }

            // ANE eval
            float *ane_out = calloc(OC * S, sizeof(float));
            ane_write_input(k, 0, xf, inSz[0]);
            ane_eval(k);  // warmup
            ane_read_output(k, 0, ane_out, outSz[0]);

            // Validate: check max abs error and relative error
            float max_abs = 0, max_rel = 0;
            int mismatches = 0;
            for (int i = 0; i < OC * S; i++) {
                float diff = fabsf(ane_out[i] - cpu_out[i]);
                float ref = fabsf(cpu_out[i]);
                if (diff > max_abs) max_abs = diff;
                float rel = ref > 1e-6f ? diff / ref : 0;
                if (rel > max_rel) max_rel = rel;
                // fp16 has ~0.1% relative error, allow generous margin
                if (rel > 0.05f && diff > 0.5f) mismatches++;
            }
            fprintf(stderr, "  correctness: max_abs=%.4f max_rel=%.4f mismatches=%d/%d %s\n",
                    max_abs, max_rel, mismatches, OC * S,
                    mismatches == 0 ? "PASS" : "FAIL");

            // Performance: measure eval latency
            int nruns = (S == 1) ? 200 : 50;
            // Warmup
            for (int i = 0; i < 5; i++) {
                ane_write_input(k, 0, xf, inSz[0]);
                ane_eval(k);
            }
            double best = 1e9, total = 0;
            for (int i = 0; i < nruns; i++) {
                ane_write_input(k, 0, xf, inSz[0]);
                double t1 = now_ms();
                ane_eval(k);
                double dt = now_ms() - t1;
                if (dt < best) best = dt;
                total += dt;
            }
            double avg = total / nruns;
            double flops = 2.0 * OC * IC * S;
            double tflops_best = flops / (best * 1e-3) / 1e12;
            double tflops_avg  = flops / (avg * 1e-3) / 1e12;
            fprintf(stderr, "  latency: best=%.3f ms  avg=%.3f ms  (%d runs)\n", best, avg, nruns);
            fprintf(stderr, "  throughput: %.2f TFLOPS (best)  %.2f TFLOPS (avg)\n", tflops_best, tflops_avg);

            // Also measure with I/O copy overhead
            double best_io = 1e9, total_io = 0;
            for (int i = 0; i < nruns; i++) {
                double t1 = now_ms();
                ane_write_input(k, 0, xf, inSz[0]);
                ane_eval(k);
                ane_read_output(k, 0, ane_out, outSz[0]);
                double dt = now_ms() - t1;
                if (dt < best_io) best_io = dt;
                total_io += dt;
            }
            fprintf(stderr, "  with I/O:  best=%.3f ms  avg=%.3f ms\n", best_io, total_io / nruns);

            fprintf(stderr, "\n");
            ane_free(k);
            free(wfp16); free(xf); free(xfp16); free(cpu_out); free(yfp16); free(ane_out);
        }

        fprintf(stderr, "=== Done ===\n");
    }
    return 0;
}
