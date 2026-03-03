#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <arm_neon.h>
#include "../training/ane_runtime.h"

static NSData *build_weight_blob(const __fp16 *fp16, int oc, int ic) {
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

static NSString *make_mil_fp32(int ic, int oc, int s) {
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

__attribute__((noinline))
static void neon_fp16_matvec(const __fp16 *W, const __fp16 *x, float *y, int oc, int ic) {
    for (int o = 0; o < oc; o++) {
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        const __fp16 *row = W + (size_t)o * ic;
        int i = 0;
        for (; i + 15 < ic; i += 16) {
            float16x8_t w0 = vld1q_f16(row + i), w1 = vld1q_f16(row + i + 8);
            float16x8_t x0 = vld1q_f16(x + i),   x1 = vld1q_f16(x + i + 8);
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(w0)), vcvt_f32_f16(vget_low_f16(x0)));
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_high_f16(w0)), vcvt_f32_f16(vget_high_f16(x0)));
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_low_f16(w1)), vcvt_f32_f16(vget_low_f16(x1)));
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(w1)), vcvt_f32_f16(vget_high_f16(x1)));
        }
        acc0 = vaddq_f32(acc0, acc1);
        y[o] = vaddvq_f32(acc0);
        for (; i < ic; i++) y[o] += (float)row[i] * (float)x[i];
    }
}

typedef struct { int ic, oc, s; const char *name; } TestCase;

int main(void) {
    @autoreleasepool {
        ane_init();

        TestCase cases[] = {
            {4096, 4096,   1, "Q/K/O decode S=1"},
            {4096, 14336,  1, "FFN up/gate decode S=1"},
            {14336, 4096,  1, "FFN down decode S=1"},
            {4096, 1024,   1, "GQA KV decode S=1"},
            {4096, 4096,  32, "Q/K/O prefill S=32"},
            {4096, 4096,  64, "Q/K/O prefill S=64"},
            {4096, 4096, 128, "Q/K/O prefill S=128"},
            {4096, 4096, 256, "Q/K/O prefill S=256"},
            {4096, 14336, 64, "FFN up prefill S=64"},
            {4096, 14336,128, "FFN up prefill S=128"},
        };
        int ncases = sizeof(cases)/sizeof(cases[0]);

        fprintf(stderr, "=== ANE vs CPU: Mistral 7B Layer Projections ===\n");
        fprintf(stderr, "%-28s %9s %9s %9s  %-6s %s\n",
                "Operation", "NEON ms", "AMX ms", "ANE ms", "Winner", "Speedup");
        fprintf(stderr, "-------------------------------------------------------------------------\n");

        for (int t = 0; t < ncases; t++) {
            int IC = cases[t].ic, OC = cases[t].oc, S = cases[t].s;
            int ANE_S = (S < 16) ? 16 : S;

            __fp16 *wfp16 = (__fp16*)malloc((size_t)OC * IC * sizeof(__fp16));
            srand(42 + t);
            for (int i = 0; i < OC * IC; i++)
                wfp16[i] = (__fp16)(0.001f * (rand() % 2000 - 1000));

            __fp16 *xh = (__fp16*)calloc((size_t)IC * ANE_S, sizeof(__fp16));
            float *xf = calloc((size_t)IC * ANE_S, sizeof(float));
            for (int i = 0; i < IC * S; i++) {
                xh[i] = (__fp16)(0.01f * (rand() % 200 - 100));
                xf[i] = (float)xh[i];
            }

            int nruns = (S == 1) ? 50 : 20;

            // --- NEON fp16 matvec (S=1 only) ---
            double neon_best = -1;
            if (S == 1) {
                float *yn = calloc(OC, sizeof(float));
                for (int i = 0; i < 3; i++) neon_fp16_matvec(wfp16, xh, yn, OC, IC);
                neon_best = 1e9;
                for (int i = 0; i < nruns; i++) {
                    double t0 = now_ms();
                    neon_fp16_matvec(wfp16, xh, yn, OC, IC);
                    double dt = now_ms() - t0;
                    if (dt < neon_best) neon_best = dt;
                }
                free(yn);
            }

            // --- AMX (cblas_sgemm) ---
            double amx_best = 1e9;
            {
                float *wf = malloc((size_t)OC * IC * sizeof(float));
                for (int i = 0; i < OC * IC; i++) wf[i] = (float)wfp16[i];
                float *ya = calloc((size_t)OC * S, sizeof(float));
                for (int i = 0; i < 3; i++)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                OC, S, IC, 1.0f, wf, IC, xf, S, 0.0f, ya, S);
                for (int i = 0; i < nruns; i++) {
                    double t0 = now_ms();
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                OC, S, IC, 1.0f, wf, IC, xf, S, 0.0f, ya, S);
                    double dt = now_ms() - t0;
                    if (dt < amx_best) amx_best = dt;
                }
                free(wf); free(ya);
            }

            // --- ANE ---
            double ane_best = -1;
            {
                NSData *wdata = build_weight_blob(wfp16, OC, IC);
                NSString *mil = make_mil_fp32(IC, OC, ANE_S);
                NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
                size_t inSz[1]  = { (size_t)IC * ANE_S * sizeof(float) };
                size_t outSz[1] = { (size_t)OC * ANE_S * sizeof(float) };
                ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
                if (k) {
                    for (int i = 0; i < 5; i++) { ane_write_input(k, 0, xf, inSz[0]); ane_eval(k); }
                    ane_best = 1e9;
                    for (int i = 0; i < nruns; i++) {
                        ane_write_input(k, 0, xf, inSz[0]);
                        double t0 = now_ms();
                        ane_eval(k);
                        double dt = now_ms() - t0;
                        if (dt < ane_best) ane_best = dt;
                    }
                    ane_free(k);
                }
            }

            // Results
            double times[3] = {neon_best, amx_best, ane_best};
            const char *names[3] = {"NEON", "AMX", "ANE"};
            int best_idx = -1;
            double best_time = 1e9;
            for (int i = 0; i < 3; i++) {
                if (times[i] > 0 && times[i] < best_time) {
                    best_time = times[i];
                    best_idx = i;
                }
            }
            double second = 1e9;
            for (int i = 0; i < 3; i++) {
                if (i != best_idx && times[i] > 0 && times[i] < second) second = times[i];
            }

            fprintf(stderr, "%-28s ", cases[t].name);
            if (neon_best > 0) fprintf(stderr, "%8.3f  ", neon_best);
            else               fprintf(stderr, "%9s ", "  -  ");
            fprintf(stderr, "%8.3f  ", amx_best);
            if (ane_best > 0)  fprintf(stderr, "%8.3f  ", ane_best);
            else               fprintf(stderr, "%9s ", "FAIL");
            fprintf(stderr, "%-6s %.1fx\n", names[best_idx], second / best_time);

            free(wfp16); free(xh); free(xf);
        }

        fprintf(stderr, "\nNEON = fp16 matvec single-thread | AMX = cblas_sgemm fp32 | ANE = fp32 I/O, S padded to 16\n");
    }
    return 0;
}
