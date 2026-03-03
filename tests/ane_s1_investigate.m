#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include "../training/ane_runtime.h"

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

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
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

int main(void) {
    @autoreleasepool {
        ane_init();
        int IC = 256, OC = 256;

        fprintf(stderr, "=== S dimension sweep (IC=%d, OC=%d) ===\n\n", IC, OC);
        int svals[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
        int nsvals = sizeof(svals)/sizeof(svals[0]);

        // Fixed weights
        _Float16 *wfp16 = malloc(OC * IC * sizeof(_Float16));
        srand(42);
        for (int i = 0; i < OC * IC; i++)
            wfp16[i] = (_Float16)(0.01f * ((rand() % 200) - 100));
        NSData *wdata = build_weight_blob(wfp16, OC, IC);

        for (int si = 0; si < nsvals; si++) {
            int S = svals[si];
            float *xf = malloc(IC * S * sizeof(float));
            for (int i = 0; i < IC * S; i++) xf[i] = 1.0f;
            float *yf = calloc(OC * S, sizeof(float));

            NSString *mil = make_mil(IC, OC, S);
            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            size_t inSz[1]  = { (size_t)IC * S * sizeof(float) };
            size_t outSz[1] = { (size_t)OC * S * sizeof(float) };

            ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
            if (!k) {
                fprintf(stderr, "S=%3d: COMPILE FAIL\n", S);
                free(xf); free(yf);
                continue;
            }

            ane_write_input(k, 0, xf, inSz[0]);
            ane_eval(k);
            ane_read_output(k, 0, yf, outSz[0]);

            // CPU reference
            float cpu_ref = 0;
            for (int ic = 0; ic < IC; ic++)
                cpu_ref += (float)wfp16[ic] * 1.0f;  // y[0][0] = sum W[0][ic] * x[ic][0]

            float diff = fabsf(yf[0] - cpu_ref);
            bool correct = diff < 1.0f;  // generous for fp16 accum

            // Perf
            int nruns = 100;
            for (int i = 0; i < 5; i++) { ane_write_input(k, 0, xf, inSz[0]); ane_eval(k); }
            double best = 1e9;
            for (int i = 0; i < nruns; i++) {
                ane_write_input(k, 0, xf, inSz[0]);
                double t0 = now_ms();
                ane_eval(k);
                double dt = now_ms() - t0;
                if (dt < best) best = dt;
            }
            double flops = 2.0 * OC * IC * S;
            double gflops = flops / (best * 1e-3) / 1e9;

            fprintf(stderr, "S=%3d: y[0]=%.4f ref=%.4f diff=%.4f %s  lat=%.3f ms  %.1f GFLOPS\n",
                    S, yf[0], cpu_ref, diff, correct ? "OK" : "BAD", best, gflops);

            ane_free(k);
            free(xf); free(yf);
        }

        // Now test S=1 with padding: send S=2 but only use first column
        fprintf(stderr, "\n=== S=1 workaround: pad to S=2, use first column ===\n");
        {
            int S = 2;
            float xf[IC * 2];
            for (int i = 0; i < IC; i++) { xf[i*2] = 1.0f; xf[i*2+1] = 0.0f; }
            float yf[OC * 2];

            NSString *mil = make_mil(IC, OC, S);
            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            size_t inSz[1]  = { (size_t)IC * S * sizeof(float) };
            size_t outSz[1] = { (size_t)OC * S * sizeof(float) };

            ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
            if (k) {
                ane_write_input(k, 0, xf, inSz[0]);
                ane_eval(k);
                ane_read_output(k, 0, yf, outSz[0]);

                float cpu_ref = 0;
                for (int ic = 0; ic < IC; ic++)
                    cpu_ref += (float)wfp16[ic] * 1.0f;

                fprintf(stderr, "y[0]=%.4f ref=%.4f diff=%.4f %s\n",
                        yf[0], cpu_ref, fabsf(yf[0] - cpu_ref),
                        fabsf(yf[0] - cpu_ref) < 1.0f ? "OK" : "BAD");
                ane_free(k);
            }
        }

        free(wfp16);
        fprintf(stderr, "\n=== Done ===\n");
    }
    return 0;
}
