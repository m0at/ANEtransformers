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

// fp32 I/O with casts
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

// Native fp16 I/O (no casts)
static NSString *make_mil_fp16(int ic, int oc, int s) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"
        "    } -> (y);\n"
        "}\n",
        ic, s, oc, ic, oc, ic, oc, s];
}

typedef struct { int ic, oc; const char *name; } Shape;

int main(void) {
    @autoreleasepool {
        ane_init();

        Shape shapes[] = {
            {4096, 4096,  "4096x4096"},
            {4096, 14336, "4096x14336"},
            {14336, 4096, "14336x4096"},
        };
        int nshapes = sizeof(shapes)/sizeof(shapes[0]);
        int svals[] = {16, 32, 64, 128, 256, 512};
        int nsvals = sizeof(svals)/sizeof(svals[0]);

        fprintf(stderr, "=== ANE Throughput Sweep ===\n");
        fprintf(stderr, "%-16s %5s  %8s %8s %8s %8s\n",
                "Shape", "S", "fp32 ms", "TFLOPS", "fp16 ms", "TFLOPS");
        fprintf(stderr, "--------------------------------------------------------------\n");

        for (int si = 0; si < nshapes; si++) {
            int IC = shapes[si].ic, OC = shapes[si].oc;

            _Float16 *wfp16 = malloc(OC * IC * sizeof(_Float16));
            srand(42);
            for (int i = 0; i < OC * IC; i++)
                wfp16[i] = (_Float16)(0.001f * (rand() % 1000));
            NSData *wdata = build_weight_blob(wfp16, OC, IC);

            for (int sj = 0; sj < nsvals; sj++) {
                int S = svals[sj];
                double flops = 2.0 * OC * IC * S;
                int nruns = S <= 64 ? 100 : 30;

                // fp32 version
                double fp32_best = -1;
                {
                    NSString *mil = make_mil_fp32(IC, OC, S);
                    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
                    size_t inSz[1]  = { (size_t)IC * S * sizeof(float) };
                    size_t outSz[1] = { (size_t)OC * S * sizeof(float) };
                    ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
                    if (k) {
                        float *xf = calloc(IC * S, sizeof(float));
                        for (int i = 0; i < IC * S; i++) xf[i] = 0.01f;
                        for (int i = 0; i < 5; i++) { ane_write_input(k, 0, xf, inSz[0]); ane_eval(k); }
                        fp32_best = 1e9;
                        for (int i = 0; i < nruns; i++) {
                            ane_write_input(k, 0, xf, inSz[0]);
                            double t0 = now_ms();
                            ane_eval(k);
                            double dt = now_ms() - t0;
                            if (dt < fp32_best) fp32_best = dt;
                        }
                        free(xf);
                        ane_free(k);
                    }
                }

                // fp16 version
                double fp16_best = -1;
                {
                    NSString *mil = make_mil_fp16(IC, OC, S);
                    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
                    size_t inSz[1]  = { (size_t)IC * S * sizeof(_Float16) };
                    size_t outSz[1] = { (size_t)OC * S * sizeof(_Float16) };
                    ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
                    if (k) {
                        _Float16 *xh = calloc(IC * S, sizeof(_Float16));
                        for (int i = 0; i < IC * S; i++) xh[i] = (_Float16)0.01f;
                        for (int i = 0; i < 5; i++) { ane_write_input(k, 0, xh, inSz[0]); ane_eval(k); }
                        fp16_best = 1e9;
                        for (int i = 0; i < nruns; i++) {
                            ane_write_input(k, 0, xh, inSz[0]);
                            double t0 = now_ms();
                            ane_eval(k);
                            double dt = now_ms() - t0;
                            if (dt < fp16_best) fp16_best = dt;
                        }
                        free(xh);
                        ane_free(k);
                    }
                }

                fprintf(stderr, "%-16s %5d  ", shapes[si].name, S);
                if (fp32_best > 0)
                    fprintf(stderr, "%8.3f %7.2fT  ", fp32_best, flops / (fp32_best * 1e-3) / 1e12);
                else
                    fprintf(stderr, "%8s %8s  ", "FAIL", "-");
                if (fp16_best > 0)
                    fprintf(stderr, "%8.3f %7.2fT", fp16_best, flops / (fp16_best * 1e-3) / 1e12);
                else
                    fprintf(stderr, "%8s %8s", "FAIL", "-");
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
            free(wfp16);
        }

        fprintf(stderr, "=== Done ===\n");
    }
    return 0;
}
