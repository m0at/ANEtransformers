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

int main(void) {
    @autoreleasepool {
        ane_init();
        int IC = 256, OC = 256;

        _Float16 *wfp16 = malloc(OC * IC * sizeof(_Float16));
        srand(42);
        for (int i = 0; i < OC * IC; i++)
            wfp16[i] = (_Float16)(0.01f * ((rand() % 200) - 100));
        NSData *wdata = build_weight_blob(wfp16, OC, IC);

        fprintf(stderr, "=== fp16 native I/O: S sweep with correctness check ===\n");
        int svals[] = {16, 32, 64, 128};

        for (int si = 0; si < 4; si++) {
            int S = svals[si];
            _Float16 *xh = malloc(IC * S * sizeof(_Float16));
            for (int i = 0; i < IC * S; i++) xh[i] = (_Float16)1.0f;

            NSString *mil = make_mil_fp16(IC, OC, S);
            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            size_t inSz[1]  = { (size_t)IC * S * sizeof(_Float16) };
            size_t outSz[1] = { (size_t)OC * S * sizeof(_Float16) };

            ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
            if (!k) { fprintf(stderr, "S=%d: COMPILE FAIL\n", S); free(xh); continue; }

            _Float16 *yh = calloc(OC * S, sizeof(_Float16));
            ane_write_input(k, 0, xh, inSz[0]);
            ane_eval(k);
            ane_read_output(k, 0, yh, outSz[0]);

            // CPU ref
            float cpu_ref = 0;
            for (int ic = 0; ic < IC; ic++) cpu_ref += (float)wfp16[ic];

            float ane_val = (float)yh[0];
            float diff = fabsf(ane_val - cpu_ref);
            bool nonzero = false;
            for (int i = 0; i < OC * S; i++) if ((float)yh[i] != 0.0f) { nonzero = true; break; }

            fprintf(stderr, "S=%3d: y[0]=%.4f ref=%.4f diff=%.4f nonzero=%s %s\n",
                    S, ane_val, cpu_ref, diff, nonzero ? "yes" : "NO",
                    (diff < 1.0f && nonzero) ? "PASS" : "FAIL");

            ane_free(k);
            free(xh); free(yh);
        }

        // Also test: fp16 input with fp32 output (cast only on output)
        fprintf(stderr, "\n=== fp16 in, fp32 out (cast only on output) ===\n");
        for (int si = 0; si < 4; si++) {
            int S = svals[si];
            int IC2 = 256, OC2 = 256;

            NSString *mil = [NSString stringWithFormat:
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
                "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
                "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"
                "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
                "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
                "    } -> (y);\n"
                "}\n",
                IC2, S, OC2, IC2, OC2, IC2, OC2, S, OC2, S];

            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            size_t inSz[1]  = { (size_t)IC2 * S * sizeof(_Float16) };
            size_t outSz[1] = { (size_t)OC2 * S * sizeof(float) };

            ANEKernel *k = ane_compile(md, wdata, 1, inSz, 1, outSz);
            if (!k) { fprintf(stderr, "S=%d: COMPILE FAIL\n", S); continue; }

            _Float16 *xh = malloc(IC2 * S * sizeof(_Float16));
            for (int i = 0; i < IC2 * S; i++) xh[i] = (_Float16)1.0f;
            float *yf = calloc(OC2 * S, sizeof(float));

            ane_write_input(k, 0, xh, inSz[0]);
            ane_eval(k);
            ane_read_output(k, 0, yf, outSz[0]);

            float cpu_ref = 0;
            for (int ic = 0; ic < IC2; ic++) cpu_ref += (float)wfp16[ic];

            float diff = fabsf(yf[0] - cpu_ref);
            bool nonzero = false;
            for (int i = 0; i < OC2 * S; i++) if (yf[i] != 0.0f) { nonzero = true; break; }

            fprintf(stderr, "S=%3d: y[0]=%.4f ref=%.4f diff=%.4f nonzero=%s %s\n",
                    S, yf[0], cpu_ref, diff, nonzero ? "yes" : "NO",
                    (diff < 1.0f && nonzero) ? "PASS" : "FAIL");

            ane_free(k);
            free(xh); free(yf);
        }

        free(wfp16);
    }
    return 0;
}
