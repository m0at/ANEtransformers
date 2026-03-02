// test_lm_head_ane.m — LM head conv on ANE vs cblas_sgemv
// Input:  [1, 768,   1, 1]
// Weight: [50257, 768, 1, 1]
// Output: [1, 50257, 1, 1]
//
// Build:
//   xcrun clang -O2 -fobjc-arc -o test_lm_head_ane test_lm_head_ane.m \
//     -framework Foundation -framework CoreML -framework IOSurface \
//     -framework Accelerate -ldl

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#define DIM   768
#define VOCAB 50257
#define N_RUNS 50

// --- ANE boilerplate (mirrors gpt2.m) ---
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I   = NSClassFromString(@"_ANEInMemoryModel");
    g_AR  = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
    mach_timebase_info(&g_tb);
}

static double tb_ms(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:         @(bytes),
        (id)kIOSurfaceHeight:        @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow:   @(bytes),
        (id)kIOSurfaceAllocSize:     @(bytes),
        (id)kIOSurfacePixelFormat:   @0
    });
}

typedef struct { id model; NSString *td; } Kern;

static Kern compile_mil(NSString *mil, NSDictionary *wd) {
    Kern k = {nil, nil};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_D, @selector(modelWithMILText:weights:optionsPlist:),
        md, wd ?: @{}, nil);
    if (!desc) { printf("  desc=NULL\n"); return k; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];

    [[NSFileManager defaultManager]
        createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    for (NSString *path in wd) {
        NSString *rel  = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        NSString *full = [td stringByAppendingPathComponent:rel];
        [[NSFileManager defaultManager]
            createDirectoryAtPath:[full stringByDeletingLastPathComponent]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [wd[path][@"data"] writeToFile:full atomically:YES];
    }

    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        NSString *msg = e ? [e localizedDescription] : @"(nil)";
        int cap = (int)[msg length]; if (cap > 400) cap = 400;
        printf("  compile FAIL: %s\n", [[msg substringToIndex:cap] UTF8String]);
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return k;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  load FAIL\n");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return k;
    }
    k.model = mdl; k.td = td;
    return k;
}

static BOOL ane_eval(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
    NSMutableArray *inArr  = [NSMutableArray array], *inIdx  = [NSMutableArray array];
    NSMutableArray *outArr = [NSMutableArray array], *outIdx = [NSMutableArray array];
    for (int i = 0; i < nin;  i++) {
        [inArr  addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ins[i])];
        [inIdx  addObject:@(i)];
    }
    for (int i = 0; i < nout; i++) {
        [outArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), outs[i])];
        [outIdx addObject:@(i)];
    }
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        inArr, inIdx, outArr, outIdx, nil, nil, @0);
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok && e) {
        NSString *desc = [e localizedDescription];
        int cap = (int)[desc length]; if (cap > 600) cap = 600;
        printf("  eval error: %s\n", [[desc substringToIndex:cap] UTF8String]);
    }
    return ok;
}

static void cleanup_kern(Kern *k) {
    if (!k->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->td error:nil];
    k->model = nil;
}

// --- Weight blob builder ---
// Layout: 128-byte header + fp16 data
//   byte 0:    version = 1
//   byte 4:    type    = 2 (fp16)
//   byte 64-67: DEADBEEF magic (LE: EF BE AD DE)
//   byte 68:   flag    = 1
//   byte 72:   data_size (uint32)
//   byte 80:   data_offset = 128 (uint32)
static NSData *make_weight_blob(_Float16 *data, int count) {
    size_t data_bytes = (size_t)count * 2;
    size_t total      = 128 + data_bytes;
    uint8_t *buf      = (uint8_t *)calloc(total, 1);
    buf[0]  = 1;
    buf[4]  = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)data_bytes;
    *(uint32_t *)(buf + 80) = 128;
    memcpy(buf + 128, data, data_bytes);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// --- IOSurface sizing for ANE ---
// ane_surf(channels, spatial) = max(49152, ((channels+63)/64)*64 * spatial * 2)
static size_t ane_surf_size(int channels, int spatial) {
    size_t aligned = (((size_t)channels + 63) / 64) * 64 * (size_t)spatial * 2;
    return aligned > 49152 ? aligned : 49152;
}

// --- MIL program ---
// Single conv: input [1,768,1,1] -> weight [50257,768,1,1] -> output [1,50257,1,1]
// fp16 throughout; no bias.
static NSString *gen_lm_head_mil(void) {
    NSMutableString *m = [NSMutableString string];
    // Header
    [m appendString:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> x) {\n", DIM];
    // Conv constants
    [m appendString:
        @"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    // Weight constant — loaded from blob file
    [m appendFormat:
        @"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
         "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE("
         "path=string(\"@model_path/weights/lm_w.bin\"), offset=uint64(64)))];\n",
         VOCAB, DIM, VOCAB, DIM];
    // Conv
    [m appendFormat:
        @"        tensor<fp16, [1,%d,1,1]> out = conv("
         "dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,"
         "weight=W,x=x)[name=string(\"lm_conv\")];\n", VOCAB];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

int main(void) {
    @autoreleasepool {
        ane_init();
        printf("=== LM Head ANE vs BLAS benchmark ===\n");
        printf("Op: [%d,%d] x [%d] -> [%d]  (fp16 weights, fp32 BLAS)\n\n",
               VOCAB, DIM, DIM, VOCAB);

        // --- Generate random fp32 input and fp16 weight matrix ---
        printf("Generating random weights (%zu MB fp16)...\n",
               (size_t)VOCAB * DIM * 2 / (1024*1024));
        float    *input_f32 = (float *)    malloc((size_t)DIM   * sizeof(float));
        _Float16 *weight_h  = (_Float16 *) malloc((size_t)VOCAB * DIM * sizeof(_Float16));
        float    *weight_f32= (float *)    malloc((size_t)VOCAB * DIM * sizeof(float));

        srand48(12345);
        for (int i = 0; i < DIM;       i++) input_f32[i]  = (float)(drand48() - 0.5);
        for (int i = 0; i < VOCAB*DIM; i++) {
            float v       = (float)(drand48() - 0.5) * 0.02f;
            weight_h[i]   = (_Float16)v;
            weight_f32[i] = (float)weight_h[i];  // use fp16-rounded values for fair comparison
        }

        // --- BLAS reference ---
        float *blas_out = (float *)malloc((size_t)VOCAB * sizeof(float));
        printf("Running BLAS warmup...\n");
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    VOCAB, DIM, 1.0f,
                    weight_f32, DIM,
                    input_f32, 1,
                    0.0f, blas_out, 1);

        printf("Timing BLAS (%d runs)...\n", N_RUNS);
        uint64_t t0 = mach_absolute_time();
        for (int r = 0; r < N_RUNS; r++) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        VOCAB, DIM, 1.0f,
                        weight_f32, DIM,
                        input_f32, 1,
                        0.0f, blas_out, 1);
        }
        double blas_ms = tb_ms(mach_absolute_time() - t0) / N_RUNS;
        printf("  BLAS avg: %.3f ms\n\n", blas_ms);

        // --- Build weight blob ---
        printf("Building weight blob (%zu bytes)...\n",
               (size_t)128 + (size_t)VOCAB * DIM * 2);
        NSData *weight_blob = make_weight_blob(weight_h, VOCAB * DIM);

        // --- Build MIL and compile ---
        NSString *mil = gen_lm_head_mil();
        printf("MIL program (first 300 chars):\n%.300s\n...\n\n",
               [mil UTF8String]);

        NSDictionary *wd = @{
            @"@model_path/weights/lm_w.bin": @{@"offset": @0, @"data": weight_blob},
        };

        printf("Compiling LM head conv for ANE...\n");
        uint64_t ct0 = mach_absolute_time();
        Kern kern = compile_mil(mil, wd);
        double compile_ms = tb_ms(mach_absolute_time() - ct0);

        if (!kern.model) {
            printf("\nANE COMPILE FAILED — likely weight tensor too large for ANE.\n");
            printf("Weight: %d x %d x fp16 = %.1f MB\n",
                   VOCAB, DIM, (double)VOCAB*DIM*2/1024/1024);
            printf("BLAS baseline: %.3f ms\n", blas_ms);
            free(input_f32); free(weight_h); free(weight_f32); free(blas_out);
            return 1;
        }
        printf("  Compiled in %.0f ms\n\n", compile_ms);

        // --- Allocate IOSurfaces ---
        // Input:  channels=DIM,   spatial=1
        // Output: channels=VOCAB, spatial=1
        size_t in_sz  = ane_surf_size(DIM,   1);
        size_t out_sz = ane_surf_size(VOCAB, 1);
        printf("IOSurface sizes: in=%zu bytes, out=%zu bytes\n", in_sz, out_sz);

        IOSurfaceRef surf_in  = make_surface(in_sz);
        IOSurfaceRef surf_out = make_surface(out_sz);
        if (!surf_in || !surf_out) {
            printf("IOSurface alloc FAILED\n");
            cleanup_kern(&kern);
            return 1;
        }

        // Fill input surface: channel-first fp16, shape [1, DIM, 1, 1]
        IOSurfaceLock(surf_in, 0, NULL);
        _Float16 *pin = (_Float16 *)IOSurfaceGetBaseAddress(surf_in);
        memset(pin, 0, in_sz);
        for (int c = 0; c < DIM; c++) pin[c] = (_Float16)input_f32[c];
        IOSurfaceUnlock(surf_in, 0, NULL);

        IOSurfaceRef ins[]  = {surf_in};
        IOSurfaceRef outs[] = {surf_out};

        // --- ANE warmup ---
        printf("ANE warmup run...\n");
        BOOL ok = ane_eval(&kern, ins, 1, outs, 1);
        if (!ok) {
            printf("ANE eval FAILED on warmup\n");
            cleanup_kern(&kern);
            CFRelease(surf_in); CFRelease(surf_out);
            free(input_f32); free(weight_h); free(weight_f32); free(blas_out);
            return 1;
        }
        printf("  ANE warmup OK\n");

        // --- Correctness check ---
        // Read ANE output, compare to BLAS
        IOSurfaceLock(surf_out, kIOSurfaceLockReadOnly, NULL);
        _Float16 *pout = (_Float16 *)IOSurfaceGetBaseAddress(surf_out);
        float max_abs_err = 0.0f, max_blas_val = 0.0f;
        int   worst_idx   = 0;
        float ane_worst   = 0.0f, blas_worst = 0.0f;
        // ANE output layout: channel-first [1, VOCAB, 1, 1]
        // The ANE pads channels to multiples of 64, but the first VOCAB values
        // correspond to logit[0..VOCAB-1].
        // With spatial=1, pout[v] = logit[v].
        for (int v = 0; v < VOCAB; v++) {
            float ane_v  = (float)pout[v];
            float blas_v = blas_out[v];
            float err    = fabsf(ane_v - blas_v);
            if (err > max_abs_err) {
                max_abs_err = err; worst_idx = v;
                ane_worst = ane_v; blas_worst = blas_v;
            }
            if (fabsf(blas_v) > max_blas_val) max_blas_val = fabsf(blas_v);
        }
        IOSurfaceUnlock(surf_out, kIOSurfaceLockReadOnly, NULL);

        printf("\nCorrectness:\n");
        printf("  Max |BLAS| value:    %.6f\n", max_blas_val);
        printf("  Max abs error:       %.6f  (at vocab idx %d)\n", max_abs_err, worst_idx);
        printf("  ANE[worst]:          %.6f\n", ane_worst);
        printf("  BLAS[worst]:         %.6f\n", blas_worst);
        float rel_err = max_blas_val > 0 ? max_abs_err / max_blas_val : 0;
        printf("  Relative error:      %.4f%%\n", rel_err * 100.0f);
        printf("  %s\n\n", rel_err < 0.01f ? "PASS (rel err < 1%)" : "WARN (rel err >= 1%)");

        // --- ANE timing ---
        printf("Timing ANE (%d runs)...\n", N_RUNS);
        t0 = mach_absolute_time();
        for (int r = 0; r < N_RUNS; r++) {
            ane_eval(&kern, ins, 1, outs, 1);
        }
        double ane_ms = tb_ms(mach_absolute_time() - t0) / N_RUNS;
        printf("  ANE avg: %.3f ms\n\n", ane_ms);

        // --- Summary ---
        printf("=== Summary ===\n");
        printf("  BLAS (fp32, sgemv):  %.3f ms\n", blas_ms);
        printf("  ANE  (fp16 conv):    %.3f ms\n", ane_ms);
        if (ane_ms < blas_ms)
            printf("  Speedup: %.2fx  (ANE wins)\n", blas_ms / ane_ms);
        else
            printf("  Slowdown: %.2fx  (BLAS wins)\n", ane_ms / blas_ms);
        printf("  Weight: %d x %d x fp16 = %.1f MB\n",
               VOCAB, DIM, (double)VOCAB*DIM*2/1024/1024);

        // Cleanup
        cleanup_kern(&kern);
        CFRelease(surf_in);
        CFRelease(surf_out);
        free(input_f32); free(weight_h); free(weight_f32); free(blas_out);
    }
    return 0;
}
