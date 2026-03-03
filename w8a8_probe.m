// w8a8_probe.m — Probe ANE W8A8 (int8 weight + int8 activation) compute
// Tests runtime quantize/dequantize ops (NOT constexpr) which enable native
// int8-int8 MAC on A17 Pro / M4 / M5 Neural Engine.
//
// Build: xcrun clang -O2 -fobjc-arc -o w8a8_probe w8a8_probe.m \
//            -framework Foundation -framework IOSurface -ldl
// Run:   ./w8a8_probe
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

// ─── ANE runtime ───

static Class g_Desc, g_IMM, g_Req, g_AIO;
static bool g_loaded = false;

static void ane_init(void) {
    if (g_loaded) return;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_IMM  = NSClassFromString(@"_ANEInMemoryModel");
    g_Req  = NSClassFromString(@"_ANERequest");
    g_AIO  = NSClassFromString(@"_ANEIOSurfaceObject");
    g_loaded = true;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

static mach_timebase_info_data_t g_tb;
static double ticks_to_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// ─── Weight blob builder (fp16 for baked weights) ───

static NSData *build_fp16_blob(int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8) = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)oc*ic; i++)
        fp16[i] = (_Float16)(0.01f * ((float)(i % 200) - 100.0f));
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ─── MIL header ───

static NSString *MIL_HEADER =
    @"program(1.3)\n"
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]\n{\n";

// ─── Test 0: FP16 baseline (same as quant_probe) ───

static NSString *gen_fp16_conv(int oc, int ic, int sp) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER, ic, sp, ic, sp,
        oc, ic, oc, ic,
        oc, sp, oc, sp];
}

// ─── Test 1: W8A8 via runtime quantize → int8 conv → dequantize ───
// This is the path that gives native int8 compute on A17 Pro / M4 / M5.
// The runtime quantize/dequantize ops are NOT constexpr — they run at inference time.
// Input fp32 → cast fp16 → quantize int8 → conv(int8 weights, int8 activations) → dequantize → fp16 → cast fp32

static NSString *gen_w8a8_conv(int oc, int ic, int sp) {
    // Per-tensor quantization (ANE requires per-channel or per-tensor, NOT per-block)
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        // Quantize activation to int8 (per-tensor)
        "        fp16 q_scale = const()[name = string(\"q_scale\"), val = fp16(0.01)];\n"
        "        int8 q_zp = const()[name = string(\"q_zp\"), val = int8(0)];\n"
        "        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
        "        int32 q_axis = const()[name = string(\"q_axis\"), val = int32(-1)];\n"
        "        tensor<int8, [1, %d, 1, %d]> x_q = quantize(input = x16, scale = q_scale, "
        "zero_point = q_zp, output_dtype = q_dtype, axis = q_axis)[name = string(\"quant_x\")];\n"
        // Conv with int8 weights (baked) and int8 activations
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x_q)[name = string(\"conv\")];\n"
        // Output
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER,
        ic, sp, ic, sp,
        ic, sp,
        oc, ic, oc, ic,
        oc, sp,
        oc, sp];
}

// ─── Test 2: W8A8 with per-channel scale (more realistic) ───

static NSString *gen_w8a8_perchannel(int oc, int ic, int sp) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        // Per-channel quantize on channel dim (axis=1 for [1,C,1,S])
        "        tensor<fp16, [1, %d, 1, 1]> q_scale = const()[name = string(\"q_scale\"), "
        "val = tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<int8, [1, %d, 1, 1]> q_zp = const()[name = string(\"q_zp\"), "
        "val = tensor<int8, [1, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
        "        int32 q_axis = const()[name = string(\"q_axis\"), val = int32(1)];\n"
        "        tensor<int8, [1, %d, 1, %d]> x_q = quantize(input = x16, scale = q_scale, "
        "zero_point = q_zp, output_dtype = q_dtype, axis = q_axis)[name = string(\"quant_x\")];\n"
        // Conv with fp16 weights, int8 activations
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x_q)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER,
        ic, sp, ic, sp,
        ic, ic,
        ic, ic, (unsigned long long)(64 + ic * 2 + 64),
        ic, sp,
        oc, ic, oc, ic, (unsigned long long)(64 + ic * 2 + 64 + ic + 64),
        oc, sp,
        oc, sp];
}

// ─── Test 3: Quantize → matmul (not conv) — alternative path ───

static NSString *gen_w8a8_matmul(int oc, int ic, int sp) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        fp16 q_scale = const()[name = string(\"q_scale\"), val = fp16(0.01)];\n"
        "        int8 q_zp = const()[name = string(\"q_zp\"), val = int8(0)];\n"
        "        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
        "        int32 q_axis = const()[name = string(\"q_axis\"), val = int32(-1)];\n"
        "        tensor<int8, [1, %d, %d]> x_q = quantize(input = x16, scale = q_scale, "
        "zero_point = q_zp, output_dtype = q_dtype, axis = q_axis)[name = string(\"quant_x\")];\n"
        // Dequantize back to fp16 to feed matmul (tests if quantize→dequantize fuses to int8 path)
        "        fp16 dq_scale = const()[name = string(\"dq_scale\"), val = fp16(0.01)];\n"
        "        int8 dq_zp = const()[name = string(\"dq_zp\"), val = int8(0)];\n"
        "        tensor<fp16, [1, %d, %d]> x_dq = dequantize(input = x_q, scale = dq_scale, "
        "zero_point = dq_zp, axis = q_axis)[name = string(\"dequant_x\")];\n"
        // Weight matrix as fp16 input
        "        tensor<fp16, [1, %d, %d]> W16 = const()[name = string(\"W16\"), "
        "val = tensor<fp16, [1, %d, %d]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
        "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
        "        tensor<fp16, [1, %d, %d]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x_dq)[name = string(\"mm\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER,
        ic, sp, ic, sp,
        ic, sp,
        ic, sp,
        oc, ic, oc, ic,
        oc, sp,
        oc, sp];
}

// ─── Test 4: Direct int8 typed conv weight (raw, no constexpr) ───

static NSString *gen_int8_weight_conv(int oc, int ic, int sp) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        // Raw int8 weight constant — will ANE accept this?
        "        tensor<int8, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER,
        ic, sp, ic, sp,
        oc, ic, oc, ic,
        oc, sp, oc, sp];
}

// ─── Test 5: ios17 target (quantize was introduced in ios17) ───

static NSString *gen_w8a8_ios17(int oc, int ic, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios17>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        fp16 q_scale = const()[name = string(\"q_scale\"), val = fp16(0.01)];\n"
        "        int8 q_zp = const()[name = string(\"q_zp\"), val = int8(0)];\n"
        "        string q_dtype = const()[name = string(\"q_dtype\"), val = string(\"int8\")];\n"
        "        int32 q_axis = const()[name = string(\"q_axis\"), val = int32(-1)];\n"
        "        tensor<int8, [1, %d, 1, %d]> x_q = quantize(input = x16, scale = q_scale, "
        "zero_point = q_zp, output_dtype = q_dtype, axis = q_axis)[name = string(\"quant_x\")];\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x_q)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        ic, sp, ic, sp,
        ic, sp,
        oc, ic, oc, ic,
        oc, sp,
        oc, sp];
}

// ─── Benchmark runner (same as quant_probe) ───

typedef struct {
    const char *name;
    double ms;
    double tflops;
    bool compiled;
    bool ran;
} TestResult;

static TestResult run_test(const char *name, NSString *mil, NSData *weightData,
                            int ic, int oc, int sp, bool is3d) {
    TestResult r = {name, 0, 0, false, false};
    ane_init();

    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);
    if (!desc) { printf("  %-32s  DESC FAIL\n", name); return r; }

    id model = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) { printf("  %-32s  MODEL FAIL\n", name); return r; }

    id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  %-32s  COMPILE FAIL: %s\n", name,
               e ? [[e localizedDescription] UTF8String] : "unknown");
        [fm removeItemAtPath:tmpDir error:nil];
        return r;
    }
    r.compiled = true;

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  %-32s  LOAD FAIL\n", name);
        [fm removeItemAtPath:tmpDir error:nil];
        return r;
    }

    size_t inBytes = is3d ? ((size_t)ic * sp * 4) : ((size_t)ic * sp * 4);
    size_t outBytes = is3d ? ((size_t)oc * sp * 4) : ((size_t)oc * sp * 4);
    IOSurfaceRef ioIn = make_surface(inBytes);
    IOSurfaceRef ioOut = make_surface(outBytes);

    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_Req,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    IOSurfaceLock(ioIn, 0, NULL);
    float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
    for (size_t i = 0; i < (size_t)ic*sp; i++) inp[i] = 0.01f;
    IOSurfaceUnlock(ioIn, 0, NULL);

    for (int i = 0; i < 5; i++) {
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) {
            printf("  %-32s  EVAL FAIL: %s\n", name,
                   e ? [[e localizedDescription] UTF8String] : "unknown");
            CFRelease(ioIn); CFRelease(ioOut);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:tmpDir error:nil];
            return r;
        }
    }
    r.ran = true;

    int iters = 100;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    double ms = ticks_to_ms(mach_absolute_time() - t0) / iters;

    double gflops = 2.0 * oc * ic * sp / 1e9;
    r.ms = ms;
    r.tflops = gflops / ms;

    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(ioIn);
    CFRelease(ioOut);
    [fm removeItemAtPath:tmpDir error:nil];

    printf("  %-32s  %8.3f ms  %7.2f TFLOPS\n", name, ms, r.tflops);
    return r;
}

// ─── Build combined weight blob for per-channel test ───

static NSData *build_perchannel_blob(int oc, int ic) {
    // Layout: [64 global] [64 chunk + ic*2 scale fp16] [64 chunk + ic zeropoint int8] [64 chunk + oc*ic*2 weight fp16]
    NSUInteger scale_size = (NSUInteger)ic * 2;
    NSUInteger zp_size = (NSUInteger)ic;
    NSUInteger w_size = (NSUInteger)oc * ic * 2;

    NSUInteger total = 64 + (64 + scale_size) + (64 + zp_size) + (64 + w_size);
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;

    // Chunk 1: scale [ic] fp16
    NSUInteger pos = 64;
    uint8_t *c1 = buf + pos;
    c1[0]=0xEF; c1[1]=0xBE; c1[2]=0xAD; c1[3]=0xDE; c1[4]=0x01;
    *(uint32_t*)(c1+8) = (uint32_t)scale_size;
    *(uint32_t*)(c1+16) = (uint32_t)(pos + 64);
    _Float16 *scales = (_Float16*)(buf + pos + 64);
    for (int i = 0; i < ic; i++) scales[i] = (_Float16)0.01f;
    pos += 64 + scale_size;

    // Chunk 2: zeropoint [ic] int8
    uint8_t *c2 = buf + pos;
    c2[0]=0xEF; c2[1]=0xBE; c2[2]=0xAD; c2[3]=0xDE; c2[4]=0x01;
    *(uint32_t*)(c2+8) = (uint32_t)zp_size;
    *(uint32_t*)(c2+16) = (uint32_t)(pos + 64);
    // zeropoints = 0 (already calloc'd)
    pos += 64 + zp_size;

    // Chunk 3: weight [oc, ic] fp16
    uint8_t *c3 = buf + pos;
    c3[0]=0xEF; c3[1]=0xBE; c3[2]=0xAD; c3[3]=0xDE; c3[4]=0x01;
    *(uint32_t*)(c3+8) = (uint32_t)w_size;
    *(uint32_t*)(c3+16) = (uint32_t)(pos + 64);
    _Float16 *wfp = (_Float16*)(buf + pos + 64);
    for (NSUInteger i = 0; i < (NSUInteger)oc*ic; i++)
        wfp[i] = (_Float16)(0.01f * ((float)(i % 200) - 100.0f));

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ─── Main ───

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();

        if (!g_Desc || !g_IMM) {
            fprintf(stderr, "ERROR: Cannot load AppleNeuralEngine.framework\n");
            return 1;
        }

        printf("=== ANE W8A8 Int8 Compute Probe ===\n");
        printf("Tests runtime quantize/dequantize ops for native int8 MAC on M5 ANE\n\n");

        int configs[][3] = {
            {768,  768,  64},
            {2048, 768,  64},
            {1024, 1024, 64},
        };
        int nconfigs = sizeof(configs) / sizeof(configs[0]);

        for (int ci = 0; ci < nconfigs; ci++) {
            int oc = configs[ci][0], ic = configs[ci][1], sp = configs[ci][2];
            printf("\n── Config: oc=%d, ic=%d, sp=%d ──\n", oc, ic, sp);

            NSData *fp16blob = build_fp16_blob(oc, ic);

            // Test 0: FP16 baseline
            run_test("FP16 baseline (conv)", gen_fp16_conv(oc, ic, sp),
                     fp16blob, ic, oc, sp, false);

            // Test 1: W8A8 per-tensor quantize → conv
            run_test("W8A8 per-tensor (conv)", gen_w8a8_conv(oc, ic, sp),
                     fp16blob, ic, oc, sp, false);

            // Test 2: W8A8 per-channel quantize → conv
            NSData *pcblob = build_perchannel_blob(oc, ic);
            run_test("W8A8 per-channel (conv)", gen_w8a8_perchannel(oc, ic, sp),
                     pcblob, ic, oc, sp, false);

            // Test 3: quantize → dequantize → matmul (3D tensors)
            run_test("W8A8 quant→dequant→matmul", gen_w8a8_matmul(oc, ic, sp),
                     fp16blob, ic, oc, sp, true);

            // Test 4: Raw int8 weight → conv (type mismatch probe)
            NSData *int8blob = build_fp16_blob(oc, ic); // reuse, ANE reads bytes
            run_test("Raw int8 weight (conv)", gen_int8_weight_conv(oc, ic, sp),
                     int8blob, ic, oc, sp, false);

            // Test 5: ios17 target (quantize introduced in ios17)
            run_test("W8A8 ios17 target (conv)", gen_w8a8_ios17(oc, ic, sp),
                     fp16blob, ic, oc, sp, false);
        }

        printf("\n=== Interpretation ===\n");
        printf("If W8A8 TFLOPS > FP16 baseline  → native int8 compute on ANE!\n");
        printf("If W8A8 TFLOPS ≈ FP16           → quantize/dequant adds overhead, no int8 gain\n");
        printf("COMPILE FAIL                     → MIL op not supported by ANE compiler\n");
        printf("EVAL FAIL                        → compiles but hardware rejects\n");
        printf("\nNote: W8A8 int8 compute is documented for A17 Pro / M4 / M5.\n");
        printf("The question is whether it works through _ANEInMemoryModel.\n");

        return 0;
    }
}
