// Test: Multi-input MIL programs with different spatial dimensions
// Validates whether ANE supports mixed-size inputs (critical for KV-cache attention)
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static IOSurfaceRef make_surface(size_t bytes) {
    // ANE requires page-aligned surfaces (16384 min)
    size_t alloc = (bytes + 16383) & ~(size_t)16383;
    if (alloc < 16384) alloc = 16384;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(alloc), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(alloc),
        (id)kIOSurfaceAllocSize:@(alloc), (id)kIOSurfacePixelFormat:@0});
}

typedef struct { id model; NSString *td; } Kern;
static Kern compile_mil(NSString *mil, NSDictionary *wd) {
    Kern k = {nil, nil};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, wd ?: @{}, nil);
    if (!desc) { printf("  desc=NULL\n"); return k; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in wd) {
        [wd[path][@"data"] writeToFile:[td stringByAppendingPathComponent:
            [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""]] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  compile FAIL: %s\n", e?[[[e localizedDescription] substringToIndex:MIN(300,(int)[[e localizedDescription] length])] UTF8String]:"");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return k;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  load FAIL\n"); [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return k;
    }
    k.model = mdl; k.td = td;
    return k;
}
static BOOL ane_eval_io(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
    NSMutableArray *inArr = [NSMutableArray array], *inIdx = [NSMutableArray array];
    NSMutableArray *outArr = [NSMutableArray array], *outIdx = [NSMutableArray array];
    for (int i = 0; i < nin; i++) {
        [inArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ins[i])];
        [inIdx addObject:@(i)];
    }
    for (int i = 0; i < nout; i++) {
        [outArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), outs[i])];
        [outIdx addObject:@(i)];
    }
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        inArr, inIdx, outArr, outIdx, nil, nil, @0);
    NSError *e = nil;
    BOOL ret = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ret && e) printf("  eval error: %s\n", [[[e localizedDescription] substringToIndex:MIN(300,(int)[[e localizedDescription] length])] UTF8String]);
    return ret;
}
static void cleanup_kern(Kern *k) {
    if (!k->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->td error:nil];
    k->model = nil;
}

// Fill IOSurface with fp16 value
static void fill_fp16(IOSurfaceRef s, _Float16 val) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(s);
    size_t n = IOSurfaceGetAllocSize(s) / sizeof(_Float16);
    for (size_t i = 0; i < n; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
}

// Read fp16 values from IOSurface
static void read_fp16(IOSurfaceRef s, _Float16 *dst, size_t count) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    memcpy(dst, IOSurfaceGetBaseAddress(s), count * sizeof(_Float16));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

#define MIL_HDR \
    "program(1.3)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, " \
    "{\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define MIL_END "\n}\n"

// ============================================================
// TEST 1: Two inputs, same spatial dims — add(a, b)
// ============================================================
static int test1_same_dims(void) {
    printf("=== TEST 1: Two inputs, same spatial [1,64,1,32] — add ===\n");

    NSString *mil = @MIL_HDR
        "    func main<ios18>(tensor<fp16, [1, 64, 1, 32]> a, tensor<fp16, [1, 64, 1, 32]> b) {\n"
        "        tensor<fp16, [1, 64, 1, 32]> out = add(x=a, y=b)[name=string(\"add\")];\n"
        "    } -> (out);\n"
    MIL_END;

    Kern k = compile_mil(mil, @{});
    if (!k.model) { printf("  FAIL (compile)\n\n"); return 0; }

    size_t sz = 1 * 64 * 1 * 32 * 2; // fp16
    IOSurfaceRef in0 = make_surface(sz);
    IOSurfaceRef in1 = make_surface(sz);
    IOSurfaceRef out = make_surface(sz);

    fill_fp16(in0, (_Float16)2.0);
    fill_fp16(in1, (_Float16)3.0);
    fill_fp16(out, (_Float16)0.0);

    IOSurfaceRef ins[] = {in0, in1};
    IOSurfaceRef outs[] = {out};
    BOOL ok = ane_eval_io(&k, ins, 2, outs, 1);
    if (!ok) { printf("  FAIL (eval)\n\n"); cleanup_kern(&k); return 0; }

    // Check output
    int n = 64 * 32;
    _Float16 *buf = malloc(n * sizeof(_Float16));
    read_fp16(out, buf, n);
    int pass = 1;
    for (int i = 0; i < n; i++) {
        float v = (float)buf[i];
        if (fabsf(v - 5.0f) > 0.1f) { printf("  MISMATCH [%d]: got %.4f expected 5.0\n", i, v); pass = 0; break; }
    }
    free(buf);

    // Benchmark
    if (pass) {
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 100; i++) ane_eval_io(&k, ins, 2, outs, 1);
        double ms = tb_ms(mach_absolute_time() - t0);
        printf("  100 iters: %.2f ms (%.3f ms/iter)\n", ms, ms / 100.0);
    }

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    cleanup_kern(&k);
    CFRelease(in0); CFRelease(in1); CFRelease(out);
    return pass;
}

// ============================================================
// TEST 2: Three inputs, DIFFERENT spatial dims — KV-cache SDPA
// q:[1,64,1,1], k:[1,64,1,128], v:[1,64,1,128]
// ============================================================
static int test2_diff_dims(void) {
    printf("=== TEST 2: Three inputs, DIFFERENT spatial dims — KV-cache SDPA ===\n");
    printf("  q:[1,64,1,1]  k:[1,64,1,128]  v:[1,64,1,128]\n");

    NSString *mil = @MIL_HDR
        "    func main<ios18>(\n"
        "        tensor<fp16, [1, 64, 1, 1]> q,\n"
        "        tensor<fp16, [1, 64, 1, 128]> k,\n"
        "        tensor<fp16, [1, 64, 1, 128]> v\n"
        "    ) {\n"
        "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1, 4, 16, 1])];\n"
        "        tensor<int32, [4]> ksh = const()[name=string(\"ksh\"), val=tensor<int32, [4]>([1, 4, 16, 128])];\n"
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0, 1, 3, 2])];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 1]> q4 = reshape(shape=qsh, x=q)[name=string(\"rq\")];\n"
        "        tensor<fp16, [1, 4, 1, 16]> qt = transpose(perm=pm, x=q4)[name=string(\"tq\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 128]> k4 = reshape(shape=ksh, x=k)[name=string(\"rk\")];\n"
        "        tensor<fp16, [1, 4, 128, 16]> kt = transpose(perm=pm, x=k4)[name=string(\"tk\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 128]> v4 = reshape(shape=ksh, x=v)[name=string(\"rv\")];\n"
        "        tensor<fp16, [1, 4, 128, 16]> vt = transpose(perm=pm, x=v4)[name=string(\"tv\")];\n"
        "\n"
        "        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"
        "        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"
        "        tensor<fp16, [1, 4, 1, 128]> scores = matmul(transpose_x=tx, transpose_y=ty, x=qt, y=kt)[name=string(\"mm1\")];\n"
        "\n"
        "        fp16 sc = const()[name=string(\"sc\"), val=fp16(0.25)];\n"
        "        tensor<fp16, [1, 4, 1, 128]> scaled = mul(x=scores, y=sc)[name=string(\"scl\")];\n"
        "\n"
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"
        "        tensor<fp16, [1, 4, 1, 128]> attn = softmax(axis=sax, x=scaled)[name=string(\"sm\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 1, 16]> a4 = matmul(transpose_x=tx, transpose_y=tx, x=attn, y=vt)[name=string(\"mm2\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 1]> at = transpose(perm=pm, x=a4)[name=string(\"ta\")];\n"
        "        tensor<int32, [4]> osh = const()[name=string(\"osh\"), val=tensor<int32, [4]>([1, 64, 1, 1])];\n"
        "        tensor<fp16, [1, 64, 1, 1]> out = reshape(shape=osh, x=at)[name=string(\"ra\")];\n"
        "    } -> (out);\n"
    MIL_END;

    Kern k = compile_mil(mil, @{});
    if (!k.model) { printf("  FAIL (compile)\n\n"); return 0; }

    size_t sz_q = 1 * 64 * 1 * 1 * 2;
    size_t sz_kv = 1 * 64 * 1 * 128 * 2;
    size_t sz_out = 1 * 64 * 1 * 1 * 2;

    IOSurfaceRef sq = make_surface(sz_q);
    IOSurfaceRef sk = make_surface(sz_kv);
    IOSurfaceRef sv = make_surface(sz_kv);
    IOSurfaceRef so = make_surface(sz_out);

    fill_fp16(sq, (_Float16)1.0);
    fill_fp16(sk, (_Float16)0.5);
    fill_fp16(sv, (_Float16)0.25);
    fill_fp16(so, (_Float16)0.0);

    IOSurfaceRef ins[] = {sq, sk, sv};
    IOSurfaceRef outs[] = {so};
    BOOL ok = ane_eval_io(&k, ins, 3, outs, 1);
    if (!ok) { printf("  FAIL (eval)\n\n"); cleanup_kern(&k); return 0; }

    // Check output is non-zero
    int n = 64;
    _Float16 *buf = malloc(n * sizeof(_Float16));
    read_fp16(so, buf, n);
    int pass = 1;
    int nonzero = 0;
    printf("  Output sample: ");
    for (int i = 0; i < MIN(8, n); i++) printf("%.4f ", (float)buf[i]);
    printf("...\n");
    for (int i = 0; i < n; i++) if (fabsf((float)buf[i]) > 1e-6f) nonzero++;
    if (nonzero == 0) { printf("  All zeros — eval produced no output\n"); pass = 0; }
    else printf("  %d/%d non-zero outputs\n", nonzero, n);
    free(buf);

    // Benchmark
    if (pass) {
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 100; i++) ane_eval_io(&k, ins, 3, outs, 1);
        double ms = tb_ms(mach_absolute_time() - t0);
        printf("  100 iters: %.2f ms (%.3f ms/iter)\n", ms, ms / 100.0);
    }

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    cleanup_kern(&k);
    CFRelease(sq); CFRelease(sk); CFRelease(sv); CFRelease(so);
    return pass;
}

// ============================================================
// TEST 3: Four inputs — KV-cache SDPA + causal mask
// q:[1,64,1,1], k:[1,64,1,128], v:[1,64,1,128], mask:[1,1,1,128]
// ============================================================
static int test3_with_mask(void) {
    printf("=== TEST 3: Four inputs — KV-cache SDPA + mask ===\n");
    printf("  q:[1,64,1,1]  k:[1,64,1,128]  v:[1,64,1,128]  mask:[1,1,1,128]\n");

    NSString *mil = @MIL_HDR
        "    func main<ios18>(\n"
        "        tensor<fp16, [1, 64, 1, 1]> q,\n"
        "        tensor<fp16, [1, 64, 1, 128]> k,\n"
        "        tensor<fp16, [1, 64, 1, 128]> v,\n"
        "        tensor<fp16, [1, 1, 1, 128]> mask\n"
        "    ) {\n"
        "        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1, 4, 16, 1])];\n"
        "        tensor<int32, [4]> ksh = const()[name=string(\"ksh\"), val=tensor<int32, [4]>([1, 4, 16, 128])];\n"
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0, 1, 3, 2])];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 1]> q4 = reshape(shape=qsh, x=q)[name=string(\"rq\")];\n"
        "        tensor<fp16, [1, 4, 1, 16]> qt = transpose(perm=pm, x=q4)[name=string(\"tq\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 128]> k4 = reshape(shape=ksh, x=k)[name=string(\"rk\")];\n"
        "        tensor<fp16, [1, 4, 128, 16]> kt = transpose(perm=pm, x=k4)[name=string(\"tk\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 128]> v4 = reshape(shape=ksh, x=v)[name=string(\"rv\")];\n"
        "        tensor<fp16, [1, 4, 128, 16]> vt = transpose(perm=pm, x=v4)[name=string(\"tv\")];\n"
        "\n"
        "        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"
        "        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"
        "        tensor<fp16, [1, 4, 1, 128]> scores = matmul(transpose_x=tx, transpose_y=ty, x=qt, y=kt)[name=string(\"mm1\")];\n"
        "\n"
        "        fp16 sc = const()[name=string(\"sc\"), val=fp16(0.25)];\n"
        "        tensor<fp16, [1, 4, 1, 128]> scaled = mul(x=scores, y=sc)[name=string(\"scl\")];\n"
        "\n"
        "        // Broadcast mask [1,1,1,128] + scores [1,4,1,128]\n"
        "        tensor<fp16, [1, 4, 1, 128]> masked = add(x=scaled, y=mask)[name=string(\"msk\")];\n"
        "\n"
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"
        "        tensor<fp16, [1, 4, 1, 128]> attn = softmax(axis=sax, x=masked)[name=string(\"sm\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 1, 16]> a4 = matmul(transpose_x=tx, transpose_y=tx, x=attn, y=vt)[name=string(\"mm2\")];\n"
        "\n"
        "        tensor<fp16, [1, 4, 16, 1]> at = transpose(perm=pm, x=a4)[name=string(\"ta\")];\n"
        "        tensor<int32, [4]> osh = const()[name=string(\"osh\"), val=tensor<int32, [4]>([1, 64, 1, 1])];\n"
        "        tensor<fp16, [1, 64, 1, 1]> out = reshape(shape=osh, x=at)[name=string(\"ra\")];\n"
        "    } -> (out);\n"
    MIL_END;

    Kern k = compile_mil(mil, @{});
    if (!k.model) { printf("  FAIL (compile)\n\n"); return 0; }

    size_t sz_q = 1 * 64 * 1 * 1 * 2;
    size_t sz_kv = 1 * 64 * 1 * 128 * 2;
    size_t sz_mask = 1 * 1 * 1 * 128 * 2;
    size_t sz_out = 1 * 64 * 1 * 1 * 2;

    IOSurfaceRef sq = make_surface(sz_q);
    IOSurfaceRef sk = make_surface(sz_kv);
    IOSurfaceRef sv = make_surface(sz_kv);
    IOSurfaceRef sm = make_surface(sz_mask);
    IOSurfaceRef so = make_surface(sz_out);

    fill_fp16(sq, (_Float16)1.0);
    fill_fp16(sk, (_Float16)0.5);
    fill_fp16(sv, (_Float16)0.25);
    // Mask: first 64 positions = 0 (attend), last 64 = -10000 (block)
    {
        IOSurfaceLock(sm, 0, NULL);
        _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(sm);
        for (int i = 0; i < 64; i++) p[i] = (_Float16)0.0;
        for (int i = 64; i < 128; i++) p[i] = (_Float16)(-10000.0);
        IOSurfaceUnlock(sm, 0, NULL);
    }
    fill_fp16(so, (_Float16)0.0);

    IOSurfaceRef ins[] = {sq, sk, sv, sm};
    IOSurfaceRef outs[] = {so};
    BOOL ok = ane_eval_io(&k, ins, 4, outs, 1);
    if (!ok) { printf("  FAIL (eval)\n\n"); cleanup_kern(&k); return 0; }

    // Check output
    int n = 64;
    _Float16 *buf = malloc(n * sizeof(_Float16));
    read_fp16(so, buf, n);
    int pass = 1;
    int nonzero = 0;
    printf("  Output sample: ");
    for (int i = 0; i < MIN(8, n); i++) printf("%.4f ", (float)buf[i]);
    printf("...\n");
    for (int i = 0; i < n; i++) if (fabsf((float)buf[i]) > 1e-6f) nonzero++;
    if (nonzero == 0) { printf("  All zeros — eval produced no output\n"); pass = 0; }
    else printf("  %d/%d non-zero outputs\n", nonzero, n);
    free(buf);

    // Benchmark
    if (pass) {
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 100; i++) ane_eval_io(&k, ins, 4, outs, 1);
        double ms = tb_ms(mach_absolute_time() - t0);
        printf("  100 iters: %.2f ms (%.3f ms/iter)\n", ms, ms / 100.0);
    }

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    cleanup_kern(&k);
    CFRelease(sq); CFRelease(sk); CFRelease(sv); CFRelease(sm); CFRelease(so);
    return pass;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();

        printf("Multi-Input ANE MIL Test\n");
        printf("========================\n\n");

        int p1 = test1_same_dims();
        int p2 = test2_diff_dims();
        int p3 = test3_with_mask();

        printf("========================\n");
        printf("Results: %d/3 passed\n", p1 + p2 + p3);
        printf("  TEST 1 (same dims, 2 inputs):    %s\n", p1 ? "PASS" : "FAIL");
        printf("  TEST 2 (diff dims, 3 inputs):     %s\n", p2 ? "PASS" : "FAIL");
        printf("  TEST 3 (diff dims+mask, 4 inputs): %s\n", p3 ? "PASS" : "FAIL");

        return (p1 && p2 && p3) ? 0 : 1;
    }
}
