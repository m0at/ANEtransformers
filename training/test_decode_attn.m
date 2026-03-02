// test_decode_attn.m — Validate 4-input decode attention MIL kernel on ANE
// Inputs: x[1,768,1,1], k_cache[1,768,1,T], v_cache[1,768,1,T], mask[1,1,1,T]
// Q projection via conv, then multi-head attention against KV cache
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#define DIM 768
#define HEADS 12
#define HD (DIM/HEADS)

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
    size_t alloc = (bytes + 16383) & ~(size_t)16383;
    if (alloc < 16384) alloc = 16384;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(alloc), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(alloc),
        (id)kIOSurfaceAllocSize:@(alloc), (id)kIOSurfacePixelFormat:@0});
}
// ANE surface layout for tensor [1, C, 1, W] in fp16:
// - Spatial dim padded to minimum 32: stride = max(W, 32)
// - Channel c, position w at byte offset: (c * stride + w) * 2
// - Surface size = C * stride * 2 bytes
// For multi-input models, ALL surfaces must be >= max surface size across all I/O tensors.
static int ane_stride(int W) { return (W < 32) ? 32 : W; }
static size_t ane_surf(int C, int W) { return (size_t)C * ane_stride(W) * 2; }

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
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        NSString *full = [td stringByAppendingPathComponent:rel];
        [[NSFileManager defaultManager] createDirectoryAtPath:[full stringByDeletingLastPathComponent]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [wd[path][@"data"] writeToFile:full atomically:YES];
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
static BOOL ane_eval(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
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

// Weight blob: 128-byte header + fp16 data
static NSData *build_blob(const float *w, int count) {
    int wsize = count * 2, total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < count; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSData *build_identity_blob(int ch) {
    float *w = (float *)calloc(ch * ch, sizeof(float));
    for (int i = 0; i < ch; i++) w[i * ch + i] = 1.0f;
    NSData *d = build_blob(w, ch * ch);
    free(w);
    return d;
}

static NSData *build_zero_blob(int count) {
    float *w = (float *)calloc(count, sizeof(float));
    NSData *d = build_blob(w, count);
    free(w);
    return d;
}

static void fill_fp16(IOSurfaceRef s, _Float16 val) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(s);
    size_t n = IOSurfaceGetAllocSize(s) / sizeof(_Float16);
    for (size_t i = 0; i < n; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
}



#define MIL_HDR \
    "program(1.3)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, " \
    "{\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// Generate decode attention MIL for cache length T
static NSString *gen_decode_attn_mil(int T) {
    NSMutableString *m = [NSMutableString string];
    [m appendFormat:@MIL_HDR];
    [m appendFormat:@"    func main<ios18>(\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, 1]> x,\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> kcache,\n", DIM, T];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> vcache,\n", DIM, T];
    [m appendFormat:@"        tensor<fp16, [1, 1, 1, %d]> mask\n", T];
    [m appendString:@"    ) {\n"];
    [m appendString:@CONV_CONST];

    // Weight and bias constants (BLOBFILE)
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", DIM, DIM, DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> Bq = const()[name=string(\"Bq\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/bq.bin\"), offset=uint64(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", DIM, DIM, DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> Bo = const()[name=string(\"Bo\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/bo.bin\"), offset=uint64(64)))];\n", DIM, DIM];

    // Q projection: conv(Wq, x) + Bq
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> qc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string(\"cq\")];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> qf = add(x=qc,y=Bq)[name=string(\"aq\")];\n", DIM];

    // Reshape Q: [1,768,1,1] -> [1,12,64,1] -> transpose -> [1,12,1,64]
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", HEADS, HD];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS, HD];

    // Reshape kcache: [1,768,1,T] -> [1,12,64,T] -> transpose -> [1,12,T,64]
    [m appendFormat:@"        tensor<int32, [4]> ksh = const()[name=string(\"ksh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, T];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=ksh,x=kcache)[name=string(\"rk\")];\n", HEADS, HD, T];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", HEADS, T, HD];

    // Reshape vcache: [1,768,1,T] -> [1,12,64,T] -> transpose -> [1,12,T,64]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=ksh,x=vcache)[name=string(\"rv\")];\n", HEADS, HD, T];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", HEADS, T, HD];

    // scores = matmul(q, k^T): [1,12,1,64] x [1,12,64,T] -> [1,12,1,T]
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, T];

    // Scale: 1/sqrt(64) = 0.125
    [m appendString:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(0.125)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, T];

    // Add mask: broadcast [1,1,1,T] -> [1,12,1,T]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ms = add(x=sc2,y=mask)[name=string(\"msk\")];\n", HEADS, T];

    // Softmax
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, T];

    // out = matmul(scores, v): [1,12,1,T] x [1,12,T,64] -> [1,12,1,64]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS, HD];

    // Transpose back: [1,12,1,64] -> [1,12,64,1]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS, HD];

    // Reshape to [1,768,1,1]
    [m appendFormat:@"        tensor<int32, [4]> osh = const()[name=string(\"osh\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> af = reshape(shape=osh,x=at)[name=string(\"ra\")];\n", DIM];

    // Wo projection + bias
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> oc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> out = add(x=oc,y=Bo)[name=string(\"ao\")];\n", DIM];

    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

static int test_decode_attn(int T) {
    printf("=== Decode Attention T=%d ===\n", T);
    printf("  x:[1,%d,1,1]  kcache:[1,%d,1,%d]  vcache:[1,%d,1,%d]  mask:[1,1,1,%d]\n", DIM, DIM, T, DIM, T, T);

    NSString *mil = gen_decode_attn_mil(T);

    // Build weights: identity Wq, Wo; zero biases
    NSData *wq_blob = build_identity_blob(DIM);
    NSData *wo_blob = build_identity_blob(DIM);
    NSData *bq_blob = build_zero_blob(DIM);
    NSData *bo_blob = build_zero_blob(DIM);

    NSDictionary *wd = @{
        @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":wq_blob},
        @"@model_path/weights/bq.bin": @{@"offset":@0, @"data":bq_blob},
        @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":wo_blob},
        @"@model_path/weights/bo.bin": @{@"offset":@0, @"data":bo_blob},
    };

    uint64_t t0 = mach_absolute_time();
    Kern k = compile_mil(mil, wd);
    double compile_ms = tb_ms(mach_absolute_time() - t0);
    if (!k.model) { printf("  FAIL (compile)\n\n"); return 0; }
    printf("  Compiled in %.1f ms\n", compile_ms);

    // All surfaces must be >= max across all I/O tensors
    size_t max_surf = ane_surf(DIM, T);  // largest: [1,DIM,1,T]
    int stride = ane_stride(T);           // spatial stride for data layout
    int stride1 = ane_stride(1);          // spatial stride for W=1 tensors (=32)
    printf("  Surface: %zu bytes, stride=%d, stride1=%d\n", max_surf, stride, stride1);

    IOSurfaceRef s_x = make_surface(max_surf);
    IOSurfaceRef s_k = make_surface(max_surf);
    IOSurfaceRef s_v = make_surface(max_surf);
    IOSurfaceRef s_m = make_surface(max_surf);
    IOSurfaceRef s_o = make_surface(max_surf);

    // ANE layout: channel c, position w -> offset c * stride + w
    // For x [1,768,1,1]: use stride1 (=32), data at c*32 + 0
    // For kv [1,768,1,T]: use stride (=max(T,32)), data at c*stride + t
    // For mask [1,1,1,T]: channel 0, data at 0*stride + t (but surface oversized)

    // Simple test: fill all with constant values, check output
    // x=1.0, kcache=1.0, vcache=0.5, mask=0.0 (attend everything)
    // With identity Wq: q = x = 1.0 everywhere
    // Each head: q=[1,...,1] (64-dim), k[t]=[1,...,1] for all t
    // scores[t] = dot(q,k[t]) = 64, scaled = 64*0.125 = 8.0 for all t
    // mask = 0 => softmax(8,8,...,8) = 1/T for each position
    // v[t] = [0.5, 0.5, ...] for all t
    // out = sum(1/T * v[t]) = 0.5 for all channels
    // With identity Wo: final output = 0.5
    fill_fp16(s_x, (_Float16)1.0);
    fill_fp16(s_k, (_Float16)1.0);
    fill_fp16(s_v, (_Float16)0.5);
    fill_fp16(s_m, (_Float16)0.0);

    fill_fp16(s_o, (_Float16)0.0);

    IOSurfaceRef ins[] = {s_x, s_k, s_v, s_m};
    IOSurfaceRef outs[] = {s_o};
    BOOL ok = ane_eval(&k, ins, 4, outs, 1);
    if (!ok) { printf("  FAIL (eval)\n\n"); cleanup_kern(&k); return 0; }

    // Expected: all channels = 0.5 (uniform attention over constant v=0.5)

    // Read output [1,768,1,1]: channel c at offset c * stride
    float out_vals[DIM];
    {
        IOSurfaceLock(s_o, kIOSurfaceLockReadOnly, NULL);
        _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(s_o);
        for (int c = 0; c < DIM; c++) out_vals[c] = (float)p[c * stride + 0];
        IOSurfaceUnlock(s_o, kIOSurfaceLockReadOnly, NULL);
    }

    int pass = 1;
    float max_err = 0;
    int mismatches = 0;
    for (int c = 0; c < DIM; c++) {
        float expected = (c % HD) * 0.01f;
        float got = out_vals[c];
        float err = fabsf(got - expected);
        if (err > max_err) max_err = err;
        if (err > 0.05f) {
            if (mismatches < 10)
                printf("  MISMATCH ch=%d: got=%.4f expected=%.4f err=%.4f\n", c, got, expected, err);
            mismatches++;
            pass = 0;
        }
    }
    if (mismatches > 10) printf("  ... (%d total mismatches)\n", mismatches);
    if (pass) printf("  Values match (max error=%.4f)\n", max_err);

    printf("  Output[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", out_vals[i]);
    printf("...\n");
    printf("  Expected[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", (i % HD) * 0.01f);
    printf("...\n");

    // Benchmark
    if (pass) {
        int iters = 100;
        t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) ane_eval(&k, ins, 4, outs, 1);
        double ms = tb_ms(mach_absolute_time() - t0);
        printf("  %d iters: %.2f ms (%.3f ms/iter)\n", iters, ms, ms / iters);
    }

    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    cleanup_kern(&k);
    CFRelease(s_x); CFRelease(s_k); CFRelease(s_v); CFRelease(s_m); CFRelease(s_o);
    return pass;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        ane_init();

        printf("Decode Attention ANE Test\n");
        printf("=========================\n\n");

        int p1 = test_decode_attn(32);
        int p2 = test_decode_attn(64);
        int p3 = test_decode_attn(128);

        printf("=========================\n");
        printf("Results: %d/3 passed\n", p1 + p2 + p3);
        printf("  T=32:  %s\n", p1 ? "PASS" : "FAIL");
        printf("  T=64:  %s\n", p2 ? "PASS" : "FAIL");
        printf("  T=128: %s\n", p3 ? "PASS" : "FAIL");

        return (p1 && p2 && p3) ? 0 : 1;
    }
}
