// test_ffn_seq1.m — Validate GPT-2 FFN MIL kernel compiles and runs on ANE with seq=1
// Key finding: IOSurface must be >= 49152 bytes even if logical tensor is smaller
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <math.h>

#define DIM 768
#define HIDDEN 3072

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// Minimum IOSurface size the ANE accepts for this model (empirically determined)
#define ANE_MIN_SURFACE 49152

static Class g_D, g_I, g_AR, g_AIO;
static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surface(size_t bytes) {
    size_t alloc = bytes < ANE_MIN_SURFACE ? ANE_MIN_SURFACE : bytes;
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
    BOOL r = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!r && e) printf("  eval error: %s\n", [[e localizedDescription] UTF8String]);
    return r;
}

static void cleanup_kern(Kern *k) {
    if (!k->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->td error:nil];
    k->model = nil;
}

// Fused FFN MIL generator -- exact copy from gpt2.m
static NSString *gen_ffn_mil(int layer, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, seq];
    [m appendString:@CONV_CONST];
    NSString *ld = [NSString stringWithFormat:@"layer_%02d", layer];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/w1.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> B1 = const()[name=string(\"B1\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/b1.bin\"), offset=uint64(64)))];\n", HIDDEN,HIDDEN,ld];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/w2.bin\"), offset=uint64(64)))];\n", DIM,HIDDEN,DIM,HIDDEN,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> B2 = const()[name=string(\"B2\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/b2.bin\"), offset=uint64(64)))];\n", DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> hc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=x)[name=string(\"c1\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h = add(x=hc,y=B1)[name=string(\"ab1\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h2 = mul(x=h,y=h)[name=string(\"h2\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = mul(x=h2,y=h)[name=string(\"h3\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c1 = const()[name=string(\"c1v\"), val=fp16(0.044715)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=h3,y=c1)[name=string(\"t1\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t2 = add(x=h,y=t1)[name=string(\"t2\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c2 = const()[name=string(\"c2v\"), val=fp16(0.7978845608)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t3 = mul(x=t2,y=c2)[name=string(\"t3\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t4 = tanh(x=t3)[name=string(\"t4\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c3 = const()[name=string(\"c3v\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t5 = add(x=t4,y=c3)[name=string(\"t5\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c4 = const()[name=string(\"c4v\"), val=fp16(0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t6 = mul(x=h,y=c4)[name=string(\"t6\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gelu = mul(x=t6,y=t5)[name=string(\"gelu\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oc2 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gelu)[name=string(\"c2\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=oc2,y=B2)[name=string(\"ab2\")];\n", DIM,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

static NSData *make_random_blob(int n_fp16) {
    int wsize = n_fp16 * 2, total = 128 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    _Float16 *weights = (_Float16*)(buf + 128);
    for (int i = 0; i < n_fp16; i++)
        weights[i] = (_Float16)((drand48() - 0.5) * 0.02);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        srand48(42);
        ane_init();

        int seq = 1;
        printf("=== test_ffn_seq1: GPT-2 FFN on ANE, seq=%d ===\n", seq);

        // Random weight blobs: W1[3072,768,1,1], B1[1,3072,1,1], W2[768,3072,1,1], B2[1,768,1,1]
        NSData *w1_blob = make_random_blob(HIDDEN * DIM);
        NSData *b1_blob = make_random_blob(HIDDEN);
        NSData *w2_blob = make_random_blob(DIM * HIDDEN);
        NSData *b2_blob = make_random_blob(DIM);

        NSDictionary *wd = @{
            @"@model_path/weights/layer_00/w1.bin": @{@"offset":@0, @"data":w1_blob},
            @"@model_path/weights/layer_00/w2.bin": @{@"offset":@0, @"data":w2_blob},
            @"@model_path/weights/layer_00/b1.bin": @{@"offset":@0, @"data":b1_blob},
            @"@model_path/weights/layer_00/b2.bin": @{@"offset":@0, @"data":b2_blob},
        };

        // Compile
        printf("Generating FFN MIL for layer=0, seq=%d (DIM=%d, HIDDEN=%d)...\n", seq, DIM, HIDDEN);
        NSString *mil = gen_ffn_mil(0, seq);
        printf("Compiling + loading...\n");
        Kern k = compile_mil(mil, wd);
        if (!k.model) {
            printf("FAIL: compile/load failed\n");
            return 1;
        }
        printf("  OK\n");

        // Allocate IOSurfaces (ANE_MIN_SURFACE ensures sufficient size)
        size_t logical = DIM * seq * 2;
        printf("IOSurface: logical=%zu bytes, alloc=%d bytes\n", logical, ANE_MIN_SURFACE);
        IOSurfaceRef surf_in  = make_surface(logical);
        IOSurfaceRef surf_out = make_surface(logical);

        // Fill input with random fp16 data at position 0 of each channel
        IOSurfaceLock(surf_in, 0, NULL);
        _Float16 *pin = (_Float16*)IOSurfaceGetBaseAddress(surf_in);
        memset(pin, 0, IOSurfaceGetAllocSize(surf_in));
        for (int c = 0; c < DIM; c++)
            pin[c * seq] = (_Float16)((drand48() - 0.5) * 2.0);
        IOSurfaceUnlock(surf_in, 0, NULL);

        // Clear output
        IOSurfaceLock(surf_out, 0, NULL);
        memset(IOSurfaceGetBaseAddress(surf_out), 0, IOSurfaceGetAllocSize(surf_out));
        IOSurfaceUnlock(surf_out, 0, NULL);

        // Evaluate on ANE
        printf("Running on ANE...\n");
        IOSurfaceRef ins[] = {surf_in}, outs[] = {surf_out};
        BOOL ok = ane_eval(&k, ins, 1, outs, 1);
        if (!ok) {
            printf("FAIL: ane_eval returned NO\n");
            cleanup_kern(&k); CFRelease(surf_in); CFRelease(surf_out);
            return 1;
        }
        printf("  OK\n");

        // Verify output is non-zero
        IOSurfaceLock(surf_out, kIOSurfaceLockReadOnly, NULL);
        _Float16 *pout = (_Float16*)IOSurfaceGetBaseAddress(surf_out);
        double sum_abs = 0;
        int nonzero = 0;
        for (int c = 0; c < DIM; c++) {
            float v = (float)pout[c * seq];
            sum_abs += fabs(v);
            if (v != 0.0f) nonzero++;
        }
        IOSurfaceUnlock(surf_out, kIOSurfaceLockReadOnly, NULL);

        printf("Output: %d/%d channels non-zero, sum_abs=%.6f\n", nonzero, DIM, sum_abs);

        cleanup_kern(&k);
        CFRelease(surf_in);
        CFRelease(surf_out);

        if (nonzero > 0 && sum_abs > 1e-6) {
            printf("PASS\n");
            return 0;
        } else {
            printf("FAIL: output is all zeros\n");
            return 1;
        }
    }
}
