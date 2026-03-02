// test_chaining.m — Explore _ANEChainingRequest for multi-layer ANE dispatch
// Goal: chain two conv kernels so output of conv1 feeds into conv2 without CPU round-trip
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static void dump_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n", name); return; }
    printf("\n=== %s ===\n", name);
    unsigned int count;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count) printf("  Class methods:\n");
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    + %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);
    methods = class_copyMethodList(cls, &count);
    if (count) printf("  Instance methods:\n");
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    - %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);
    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    if (pcount) printf("  Properties:\n");
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        const char *pattr = property_getAttributes(props[i]);
        printf("    @property %s  [%s]\n", pname, pattr ? pattr : "?");
    }
    free(props);
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Compile a single conv kernel (fp32 I/O with cast, known-working pattern)
typedef struct {
    id model;           // _ANEInMemoryModel
    IOSurfaceRef ioIn;
    IOSurfaceRef ioOut;
    id request;         // _ANERequest
    NSString *tmpDir;
    int ch, sp;
} TestKern;

static Class g_D, g_I, g_AR, g_AIO;

static TestKern *compile_conv(int ch, int sp, _Float16 *weights) {
    int ws = ch*ch*2, tot = 128+ws;
    uint8_t *blob = (uint8_t*)calloc(tot,1);
    blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
    *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
    memcpy(blob+128, weights, ws);
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n", ch, sp, ch, sp, ch, ch, ch, ch, ch, sp, ch, sp];

    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}};

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, wdict, nil);
    if (!desc) return NULL;
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e))
        return NULL;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e))
        return NULL;

    int ioBytes = ch * sp * 4;
    TestKern *k = (TestKern*)calloc(1, sizeof(TestKern));
    k->model = mdl;
    k->ioIn = make_surface(ioBytes);
    k->ioOut = make_surface(ioBytes);
    k->tmpDir = td;
    k->ch = ch;
    k->sp = sp;

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return k;
}

static void eval_kern(TestKern *k) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
}

static void free_kern(TestKern *k) {
    if (!k) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    free(k);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_I  = NSClassFromString(@"_ANEInMemoryModel");
        g_AR = NSClassFromString(@"_ANERequest");
        g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        printf("=== ANE Chaining & Real-Time API Exploration ===\n");

        // ---- Part 0: Dump unexplored classes ----
        printf("\n--- Unexplored Chaining Classes ---\n");
        dump_class("_ANEIOSurfaceOutputSets");
        dump_class("_ANEInputBuffersReady");
        dump_class("_ANEOutputSetEnqueue");
        dump_class("_ANEModelInstanceParameters");
        dump_class("_ANEBuffer");
        dump_class("_ANEProgramForEvaluation");
        dump_class("_ANEProgramIOSurfacesMapper");
        dump_class("_ANEDeviceController");

        // ---- Part 1: Baseline — two sequential evals (CPU round-trip) ----
        printf("\n--- Part 1: Baseline — two sequential conv evals ---\n");
        int CH = 64, SP = 32;

        // Conv1: 2x identity
        _Float16 *w1 = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) w1[i*CH+i] = (_Float16)2.0f;

        // Conv2: 3x identity
        _Float16 *w2 = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) w2[i*CH+i] = (_Float16)3.0f;

        TestKern *k1 = compile_conv(CH, SP, w1);
        TestKern *k2 = compile_conv(CH, SP, w2);
        free(w1); free(w2);

        if (!k1 || !k2) { printf("FAIL: compile\n"); return 1; }
        printf("  Compiled two %dx%d conv kernels\n", CH, CH);

        // Write input: 1.0 everywhere
        IOSurfaceLock(k1->ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(k1->ioIn);
        for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        IOSurfaceUnlock(k1->ioIn, 0, NULL);

        // Sequential: k1 → read → write → k2
        uint64_t t0 = mach_absolute_time();
        int N_ITER = 100;
        for (int i = 0; i < N_ITER; i++) {
            eval_kern(k1);
            // Read k1 output, write to k2 input
            IOSurfaceLock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
            IOSurfaceLock(k2->ioIn, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(k2->ioIn), IOSurfaceGetBaseAddress(k1->ioOut), CH*SP*4);
            IOSurfaceUnlock(k2->ioIn, 0, NULL);
            IOSurfaceUnlock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
            eval_kern(k2);
        }
        double seq_ms = tb_ms(mach_absolute_time() - t0);
        printf("  Sequential (2 evals + memcpy): %.1fms / %d iter = %.3fms/iter\n", seq_ms, N_ITER, seq_ms/N_ITER);

        // Verify: input=1.0, conv1=2x, conv2=3x → expected output = 6.0
        IOSurfaceLock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);
        float *out = (float*)IOSurfaceGetBaseAddress(k2->ioOut);
        printf("  Expected 6.0, got: %.4f (first elem)\n", out[0]);
        IOSurfaceUnlock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);

        // Shared IOSurface: k1 output IS k2 input (zero-copy)
        printf("\n--- Part 1b: Shared IOSurface (k1.ioOut == k2.ioIn) ---\n");
        // Rebuild k2 request to use k1's output surface as input
        id wI2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k1->ioOut);
        id wO2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k2->ioOut);
        id req_shared = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI2], @[@0], @[wO2], @[@0], nil, nil, @0);

        // Re-write input
        IOSurfaceLock(k1->ioIn, 0, NULL);
        inp = (float*)IOSurfaceGetBaseAddress(k1->ioIn);
        for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        IOSurfaceUnlock(k1->ioIn, 0, NULL);

        t0 = mach_absolute_time();
        for (int i = 0; i < N_ITER; i++) {
            eval_kern(k1);
            // No memcpy! k2 reads directly from k1's output surface
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                k2->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req_shared, &e);
        }
        double shared_ms = tb_ms(mach_absolute_time() - t0);
        printf("  Shared surface (no memcpy): %.1fms / %d iter = %.3fms/iter\n", shared_ms, N_ITER, shared_ms/N_ITER);
        printf("  Speedup: %.2fx\n", seq_ms / shared_ms);

        IOSurfaceLock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);
        out = (float*)IOSurfaceGetBaseAddress(k2->ioOut);
        printf("  Expected 6.0, got: %.4f\n", out[0]);
        IOSurfaceUnlock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);

        // ---- Part 2: Try _ANEClient chaining API ----
        printf("\n--- Part 2: _ANEChainingRequest ---\n");
        Class chainClass = NSClassFromString(@"_ANEChainingRequest");
        Class clientClass = NSClassFromString(@"_ANEClient");
        Class bufClass = NSClassFromString(@"_ANEBuffer");
        Class outSetsClass = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class inBufReadyClass = NSClassFromString(@"_ANEInputBuffersReady");
        Class outEnqClass = NSClassFromString(@"_ANEOutputSetEnqueue");

        if (!chainClass) { printf("  _ANEChainingRequest: NOT FOUND\n"); goto part3; }

        {
            id client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, @selector(sharedConnection));
            printf("  Client: %s\n", client ? "OK" : "nil");

            // _ANEBuffer: bufferWithIOSurfaceObject:symbolIndex:source:
            // source: 0=input (ANEBufferProducerAgent=0), 1=output (ANEBufferProducerAgent=1)
            id inSurf = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k1->ioIn);
            id outSurf1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k1->ioOut);
            id outSurf2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k2->ioOut);

            id inBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(bufClass,
                @selector(bufferWithIOSurfaceObject:symbolIndex:source:), inSurf, @0, 0LL);
            id outBuf1 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(bufClass,
                @selector(bufferWithIOSurfaceObject:symbolIndex:source:), outSurf1, @0, 1LL);
            id outBuf2 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(bufClass,
                @selector(bufferWithIOSurfaceObject:symbolIndex:source:), outSurf2, @0, 1LL);

            // _ANEIOSurfaceOutputSets requires non-NULL statsSurRef
            IOSurfaceRef statsSurf = make_surface(4096);
            id oset1 = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(outSetsClass,
                @selector(objectWithstatsSurRef:outputBuffer:), statsSurf, @[outBuf1]);
            id oset2 = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(outSetsClass,
                @selector(objectWithstatsSurRef:outputBuffer:), statsSurf, @[outBuf2]);
            printf("  oset1: %s  oset2: %s\n", oset1?"OK":"nil", oset2?"OK":"nil");

            // --- Config A: With loopback (chain: output sym 0 → input sym 0) ---
            printf("\n  --- Config A: With loopback ---\n");
            @try {
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[inBuf], @[oset1, oset2], @[@0], @[@0], @0, @[], nil, @0, @0);
                printf("    Created: %s\n", chainReq ? "OK" : "nil");
                if (chainReq) {
                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq, @selector(validate));
                    printf("    Validate: %s\n", valid ? "YES" : "NO");

                    id aneModel = ((id(*)(id,SEL))objc_msgSend)(k1->model, @selector(model));
                    printf("    _ANEModel: %s\n", aneModel ? "OK" : "nil");
                    if (aneModel && client) {
                        NSError *e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                            aneModel, @{}, chainReq, 21, &e);
                        printf("    prepareChaining: %s\n", ok ? "OK" : "FAIL");
                        if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                        if (ok) {
                            // enqueueSets: _ANEOutputSetEnqueue objects
                            id oseq0 = ((id(*)(Class,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                                outEnqClass, @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                                0, 0, 1ULL, NO, NO);
                            id oseq1 = ((id(*)(Class,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                                outEnqClass, @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                                0, 1, 2ULL, NO, NO);

                            e = nil;
                            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                aneModel, @[oseq0, oseq1], @{}, 21, &e);
                            printf("    enqueueSets: %s\n", ok ? "OK" : "FAIL");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                            // buffersReady: _ANEInputBuffersReady
                            id ibr = ((id(*)(Class,SEL,unsigned int,id,id,unsigned long long))objc_msgSend)(
                                inBufReadyClass, @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                                0, @[@0], @[@0], 0ULL);

                            e = nil;
                            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                aneModel, ibr, @{}, 21, &e);
                            printf("    buffersReady: %s\n", ok ? "OK" : "FAIL");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                            IOSurfaceLock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
                            float *r1 = (float*)IOSurfaceGetBaseAddress(k1->ioOut);
                            printf("    k1 out[0]: %.4f (expected 2.0)\n", r1[0]);
                            IOSurfaceUnlock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
                            IOSurfaceLock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);
                            float *r2 = (float*)IOSurfaceGetBaseAddress(k2->ioOut);
                            printf("    k2 out[0]: %.4f (expected 6.0 if chained)\n", r2[0]);
                            IOSurfaceUnlock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);

                            if (ok) {
                                // Benchmark chaining
                                t0 = mach_absolute_time();
                                for (int i = 0; i < N_ITER; i++) {
                                    e = nil;
                                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                        client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                        aneModel, @[oseq0, oseq1], @{}, 21, &e);
                                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                        client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                        aneModel, ibr, @{}, 21, &e);
                                }
                                double chain_ms = tb_ms(mach_absolute_time() - t0);
                                printf("    Chaining: %.1fms / %d = %.3fms/iter\n", chain_ms, N_ITER, chain_ms/N_ITER);
                                printf("    vs Sequential: %.3fms/iter\n", seq_ms/N_ITER);
                            }
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            // --- Config B: No loopback, single output ---
            printf("\n  --- Config B: No loopback ---\n");
            @try {
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[inBuf], @[oset1], @[], @[], @0, @[], nil, @0, @0);
                printf("    Created: %s\n", chainReq ? "OK" : "nil");
                if (chainReq) {
                    BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq, @selector(validate));
                    printf("    Validate: %s\n", valid ? "YES" : "NO");

                    id aneModel = ((id(*)(id,SEL))objc_msgSend)(k1->model, @selector(model));
                    if (aneModel && client) {
                        NSError *e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                            aneModel, @{}, chainReq, 21, &e);
                        printf("    prepareChaining: %s\n", ok ? "OK" : "FAIL");
                        if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                        if (ok) {
                            id oseq = ((id(*)(Class,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                                outEnqClass, @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                                0, 0, 1ULL, NO, NO);
                            e = nil;
                            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                aneModel, @[oseq], @{}, 21, &e);
                            printf("    enqueueSets: %s\n", ok ? "OK" : "FAIL");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                            id ibr = ((id(*)(Class,SEL,unsigned int,id,id,unsigned long long))objc_msgSend)(
                                inBufReadyClass, @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                                0, @[@0], @[@0], 0ULL);
                            e = nil;
                            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                aneModel, ibr, @{}, 21, &e);
                            printf("    buffersReady: %s\n", ok ? "OK" : "FAIL");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                            IOSurfaceLock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
                            float *r = (float*)IOSurfaceGetBaseAddress(k1->ioOut);
                            printf("    k1 out[0]: %.4f (expected 2.0)\n", r[0]);
                            IOSurfaceUnlock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            CFRelease(statsSurf);
        }

        // ---- Part 3: Real-Time eval path ----
        part3:
        printf("\n--- Part 3: Real-Time Eval Path ---\n");
        {
            id client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, @selector(sharedConnection));
            if (!client) { printf("  No client\n"); goto part4; }

            // beginRealTimeTask
            @try {
                BOOL ok = ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
                printf("  beginRealTimeTask: %s\n", ok ? "OK" : "FAIL");
            } @catch (NSException *ex) {
                printf("  beginRealTimeTask exception: %s\n", [[ex reason] UTF8String]);
            }

            // Try evaluateRealTimeWithModel
            @try {
                id aneModel = ((id(*)(id,SEL))objc_msgSend)(k1->model, @selector(model));
                if (aneModel) {
                    // Write input
                    IOSurfaceLock(k1->ioIn, 0, NULL);
                    float *p = (float*)IOSurfaceGetBaseAddress(k1->ioIn);
                    for (int i = 0; i < CH*SP; i++) p[i] = 1.0f;
                    IOSurfaceUnlock(k1->ioIn, 0, NULL);

                    // Map IO surfaces
                    NSError *e = nil;
                    printf("  Mapping IOSurfaces...\n");
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,BOOL,NSError**))objc_msgSend)(
                        client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
                        aneModel, k1->request, NO, &e);
                    printf("  mapIOSurfaces: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                    printf("  evaluateRealTimeWithModel...\n");
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        client, @selector(evaluateRealTimeWithModel:options:request:error:),
                        aneModel, @{}, k1->request, &e);
                    printf("  evaluateRealTime: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        IOSurfaceLock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);
                        float *r = (float*)IOSurfaceGetBaseAddress(k1->ioOut);
                        printf("  Output: %.4f (expected 2.0)\n", r[0]);
                        IOSurfaceUnlock(k1->ioOut, kIOSurfaceLockReadOnly, NULL);

                        // Benchmark RT eval
                        t0 = mach_absolute_time();
                        for (int i = 0; i < N_ITER; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                                client, @selector(evaluateRealTimeWithModel:options:request:error:),
                                aneModel, @{}, k1->request, &e);
                        }
                        double rt_ms = tb_ms(mach_absolute_time() - t0);
                        printf("  RT eval: %.1fms / %d = %.3fms/eval\n", rt_ms, N_ITER, rt_ms/N_ITER);

                        // Compare with normal eval
                        t0 = mach_absolute_time();
                        for (int i = 0; i < N_ITER; i++) {
                            eval_kern(k1);
                        }
                        double normal_ms = tb_ms(mach_absolute_time() - t0);
                        printf("  Normal eval: %.1fms / %d = %.3fms/eval\n", normal_ms, N_ITER, normal_ms/N_ITER);
                        printf("  RT speedup: %.2fx\n", normal_ms / rt_ms);
                    }

                    // Unmap
                    ((void(*)(id,SEL,id,id))objc_msgSend)(
                        client, @selector(unmapIOSurfacesWithModel:request:), aneModel, k1->request);
                }
            } @catch (NSException *ex) {
                printf("  RT eval exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            @try {
                BOOL ok = ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
                printf("  endRealTimeTask: %s\n", ok ? "OK" : "FAIL");
            } @catch (NSException *ex) {
                printf("  endRealTimeTask exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ---- Part 4: Multi-procedure MIL (two convs in one program) ----
        part4:
        printf("\n--- Part 4: Two convs in one MIL program ---\n");
        {
            // The most reliable chaining: put both convolutions in one MIL program
            // Output = W2 @ (W1 @ x) = (W2*W1) @ x
            _Float16 *wa = (_Float16*)calloc(CH*CH, sizeof(_Float16));
            _Float16 *wb = (_Float16*)calloc(CH*CH, sizeof(_Float16));
            for (int i = 0; i < CH; i++) { wa[i*CH+i] = (_Float16)2.0f; wb[i*CH+i] = (_Float16)3.0f; }

            int ws = CH*CH*2, tot = 128+ws;
            uint8_t *blobA = (uint8_t*)calloc(tot,1);
            blobA[0]=1; blobA[4]=2; blobA[64]=0xEF; blobA[65]=0xBE; blobA[66]=0xAD; blobA[67]=0xDE; blobA[68]=1;
            *(uint32_t*)(blobA+72)=ws; *(uint32_t*)(blobA+80)=128;
            memcpy(blobA+128, wa, ws);
            NSData *wdA = [NSData dataWithBytesNoCopy:blobA length:tot freeWhenDone:YES];

            uint8_t *blobB = (uint8_t*)calloc(tot,1);
            blobB[0]=1; blobB[4]=2; blobB[64]=0xEF; blobB[65]=0xBE; blobB[66]=0xAD; blobB[67]=0xDE; blobB[68]=1;
            *(uint32_t*)(blobB+72)=ws; *(uint32_t*)(blobB+80)=128;
            memcpy(blobB+128, wb, ws);
            NSData *wdB = [NSData dataWithBytesNoCopy:blobB length:tot freeWhenDone:YES];
            free(wa); free(wb);

            NSString *mil2 = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
                "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
                "        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n"
                "        tensor<fp16, [1,%d,1,%d]> h = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=x16)"
                "[name=string(\"conv1\")];\n"
                "        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n"
                "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=h)"
                "[name=string(\"conv2\")];\n"
                "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
                "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
                "    } -> (y);\n"
                "}\n", CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];

            NSData *md2 = [mil2 dataUsingEncoding:NSUTF8StringEncoding];
            NSDictionary *wdict2 = @{
                @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":wdA},
                @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":wdB}
            };

            id desc2 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md2, wdict2, nil);
            if (!desc2) { printf("  desc=nil\n"); goto done; }
            id mdl2 = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc2);
            id hx2 = ((id(*)(id,SEL))objc_msgSend)(mdl2, @selector(hexStringIdentifier));
            NSString *td2 = [NSTemporaryDirectory() stringByAppendingPathComponent:hx2];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[td2 stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
            [md2 writeToFile:[td2 stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [wdA writeToFile:[td2 stringByAppendingPathComponent:@"weights/w1.bin"] atomically:YES];
            [wdB writeToFile:[td2 stringByAppendingPathComponent:@"weights/w2.bin"] atomically:YES];

            NSError *e = nil;
            uint64_t tc = mach_absolute_time();
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl2, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            double ctime = tb_ms(mach_absolute_time() - tc);
            printf("  Compile (2 convs in 1 MIL): %s (%.1fms)\n", ok ? "OK" : "FAIL", ctime);
            if (!ok && e) printf("    Error: %s\n", [[e description] UTF8String]);
            if (!ok) goto done;

            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl2, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("  Load: %s\n", ok ? "OK" : "FAIL");
            if (!ok) goto done;

            int ioBytes = CH * SP * 4;
            IOSurfaceRef ioIn2 = make_surface(ioBytes);
            IOSurfaceRef ioOut2 = make_surface(ioBytes);
            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn2);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut2);
            id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            IOSurfaceLock(ioIn2, 0, NULL);
            float *p = (float*)IOSurfaceGetBaseAddress(ioIn2);
            for (int i = 0; i < CH*SP; i++) p[i] = 1.0f;
            IOSurfaceUnlock(ioIn2, 0, NULL);

            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
            printf("  Eval: %s\n", ok ? "OK" : "FAIL");

            if (ok) {
                IOSurfaceLock(ioOut2, kIOSurfaceLockReadOnly, NULL);
                float *r = (float*)IOSurfaceGetBaseAddress(ioOut2);
                printf("  Output: %.4f (expected 6.0 = 2x * 3x)\n", r[0]);
                IOSurfaceUnlock(ioOut2, kIOSurfaceLockReadOnly, NULL);

                // Benchmark fused vs sequential
                t0 = mach_absolute_time();
                for (int i = 0; i < N_ITER; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
                }
                double fused_ms = tb_ms(mach_absolute_time() - t0);
                printf("  Fused (1 dispatch, 2 convs): %.1fms / %d = %.3fms/iter\n", fused_ms, N_ITER, fused_ms/N_ITER);
                printf("  vs Sequential (2 dispatches): %.3fms/iter\n", seq_ms/N_ITER);
                printf("  vs Shared surface: %.3fms/iter\n", shared_ms/N_ITER);
                printf("  Fused speedup over sequential: %.2fx\n", seq_ms / fused_ms);
                printf("  Fused speedup over shared: %.2fx\n", shared_ms / fused_ms);
            }

            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl2, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:td2 error:nil];
            CFRelease(ioIn2); CFRelease(ioOut2);
        }

        done:
        printf("\n--- Cleanup ---\n");
        free_kern(k1);
        free_kern(k2);
        printf("Done.\n");
    }
    return 0;
}
