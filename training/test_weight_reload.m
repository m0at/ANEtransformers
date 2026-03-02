// test_weight_reload.m — Can we skip recompilation by rewriting weight blobs on disk?
// Compile a conv kernel with weights A, eval, verify output.
// Overwrite weights/weight.bin in tmpDir with weights B.
// unloadWithQoS: then loadWithQoS: (no recompile).
// Eval again — if output matches B @ x, compilation bottleneck is eliminated.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        if (!g_D || !g_I || !g_AR || !g_AIO) {
            printf("FAIL: ANE classes not found\n");
            return 1;
        }

        // Small test: 4x4 conv kernel, spatial=4
        int IC = 4, OC = 4, SP = 4;

        // Weight set A: identity matrix
        _Float16 weightsA[16];
        for (int i = 0; i < IC*OC; i++) weightsA[i] = (i / OC == i % OC) ? (_Float16)1.0f : (_Float16)0.0f;

        // Weight set B: 2x identity
        _Float16 weightsB[16];
        for (int i = 0; i < IC*OC; i++) weightsB[i] = (i / OC == i % OC) ? (_Float16)2.0f : (_Float16)0.0f;

        // Build weight blob for A
        int ws = IC * OC * 2;
        int tot = 128 + ws;
        uint8_t *blob = (uint8_t*)calloc(tot, 1);
        blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
        *(uint32_t*)(blob+72) = ws;
        *(uint32_t*)(blob+80) = 128;
        memcpy(blob + 128, weightsA, ws);
        NSData *wdataA = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

        // MIL for a simple conv
        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
            "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
            "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
            "[name=string(\"conv\")];\n"
            "    } -> (y);\n"
            "}\n", IC, SP, OC, IC, OC, IC, OC, SP];

        NSDictionary *weights = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wdataA}
        };

        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

        // === Compile with weights A ===
        printf("=== Step 1: Compile with weights A (identity) ===\n");
        uint64_t t0 = mach_absolute_time();
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
        if (!desc) { printf("FAIL: desc=NULL\n"); return 1; }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdataA writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { printf("FAIL: compile: %s\n", [[e description] UTF8String]); return 1; }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { printf("FAIL: load: %s\n", [[e description] UTF8String]); return 1; }
        printf("  Compile+load: %.1fms\n", tb_ms(mach_absolute_time() - t0));
        printf("  tmpDir: %s\n", [td UTF8String]);

        // Build request and IOSurfaces
        int inBytes = IC * SP * 2;
        int outBytes = OC * SP * 2;
        IOSurfaceRef ioIn = make_surface(inBytes);
        IOSurfaceRef ioOut = make_surface(outBytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        // Write input: [1, 2, 3, 4] repeated across channels
        IOSurfaceLock(ioIn, 0, NULL);
        _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < IC; c++)
            for (int s = 0; s < SP; s++)
                inp[c * SP + s] = (_Float16)(s + 1.0f);
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Eval with weights A
        printf("\n=== Step 2: Eval with weights A ===\n");
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) { printf("FAIL: eval: %s\n", e ? [[e description] UTF8String] : "?"); return 1; }

        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *outA = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
        printf("  Output A (identity @ [1,2,3,4]):");
        for (int c = 0; c < OC; c++) {
            printf(" [");
            for (int s = 0; s < SP; s++) printf("%.1f%s", (float)outA[c*SP+s], s<SP-1?",":"");
            printf("]");
        }
        printf("\n");
        // Copy output A
        _Float16 outA_copy[64];
        memcpy(outA_copy, outA, outBytes);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        // === Step 3: Overwrite weight file with B, unload+load ===
        printf("\n=== Step 3: Overwrite weight.bin with B (2x identity), unload+load ===\n");
        uint8_t *blobB = (uint8_t*)calloc(tot, 1);
        blobB[0]=1; blobB[4]=2; blobB[64]=0xEF; blobB[65]=0xBE; blobB[66]=0xAD; blobB[67]=0xDE; blobB[68]=1;
        *(uint32_t*)(blobB+72) = ws;
        *(uint32_t*)(blobB+80) = 128;
        memcpy(blobB + 128, weightsB, ws);
        NSData *wdataB = [NSData dataWithBytesNoCopy:blobB length:tot freeWhenDone:YES];

        NSString *weightPath = [td stringByAppendingPathComponent:@"weights/weight.bin"];
        [wdataB writeToFile:weightPath atomically:YES];
        printf("  Wrote new weight.bin (%d bytes)\n", tot);

        // Unload
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        printf("  Unload: %s (%.2fms)\n", ok ? "OK" : "FAIL", tb_ms(mach_absolute_time() - t0));

        // Reload (no compile!)
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("  Load: %s (%.2fms)\n", ok ? "OK" : [[e description] UTF8String], tb_ms(mach_absolute_time() - t0));

        if (!ok) {
            printf("\n*** Load-after-overwrite FAILED — trying compile+load ***\n");
            t0 = mach_absolute_time();
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            printf("  Re-compile: %s (%.2fms)\n", ok ? "OK" : "FAIL", tb_ms(mach_absolute_time() - t0));
            t0 = mach_absolute_time();
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("  Re-load: %s (%.2fms)\n", ok ? "OK" : "FAIL", tb_ms(mach_absolute_time() - t0));
        }

        // Need new request with new IOSurface objects (re-use same surfaces)
        wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        // Re-write same input
        IOSurfaceLock(ioIn, 0, NULL);
        inp = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < IC; c++)
            for (int s = 0; s < SP; s++)
                inp[c * SP + s] = (_Float16)(s + 1.0f);
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Eval with (possibly reloaded) weights B
        printf("\n=== Step 4: Eval after reload ===\n");
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) { printf("FAIL: eval after reload: %s\n", e ? [[e description] UTF8String] : "?"); return 1; }

        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *outB = (_Float16*)IOSurfaceGetBaseAddress(ioOut);
        printf("  Output B (2x identity @ [1,2,3,4]):");
        for (int c = 0; c < OC; c++) {
            printf(" [");
            for (int s = 0; s < SP; s++) printf("%.1f%s", (float)outB[c*SP+s], s<SP-1?",":"");
            printf("]");
        }
        printf("\n");

        // Check: did the output change?
        bool changed = false;
        for (int i = 0; i < OC * SP; i++) {
            if (fabsf((float)outB[i] - (float)outA_copy[i]) > 0.01f) { changed = true; break; }
        }
        // Expected output B should be 2x output A if weight reload worked
        bool correct = true;
        for (int i = 0; i < OC * SP; i++) {
            float expected = (float)outA_copy[i] * 2.0f;
            if (fabsf((float)outB[i] - expected) > 0.1f) { correct = false; break; }
        }
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        printf("\n=== RESULT ===\n");
        if (changed && correct) {
            printf("SUCCESS: Weight reload works! Output changed to match new weights.\n");
            printf(">>> Compilation bottleneck can be eliminated <<<\n");
        } else if (changed && !correct) {
            printf("PARTIAL: Output changed but doesn't match expected 2x. Weights may be partially updated.\n");
            printf("  Expected 2x of A, got different values.\n");
        } else {
            printf("FAIL: Output did NOT change. Weight reload does not work.\n");
            printf("  Output is still the same as weights A. ANE cached the compiled model.\n");
            printf(">>> Need alternative approach (weightsBuffer IOSurface or async recompile) <<<\n");
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        [fm removeItemAtPath:td error:nil];
        CFRelease(ioIn); CFRelease(ioOut);
    }
    return 0;
}
