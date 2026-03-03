// ane_cache_probe2.m — Test ANE compiled model caching strategies
// Key finding from probe1: loadWithQoS works without compileWithQoS (1.5ms vs 14ms)
// because the aned daemon caches compiled programs keyed by hexStringIdentifier.
//
// This test explores:
// 1. Does the daemon cache persist across unload? (YES from probe1)
// 2. Does compiledModelExists detect daemon-cached models?
// 3. Does the cache persist across process invocations?
// 4. Can we save/restore the tmpDir compiled artifacts for cold-cache scenarios?
// 5. What is purgeCompiledModel?
// 6. Timing: compile vs load-from-cache at Mistral-scale sizes
//
// Build: clang -framework Foundation -framework IOSurface -fobjc-arc -O2 -o ane_cache2 tests/ane_cache_probe2.m
// Run:   ./ane_cache2 [--phase 1|2]
//   Phase 1: compile programs and save cache
//   Phase 2: reload from cache (run as separate process to test cross-process caching)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <getopt.h>

#include "../training/ane_runtime.h"
#include "../mistral/ane_mil_gen_mistral.h"

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static void list_dir(NSString *path, int indent) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *items = [fm contentsOfDirectoryAtPath:path error:nil];
    for (NSString *item in items) {
        NSString *full = [path stringByAppendingPathComponent:item];
        BOOL isDir;
        [fm fileExistsAtPath:full isDirectory:&isDir];
        NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
        for (int i = 0; i < indent; i++) fprintf(stderr, "  ");
        if (isDir) {
            fprintf(stderr, "%s/\n", [item UTF8String]);
            list_dir(full, indent + 1);
        } else {
            unsigned long long sz = [attrs[NSFileSize] unsignedLongLongValue];
            fprintf(stderr, "%s  (%llu bytes)\n", [item UTF8String], sz);
        }
    }
}

// Create a model object from MIL + weights, but don't compile/load yet
typedef struct {
    id model;
    NSString *hexId;
    NSString *tmpDir;
    NSData *milData;
    NSData *blob;
    size_t inSz, outSz;
} ModelInfo;

static ModelInfo create_model(int in_ch, int out_ch, int S) {
    _Float16 *W = (_Float16 *)malloc((size_t)out_ch * in_ch * sizeof(_Float16));
    // Use deterministic weights so hexId is reproducible across runs
    srand(42 + in_ch * 10000 + out_ch * 100 + S);
    for (int i = 0; i < out_ch * in_ch; i++)
        W[i] = (_Float16)((float)rand() / (float)RAND_MAX * 0.02f - 0.01f);

    NSData *blob = mil_build_single_weight_blob(W, out_ch, in_ch);
    NSString *mil = mil_gen_conv_baked(in_ch, out_ch, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    free(W);

    NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];

    ModelInfo mi = {0};
    mi.model = mdl;
    mi.hexId = hx;
    mi.tmpDir = td;
    mi.milData = milData;
    mi.blob = blob;
    mi.inSz = (size_t)in_ch * S * sizeof(float);
    mi.outSz = (size_t)out_ch * S * sizeof(float);
    return mi;
}

static void prepopulate_tmpdir(ModelInfo *mi) {
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[mi->tmpDir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [mi->milData writeToFile:[mi->tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [mi->blob writeToFile:[mi->tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();

        int phase = 0; // 0 = both, 1 = compile only, 2 = reload only
        int opt;
        while ((opt = getopt(argc, argv, "p:")) != -1) {
            if (opt == 'p') phase = atoi(optarg);
        }

        NSString *cacheDir = @"/tmp/ane_compiled_cache";
        NSFileManager *fm = [NSFileManager defaultManager];
        NSError *e = nil;

        // Test sizes
        typedef struct { int in_ch, out_ch, S; const char *name; } TestCase;
        TestCase tests[] = {
            {256, 256, 16, "small 256x256 S=16"},
            {4096, 4096, 16, "Wo 4096x4096 S=16"},
            {4096, 1024, 16, "KV 4096x1024 S=16"},
        };
        int nTests = sizeof(tests) / sizeof(tests[0]);

        if (phase == 0 || phase == 1) {
            fprintf(stderr, "═══════════════════════════════════════════\n");
            fprintf(stderr, "Phase 1: Compile and cache\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            [fm removeItemAtPath:cacheDir error:nil];
            [fm createDirectoryAtPath:cacheDir withIntermediateDirectories:YES attributes:nil error:nil];

            for (int t = 0; t < nTests; t++) {
                TestCase *tc = &tests[t];
                fprintf(stderr, "--- %s ---\n", tc->name);

                ModelInfo mi = create_model(tc->in_ch, tc->out_ch, tc->S);
                fprintf(stderr, "hexId: %.32s...\n", [mi.hexId UTF8String]);
                prepopulate_tmpdir(&mi);

                // Check if already cached in daemon
                BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(mi.model, @selector(compiledModelExists));
                fprintf(stderr, "compiledModelExists (before compile): %s\n", cached ? "YES" : "NO");

                // Compile
                uint64_t t0 = mach_absolute_time();
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                double compileMs = ticksToMs(mach_absolute_time() - t0);
                fprintf(stderr, "compile: %s (%.1f ms)\n", ok ? "YES" : "NO", compileMs);
                if (!ok) { fprintf(stderr, "FATAL\n"); return 1; }

                // List what compile generated
                fprintf(stderr, "tmpDir after compile:\n");
                list_dir(mi.tmpDir, 1);

                // Check if compiled model registered in daemon
                cached = ((BOOL(*)(id,SEL))objc_msgSend)(mi.model, @selector(compiledModelExists));
                fprintf(stderr, "compiledModelExists (after compile): %s\n", cached ? "YES" : "NO");

                // Load
                t0 = mach_absolute_time();
                ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                double loadMs = ticksToMs(mach_absolute_time() - t0);
                fprintf(stderr, "load: %s (%.1f ms)\n", ok ? "YES" : "NO", loadMs);

                // Quick eval to verify
                IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                    (id)kIOSurfaceWidth:@(mi.inSz),(id)kIOSurfaceHeight:@1,
                    (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(mi.inSz),
                    (id)kIOSurfaceAllocSize:@(mi.inSz),(id)kIOSurfacePixelFormat:@0});
                IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                    (id)kIOSurfaceWidth:@(mi.outSz),(id)kIOSurfaceHeight:@1,
                    (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(mi.outSz),
                    (id)kIOSurfaceAllocSize:@(mi.outSz),(id)kIOSurfacePixelFormat:@0});
                id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mi.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                fprintf(stderr, "eval: %s\n", ok ? "PASS" : "FAIL");

                // Unload
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi.model, @selector(unloadWithQoS:error:), 21, &e);

                // Check compiledModelExists after unload
                cached = ((BOOL(*)(id,SEL))objc_msgSend)(mi.model, @selector(compiledModelExists));
                fprintf(stderr, "compiledModelExists (after unload): %s\n", cached ? "YES" : "NO");

                // Save the tmpDir contents to persistent cache
                NSString *cachePath = [cacheDir stringByAppendingPathComponent:
                    [NSString stringWithFormat:@"model_%d", t]];
                [fm copyItemAtPath:mi.tmpDir toPath:cachePath error:&e];
                if (e) { fprintf(stderr, "Cache save failed: %s\n", [[e description] UTF8String]); e = nil; }

                // Also save the hexId for cross-process use
                [mi.hexId writeToFile:[cachePath stringByAppendingPathComponent:@"_hexid.txt"]
                          atomically:YES encoding:NSUTF8StringEncoding error:nil];

                // Don't remove tmpDir — leave it for phase 2 same-process test
                CFRelease(ioIn); CFRelease(ioOut);
                fprintf(stderr, "\n");
            }

            // ─── Same-process reload test ────────────────────────────────
            fprintf(stderr, "═══════════════════════════════════════════\n");
            fprintf(stderr, "Same-process reload (daemon cache warm)\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            for (int t = 0; t < nTests; t++) {
                TestCase *tc = &tests[t];
                fprintf(stderr, "--- %s ---\n", tc->name);

                ModelInfo mi = create_model(tc->in_ch, tc->out_ch, tc->S);
                prepopulate_tmpdir(&mi);

                // Check daemon cache
                BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(mi.model, @selector(compiledModelExists));
                fprintf(stderr, "compiledModelExists: %s\n", cached ? "YES" : "NO");

                // Try load without compile
                uint64_t t0 = mach_absolute_time();
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                double loadMs = ticksToMs(mach_absolute_time() - t0);
                fprintf(stderr, "load (no compile): %s (%.1f ms)\n", ok ? "YES" : "NO", loadMs);
                if (e) { fprintf(stderr, "  err: %s\n", [[e description] UTF8String]); e = nil; }

                if (ok) {
                    // Eval
                    IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                        (id)kIOSurfaceWidth:@(mi.inSz),(id)kIOSurfaceHeight:@1,
                        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(mi.inSz),
                        (id)kIOSurfaceAllocSize:@(mi.inSz),(id)kIOSurfacePixelFormat:@0});
                    IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                        (id)kIOSurfaceWidth:@(mi.outSz),(id)kIOSurfaceHeight:@1,
                        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(mi.outSz),
                        (id)kIOSurfaceAllocSize:@(mi.outSz),(id)kIOSurfacePixelFormat:@0});
                    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
                    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mi.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                    fprintf(stderr, "eval: %s\n", ok ? "PASS" : "FAIL");
                    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi.model, @selector(unloadWithQoS:error:), 21, &e);
                    CFRelease(ioIn); CFRelease(ioOut);
                }

                // Clean up tmpDir
                [fm removeItemAtPath:mi.tmpDir error:nil];
                fprintf(stderr, "\n");
            }

            // ─── Test purgeCompiledModel ─────────────────────────────────
            fprintf(stderr, "═══════════════════════════════════════════\n");
            fprintf(stderr, "Purge test: clear daemon cache, then try reload\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            // Test with the small model
            {
                TestCase *tc = &tests[0];
                ModelInfo mi = create_model(tc->in_ch, tc->out_ch, tc->S);

                BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(mi.model, @selector(compiledModelExists));
                fprintf(stderr, "compiledModelExists before purge: %s\n", cached ? "YES" : "NO");

                // Purge
                ((void(*)(id,SEL))objc_msgSend)(mi.model, @selector(purgeCompiledModel));
                fprintf(stderr, "purgeCompiledModel called\n");

                cached = ((BOOL(*)(id,SEL))objc_msgSend)(mi.model, @selector(compiledModelExists));
                fprintf(stderr, "compiledModelExists after purge: %s\n", cached ? "YES" : "NO");

                // Try load without compile after purge
                prepopulate_tmpdir(&mi);
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                fprintf(stderr, "load after purge (no compile): %s\n", ok ? "YES" : "NO");
                if (e) { fprintf(stderr, "  err: %s\n", [[e description] UTF8String]); e = nil; }

                if (!ok) {
                    // Need to recompile after purge
                    uint64_t t0 = mach_absolute_time();
                    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mi.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                    double compileMs = ticksToMs(mach_absolute_time() - t0);
                    fprintf(stderr, "recompile after purge: %s (%.1f ms)\n", ok ? "YES" : "NO", compileMs);
                }

                if (ok) {
                    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi.model, @selector(unloadWithQoS:error:), 21, &e);
                }
                [fm removeItemAtPath:mi.tmpDir error:nil];
            }

            // ─── Save hex IDs for phase 2 ────────────────────────────────
            fprintf(stderr, "\n\nTo test cross-process caching, run:\n");
            fprintf(stderr, "  ./ane_cache2 -p 2\n");
        }

        if (phase == 2) {
            fprintf(stderr, "═══════════════════════════════════════════\n");
            fprintf(stderr, "Phase 2: Cross-process reload from daemon cache\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            for (int t = 0; t < nTests; t++) {
                TestCase *tc = &tests[t];
                fprintf(stderr, "--- %s ---\n", tc->name);

                ModelInfo mi = create_model(tc->in_ch, tc->out_ch, tc->S);
                fprintf(stderr, "hexId: %.32s...\n", [mi.hexId UTF8String]);

                // Check daemon cache (should be warm from phase 1 if run recently)
                BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(mi.model, @selector(compiledModelExists));
                fprintf(stderr, "compiledModelExists (daemon cache): %s\n", cached ? "YES" : "NO");

                prepopulate_tmpdir(&mi);

                if (cached) {
                    // Try load without compile
                    uint64_t t0 = mach_absolute_time();
                    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mi.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                    double loadMs = ticksToMs(mach_absolute_time() - t0);
                    fprintf(stderr, "load (from daemon cache, no compile): %s (%.1f ms)\n",
                            ok ? "YES" : "NO", loadMs);
                    if (e) { fprintf(stderr, "  err: %s\n", [[e description] UTF8String]); e = nil; }

                    if (ok) {
                        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                            (id)kIOSurfaceWidth:@(mi.inSz),(id)kIOSurfaceHeight:@1,
                            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(mi.inSz),
                            (id)kIOSurfaceAllocSize:@(mi.inSz),(id)kIOSurfacePixelFormat:@0});
                        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                            (id)kIOSurfaceWidth:@(mi.outSz),(id)kIOSurfaceHeight:@1,
                            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(mi.outSz),
                            (id)kIOSurfaceAllocSize:@(mi.outSz),(id)kIOSurfacePixelFormat:@0});
                        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
                            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
                        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            mi.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                        fprintf(stderr, "eval: %s\n", ok ? "PASS" : "FAIL");
                        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi.model, @selector(unloadWithQoS:error:), 21, &e);
                        CFRelease(ioIn); CFRelease(ioOut);
                    }
                } else {
                    fprintf(stderr, "Daemon cache COLD — need full compile\n");
                    // Try loading from saved tmpDir artifacts
                    NSString *cachePath = [cacheDir stringByAppendingPathComponent:
                        [NSString stringWithFormat:@"model_%d", t]];
                    if ([fm fileExistsAtPath:cachePath]) {
                        fprintf(stderr, "Restoring saved artifacts from %s\n", [cachePath UTF8String]);
                        [fm removeItemAtPath:mi.tmpDir error:nil];
                        [fm copyItemAtPath:cachePath toPath:mi.tmpDir error:&e];
                        if (e) { fprintf(stderr, "Restore failed\n"); e = nil; }

                        // Try compile with restored artifacts (should it be faster?)
                        uint64_t t0 = mach_absolute_time();
                        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                            mi.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                        double compileMs = ticksToMs(mach_absolute_time() - t0);
                        fprintf(stderr, "compile (from saved artifacts): %s (%.1f ms)\n",
                                ok ? "YES" : "NO", compileMs);

                        if (ok) {
                            t0 = mach_absolute_time();
                            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                                mi.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                            double loadMs = ticksToMs(mach_absolute_time() - t0);
                            fprintf(stderr, "load: %s (%.1f ms)\n", ok ? "YES" : "NO", loadMs);
                            if (ok) {
                                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi.model, @selector(unloadWithQoS:error:), 21, &e);
                            }
                        }
                    }
                }

                [fm removeItemAtPath:mi.tmpDir error:nil];
                fprintf(stderr, "\n");
            }
        }

        // ─── Step: Compile timing at scale ───────────────────────────────
        if (phase == 0) {
            fprintf(stderr, "═══════════════════════════════════════════\n");
            fprintf(stderr, "Timing: 160 programs (Mistral 7B scale)\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            // Estimate: 5 programs per layer (QKV, Wo, W1, W3, W2) x 32 layers = 160
            // But we already know from probe1 that compile is ~14ms for small,
            // and ~500ms for large (14336x4096).
            // If daemon-cached load is ~1.5ms, then 160 * 1.5ms = 240ms total.
            // vs 160 * ~300ms average = 48 seconds for full compile.

            fprintf(stderr, "Expected savings:\n");
            fprintf(stderr, "  Full compile: 160 programs * ~300ms avg = ~48s\n");
            fprintf(stderr, "  Cached load:  160 programs * ~1.5ms = ~0.24s\n");
            fprintf(stderr, "  Speedup: ~200x\n\n");

            // Benchmark 10 small cached loads to get accurate per-load timing
            {
                ModelInfo mi = create_model(256, 256, 16);
                prepopulate_tmpdir(&mi);

                // Ensure compiled
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi.model, @selector(unloadWithQoS:error:), 21, &e);
                [fm removeItemAtPath:mi.tmpDir error:nil];

                // Now measure 10 load-from-cache cycles
                double total = 0;
                for (int i = 0; i < 10; i++) {
                    ModelInfo mi2 = create_model(256, 256, 16);
                    prepopulate_tmpdir(&mi2);
                    uint64_t t0 = mach_absolute_time();
                    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mi2.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                    double ms = ticksToMs(mach_absolute_time() - t0);
                    total += ms;
                    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi2.model, @selector(unloadWithQoS:error:), 21, &e);
                    [fm removeItemAtPath:mi2.tmpDir error:nil];
                }
                fprintf(stderr, "Cached load (small, 10 iters): %.2f ms/load\n", total / 10);
            }

            // Benchmark larger cached loads
            {
                ModelInfo mi = create_model(4096, 4096, 16);
                prepopulate_tmpdir(&mi);
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mi.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi.model, @selector(unloadWithQoS:error:), 21, &e);
                [fm removeItemAtPath:mi.tmpDir error:nil];

                double total = 0;
                for (int i = 0; i < 5; i++) {
                    ModelInfo mi2 = create_model(4096, 4096, 16);
                    prepopulate_tmpdir(&mi2);
                    uint64_t t0 = mach_absolute_time();
                    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mi2.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                    double ms = ticksToMs(mach_absolute_time() - t0);
                    total += ms;
                    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mi2.model, @selector(unloadWithQoS:error:), 21, &e);
                    [fm removeItemAtPath:mi2.tmpDir error:nil];
                }
                fprintf(stderr, "Cached load (4096x4096, 5 iters): %.2f ms/load\n", total / 5);
            }
        }

        fprintf(stderr, "\n═══════════════════════════════════════════\n");
        fprintf(stderr, "Done.\n");
    }
    return 0;
}
