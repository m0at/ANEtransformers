// ane_cache_probe.m — Test if ANE compiled programs can be cached and reloaded
// Key questions:
// 1. What files does compileWithQoS generate in the temp directory?
// 2. Can we skip compileWithQoS by restoring those files and calling loadWithQoS directly?
// 3. Does _ANEInMemoryModel have any compiled-model-related methods?
// 4. Does macOS cache compiled ANE programs in /var/folders/?
//
// Build: clang -framework Foundation -framework IOSurface -fobjc-arc -O2 -o ane_cache_probe tests/ane_cache_probe.m
// Run:   ./ane_cache_probe

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

#include "../training/ane_runtime.h"
#include "../mistral/ane_mil_gen_mistral.h"

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// List directory contents recursively with file sizes
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

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "ANE Cache Probe — Can we skip recompilation?\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        // ─── Step 1: Enumerate _ANEInMemoryModel methods ────────────────
        fprintf(stderr, "=== Step 1: _ANEInMemoryModel API ===\n");
        ane_init();

        fprintf(stderr, "\nClass methods:\n");
        unsigned int count;
        Method *methods = class_copyMethodList(object_getClass(g_ANEInMem), &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL s = method_getName(methods[i]);
            const char *enc = method_getTypeEncoding(methods[i]);
            fprintf(stderr, "  + %s  [%s]\n", sel_getName(s), enc ? enc : "?");
        }
        free(methods);

        fprintf(stderr, "\nInstance methods:\n");
        methods = class_copyMethodList(g_ANEInMem, &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL s = method_getName(methods[i]);
            const char *enc = method_getTypeEncoding(methods[i]);
            fprintf(stderr, "  - %s  [%s]\n", sel_getName(s), enc ? enc : "?");
        }
        free(methods);

        // Also check descriptor class
        fprintf(stderr, "\n_ANEInMemoryModelDescriptor class methods:\n");
        methods = class_copyMethodList(object_getClass(g_ANEDesc), &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL s = method_getName(methods[i]);
            fprintf(stderr, "  + %s\n", sel_getName(s));
        }
        free(methods);

        fprintf(stderr, "\n_ANEInMemoryModelDescriptor instance methods:\n");
        methods = class_copyMethodList(g_ANEDesc, &count);
        for (unsigned int i = 0; i < count; i++) {
            SEL s = method_getName(methods[i]);
            fprintf(stderr, "  - %s\n", sel_getName(s));
        }
        free(methods);

        // ─── Step 2: Compile a small program and inspect tmpDir ─────────
        fprintf(stderr, "\n=== Step 2: Compile and inspect temp directory ===\n");

        int in_ch = 256, out_ch = 256, S = 16;
        _Float16 *W = (_Float16 *)malloc((size_t)out_ch * in_ch * sizeof(_Float16));
        for (int i = 0; i < out_ch * in_ch; i++)
            W[i] = (_Float16)((float)arc4random() / (float)UINT32_MAX * 0.02f - 0.01f);

        NSData *blob = mil_build_single_weight_blob(W, out_ch, in_ch);
        NSString *mil = mil_gen_conv_baked(in_ch, out_ch, S);
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

        size_t inSz = (size_t)in_ch * S * sizeof(float);
        size_t outSz = (size_t)out_ch * S * sizeof(float);

        // Manual compile flow to capture tmpDir
        NSError *e = nil;
        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) { fprintf(stderr, "FATAL: descriptor creation failed\n"); return 1; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "FATAL: model creation failed\n"); return 1; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        fprintf(stderr, "hexStringIdentifier: %s\n", [hx UTF8String]);
        fprintf(stderr, "tmpDir: %s\n", [td UTF8String]);

        // Query model state before compile
        NSUInteger state = ((NSUInteger(*)(id,SEL))objc_msgSend)(mdl, @selector(state));
        fprintf(stderr, "State before compile: %lu\n", state);

        // Pre-populate temp dir
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        fprintf(stderr, "\n--- Before compile ---\n");
        list_dir(td, 1);

        // Compile
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        double compileMs = ticksToMs(mach_absolute_time() - t0);
        fprintf(stderr, "\ncompile: %s (%.1f ms)\n", ok ? "YES" : "NO", compileMs);
        if (e) fprintf(stderr, "  err: %s\n", [[e description] UTF8String]);
        if (!ok) return 1;

        state = ((NSUInteger(*)(id,SEL))objc_msgSend)(mdl, @selector(state));
        fprintf(stderr, "State after compile: %lu\n", state);

        fprintf(stderr, "\n--- After compile (before load) ---\n");
        list_dir(td, 1);

        // Load
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double loadMs = ticksToMs(mach_absolute_time() - t0);
        fprintf(stderr, "\nload: %s (%.1f ms)\n", ok ? "YES" : "NO", loadMs);
        if (e) fprintf(stderr, "  err: %s\n", [[e description] UTF8String]);
        if (!ok) return 1;

        state = ((NSUInteger(*)(id,SEL))objc_msgSend)(mdl, @selector(state));
        fprintf(stderr, "State after load: %lu\n", state);

        fprintf(stderr, "\n--- After load ---\n");
        list_dir(td, 1);

        // Quick eval to verify it works
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(inSz),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(inSz),
            (id)kIOSurfaceAllocSize:@(inSz),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(outSz),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(outSz),
            (id)kIOSurfaceAllocSize:@(outSz),(id)kIOSurfacePixelFormat:@0});

        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        fprintf(stderr, "\nOriginal eval: %s\n", ok ? "PASS" : "FAIL");

        // Unload
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);

        // ─── Step 3: Save compiled artifacts to a persistent cache ──────
        fprintf(stderr, "\n=== Step 3: Copy compiled artifacts to cache ===\n");
        NSString *cacheDir = @"/tmp/ane_cache_test";
        [fm removeItemAtPath:cacheDir error:nil];
        [fm copyItemAtPath:td toPath:cacheDir error:&e];
        if (e) {
            fprintf(stderr, "Copy failed: %s\n", [[e description] UTF8String]);
            e = nil;
        }
        fprintf(stderr, "Cached to: %s\n", [cacheDir UTF8String]);
        list_dir(cacheDir, 1);

        // Clean up original tmpDir
        [fm removeItemAtPath:td error:nil];

        // ─── Step 4: Try to load from cache without recompiling ─────────
        fprintf(stderr, "\n=== Step 4: Reload from cache (skip compile) ===\n");

        // Create a new model with the same MIL/weights
        id desc2 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        id mdl2 = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc2);

        id hx2 = ((id(*)(id,SEL))objc_msgSend)(mdl2, @selector(hexStringIdentifier));
        NSString *td2 = [NSTemporaryDirectory() stringByAppendingPathComponent:hx2];
        fprintf(stderr, "New model hexId: %s\n", [hx2 UTF8String]);
        fprintf(stderr, "Same as original? %s\n", [hx isEqual:hx2] ? "YES" : "NO");

        // Restore cached contents to the new tmpDir
        [fm removeItemAtPath:td2 error:nil];
        [fm copyItemAtPath:cacheDir toPath:td2 error:&e];
        if (e) {
            fprintf(stderr, "Restore failed: %s\n", [[e description] UTF8String]);
            e = nil;
        }
        fprintf(stderr, "Restored to: %s\n", [td2 UTF8String]);
        list_dir(td2, 1);

        // Try loading WITHOUT compiling
        fprintf(stderr, "\nAttempting loadWithQoS without compileWithQoS...\n");
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl2, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double reloadMs = ticksToMs(mach_absolute_time() - t0);
        fprintf(stderr, "load (from cache): %s (%.1f ms)\n", ok ? "YES" : "NO", reloadMs);
        if (e) { fprintf(stderr, "  err: %s\n", [[e description] UTF8String]); e = nil; }

        if (ok) {
            state = ((NSUInteger(*)(id,SEL))objc_msgSend)(mdl2, @selector(state));
            fprintf(stderr, "State: %lu\n", state);

            // Try eval
            IOSurfaceRef ioIn2 = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                (id)kIOSurfaceWidth:@(inSz),(id)kIOSurfaceHeight:@1,
                (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(inSz),
                (id)kIOSurfaceAllocSize:@(inSz),(id)kIOSurfacePixelFormat:@0});
            IOSurfaceRef ioOut2 = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                (id)kIOSurfaceWidth:@(outSz),(id)kIOSurfaceHeight:@1,
                (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(outSz),
                (id)kIOSurfaceAllocSize:@(outSz),(id)kIOSurfacePixelFormat:@0});

            id wIn2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn2);
            id wOut2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut2);
            id req2 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wIn2], @[@0], @[wOut2], @[@0], nil, nil, @0);

            // Write input data
            IOSurfaceLock(ioIn2, 0, NULL);
            float *inp = (float *)IOSurfaceGetBaseAddress(ioIn2);
            for (int i = 0; i < in_ch * S; i++) inp[i] = 1.0f;
            IOSurfaceUnlock(ioIn2, 0, NULL);

            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req2, &e);
            fprintf(stderr, "Cached eval: %s\n", ok ? "PASS" : "FAIL");
            if (e) { fprintf(stderr, "  err: %s\n", [[e description] UTF8String]); e = nil; }

            if (ok) {
                IOSurfaceLock(ioOut2, kIOSurfaceLockReadOnly, NULL);
                float *out = (float *)IOSurfaceGetBaseAddress(ioOut2);
                float sum = 0;
                for (int i = 0; i < out_ch * S && i < 100; i++) sum += fabsf(out[i]);
                IOSurfaceUnlock(ioOut2, kIOSurfaceLockReadOnly, NULL);
                fprintf(stderr, "Output check: sum(|y[0:100]|)=%.6f %s\n", sum, sum > 0 ? "NON-ZERO" : "ZERO!");
            }

            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl2, @selector(unloadWithQoS:error:), 21, &e);
            CFRelease(ioIn2); CFRelease(ioOut2);
        }

        // ─── Step 4b: If load-only failed, try compile+load with cached dir ──
        if (!ok) {
            fprintf(stderr, "\n=== Step 4b: Try compile with cached dir pre-populated ===\n");
            fprintf(stderr, "(The compile should be instant if the compiler detects existing output)\n");

            id desc3 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                milData, wdict, nil);
            id mdl3 = ((id(*)(Class,SEL,id))objc_msgSend)(
                g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc3);
            id hx3 = ((id(*)(id,SEL))objc_msgSend)(mdl3, @selector(hexStringIdentifier));
            NSString *td3 = [NSTemporaryDirectory() stringByAppendingPathComponent:hx3];

            // Restore cached contents
            [fm removeItemAtPath:td3 error:nil];
            [fm copyItemAtPath:cacheDir toPath:td3 error:&e];
            if (e) { fprintf(stderr, "Restore failed\n"); e = nil; }

            // Compile (should it be fast with cached artifacts?)
            t0 = mach_absolute_time();
            BOOL ok3 = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl3, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            double recompMs = ticksToMs(mach_absolute_time() - t0);
            fprintf(stderr, "compile (with cached dir): %s (%.1f ms vs original %.1f ms)\n",
                    ok3 ? "YES" : "NO", recompMs, compileMs);
            if (e) { fprintf(stderr, "  err: %s\n", [[e description] UTF8String]); e = nil; }

            if (ok3) {
                t0 = mach_absolute_time();
                ok3 = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl3, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                double ld = ticksToMs(mach_absolute_time() - t0);
                fprintf(stderr, "load: %s (%.1f ms)\n", ok3 ? "YES" : "NO", ld);

                if (ok3) {
                    fprintf(stderr, "Speedup from cached dir: %.2fx\n", compileMs / recompMs);
                }

                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl3, @selector(unloadWithQoS:error:), 21, &e);
            }

            [fm removeItemAtPath:td3 error:nil];
        }

        // ─── Step 5: Check /var/folders for ANE caches ──────────────────
        fprintf(stderr, "\n=== Step 5: System ANE caches ===\n");

        // Check common cache locations
        NSString *varFolders = @"/var/folders";
        NSString *tmpBase = NSTemporaryDirectory();
        fprintf(stderr, "NSTemporaryDirectory: %s\n", [tmpBase UTF8String]);

        // Look for ANE-related dirs in the user's temp/cache area
        NSString *userTmp = [tmpBase stringByDeletingLastPathComponent]; // parent of /tmp/ subfolder
        fprintf(stderr, "Scanning %s for ANE-related files...\n", [userTmp UTF8String]);

        NSDirectoryEnumerator *en = [fm enumeratorAtPath:userTmp];
        en.skipDescendants; // don't recurse deeply
        NSString *f;
        int found = 0;
        while ((f = [en nextObject]) && found < 50) {
            NSString *lower = [f lowercaseString];
            if ([lower containsString:@"ane"] || [lower containsString:@"neural"] ||
                [lower containsString:@"espresso"] || [lower containsString:@"coreml"]) {
                NSString *full = [userTmp stringByAppendingPathComponent:f];
                BOOL isDir;
                [fm fileExistsAtPath:full isDirectory:&isDir];
                fprintf(stderr, "  %s%s\n", [f UTF8String], isDir ? "/" : "");
                found++;
            }
        }
        if (found == 0) fprintf(stderr, "  (none found)\n");

        // Also check ~/Library/Caches for ANE
        NSString *cachesDir = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject];
        fprintf(stderr, "\nScanning %s for ANE caches...\n", [cachesDir UTF8String]);
        en = [fm enumeratorAtPath:cachesDir];
        found = 0;
        while ((f = [en nextObject]) && found < 30) {
            NSString *lower = [f lowercaseString];
            if ([lower containsString:@"ane"] || [lower containsString:@"neural"] ||
                [lower containsString:@"espresso"] || [lower containsString:@"coreml"]) {
                fprintf(stderr, "  %s\n", [f UTF8String]);
                found++;
            }
        }
        if (found == 0) fprintf(stderr, "  (none found)\n");

        // ─── Step 6: Check _ANEModel (non-InMemory) for cache methods ──
        fprintf(stderr, "\n=== Step 6: _ANEModel class (non-InMemory) ===\n");
        Class aneModel = NSClassFromString(@"_ANEModel");
        if (aneModel) {
            fprintf(stderr, "Class methods:\n");
            methods = class_copyMethodList(object_getClass(aneModel), &count);
            for (unsigned int i = 0; i < count; i++)
                fprintf(stderr, "  + %s\n", sel_getName(method_getName(methods[i])));
            free(methods);

            fprintf(stderr, "Instance methods:\n");
            methods = class_copyMethodList(aneModel, &count);
            for (unsigned int i = 0; i < count; i++)
                fprintf(stderr, "  - %s\n", sel_getName(method_getName(methods[i])));
            free(methods);
        } else {
            fprintf(stderr, "  _ANEModel NOT FOUND\n");
        }

        // ─── Step 7: Check _ANECompiler class ───────────────────────────
        fprintf(stderr, "\n=== Step 7: _ANECompiler / related classes ===\n");
        NSArray *classNames = @[@"_ANECompiler", @"_ANEClient", @"_ANEDeviceController",
                                @"_ANEDaemonConnection", @"_ANEModelCacheEntry",
                                @"_ANECompiledModel", @"_ANEProgram", @"_ANEProgramCache"];
        for (NSString *cn in classNames) {
            Class cls = NSClassFromString(cn);
            if (cls) {
                fprintf(stderr, "\n%s:\n", [cn UTF8String]);
                methods = class_copyMethodList(object_getClass(cls), &count);
                for (unsigned int i = 0; i < count; i++)
                    fprintf(stderr, "  + %s\n", sel_getName(method_getName(methods[i])));
                free(methods);
                methods = class_copyMethodList(cls, &count);
                for (unsigned int i = 0; i < count; i++)
                    fprintf(stderr, "  - %s\n", sel_getName(method_getName(methods[i])));
                free(methods);
            }
        }

        // Cleanup
        [fm removeItemAtPath:cacheDir error:nil];
        [fm removeItemAtPath:td error:nil];
        free(W);
        CFRelease(ioIn); CFRelease(ioOut);

        fprintf(stderr, "\n═══════════════════════════════════════════\n");
        fprintf(stderr, "Done.\n");
    }
    return 0;
}
