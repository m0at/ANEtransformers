// ane_blob_cache_probe.m — Test pre-materialized fp16 blob caching
//
// Problem: per-layer ANE compile does dequant(~100ms) + blob_build(~50ms) + SHA256(~100ms)
// on every prefill because we can't keep 14GB of NSData in memory.
//
// Solution: save DEADBEEF blobs to ~/.cache/ane_mistral/ on first run, mmap on reload.
// SHA256 is still computed (can't avoid it), but dequant+blob_build are eliminated.
//
// This probe tests with the FFN blob (352MB, worst case) using synthetic weights.
//
// Build: clang -framework Foundation -framework IOSurface -framework Accelerate -fobjc-arc -O2 \
//        -o ane_blob_cache_probe tests/ane_blob_cache_probe.m
// Run:   ./ane_blob_cache_probe          (first run: build+save+time)
//        ./ane_blob_cache_probe          (second run: mmap+time)
//        ./ane_blob_cache_probe --clean  (delete cache)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import <fcntl.h>

#include "../training/ane_runtime.h"
#include "../mistral/ane_mil_gen_mistral.h"

static mach_timebase_info_data_t g_tb;
static double ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// ─── Blob disk cache ────────────────────────────────────────────────────────

static NSString *g_cache_dir = nil;

static NSString *cache_dir(void) {
    if (!g_cache_dir) {
        NSString *home = NSHomeDirectory();
        g_cache_dir = [home stringByAppendingPathComponent:@".cache/ane_mistral"];
    }
    return g_cache_dir;
}

// Save a blob + its hexId to disk
static bool blob_cache_save(NSData *blob, NSData *milData, NSString *name) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *dir = [cache_dir() stringByAppendingPathComponent:name];
    [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];

    NSString *blobPath = [dir stringByAppendingPathComponent:@"weight.bin"];
    NSString *milPath = [dir stringByAppendingPathComponent:@"model.mil"];

    // Save blob
    if (![blob writeToFile:blobPath atomically:YES]) {
        fprintf(stderr, "Failed to save blob to %s\n", [blobPath UTF8String]);
        return false;
    }

    // Save MIL text
    [milData writeToFile:milPath atomically:YES];

    // Create model to get hexId, save it
    NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));

    NSString *hexPath = [dir stringByAppendingPathComponent:@"hexid.txt"];
    [(NSString*)hx writeToFile:hexPath atomically:YES encoding:NSUTF8StringEncoding error:nil];

    unsigned long long sz = [[fm attributesOfItemAtPath:blobPath error:nil][NSFileSize] unsignedLongLongValue];
    fprintf(stderr, "  Saved: %s (%.1f MB, hexId=%.16s...)\n",
            [name UTF8String], sz / 1e6, [(NSString*)hx UTF8String]);
    return true;
}

// Load blob via mmap, return NSData (mapped, not copied)
static NSData *blob_cache_load_mmap(NSString *name) {
    NSString *blobPath = [[cache_dir() stringByAppendingPathComponent:name]
                           stringByAppendingPathComponent:@"weight.bin"];
    return [NSData dataWithContentsOfFile:blobPath
                                 options:NSDataReadingMappedIfSafe
                                   error:nil];
}

// Load MIL text
static NSData *blob_cache_load_mil(NSString *name) {
    NSString *milPath = [[cache_dir() stringByAppendingPathComponent:name]
                          stringByAppendingPathComponent:@"model.mil"];
    return [NSData dataWithContentsOfFile:milPath];
}

// Load saved hexId
static NSString *blob_cache_load_hexid(NSString *name) {
    NSString *hexPath = [[cache_dir() stringByAppendingPathComponent:name]
                          stringByAppendingPathComponent:@"hexid.txt"];
    return [NSString stringWithContentsOfFile:hexPath encoding:NSUTF8StringEncoding error:nil];
}

// Check if cached
static bool blob_cache_exists(NSString *name) {
    NSString *blobPath = [[cache_dir() stringByAppendingPathComponent:name]
                           stringByAppendingPathComponent:@"weight.bin"];
    return [[NSFileManager defaultManager] fileExistsAtPath:blobPath];
}

// ─── Test helpers ───────────────────────────────────────────────────────────

// Build FFN blob from synthetic weights (deterministic for reproducible hexId)
static void fill_synthetic_fp16(_Float16 *buf, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        buf[i] = (_Float16)((float)rand() / (float)RAND_MAX * 0.02f - 0.01f);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();

        if (argc > 1 && strcmp(argv[1], "--clean") == 0) {
            [[NSFileManager defaultManager] removeItemAtPath:cache_dir() error:nil];
            fprintf(stderr, "Cleaned cache: %s\n", [cache_dir() UTF8String]);
            return 0;
        }

        int dim = 4096;
        int hidden = 14336;
        int S = 64;

        NSString *mil_ffn_str = mil_gen_ffn_fused(dim, hidden, S);
        NSData *milData = [mil_ffn_str dataUsingEncoding:NSUTF8StringEncoding];

        size_t in_sz = (size_t)dim * S * sizeof(float);
        size_t out_sz = (size_t)dim * S * sizeof(float);

        NSString *blobName = @"L00_ffn";

        // ═══════════════════════════════════════════════════════════════
        // PATH A: Blob cached on disk → mmap load
        // ═══════════════════════════════════════════════════════════════
        if (blob_cache_exists(blobName)) {
            fprintf(stderr, "═══════════════════════════════════════════\n");
            fprintf(stderr, "CACHED PATH: mmap blob from disk\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            // 1. mmap blob
            uint64_t t0 = mach_absolute_time();
            NSData *blob = blob_cache_load_mmap(blobName);
            double t_mmap = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  mmap blob: %.2f ms (%.1f MB)\n", t_mmap, blob.length / 1e6);

            // 2. Load MIL
            t0 = mach_absolute_time();
            NSData *mil = blob_cache_load_mil(blobName);
            double t_mil = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  load MIL:  %.2f ms\n", t_mil);

            // 3. Create descriptor + model (triggers hexStringIdentifier/SHA256)
            t0 = mach_absolute_time();
            NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                mil, wdict, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
                g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
            double t_model = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  create model: %.2f ms\n", t_model);

            // 4. Get hexId (may trigger SHA256 computation)
            t0 = mach_absolute_time();
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            double t_hex = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  hexStringIdentifier: %.2f ms\n", t_hex);

            // Verify hexId matches saved
            NSString *saved_hex = blob_cache_load_hexid(blobName);
            bool match = [(NSString*)hx isEqualToString:saved_hex];
            fprintf(stderr, "  hexId match: %s\n", match ? "YES" : "NO");
            if (!match) {
                fprintf(stderr, "  FATAL: hexId mismatch! got=%.32s saved=%.32s\n",
                        [(NSString*)hx UTF8String], [saved_hex UTF8String]);
                return 1;
            }

            // 5. Check daemon cache
            t0 = mach_absolute_time();
            BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(mdl, @selector(compiledModelExists));
            double t_check = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  compiledModelExists: %s (%.2f ms)\n", cached ? "YES" : "NO", t_check);

            // 6. Populate tmpDir
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:(NSString*)hx];
            NSFileManager *fm = [NSFileManager defaultManager];
            t0 = mach_absolute_time();
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
            double t_tmpdir = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  populate tmpDir: %.2f ms\n", t_tmpdir);

            // 7. Compile if needed, then load
            NSError *e = nil;
            double t_compile = 0;
            if (!cached) {
                t0 = mach_absolute_time();
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                t_compile = ms(mach_absolute_time() - t0);
                fprintf(stderr, "  compile: %s (%.1f ms)\n", ok ? "YES" : "NO", t_compile);
                if (!ok) { fprintf(stderr, "FATAL: %s\n", [[e description] UTF8String]); return 1; }
            }

            t0 = mach_absolute_time();
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            double t_load = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  load: %s (%.2f ms)\n", ok ? "YES" : "NO", t_load);

            // 8. Quick eval
            if (ok) {
                IOSurfaceRef ioIn = ane_create_surface(in_sz);
                IOSurfaceRef ioOut = ane_create_surface(out_sz);
                id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

                t0 = mach_absolute_time();
                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                double t_eval = ms(mach_absolute_time() - t0);
                fprintf(stderr, "  eval: %s (%.2f ms)\n", ok ? "PASS" : "FAIL", t_eval);

                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
                CFRelease(ioIn); CFRelease(ioOut);
            }

            double total_cached = t_mmap + t_mil + t_model + t_hex + t_check + t_tmpdir + t_compile + t_load;
            fprintf(stderr, "\n  TOTAL (cached path): %.1f ms\n", total_cached);
            fprintf(stderr, "    (excl compile: %.1f ms)\n", total_cached - t_compile);

            // Also test: what if we skip model creation and just use saved hexId?
            fprintf(stderr, "\n═══════════════════════════════════════════\n");
            fprintf(stderr, "BYPASS TEST: use saved hexId, skip SHA256\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            // Can we create a model from mmap'd blob, set up tmpDir from saved hexId,
            // and call loadWithQoS without computing hexStringIdentifier ourselves?
            t0 = mach_absolute_time();
            NSData *blob2 = blob_cache_load_mmap(blobName);
            NSData *mil2 = blob_cache_load_mil(blobName);
            NSDictionary *wdict2 = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob2}};
            id desc2 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                mil2, wdict2, nil);
            id mdl2 = ((id(*)(Class,SEL,id))objc_msgSend)(
                g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc2);
            double t_create2 = ms(mach_absolute_time() - t0);

            // Use saved hexId to populate tmpDir WITHOUT calling hexStringIdentifier
            t0 = mach_absolute_time();
            NSString *td2 = [NSTemporaryDirectory() stringByAppendingPathComponent:saved_hex];
            [fm createDirectoryAtPath:[td2 stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [mil2 writeToFile:[td2 stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [blob2 writeToFile:[td2 stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
            double t_tmpdir2 = ms(mach_absolute_time() - t0);

            // Try loadWithQoS — internally it will compute hexStringIdentifier
            t0 = mach_absolute_time();
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl2, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            double t_load2 = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  create model: %.2f ms\n", t_create2);
            fprintf(stderr, "  populate tmpDir (saved hexId): %.2f ms\n", t_tmpdir2);
            fprintf(stderr, "  loadWithQoS: %s (%.2f ms)\n", ok ? "YES" : "NO", t_load2);
            if (e) fprintf(stderr, "    err: %s\n", [[e description] UTF8String]);

            if (ok) {
                fprintf(stderr, "  BYPASS TOTAL: %.1f ms\n", t_create2 + t_tmpdir2 + t_load2);
                fprintf(stderr, "  NOTE: loadWithQoS internally computes hexStringIdentifier (SHA256)\n");
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl2, @selector(unloadWithQoS:error:), 21, &e);
            }

            [fm removeItemAtPath:td error:nil];
            [fm removeItemAtPath:td2 error:nil];

            // ═══════════════════════════════════════════════════════════
            // SYMLINK TEST: tmpDir/weights/weight.bin → cache file
            // Avoids 30ms write of 352MB blob to tmpDir
            // ═══════════════════════════════════════════════════════════
            fprintf(stderr, "\n═══════════════════════════════════════════\n");
            fprintf(stderr, "SYMLINK TEST: link tmpDir → cache blobs\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            {
                NSData *blob3 = blob_cache_load_mmap(blobName);
                NSData *mil3 = blob_cache_load_mil(blobName);
                NSDictionary *wdict3 = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob3}};
                id desc3 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                    g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                    mil3, wdict3, nil);
                id mdl3 = ((id(*)(Class,SEL,id))objc_msgSend)(
                    g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc3);
                id hx3 = ((id(*)(id,SEL))objc_msgSend)(mdl3, @selector(hexStringIdentifier));

                NSString *td3 = [NSTemporaryDirectory() stringByAppendingPathComponent:(NSString*)hx3];
                [fm removeItemAtPath:td3 error:nil];

                t0 = mach_absolute_time();
                [fm createDirectoryAtPath:[td3 stringByAppendingPathComponent:@"weights"]
                    withIntermediateDirectories:YES attributes:nil error:nil];
                // Write MIL (small, ~2KB)
                [mil3 writeToFile:[td3 stringByAppendingPathComponent:@"model.mil"] atomically:YES];
                // SYMLINK the weight blob instead of copying
                NSString *cacheBlobPath = [[cache_dir() stringByAppendingPathComponent:blobName]
                                            stringByAppendingPathComponent:@"weight.bin"];
                NSString *tmpBlobPath = [td3 stringByAppendingPathComponent:@"weights/weight.bin"];
                [fm createSymbolicLinkAtPath:tmpBlobPath withDestinationPath:cacheBlobPath error:&e];
                double t_sym = ms(mach_absolute_time() - t0);
                fprintf(stderr, "  symlink tmpDir setup: %.2f ms\n", t_sym);
                if (e) { fprintf(stderr, "  symlink err: %s\n", [[e description] UTF8String]); e = nil; }

                // Try load
                t0 = mach_absolute_time();
                ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl3, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                double t_load3 = ms(mach_absolute_time() - t0);
                fprintf(stderr, "  loadWithQoS: %s (%.2f ms)\n", ok ? "YES" : "NO", t_load3);
                if (e) { fprintf(stderr, "  err: %s\n", [[e description] UTF8String]); e = nil; }

                if (ok) {
                    IOSurfaceRef ioIn = ane_create_surface(in_sz);
                    IOSurfaceRef ioOut = ane_create_surface(out_sz);
                    id wIn3 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                    id wOut3 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                    id req3 = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wIn3], @[@0], @[wOut3], @[@0], nil, nil, @0);
                    t0 = mach_absolute_time();
                    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl3, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req3, &e);
                    double t_eval3 = ms(mach_absolute_time() - t0);
                    fprintf(stderr, "  eval: %s (%.2f ms)\n", ok ? "PASS" : "FAIL", t_eval3);
                    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl3, @selector(unloadWithQoS:error:), 21, &e);
                    CFRelease(ioIn); CFRelease(ioOut);
                }

                [fm removeItemAtPath:td3 error:nil];
            }

            // ═══════════════════════════════════════════════════════════
            // ALL 5 PROGRAMS: measure total for a full layer (cached)
            // ═══════════════════════════════════════════════════════════
            fprintf(stderr, "\n═══════════════════════════════════════════\n");
            fprintf(stderr, "FULL LAYER SIM: 5 programs from cache\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            // We only have FFN cached. Simulate the others by timing model creation
            // with smaller blobs (Q/K/V/Wo).
            {
                typedef struct { const char *name; int out_ch; int in_ch; } ProgSpec;
                ProgSpec progs[] = {
                    {"Q",  dim,  dim},
                    {"K",  1024, dim},
                    {"V",  1024, dim},
                    {"Wo", dim,  dim},
                };
                double total_create = 0;
                for (int p = 0; p < 4; p++) {
                    size_t wn = (size_t)progs[p].out_ch * progs[p].in_ch;
                    _Float16 *w = (_Float16 *)malloc(wn * sizeof(_Float16));
                    fill_synthetic_fp16(w, (int)wn, 2000 + p);
                    NSData *b = mil_build_single_weight_blob(w, progs[p].out_ch, progs[p].in_ch);
                    free(w);
                    NSString *m = mil_gen_conv_baked(progs[p].in_ch, progs[p].out_ch, S);
                    NSData *md = [m dataUsingEncoding:NSUTF8StringEncoding];

                    t0 = mach_absolute_time();
                    NSDictionary *wd = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": b}};
                    id d = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:), md, wd, nil);
                    id mm = ((id(*)(Class,SEL,id))objc_msgSend)(
                        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), d);
                    ((id(*)(id,SEL))objc_msgSend)(mm, @selector(hexStringIdentifier));
                    double dt = ms(mach_absolute_time() - t0);
                    total_create += dt;
                    fprintf(stderr, "  %s (%dx%d): model+SHA256 = %.1f ms, blob = %.1f MB\n",
                            progs[p].name, progs[p].out_ch, progs[p].in_ch, dt, b.length / 1e6);
                }
                fprintf(stderr, "  FFN: model+SHA256 = ~125 ms, blob = 352.3 MB\n");
                total_create += 125;
                fprintf(stderr, "  ────────────────────\n");
                fprintf(stderr, "  Total model creation (5 progs): %.1f ms\n", total_create);
                fprintf(stderr, "  + load (~8ms x5): ~40ms\n");
                fprintf(stderr, "  Per-layer total (cached): ~%.0f ms\n", total_create + 40);
                fprintf(stderr, "  32 layers: ~%.1f s\n", (total_create + 40) * 32 / 1000);
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // PATH B: No cache → dequant + build + save
        // ═══════════════════════════════════════════════════════════════
        else {
            fprintf(stderr, "═══════════════════════════════════════════\n");
            fprintf(stderr, "FRESH PATH: dequant + build blob + save to cache\n");
            fprintf(stderr, "═══════════════════════════════════════════\n\n");

            // Allocate synthetic fp16 weights (simulating dequant output)
            size_t w_up_n = (size_t)hidden * dim;
            size_t w_dn_n = (size_t)dim * hidden;

            _Float16 *w1 = (_Float16 *)malloc(w_up_n * sizeof(_Float16));
            _Float16 *w3 = (_Float16 *)malloc(w_up_n * sizeof(_Float16));
            _Float16 *w2 = (_Float16 *)malloc(w_dn_n * sizeof(_Float16));

            // Simulate dequant time
            uint64_t t0 = mach_absolute_time();
            fill_synthetic_fp16(w1, (int)w_up_n, 1001);
            fill_synthetic_fp16(w3, (int)w_up_n, 1002);
            fill_synthetic_fp16(w2, (int)w_dn_n, 1003);
            double t_dequant = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  synthetic weight gen (simulating dequant): %.1f ms\n", t_dequant);

            // Build blob
            t0 = mach_absolute_time();
            NSData *blob = mil_build_ffn_fused_blob(w1, w3, hidden, w2, dim);
            double t_blob = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  build blob: %.1f ms (%.1f MB)\n", t_blob, blob.length / 1e6);

            free(w1); free(w3); free(w2);

            // Create model + get hexId (triggers SHA256)
            t0 = mach_absolute_time();
            NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                milData, wdict, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
                g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
            double t_model = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  create model: %.2f ms\n", t_model);

            t0 = mach_absolute_time();
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            double t_hex = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  hexStringIdentifier (SHA256): %.1f ms\n", t_hex);

            // Save to cache
            t0 = mach_absolute_time();
            blob_cache_save(blob, milData, blobName);
            double t_save = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  save to cache: %.1f ms\n", t_save);

            // Compile + load + eval (full path for reference)
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:(NSString*)hx];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [blob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

            NSError *e = nil;
            t0 = mach_absolute_time();
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            double t_compile = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  compile: %s (%.1f ms)\n", ok ? "YES" : "NO", t_compile);

            t0 = mach_absolute_time();
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            double t_load = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  load: %s (%.2f ms)\n", ok ? "YES" : "NO", t_load);

            if (ok) {
                IOSurfaceRef ioIn = ane_create_surface(in_sz);
                IOSurfaceRef ioOut = ane_create_surface(out_sz);
                id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
                t0 = mach_absolute_time();
                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                double t_eval = ms(mach_absolute_time() - t0);
                fprintf(stderr, "  eval: %s (%.2f ms)\n", ok ? "PASS" : "FAIL", t_eval);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
                CFRelease(ioIn); CFRelease(ioOut);
            }

            double total_fresh = t_dequant + t_blob + t_model + t_hex + t_compile + t_load;
            fprintf(stderr, "\n  TOTAL (fresh path): %.1f ms\n", total_fresh);
            fprintf(stderr, "\nRun again to test cached path!\n");

            [fm removeItemAtPath:td error:nil];
        }

        // ═══════════════════════════════════════════════════════════════
        // Disk space analysis
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "\n═══════════════════════════════════════════\n");
        fprintf(stderr, "DISK SPACE ANALYSIS (32 layers x 5 programs)\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        // Per-layer blob sizes (fp16 data + DEADBEEF headers)
        size_t q_blob  = 64 + 64 + (size_t)dim * dim * 2;          // Wq: 4096x4096
        size_t k_blob  = 64 + 64 + (size_t)1024 * dim * 2;         // Wk: 1024x4096
        size_t v_blob  = 64 + 64 + (size_t)1024 * dim * 2;         // Wv: 1024x4096
        size_t wo_blob = 64 + 64 + (size_t)dim * dim * 2;          // Wo: 4096x4096
        // FFN: 3 weights with headers
        size_t w_up = (size_t)hidden * dim * 2;
        size_t w_dn = (size_t)dim * hidden * 2;
        size_t ffn_blob = 64 + (64 + w_up) + (64 + w_up) + (64 + w_dn); // W1+W3+W2

        size_t per_layer = q_blob + k_blob + v_blob + wo_blob + ffn_blob;
        size_t total = per_layer * 32;

        fprintf(stderr, "  Per layer:\n");
        fprintf(stderr, "    Q  blob: %7.1f MB\n", q_blob / 1e6);
        fprintf(stderr, "    K  blob: %7.1f MB\n", k_blob / 1e6);
        fprintf(stderr, "    V  blob: %7.1f MB\n", v_blob / 1e6);
        fprintf(stderr, "    Wo blob: %7.1f MB\n", wo_blob / 1e6);
        fprintf(stderr, "    FFN blob:%7.1f MB\n", ffn_blob / 1e6);
        fprintf(stderr, "    ──────────────────\n");
        fprintf(stderr, "    Total:   %7.1f MB\n", per_layer / 1e6);
        fprintf(stderr, "\n  32 layers: %.1f GB\n", total / 1e9);

        // Also show per-layer time estimates
        fprintf(stderr, "\n  Time estimates per layer (5 programs):\n");
        fprintf(stderr, "    Current:  dequant~100ms + blob~50ms + SHA256~100ms = ~250ms\n");
        fprintf(stderr, "    Cached:   mmap~0.1ms + SHA256~100ms = ~100ms\n");
        fprintf(stderr, "    Savings:  ~150ms/layer x 32 layers = ~4.8s total\n");
        fprintf(stderr, "\n  If SHA256 can be bypassed (saved hexId):\n");
        fprintf(stderr, "    Cached:   mmap~0.1ms + loadWithQoS~2ms = ~2ms\n");
        fprintf(stderr, "    Savings:  ~248ms/layer x 32 layers = ~7.9s total\n");

        fprintf(stderr, "\nDone.\n");
    }
    return 0;
}
