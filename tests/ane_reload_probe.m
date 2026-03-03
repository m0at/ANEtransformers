// ane_reload_probe.m — Can we reload ANE programs from daemon cache WITHOUT weight data?
//
// The key question for Mistral 7B pre-compilation:
//   After compiling a program (which sends it to aned), can we later create a NEW
//   model object WITHOUT providing the 14GB of fp16 weights, and still load+eval?
//
// Tests:
//   1. Baseline: compile with weights, verify eval works
//   2. Nil weights: create model with same MIL but nil weights — does hexId match?
//   3. Dummy weights: create model with 1-byte dummy weights — does hexId match?
//   4. KVC: can we force-set hexStringIdentifier on a model object?
//   5. Reconstruct tmpDir: populate tmpDir from saved hexId, skip model creation
//   6. Descriptor from path: does _ANEInMemoryModelDescriptor have a path-based init?
//   7. Weightless reload: compile with weights, unload, free weights, reload with nil
//   8. Minimal blob: use a tiny valid DEADBEEF blob instead of full weights
//
// Build: cd /Users/andy/ANEtransformers/mistral && xcrun clang -O2 -ffast-math -fobjc-arc \
//        -Wall -Wno-unused-function -Wno-unused-variable -mcpu=apple-m4 -I. \
//        -o reload_probe ../tests/ane_reload_probe.m \
//        -framework Foundation -framework IOSurface -framework Accelerate -ldl

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

#include "../training/ane_runtime.h"
#include "ane_mil_gen_mistral.h"

static mach_timebase_info_data_t g_tb;
static double ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// ─── Dump all methods on a class ────────────────────────────────────────────
static void dump_methods(Class cls) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    fprintf(stderr, "  %s: %u methods\n", class_getName(cls), count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        fprintf(stderr, "    %s\n", sel_getName(sel));
    }
    free(methods);
}

// ─── Dump all properties on a class ─────────────────────────────────────────
static void dump_properties(Class cls) {
    unsigned int count = 0;
    objc_property_t *props = class_copyPropertyList(cls, &count);
    fprintf(stderr, "  %s: %u properties\n", class_getName(cls), count);
    for (unsigned int i = 0; i < count; i++) {
        fprintf(stderr, "    %s: %s\n", property_getName(props[i]),
                property_getAttributes(props[i]));
    }
    free(props);
}

// ─── Dump all ivars on a class ──────────────────────────────────────────────
static void dump_ivars(Class cls) {
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    fprintf(stderr, "  %s: %u ivars\n", class_getName(cls), count);
    for (unsigned int i = 0; i < count; i++) {
        const char *name = ivar_getName(ivars[i]);
        const char *type = ivar_getTypeEncoding(ivars[i]);
        fprintf(stderr, "    %s (%s)\n", name, type ? type : "?");
    }
    free(ivars);
}

// ─── Create a simple conv program with known weights ────────────────────────
typedef struct {
    id model;
    NSString *hexId;
    NSString *tmpDir;
    NSData *milData;
    NSData *blob;
    size_t inSz, outSz;
    int in_ch, out_ch, S;
} TestModel;

static TestModel make_model(int in_ch, int out_ch, int S, bool with_weights) {
    // Generate MIL text
    NSString *mil_str = mil_gen_conv_baked(in_ch, out_ch, S);
    NSData *milData = [mil_str dataUsingEncoding:NSUTF8StringEncoding];

    // Generate deterministic fp16 weights
    NSData *blob = nil;
    if (with_weights) {
        _Float16 *W = (_Float16 *)malloc((size_t)out_ch * in_ch * sizeof(_Float16));
        srand(42 + in_ch * 10000 + out_ch * 100 + S);
        for (int i = 0; i < out_ch * in_ch; i++)
            W[i] = (_Float16)((float)rand() / (float)RAND_MAX * 0.02f - 0.01f);
        blob = mil_build_single_weight_blob(W, out_ch, in_ch);
        free(W);
    }

    // Create descriptor + model
    NSDictionary *wdict = nil;
    if (blob) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
    }
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);

    id mdl = nil;
    NSString *hexId = nil;
    NSString *tmpDir = nil;

    if (desc) {
        mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (mdl) {
            hexId = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        }
    }

    TestModel tm = {0};
    tm.model = mdl;
    tm.hexId = hexId;
    tm.tmpDir = tmpDir;
    tm.milData = milData;
    tm.blob = blob;
    tm.inSz = (size_t)in_ch * S * sizeof(float);
    tm.outSz = (size_t)out_ch * S * sizeof(float);
    tm.in_ch = in_ch;
    tm.out_ch = out_ch;
    tm.S = S;
    return tm;
}

static void populate_tmpdir(TestModel *tm) {
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tm->tmpDir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [tm->milData writeToFile:[tm->tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (tm->blob)
        [tm->blob writeToFile:[tm->tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
}

static bool compile_and_load(TestModel *tm) {
    NSError *e = nil;
    populate_tmpdir(tm);
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        tm->model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { fprintf(stderr, "  compile FAILED: %s\n", [[e description] UTF8String]); return false; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        tm->model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { fprintf(stderr, "  load FAILED: %s\n", [[e description] UTF8String]); return false; }
    return true;
}

static bool eval_model(TestModel *tm) {
    IOSurfaceRef ioIn = ane_create_surface(tm->inSz);
    IOSurfaceRef ioOut = ane_create_surface(tm->outSz);

    // Fill input with known pattern
    IOSurfaceLock(ioIn, 0, NULL);
    float *inp = (float *)IOSurfaceGetBaseAddress(ioIn);
    for (size_t i = 0; i < tm->inSz / sizeof(float); i++) inp[i] = 0.01f * (i % 100);
    IOSurfaceUnlock(ioIn, 0, NULL);

    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_ANEIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        tm->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

    if (ok) {
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        float *out = (float *)IOSurfaceGetBaseAddress(ioOut);
        float sum = 0;
        for (int i = 0; i < 8; i++) sum += out[i];
        fprintf(stderr, "  output[0..7] sum=%.6f (first=%.6f)\n", sum, out[0]);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
    } else {
        fprintf(stderr, "  eval FAILED: %s\n", e ? [[e description] UTF8String] : "unknown");
    }

    CFRelease(ioIn); CFRelease(ioOut);
    return ok;
}

static void unload_model(id model) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();
        NSError *e = nil;
        NSFileManager *fm = [NSFileManager defaultManager];

        int in_ch = 256, out_ch = 256, S = 16;

        // ═══════════════════════════════════════════════════════════════
        // Step 0: Enumerate methods on ANE classes
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 0: ANE class introspection\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        dump_methods(g_ANEDesc);
        dump_properties(g_ANEDesc);
        dump_ivars(g_ANEDesc);
        fprintf(stderr, "\n");
        dump_methods(g_ANEInMem);
        dump_properties(g_ANEInMem);
        dump_ivars(g_ANEInMem);
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 1: Baseline — compile with weights, verify eval
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 1: Baseline compile with weights\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        TestModel baseline = make_model(in_ch, out_ch, S, true);
        fprintf(stderr, "  hexId: %s\n", [baseline.hexId UTF8String]);
        if (!compile_and_load(&baseline)) return 1;
        fprintf(stderr, "  compile+load: OK\n");
        bool ok = eval_model(&baseline);
        fprintf(stderr, "  eval: %s\n", ok ? "PASS" : "FAIL");

        // Save reference values
        NSString *ref_hexId = [baseline.hexId copy];
        NSData *ref_milData = [baseline.milData copy];
        NSData *ref_blob = [baseline.blob copy];

        // Unload but keep tmpDir
        unload_model(baseline.model);
        fprintf(stderr, "  unloaded\n\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 2: Nil weights — does model even create?
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 2: Create model with nil weights\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        TestModel nil_weights = make_model(in_ch, out_ch, S, false);
        if (nil_weights.model) {
            fprintf(stderr, "  model created: YES\n");
            fprintf(stderr, "  hexId: %s\n", [nil_weights.hexId UTF8String]);
            fprintf(stderr, "  hexId matches baseline: %s\n",
                    [nil_weights.hexId isEqualToString:ref_hexId] ? "YES" : "NO");

            // Check compiledModelExists — does the daemon see our cached program?
            BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(nil_weights.model, @selector(compiledModelExists));
            fprintf(stderr, "  compiledModelExists: %s\n", cached ? "YES" : "NO");

            if (cached) {
                // Try load without compile
                NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:nil_weights.hexId];
                [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                    withIntermediateDirectories:YES attributes:nil error:nil];
                [nil_weights.milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

                BOOL lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    nil_weights.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                fprintf(stderr, "  loadWithQoS (no weights): %s\n", lok ? "YES" : "NO");
                if (e) { fprintf(stderr, "    err: %s\n", [[e description] UTF8String]); e = nil; }
                if (lok) {
                    ok = eval_model(&nil_weights);
                    fprintf(stderr, "  eval: %s\n", ok ? "PASS" : "FAIL");
                    unload_model(nil_weights.model);
                }
                [fm removeItemAtPath:td error:nil];
            }
        } else {
            fprintf(stderr, "  model created: NO (descriptor is nil with nil weights)\n");

            // Try with empty dict instead of nil
            id desc2 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                ref_milData, @{}, nil);
            fprintf(stderr, "  descriptor with empty dict: %s\n", desc2 ? "non-nil" : "nil");
            if (desc2) {
                id mdl2 = ((id(*)(Class,SEL,id))objc_msgSend)(
                    g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc2);
                if (mdl2) {
                    id hx2 = ((id(*)(id,SEL))objc_msgSend)(mdl2, @selector(hexStringIdentifier));
                    fprintf(stderr, "  hexId (empty dict): %s\n", [(NSString*)hx2 UTF8String]);
                    fprintf(stderr, "  matches baseline: %s\n",
                            [(NSString*)hx2 isEqualToString:ref_hexId] ? "YES" : "NO");
                }
            }
        }
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 3: Dummy weights — minimal valid DEADBEEF blob
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 3: Create model with dummy 128-byte DEADBEEF blob\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        {
            // Create a minimal DEADBEEF blob: 64B global header + 64B chunk header + 2 bytes fp16
            uint8_t dummy_blob[130];
            memset(dummy_blob, 0, sizeof(dummy_blob));
            write_chunk_header(dummy_blob + 64, 2, 128); // 1 fp16 value at offset 128
            NSData *dummyData = [NSData dataWithBytes:dummy_blob length:sizeof(dummy_blob)];

            NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": dummyData}};
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                ref_milData, wdict, nil);
            fprintf(stderr, "  descriptor: %s\n", desc ? "non-nil" : "nil");
            if (desc) {
                id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
                    g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
                if (mdl) {
                    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
                    fprintf(stderr, "  hexId: %s\n", [(NSString*)hx UTF8String]);
                    fprintf(stderr, "  matches baseline: %s\n",
                            [(NSString*)hx isEqualToString:ref_hexId] ? "YES" : "NO");
                }
            }
        }
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 4: KVC — can we set hexStringIdentifier?
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 4: Try KVC to set hexStringIdentifier\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        {
            // Create a nil-weights model (or dummy-weights model)
            TestModel dummy = make_model(in_ch, out_ch, S, false);
            id target = dummy.model;
            if (!target) {
                // If nil weights fails, use dummy 128-byte blob
                uint8_t db[130]; memset(db, 0, sizeof(db));
                write_chunk_header(db + 64, 2, 128);
                NSData *dd = [NSData dataWithBytes:db length:sizeof(db)];
                NSDictionary *wd = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": dd}};
                id d = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                    g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                    ref_milData, wd, nil);
                target = ((id(*)(Class,SEL,id))objc_msgSend)(
                    g_ANEInMem, @selector(inMemoryModelWithDescriptor:), d);
            }

            if (target) {
                id origHex = ((id(*)(id,SEL))objc_msgSend)(target, @selector(hexStringIdentifier));
                fprintf(stderr, "  original hexId: %s\n", [(NSString*)origHex UTF8String]);

                // Try setValue:forKey:
                @try {
                    [target setValue:ref_hexId forKey:@"hexStringIdentifier"];
                    id newHex = ((id(*)(id,SEL))objc_msgSend)(target, @selector(hexStringIdentifier));
                    fprintf(stderr, "  KVC set hexStringIdentifier: SUCCESS\n");
                    fprintf(stderr, "  new hexId: %s\n", [(NSString*)newHex UTF8String]);
                    fprintf(stderr, "  matches baseline: %s\n",
                            [(NSString*)newHex isEqualToString:ref_hexId] ? "YES" : "NO");

                    if ([(NSString*)newHex isEqualToString:ref_hexId]) {
                        // Try to load with the forced hexId
                        BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(target, @selector(compiledModelExists));
                        fprintf(stderr, "  compiledModelExists: %s\n", cached ? "YES" : "NO");

                        if (cached) {
                            // Ensure tmpDir exists with MIL
                            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:ref_hexId];
                            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                                withIntermediateDirectories:YES attributes:nil error:nil];
                            [ref_milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
                            // NO weight.bin file!

                            BOOL lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                                target, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                            fprintf(stderr, "  loadWithQoS (KVC hexId, no weights file): %s\n", lok ? "YES" : "NO");
                            if (e) { fprintf(stderr, "    err: %s\n", [[e description] UTF8String]); e = nil; }

                            if (lok) {
                                // Build a temporary TestModel struct for eval
                                TestModel kvc_tm = {0};
                                kvc_tm.model = target;
                                kvc_tm.inSz = (size_t)in_ch * S * sizeof(float);
                                kvc_tm.outSz = (size_t)out_ch * S * sizeof(float);
                                ok = eval_model(&kvc_tm);
                                fprintf(stderr, "  eval (KVC reload): %s\n", ok ? "PASS" : "FAIL");
                                unload_model(target);
                            }
                            [fm removeItemAtPath:td error:nil];
                        }
                    }
                } @catch (NSException *ex) {
                    fprintf(stderr, "  KVC set hexStringIdentifier: EXCEPTION: %s\n",
                            [[ex reason] UTF8String]);

                    // Try direct ivar access
                    Ivar ivar = class_getInstanceVariable([target class], "_hexStringIdentifier");
                    if (!ivar) ivar = class_getInstanceVariable([target class], "hexStringIdentifier");
                    if (!ivar) ivar = class_getInstanceVariable([target class], "_hexId");
                    if (!ivar) ivar = class_getInstanceVariable([target class], "_identifier");
                    fprintf(stderr, "  ivar search: %s\n", ivar ? ivar_getName(ivar) : "NOT FOUND");

                    if (ivar) {
                        object_setIvar(target, ivar, ref_hexId);
                        id newHex = ((id(*)(id,SEL))objc_msgSend)(target, @selector(hexStringIdentifier));
                        fprintf(stderr, "  after ivar set: %s\n", [(NSString*)newHex UTF8String]);
                        fprintf(stderr, "  matches baseline: %s\n",
                                [(NSString*)newHex isEqualToString:ref_hexId] ? "YES" : "NO");
                    }
                }
            }
        }
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 5: Same model object — compile, unload, free blob, reload
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 5: Compile, unload, reload SAME model object\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        {
            TestModel tm = make_model(in_ch, out_ch, S, true);
            fprintf(stderr, "  hexId: %s\n", [tm.hexId UTF8String]);
            populate_tmpdir(&tm);

            // Compile + load
            BOOL cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            fprintf(stderr, "  compile: %s\n", cok ? "YES" : "NO");

            BOOL lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            fprintf(stderr, "  load: %s\n", lok ? "YES" : "NO");

            ok = eval_model(&tm);
            fprintf(stderr, "  eval #1: %s\n", ok ? "PASS" : "FAIL");

            // Unload
            unload_model(tm.model);
            fprintf(stderr, "  unloaded\n");

            // Delete the weight file from tmpDir
            [fm removeItemAtPath:[tm.tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] error:nil];
            fprintf(stderr, "  deleted weight.bin from tmpDir\n");

            // Check daemon cache
            BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(tm.model, @selector(compiledModelExists));
            fprintf(stderr, "  compiledModelExists: %s\n", cached ? "YES" : "NO");

            // Try reload on same model object (no recompile)
            uint64_t t0 = mach_absolute_time();
            lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            double t_load = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  reload (same object, no weights file): %s (%.2f ms)\n",
                    lok ? "YES" : "NO", t_load);
            if (e) { fprintf(stderr, "    err: %s\n", [[e description] UTF8String]); e = nil; }

            if (lok) {
                ok = eval_model(&tm);
                fprintf(stderr, "  eval #2 (after weightless reload): %s\n", ok ? "PASS" : "FAIL");
                unload_model(tm.model);
            }
            [fm removeItemAtPath:tm.tmpDir error:nil];
        }
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 6: New model object with same weights — check daemon cache
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 6: New model object, same weights, skip compile\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        {
            // First compile
            TestModel tm1 = make_model(in_ch, out_ch, S, true);
            populate_tmpdir(&tm1);
            BOOL cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm1.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            BOOL lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm1.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            fprintf(stderr, "  model1 compile+load: %s\n", (cok && lok) ? "OK" : "FAIL");
            unload_model(tm1.model);
            // Keep tmpDir

            // Create new model object with SAME weights
            TestModel tm2 = make_model(in_ch, out_ch, S, true);
            fprintf(stderr, "  model2 hexId: %s\n", [tm2.hexId UTF8String]);
            fprintf(stderr, "  hexIds match: %s\n",
                    [tm2.hexId isEqualToString:tm1.hexId] ? "YES" : "NO");

            BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(tm2.model, @selector(compiledModelExists));
            fprintf(stderr, "  compiledModelExists: %s\n", cached ? "YES" : "NO");

            // Populate tmpDir (re-uses same hexId, so same path)
            populate_tmpdir(&tm2);

            // Try load WITHOUT compile
            uint64_t t0 = mach_absolute_time();
            lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm2.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            double t_load = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  loadWithQoS (no compile): %s (%.2f ms)\n", lok ? "YES" : "NO", t_load);
            if (e) { fprintf(stderr, "    err: %s\n", [[e description] UTF8String]); e = nil; }

            if (lok) {
                ok = eval_model(&tm2);
                fprintf(stderr, "  eval: %s\n", ok ? "PASS" : "FAIL");
                unload_model(tm2.model);
            }

            // Now try: new model, populate tmpDir, DELETE weight.bin, then load
            TestModel tm3 = make_model(in_ch, out_ch, S, true);
            populate_tmpdir(&tm3);
            [fm removeItemAtPath:[tm3.tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] error:nil];
            fprintf(stderr, "\n  model3: tmpDir populated but weight.bin DELETED\n");

            t0 = mach_absolute_time();
            lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                tm3.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            t_load = ms(mach_absolute_time() - t0);
            fprintf(stderr, "  loadWithQoS (no weight file): %s (%.2f ms)\n", lok ? "YES" : "NO", t_load);
            if (e) { fprintf(stderr, "    err: %s\n", [[e description] UTF8String]); e = nil; }

            if (lok) {
                ok = eval_model(&tm3);
                fprintf(stderr, "  eval: %s\n", ok ? "PASS" : "FAIL");
                unload_model(tm3.model);
            }

            [fm removeItemAtPath:tm1.tmpDir error:nil];
        }
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 7: Check for alternative init methods on descriptor/model
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 7: Look for path-based or ID-based initializers\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        {
            // Check if there's a class method like modelWithPath: or modelWithIdentifier:
            SEL path_sels[] = {
                @selector(modelWithPath:),
                @selector(modelWithModelPath:),
                @selector(inMemoryModelWithPath:),
                @selector(inMemoryModelWithIdentifier:),
                @selector(modelWithIdentifier:),
                @selector(initWithPath:),
                @selector(initWithModelPath:),
            };
            for (int i = 0; i < (int)(sizeof(path_sels)/sizeof(path_sels[0])); i++) {
                BOOL responds_desc = [g_ANEDesc respondsToSelector:path_sels[i]];
                BOOL responds_model = [g_ANEInMem respondsToSelector:path_sels[i]];
                if (responds_desc || responds_model) {
                    fprintf(stderr, "  FOUND: %s responds to %s\n",
                            responds_desc ? "Descriptor" : "Model",
                            sel_getName(path_sels[i]));
                }
            }

            // Check for descriptor methods that might help
            SEL desc_sels[] = {
                @selector(descriptorWithModelPath:),
                @selector(modelDescriptorWithPath:),
                @selector(setWeights:),
                @selector(setMILText:),
                @selector(setHexStringIdentifier:),
                @selector(hexStringIdentifier),
                @selector(weights),
                @selector(milText),
                @selector(modelPath),
                @selector(model_path),
            };
            for (int i = 0; i < (int)(sizeof(desc_sels)/sizeof(desc_sels[0])); i++) {
                BOOL resp_cls_desc = [g_ANEDesc respondsToSelector:desc_sels[i]];
                BOOL resp_cls_mdl = [g_ANEInMem respondsToSelector:desc_sels[i]];

                // Also check instance methods on a real descriptor
                TestModel tmp = make_model(in_ch, out_ch, S, true);
                BOOL resp_inst = tmp.model ? [tmp.model respondsToSelector:desc_sels[i]] : NO;

                if (resp_cls_desc || resp_cls_mdl || resp_inst) {
                    fprintf(stderr, "  FOUND: %s%s%s responds to %s\n",
                            resp_cls_desc ? "Desc(cls) " : "",
                            resp_cls_mdl ? "Model(cls) " : "",
                            resp_inst ? "Model(inst) " : "",
                            sel_getName(desc_sels[i]));
                }
            }
        }
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Step 8: The money shot — compile all, free all, reload one-by-one
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "Step 8: Compile 3 programs, free weights, reload from cache\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");

        {
            typedef struct { int in_ch, out_ch; const char *name; } Prog;
            Prog progs[] = {
                {256, 256, "small_256x256"},
                {256, 128, "medium_256x128"},
                {128, 256, "medium_128x256"},
            };
            int nProgs = 3;

            NSMutableArray *hexIds = [NSMutableArray array];
            NSMutableArray *milDatas = [NSMutableArray array];
            NSMutableArray *blobs = [NSMutableArray array];

            // Phase A: compile all
            fprintf(stderr, "  --- Phase A: compile all ---\n");
            for (int i = 0; i < nProgs; i++) {
                TestModel tm = make_model(progs[i].in_ch, progs[i].out_ch, S, true);
                fprintf(stderr, "  [%d] %s hexId=%.16s...\n", i, progs[i].name, [tm.hexId UTF8String]);
                populate_tmpdir(&tm);

                BOOL cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    tm.model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                BOOL lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    tm.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                fprintf(stderr, "      compile=%s load=%s\n", cok?"Y":"N", lok?"Y":"N");

                // Quick eval to verify
                ok = eval_model(&tm);
                unload_model(tm.model);

                [hexIds addObject:tm.hexId];
                [milDatas addObject:tm.milData];
                [blobs addObject:tm.blob];

                // Keep tmpDir but delete weight.bin to simulate freeing memory
                [fm removeItemAtPath:[tm.tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] error:nil];
                // model object goes out of scope / gets released (ARC)
            }

            // Phase B: reload WITHOUT re-creating weight data
            fprintf(stderr, "\n  --- Phase B: reload from daemon cache (no weights) ---\n");
            for (int i = 0; i < nProgs; i++) {
                fprintf(stderr, "  [%d] %s\n", i, progs[i].name);

                // Re-create model with SAME weights (to get same hexId)
                TestModel tm = make_model(progs[i].in_ch, progs[i].out_ch, S, true);
                BOOL match = [tm.hexId isEqualToString:hexIds[i]];
                fprintf(stderr, "      hexId match: %s\n", match ? "YES" : "NO");

                BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(tm.model, @selector(compiledModelExists));
                fprintf(stderr, "      compiledModelExists: %s\n", cached ? "YES" : "NO");

                // Populate tmpDir with weights (must provide for hexId computation)
                populate_tmpdir(&tm);

                // BUT: delete weight.bin before load to prove load doesn't need it
                [fm removeItemAtPath:[tm.tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] error:nil];

                uint64_t t0 = mach_absolute_time();
                BOOL lok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    tm.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                double t_load = ms(mach_absolute_time() - t0);
                fprintf(stderr, "      loadWithQoS: %s (%.2f ms)\n", lok ? "YES" : "NO", t_load);
                if (e) { fprintf(stderr, "      err: %s\n", [[e description] UTF8String]); e = nil; }

                if (lok) {
                    ok = eval_model(&tm);
                    fprintf(stderr, "      eval: %s\n", ok ? "PASS" : "FAIL");
                    unload_model(tm.model);
                }
                [fm removeItemAtPath:tm.tmpDir error:nil];
            }

            // Phase C: try with model created from nil weights + KVC hexId
            fprintf(stderr, "\n  --- Phase C: nil weights + KVC hexId ---\n");
            for (int i = 0; i < nProgs; i++) {
                fprintf(stderr, "  [%d] %s\n", i, progs[i].name);

                // Create model without weights
                TestModel tm = make_model(progs[i].in_ch, progs[i].out_ch, S, false);
                if (!tm.model) {
                    fprintf(stderr, "      can't create model without weights, trying KVC on dummy...\n");
                    // Create with tiny dummy blob
                    uint8_t db[130]; memset(db, 0, sizeof(db));
                    write_chunk_header(db + 64, 2, 128);
                    NSData *dd = [NSData dataWithBytes:db length:sizeof(db)];
                    NSDictionary *wd = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": dd}};
                    id d = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                        milDatas[i], wd, nil);
                    if (!d) { fprintf(stderr, "      descriptor nil\n"); continue; }
                    tm.model = ((id(*)(Class,SEL,id))objc_msgSend)(
                        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), d);
                    if (!tm.model) { fprintf(stderr, "      model nil\n"); continue; }
                    tm.hexId = ((id(*)(id,SEL))objc_msgSend)(tm.model, @selector(hexStringIdentifier));
                    tm.tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:tm.hexId];
                    tm.inSz = (size_t)progs[i].in_ch * S * sizeof(float);
                    tm.outSz = (size_t)progs[i].out_ch * S * sizeof(float);
                }

                fprintf(stderr, "      current hexId: %.16s...\n", [tm.hexId UTF8String]);
                fprintf(stderr, "      target  hexId: %.16s...\n", [(NSString*)hexIds[i] UTF8String]);

                // Try to force the hexId
                @try {
                    [tm.model setValue:hexIds[i] forKey:@"hexStringIdentifier"];
                    id newHex = ((id(*)(id,SEL))objc_msgSend)(tm.model, @selector(hexStringIdentifier));
                    fprintf(stderr, "      KVC set: %s\n",
                            [(NSString*)newHex isEqualToString:hexIds[i]] ? "SUCCESS" : "FAILED (computed)");
                } @catch (NSException *ex) {
                    fprintf(stderr, "      KVC set: EXCEPTION\n");
                }

                [fm removeItemAtPath:tm.tmpDir error:nil];
            }
        }
        fprintf(stderr, "\n");

        // ═══════════════════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════════════════
        fprintf(stderr, "═══════════════════════════════════════════\n");
        fprintf(stderr, "SUMMARY\n");
        fprintf(stderr, "═══════════════════════════════════════════\n\n");
        fprintf(stderr, "Key question: can we reload without weight data?\n");
        fprintf(stderr, "Answer depends on results above.\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "If Step 5 reload works: same model object can reload (easy path)\n");
        fprintf(stderr, "If Step 6 no-weight-file works: tmpDir doesn't need weight.bin (easy path)\n");
        fprintf(stderr, "If Step 4 KVC works: forge hexId on lightweight model (medium path)\n");
        fprintf(stderr, "If none work: must provide full weight data for hexId computation (hard path)\n");
        fprintf(stderr, "  -> But can mmap from disk cache to avoid RAM (blob_cache_probe approach)\n");

        // Clean up baseline tmpDir
        [fm removeItemAtPath:baseline.tmpDir error:nil];

        fprintf(stderr, "\nDone.\n");
    }
    return 0;
}
