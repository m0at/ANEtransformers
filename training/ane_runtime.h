// ane_runtime.h — Reusable ANE in-memory compile/load/eval wrapper
// Uses _ANEInMemoryModel via private AppleNeuralEngine.framework
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

typedef struct {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;             // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
} ANEKernel;

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;
static bool g_ane_loaded = false;

static void ane_init(void) {
    if (g_ane_loaded) return;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ane_loaded = true;
}

static IOSurfaceRef ane_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

// Setup IOSurfaces and build ANE request for a compiled+loaded model
static ANEKernel *ane_setup_io(id mdl, NSString *tmpDir,
                                 int nInputs, size_t *inputSizes,
                                 int nOutputs, size_t *outputSizes) {
    ANEKernel *k = calloc(1, sizeof(ANEKernel));
    k->model = mdl;
    k->tmpDir = tmpDir;
    k->nInputs = nInputs;
    k->nOutputs = nOutputs;
    k->inputBytes = malloc(nInputs * sizeof(size_t));
    k->outputBytes = malloc(nOutputs * sizeof(size_t));
    memcpy(k->inputBytes, inputSizes, nInputs * sizeof(size_t));
    memcpy(k->outputBytes, outputSizes, nOutputs * sizeof(size_t));

    k->ioInputs = malloc(nInputs * sizeof(IOSurfaceRef));
    k->ioOutputs = malloc(nOutputs * sizeof(IOSurfaceRef));
    for (int i = 0; i < nInputs; i++)
        k->ioInputs[i] = ane_create_surface(inputSizes[i]);
    for (int i = 0; i < nOutputs; i++)
        k->ioOutputs[i] = ane_create_surface(outputSizes[i]);

    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:nInputs];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:nInputs];
    for (int i = 0; i < nInputs; i++) {
        [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:nOutputs];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:nOutputs];
    for (int i = 0; i < nOutputs; i++) {
        [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
        [oIdx addObject:@(i)];
    }
    k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);

    return k;
}

// Compile a MIL graph with weight blob into an ANE kernel.
// milText: NSData of MIL text
// weightData: NSData of raw weight blob (can be nil)
// inputSizes/outputSizes: arrays of byte sizes for each I/O tensor
static ANEKernel *ane_compile(NSData *milText, NSData *weightData,
                               int nInputs, size_t *inputSizes,
                               int nOutputs, size_t *outputSizes) {
    ane_init();
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milText, wdict, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    // Pre-populate temp dir with MIL + weights
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE compile failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE load failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    return ane_setup_io(mdl, td, nInputs, inputSizes, nOutputs, outputSizes);
}

// Compile or load-from-daemon-cache. If the aned daemon has already compiled
// a program with the same MIL+weights hash, skip compilation and just load (~2ms).
// Otherwise compile (~15-50ms) then load. The daemon caches compiled programs
// persistently keyed by SHA256(milText)_SHA256(weights)_SHA256(options).
// weightBlobPath: if non-nil, symlink tmpDir/weights/weight.bin to this path
//   instead of copying the blob (~0.4ms vs ~130ms for 352MB FFN blobs).
static ANEKernel *ane_compile_cached_ex(NSData *milText, NSData *weightData,
                                         NSString *weightBlobPath,
                                         int nInputs, size_t *inputSizes,
                                         int nOutputs, size_t *outputSizes) {
    ane_init();
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milText, wdict, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];

    // Helper: populate tmpDir with MIL + weights (symlink if path available)
    void (^populateTmpDir)(void) = ^{
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        NSString *wbPath = [td stringByAppendingPathComponent:@"weights/weight.bin"];
        if (weightBlobPath && [fm fileExistsAtPath:weightBlobPath]) {
            // Symlink: ~0.4ms vs ~130ms copy for large blobs
            [fm createSymbolicLinkAtPath:wbPath withDestinationPath:weightBlobPath error:nil];
        } else if (weightData) {
            [weightData writeToFile:wbPath atomically:YES];
        }
    };

    BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(mdl, @selector(compiledModelExists));
    if (!cached) {
        populateTmpDir();
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ANE compile failed: %s\n", [[e description] UTF8String]);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
    } else {
        if (![fm fileExistsAtPath:[td stringByAppendingPathComponent:@"model.mil"]]) {
            populateTmpDir();
        }
    }

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE load failed: %s\n", [[e description] UTF8String]);
        return NULL;
    }

    return ane_setup_io(mdl, td, nInputs, inputSizes, nOutputs, outputSizes);
}

// Backward-compatible wrapper (no symlink optimization)
static ANEKernel *ane_compile_cached(NSData *milText, NSData *weightData,
                                      int nInputs, size_t *inputSizes,
                                      int nOutputs, size_t *outputSizes) {
    return ane_compile_cached_ex(milText, weightData, nil, nInputs, inputSizes, nOutputs, outputSizes);
}

// Load a previously-compiled program from daemon cache using a forged hexId.
// Creates a dummy model object, sets _hexStringIdentifier via KVC, then loads.
// ~1.7ms per program — no dequant, no blob building, no SHA256 hashing.
static ANEKernel *ane_load_forged(NSData *milText, NSString *hexId,
                                    int nInputs, size_t *inputSizes,
                                    int nOutputs, size_t *outputSizes) {
    ane_init();
    NSError *e = nil;

    // Create dummy 130-byte DEADBEEF blob (minimal valid blob)
    uint8_t dummy[130] = {0};
    dummy[0] = 0x01; dummy[4] = 0x02;
    dummy[64] = 0xEF; dummy[65] = 0xBE; dummy[66] = 0xAD; dummy[67] = 0xDE;
    dummy[68] = 0x01;
    *(uint32_t*)(dummy + 72) = 2;    // data_size = 2 bytes
    *(uint32_t*)(dummy + 80) = 128;  // data_offset
    NSData *dummyBlob = [NSData dataWithBytes:dummy length:130];
    NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": dummyBlob}};

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milText, wdict, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    // KVC forge: overwrite _hexStringIdentifier with the real hexId
    [mdl setValue:hexId forKey:@"hexStringIdentifier"];

    // Ensure tmpDir exists (loadWithQoS needs it even from cache)
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:[td stringByAppendingPathComponent:@"model.mil"]]) {
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        // Write dummy weight file (daemon won't read it, but path must exist)
        [dummyBlob writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    }

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE forged load failed (hexId=%s): %s\n",
                [hexId UTF8String], [[e description] UTF8String]);
        return NULL;
    }

    return ane_setup_io(mdl, td, nInputs, inputSizes, nOutputs, outputSizes);
}

// Compile a program and return its hexId (for later ane_load_forged use).
// Returns the hexId string, or nil on failure. Caller should save this.
static NSString *ane_compile_and_get_hexid(NSData *milText, NSData *weightData) {
    ane_init();
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milText, wdict, nil);
    if (!desc) return nil;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    NSString *hexId = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];

    BOOL cached = ((BOOL(*)(id,SEL))objc_msgSend)(mdl, @selector(compiledModelExists));
    if (!cached) {
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weightData)
            [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ANE compile failed: %s\n", [[e description] UTF8String]);
            return nil;
        }
    }

    return hexId;
}

static void ane_write_input(ANEKernel *k, int idx, const void *data, size_t bytes) {
    IOSurfaceLock(k->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(k->ioInputs[idx], 0, NULL);
}

static void ane_read_output(ANEKernel *k, int idx, void *data, size_t bytes) {
    IOSurfaceLock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(k->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

static bool ane_eval(ANEKernel *k) {
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, k->request, &e);
}

// keepTmpDir: if true, don't delete tmpDir (preserve for daemon cache reuse)
static void ane_free_ex(ANEKernel *k, bool keepTmpDir) {
    if (!k) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    for (int i = 0; i < k->nInputs; i++) CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++) CFRelease(k->ioOutputs[i]);
    if (!keepTmpDir)
        [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    free(k->ioInputs); free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    free(k);
}

static void ane_free(ANEKernel *k) {
    ane_free_ex(k, false);
}
