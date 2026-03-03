// quant_probe.m — Probe ANE quantization support via private API
// Tests: FP16 baseline, INT8 dequant, UINT4 dequant, UINT4 blockwise, LUT palettization
// Build: xcrun clang -O2 -fobjc-arc -o quant_probe quant_probe.m \
//            -framework Foundation -framework IOSurface -ldl
// Run:   ./quant_probe
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

// ─── ANE runtime (inline, matching training/ane_runtime.h) ───

static Class g_Desc, g_IMM, g_Req, g_AIO;
static bool g_loaded = false;

static void ane_init(void) {
    if (g_loaded) return;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_IMM  = NSClassFromString(@"_ANEInMemoryModel");
    g_Req  = NSClassFromString(@"_ANERequest");
    g_AIO  = NSClassFromString(@"_ANEIOSurfaceObject");
    g_loaded = true;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

static mach_timebase_info_data_t g_tb;
static double ticks_to_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// ─── Weight blob builders ───

// Standard blob header: 64-byte global + 64-byte chunk header + data
// Global: [0]=0x01 [4]=0x02
// Chunk:  [0..3]=0xDEADBEEF [4]=0x01 [8..11]=data_size [16..19]=data_offset_from_file_start
static NSData *build_fp16_blob(int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128; // data starts at byte 128
    // Fill with small random fp16
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)oc*ic; i++)
        fp16[i] = (_Float16)(0.01f * ((float)(i % 200) - 100.0f));
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// INT8 weight blob: same header structure, data is int8
static NSData *build_int8_blob(int oc, int ic) {
    NSUInteger wsize = (NSUInteger)oc * ic;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    int8_t *data = (int8_t*)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)oc*ic; i++)
        data[i] = (int8_t)((i % 256) - 128);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// UINT4 weight blob: packed 2 values per byte (high nibble first)
static NSData *build_uint4_blob(int oc, int ic) {
    NSUInteger nels = (NSUInteger)oc * ic;
    NSUInteger wsize = (nels + 1) / 2; // packed
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    uint8_t *data = buf + 128;
    for (NSUInteger i = 0; i < nels/2; i++)
        data[i] = (uint8_t)(((i*2) % 16) << 4 | ((i*2+1) % 16));
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Scale blob: per-channel fp16 scales
static NSData *build_scale_blob(int oc, float val) {
    NSUInteger wsize = (NSUInteger)oc * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (int i = 0; i < oc; i++) fp16[i] = (_Float16)val;
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Zero-point blob: per-channel int8 or uint4
static NSData *build_zp_blob_int8(int oc) {
    NSUInteger wsize = (NSUInteger)oc;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    // zero-points = 0
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSData *build_zp_blob_uint4(int oc) {
    NSUInteger wsize = (oc + 1) / 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// LUT blob: 16-entry fp16 lookup table
static NSData *build_lut_blob(int oc) {
    // Per-channel LUT: [oc, 1, 16] fp16 = oc * 16 * 2 bytes
    NSUInteger nentries = 16;
    NSUInteger wsize = (NSUInteger)oc * nentries * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (NSUInteger c = 0; c < (NSUInteger)oc; c++)
        for (NSUInteger e = 0; e < nentries; e++)
            fp16[c * nentries + e] = (_Float16)(0.01f * (float)e - 0.08f);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// LUT index blob: 4-bit indices packed into uint8, shaped [oc, ic/2]
static NSData *build_lut_index_blob(int oc, int ic) {
    NSUInteger nels = (NSUInteger)oc * ic;
    NSUInteger wsize = (nels + 1) / 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE; chunk[4]=0x01;
    *(uint32_t*)(chunk+8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk+16) = 128;
    uint8_t *data = buf + 128;
    for (NSUInteger i = 0; i < nels/2; i++)
        data[i] = (uint8_t)(((i*2) % 16) << 4 | ((i*2+1) % 16));
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ─── Multi-blob combiner ───
// Combines multiple blobs into a single weight file with correct offsets.
// Returns NSData and fills offsets[] with the data_offset for each blob's payload.
static NSData *combine_blobs(NSArray<NSData*> *blobs, uint64_t *offsets) {
    // Layout: 64-byte global header, then for each blob: 64-byte chunk + data
    NSUInteger total = 64;
    NSUInteger *data_sizes = calloc(blobs.count, sizeof(NSUInteger));
    for (NSUInteger i = 0; i < blobs.count; i++) {
        // Each source blob is 64 (global) + 64 (chunk) + data
        // Extract data_size from chunk header at offset 64+8
        const uint8_t *src = blobs[i].bytes;
        data_sizes[i] = *(uint32_t*)(src + 64 + 8);
        total += 64 + data_sizes[i]; // chunk header + data for each
    }
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    NSUInteger pos = 64;
    for (NSUInteger i = 0; i < blobs.count; i++) {
        const uint8_t *src = blobs[i].bytes;
        // Copy chunk header
        memcpy(buf + pos, src + 64, 64);
        // Fix data_offset to point to correct absolute position
        *(uint32_t*)(buf + pos + 16) = (uint32_t)(pos + 64);
        offsets[i] = pos + 64;
        // Copy data
        memcpy(buf + pos + 64, src + 128, data_sizes[i]);
        pos += 64 + data_sizes[i];
    }
    free(data_sizes);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ─── MIL generators ───

static NSString *MIL_HEADER =
    @"program(1.3)\n"
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]\n{\n";

// Test 1: FP16 baseline conv (baked weights)
static NSString *gen_fp16_conv(int oc, int ic, int sp) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER, ic, sp, ic, sp, oc, ic, oc, ic, oc, sp, oc, sp];
}

// Test 2: INT8 per-channel dequantize → conv
static NSString *gen_int8_dequant_conv(int oc, int ic, int sp,
                                        uint64_t w_off, uint64_t s_off, uint64_t zp_off) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        int32 ax = const()[name = string(\"ax\"), val = int32(0)];\n"
        "        tensor<int8, [%d, %d, 1, 1]> qw = const()[name = string(\"qw\"), "
        "val = tensor<int8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, 1, 1, 1]> sc = const()[name = string(\"sc\"), "
        "val = tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<int8, [%d, 1, 1, 1]> zp = const()[name = string(\"zp\"), "
        "val = tensor<int8, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize("
        "quantized_data = qw, zero_point = zp, scale = sc, axis = ax)"
        "[name = string(\"dequant\")];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER, ic, sp,
        oc, ic, oc, ic, (unsigned long long)w_off,
        oc, oc, (unsigned long long)s_off,
        oc, oc, (unsigned long long)zp_off,
        oc, ic,
        ic, sp, oc, sp, oc, sp];
}

// Test 3: UINT4 per-channel dequantize → conv
static NSString *gen_uint4_dequant_conv(int oc, int ic, int sp,
                                         uint64_t w_off, uint64_t s_off, uint64_t zp_off) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        int32 ax = const()[name = string(\"ax\"), val = int32(0)];\n"
        "        tensor<uint4, [%d, %d, 1, 1]> qw = const()[name = string(\"qw\"), "
        "val = tensor<uint4, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, 1, 1, 1]> sc = const()[name = string(\"sc\"), "
        "val = tensor<fp16, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<uint4, [%d, 1, 1, 1]> zp = const()[name = string(\"zp\"), "
        "val = tensor<uint4, [%d, 1, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_affine_dequantize("
        "quantized_data = qw, zero_point = zp, scale = sc, axis = ax)"
        "[name = string(\"dequant\")];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER, ic, sp,
        oc, ic, oc, ic, (unsigned long long)w_off,
        oc, oc, (unsigned long long)s_off,
        oc, oc, (unsigned long long)zp_off,
        oc, ic,
        ic, sp, oc, sp, oc, sp];
}

// Test 4: UINT4 blockwise shift-scale (iOS18)
static NSString *gen_uint4_blockwise_conv(int oc, int ic, int sp,
                                           uint64_t w_off, uint64_t s_off, uint64_t zp_off,
                                           int block_size) {
    int nblocks = (ic + block_size - 1) / block_size;
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<uint4, [%d, %d, 1, 1]> qw = const()[name = string(\"qw\"), "
        "val = tensor<uint4, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> sc = const()[name = string(\"sc\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> off = const()[name = string(\"off\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_blockwise_shift_scale("
        "data = qw, scale = sc, offset = off)"
        "[name = string(\"dequant\")];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER, ic, sp,
        oc, ic, oc, ic, (unsigned long long)w_off,
        oc, nblocks, oc, nblocks, (unsigned long long)s_off,
        oc, nblocks, oc, nblocks, (unsigned long long)zp_off,
        oc, ic,
        ic, sp, oc, sp, oc, sp];
}

// Test 5: LUT palettization (4-bit, 16 entries) via constexpr_lut_to_dense
static NSString *gen_lut_conv(int oc, int ic, int sp,
                               uint64_t idx_off, uint64_t lut_off) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<uint8, [%d, %d, 1, 1]> idx = const()[name = string(\"idx\"), "
        "val = tensor<uint8, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<fp16, [%d, 1, 16]> lut = const()[name = string(\"lut\"), "
        "val = tensor<fp16, [%d, 1, 16]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu)))];\n"
        "        tensor<int32, [4]> shape = const()[name = string(\"shape\"), val = tensor<int32, [4]>([%d, %d, 1, 1])];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = constexpr_lut_to_dense("
        "indices = idx, lut = lut, shape = shape)"
        "[name = string(\"lut_dense\")];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        MIL_HEADER, ic, sp,
        oc, (ic+1)/2, oc, (ic+1)/2, (unsigned long long)idx_off,
        oc, oc, (unsigned long long)lut_off,
        oc, ic,
        oc, ic,
        ic, sp, oc, sp, oc, sp];
}

// ─── Benchmark runner ───

typedef struct {
    const char *name;
    double ms;
    double tflops;
    bool compiled;
    bool ran;
} TestResult;

static TestResult run_test(const char *name, NSString *mil, NSData *weightData,
                            int ic, int oc, int sp) {
    TestResult r = {name, 0, 0, false, false};
    ane_init();

    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);
    if (!desc) { printf("  %-28s  DESC FAIL\n", name); return r; }

    id model = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) { printf("  %-28s  MODEL FAIL\n", name); return r; }

    id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  %-28s  COMPILE FAIL: %s\n", name, e ? [[e description] UTF8String] : "unknown");
        [fm removeItemAtPath:tmpDir error:nil];
        return r;
    }
    r.compiled = true;

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  %-28s  LOAD FAIL\n", name);
        [fm removeItemAtPath:tmpDir error:nil];
        return r;
    }

    // Create I/O surfaces
    size_t inBytes = (size_t)ic * sp * 4;  // fp32 input
    size_t outBytes = (size_t)oc * sp * 4; // fp32 output
    IOSurfaceRef ioIn = make_surface(inBytes);
    IOSurfaceRef ioOut = make_surface(outBytes);

    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_Req,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    // Fill input with small values
    IOSurfaceLock(ioIn, 0, NULL);
    float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
    for (size_t i = 0; i < (size_t)ic*sp; i++) inp[i] = 0.01f;
    IOSurfaceUnlock(ioIn, 0, NULL);

    // Warmup
    for (int i = 0; i < 5; i++) {
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        if (!ok) {
            printf("  %-28s  EVAL FAIL\n", name);
            CFRelease(ioIn); CFRelease(ioOut);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
            [fm removeItemAtPath:tmpDir error:nil];
            return r;
        }
    }
    r.ran = true;

    // Benchmark
    int iters = 100;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    double ms = ticks_to_ms(mach_absolute_time() - t0) / iters;

    double gflops = 2.0 * oc * ic * sp / 1e9;
    r.ms = ms;
    r.tflops = gflops / ms;

    // Cleanup
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(ioIn);
    CFRelease(ioOut);
    [fm removeItemAtPath:tmpDir error:nil];

    printf("  %-28s  %8.3f ms  %7.2f TFLOPS\n", name, ms, r.tflops);
    return r;
}

// ─── Main ───

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        ane_init();

        if (!g_Desc || !g_IMM) {
            fprintf(stderr, "ERROR: Cannot load AppleNeuralEngine.framework\n");
            return 1;
        }

        printf("=== ANE Quantization Probe ===\n");
        printf("Tests: FP16, INT8 dequant, UINT4 dequant, UINT4 blockwise, LUT palettization\n\n");

        // Test configurations: (oc, ic, spatial) — matching real transformer shapes
        int configs[][3] = {
            {768,  768,  64},   // attention projection
            {2048, 768,  64},   // FFN up
            {768,  2048, 64},   // FFN down
            {1024, 1024, 64},   // medium square
            {2048, 2048, 64},   // large square
        };
        int nconfigs = sizeof(configs) / sizeof(configs[0]);

        for (int ci = 0; ci < nconfigs; ci++) {
            int oc = configs[ci][0], ic = configs[ci][1], sp = configs[ci][2];
            printf("\n── Config: oc=%d, ic=%d, sp=%d  (%.1f MB fp16 weights) ──\n",
                   oc, ic, sp, (double)oc*ic*2/1024/1024);

            // Test 1: FP16 baseline
            {
                NSData *blob = build_fp16_blob(oc, ic);
                NSString *mil = gen_fp16_conv(oc, ic, sp);
                run_test("FP16 baseline", mil, blob, ic, oc, sp);
            }

            // Test 2: INT8 per-channel dequantize
            {
                NSData *wblob = build_int8_blob(oc, ic);
                NSData *sblob = build_scale_blob(oc, 0.01f);
                NSData *zblob = build_zp_blob_int8(oc);
                uint64_t offsets[3];
                NSData *combined = combine_blobs(@[wblob, sblob, zblob], offsets);
                NSString *mil = gen_int8_dequant_conv(oc, ic, sp, offsets[0], offsets[1], offsets[2]);
                run_test("INT8 per-ch dequant", mil, combined, ic, oc, sp);
            }

            // Test 3: UINT4 per-channel dequantize
            {
                NSData *wblob = build_uint4_blob(oc, ic);
                NSData *sblob = build_scale_blob(oc, 0.01f);
                NSData *zblob = build_zp_blob_uint4(oc);
                uint64_t offsets[3];
                NSData *combined = combine_blobs(@[wblob, sblob, zblob], offsets);
                NSString *mil = gen_uint4_dequant_conv(oc, ic, sp, offsets[0], offsets[1], offsets[2]);
                run_test("UINT4 per-ch dequant", mil, combined, ic, oc, sp);
            }

            // Test 4: UINT4 blockwise (iOS18, block_size=32)
            {
                int block_size = 32;
                int nblocks = (ic + block_size - 1) / block_size;
                NSData *wblob = build_uint4_blob(oc, ic);
                // Blockwise scale/offset: [oc, nblocks, 1, 1] fp16
                NSUInteger bs_size = (NSUInteger)oc * nblocks * 2;
                NSUInteger bs_total = 64 + 64 + bs_size;
                uint8_t *sbuf = (uint8_t*)calloc(bs_total, 1);
                sbuf[0] = 0x01; sbuf[4] = 0x02;
                uint8_t *sc = sbuf + 64;
                sc[0]=0xEF; sc[1]=0xBE; sc[2]=0xAD; sc[3]=0xDE; sc[4]=0x01;
                *(uint32_t*)(sc+8) = (uint32_t)bs_size;
                *(uint32_t*)(sc+16) = 128;
                _Float16 *sfp = (_Float16*)(sbuf + 128);
                for (NSUInteger i = 0; i < (NSUInteger)oc*nblocks; i++) sfp[i] = (_Float16)0.01f;
                NSData *sblob = [NSData dataWithBytesNoCopy:sbuf length:bs_total freeWhenDone:YES];

                uint8_t *obuf = (uint8_t*)calloc(bs_total, 1);
                obuf[0] = 0x01; obuf[4] = 0x02;
                uint8_t *oc2 = obuf + 64;
                oc2[0]=0xEF; oc2[1]=0xBE; oc2[2]=0xAD; oc2[3]=0xDE; oc2[4]=0x01;
                *(uint32_t*)(oc2+8) = (uint32_t)bs_size;
                *(uint32_t*)(oc2+16) = 128;
                NSData *oblob = [NSData dataWithBytesNoCopy:obuf length:bs_total freeWhenDone:YES];

                uint64_t offsets[3];
                NSData *combined = combine_blobs(@[wblob, sblob, oblob], offsets);
                NSString *mil = gen_uint4_blockwise_conv(oc, ic, sp, offsets[0], offsets[1], offsets[2], block_size);
                run_test("UINT4 blockwise (bs=32)", mil, combined, ic, oc, sp);
            }

            // Test 5: LUT palettization (4-bit, 16 entries)
            {
                NSData *iblob = build_lut_index_blob(oc, ic);
                NSData *lblob = build_lut_blob(oc);
                uint64_t offsets[2];
                NSData *combined = combine_blobs(@[iblob, lblob], offsets);
                NSString *mil = gen_lut_conv(oc, ic, sp, offsets[0], offsets[1]);
                run_test("LUT 4-bit palettize", mil, combined, ic, oc, sp);
            }
        }

        printf("\n=== Interpretation ===\n");
        printf("If INT8 TFLOPS ~= 2x FP16 → native int8 compute\n");
        printf("If UINT4 TFLOPS ~= 4x FP16 → native int4 compute\n");
        printf("If same TFLOPS as FP16 → dequant at compile time (memory savings only)\n");
        printf("COMPILE FAIL → MIL type/op not supported by ANE compiler\n");
        printf("EVAL FAIL → compiles but hardware rejects at dispatch\n");

        return 0;
    }
}
