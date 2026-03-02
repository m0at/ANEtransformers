// bench_fused.m — 3-tier fused ANE benchmark
// Tier 1: Dispatch overhead scaling curve
// Tier 2: Peak TFLOPS across configs
// Tier 3: Transformer block (QKV, Attention, FFN, Full layer)
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

#define DIM 768
#define HEADS 12
#define HD (DIM/HEADS)
#define HIDDEN 2048
#define SEQ 256

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
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}
static NSData *build_blob(const float *w, int oc, int ic) {
    int wsize = oc*ic*2, total = 128+wsize;
    uint8_t *buf = (uint8_t*)calloc(total,1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    _Float16 *fp16 = (_Float16*)(buf+128);
    for (int i = 0; i < oc*ic; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *data, int count) {
    int wsize = count*2, total = 128+wsize;
    uint8_t *buf = (uint8_t*)calloc(total,1);
    buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
    *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
    memcpy(buf+128, data, wsize);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}
// Build identity weight: single matrix using build_blob format
static NSData *build_identity_weight(int ch) {
    float *w = (float*)calloc(ch*ch, sizeof(float));
    for (int i = 0; i < ch; i++) w[i*ch+i] = 1.0f;
    NSData *d = build_blob(w, ch, ch);
    free(w);
    return d;
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
        [wd[path][@"data"] writeToFile:[td stringByAppendingPathComponent:
            [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""]] atomically:YES];
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
static BOOL ane_eval_io(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
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
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static void cleanup_kern(Kern *k) {
    if (!k->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->td error:nil];
    k->model = nil;
}

// --- MIL generators ---

// Fused N-layer conv chain (fp32 I/O, fp16 internal) — single weight.bin
static NSString *gen_fused_chain(int ch, int sp, int depth) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ch, sp];
    [m appendString:@CONV_CONST];
    [m appendString:@"        string tofp16 = const()[name=string(\"tofp16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> xh = cast(dtype=tofp16, x=x)[name=string(\"cast_in\")];\n", ch, sp];
    NSString *prev = @"xh";
    for (int i = 0; i < depth; i++) {
        [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name=string(\"W%d\"), "
            "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n",
            ch, ch, i, i, ch, ch];
        NSString *out = [NSString stringWithFormat:@"c%d", i];
        [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> %@ = conv(dilations=dl, groups=gr, pad=pd, "
            "pad_type=pt, strides=st, weight=W%d, x=%@)[name=string(\"%@\")];\n",
            ch, sp, out, i, prev, out];
        prev = out;
    }
    [m appendString:@"        string tofp32 = const()[name=string(\"tofp32\"), val=string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> out = cast(dtype=tofp32, x=%@)[name=string(\"cast_out\")];\n", ch, sp, prev];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Single conv kernel (fp32 I/O) for sequential dispatch
static NSString *gen_single_conv(int ch, int sp) {
    return gen_fused_chain(ch, sp, 1);
}

// Fused QKV: 3 parallel convs on same input
static NSString *gen_fused_qkv(int dim, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> q = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string(\"cq\")];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> k = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=x)[name=string(\"ck\")];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> v = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=x)[name=string(\"cv\")];\n", dim,seq];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(q,k,v))[name=string(\"cat\")];\n", 3*dim,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Single conv kernel (fp16 I/O)
static NSString *gen_single_conv_fp16(int oc, int ic, int seq, NSString *wname) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@.bin\"), offset=uint64(64)))];\n", oc,ic,oc,ic,wname];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"co\")];\n", oc,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Full fused attention: QKV → reshape → matmul → scale → mask → softmax → matmul → reshape → Wo
static NSString *gen_fused_attention(int dim, int heads, int hd, int seq) {
    float scale_val = 1.0f / sqrtf((float)hd);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendString:@CONV_CONST];
    // Weights
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    // QKV projections
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string(\"cq\")];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=x)[name=string(\"ck\")];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=x)[name=string(\"cv\")];\n", dim,seq];
    // Reshape + transpose to multi-head
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads,hd,seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", heads,hd,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", heads,seq,hd];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];\n", heads,hd,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", heads,seq,hd];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];\n", heads,hd,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", heads,seq,hd];
    // Q @ K^T
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", heads,seq,seq];
    // Scale
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", scale_val];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", heads,seq,seq];
    // Causal mask
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", seq,seq,seq,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", heads,seq,seq];
    // Softmax
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", heads,seq,seq];
    // scores @ V
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", heads,seq,hd];
    // Reshape back
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", heads,hd,seq];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", dim,seq];
    // Wo projection
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", dim,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Fused FFN: W1,W3 parallel → sigmoid → mul (SiLU) → mul (gate) → W2
static NSString *gen_fused_ffn(int dim, int hidden, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", hidden,dim,hidden,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", hidden,dim,hidden,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", dim,hidden,dim,hidden];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=x)[name=string(\"c1\")];\n", hidden,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=x)[name=string(\"c3\")];\n", hidden,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", hidden,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", hidden,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", hidden,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];\n", dim,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ========================== MAIN ==========================

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        // ==================== TIER 1: Dispatch Overhead Scaling ====================
        printf("=== Tier 1: Dispatch Overhead Scaling ===\n");
        printf("%-8s %10s %10s %8s %16s\n", "Layers", "Fused(ms)", "Seq(ms)", "Speedup", "Overhead/layer(us)");
        printf("--------------------------------------------------------------\n");
        {
            int ch = 256, sp = 64;
            int layers[] = {1, 2, 4, 8, 16, 32};
            int nlayers = 6;
            int WARMUP = 50, ITERS = 1000;

            // Compile single-conv kernel for sequential
            NSData *id_blob = build_identity_weight(ch);
            NSDictionary *w_wd = @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":id_blob}};
            NSString *single_mil = gen_single_conv(ch, sp);
            Kern seq_k = compile_mil(single_mil, w_wd);
            if (!seq_k.model) { printf("  Sequential kernel compile FAIL\n"); goto tier2; }

            size_t io_bytes = ch * sp * 4; // fp32
            IOSurfaceRef seqIn = make_surface(io_bytes), seqOut = make_surface(io_bytes);

            for (int li = 0; li < nlayers; li++) {
                int L = layers[li];

                // --- Fused ---
                NSString *fused_mil = gen_fused_chain(ch, sp, L);
                Kern fused_k = compile_mil(fused_mil, w_wd);
                if (!fused_k.model) { printf("  Fused L=%d compile FAIL\n", L); continue; }

                IOSurfaceRef fIn = make_surface(io_bytes), fOut = make_surface(io_bytes);

                // Fill input with 1.0
                IOSurfaceLock(fIn, 0, NULL);
                float *fp = (float*)IOSurfaceGetBaseAddress(fIn);
                for (int i = 0; i < ch*sp; i++) fp[i] = 1.0f;
                IOSurfaceUnlock(fIn, 0, NULL);

                IOSurfaceRef fins[] = {fIn}, fouts[] = {fOut};
                for (int i = 0; i < WARMUP; i++) ane_eval_io(&fused_k, fins, 1, fouts, 1);
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < ITERS; i++) ane_eval_io(&fused_k, fins, 1, fouts, 1);
                double fused_ms = tb_ms(mach_absolute_time() - t0) / ITERS;

                // Correctness check
                IOSurfaceLock(fOut, kIOSurfaceLockReadOnly, NULL);
                float *out = (float*)IOSurfaceGetBaseAddress(fOut);
                float maxdiff = 0;
                for (int i = 0; i < ch*sp; i++) { float d = fabsf(out[i] - 1.0f); if (d > maxdiff) maxdiff = d; }
                IOSurfaceUnlock(fOut, kIOSurfaceLockReadOnly, NULL);
                if (maxdiff > 0.01f) printf("  WARNING: L=%d fused maxdiff=%.4f\n", L, maxdiff);

                CFRelease(fIn); CFRelease(fOut);
                cleanup_kern(&fused_k);

                // --- Sequential ---
                // Fill sequential input with 1.0
                IOSurfaceLock(seqIn, 0, NULL);
                fp = (float*)IOSurfaceGetBaseAddress(seqIn);
                for (int i = 0; i < ch*sp; i++) fp[i] = 1.0f;
                IOSurfaceUnlock(seqIn, 0, NULL);

                IOSurfaceRef sins[] = {seqIn}, souts[] = {seqOut};
                // Warmup
                for (int i = 0; i < WARMUP; i++) {
                    for (int l = 0; l < L; l++) ane_eval_io(&seq_k, sins, 1, souts, 1);
                    // Swap for next iteration (output becomes input)
                    if (L > 1) {
                        IOSurfaceLock(seqOut, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceLock(seqIn, 0, NULL);
                        memcpy(IOSurfaceGetBaseAddress(seqIn), IOSurfaceGetBaseAddress(seqOut), io_bytes);
                        IOSurfaceUnlock(seqIn, 0, NULL);
                        IOSurfaceUnlock(seqOut, kIOSurfaceLockReadOnly, NULL);
                    }
                }
                t0 = mach_absolute_time();
                for (int i = 0; i < ITERS; i++) {
                    // Reset input each iter
                    IOSurfaceLock(seqIn, 0, NULL);
                    fp = (float*)IOSurfaceGetBaseAddress(seqIn);
                    for (int j = 0; j < ch*sp; j++) fp[j] = 1.0f;
                    IOSurfaceUnlock(seqIn, 0, NULL);
                    for (int l = 0; l < L; l++) {
                        ane_eval_io(&seq_k, sins, 1, souts, 1);
                        if (l < L-1) {
                            IOSurfaceLock(seqOut, kIOSurfaceLockReadOnly, NULL);
                            IOSurfaceLock(seqIn, 0, NULL);
                            memcpy(IOSurfaceGetBaseAddress(seqIn), IOSurfaceGetBaseAddress(seqOut), io_bytes);
                            IOSurfaceUnlock(seqIn, 0, NULL);
                            IOSurfaceUnlock(seqOut, kIOSurfaceLockReadOnly, NULL);
                        }
                    }
                }
                double seq_ms = tb_ms(mach_absolute_time() - t0) / ITERS;

                double speedup = seq_ms / fused_ms;
                double overhead_us = (seq_ms - fused_ms) * 1000.0 / (L > 1 ? L-1 : 1);

                printf("%5d    %9.3f  %9.3f  %6.1fx  %14.1f\n", L, fused_ms, seq_ms, speedup, overhead_us);
            }

            CFRelease(seqIn); CFRelease(seqOut);
            cleanup_kern(&seq_k);
        }
tier2:

        // ==================== TIER 2: Peak TFLOPS ====================
        printf("\n=== Tier 2: Peak TFLOPS ===\n");
        printf("%-6s %-6s %-8s %10s %8s %8s\n", "C", "S", "Layers", "ms/eval", "TFLOPS", "Eff%");
        printf("--------------------------------------------------\n");
        {
            int channels[] = {256, 512, 768, 1024};
            int spatials[] = {64, 128, 256};
            int depths[] = {1, 4, 8};
            int WARMUP = 20, ITERS = 500;
            double peak_single = 0, peak_any = 0;

            for (int ci = 0; ci < 4; ci++) {
                for (int si = 0; si < 3; si++) {
                    for (int di = 0; di < 3; di++) {
                        int c = channels[ci], s = spatials[si], d = depths[di];
                        // Build random weight (all layers share same weight)
                        float *rw = (float*)malloc(c*c*4);
                        float rsc = 1.0f/sqrtf(c);
                        for (int j=0;j<c*c;j++) rw[j]=rsc*(2*drand48()-1);
                        NSData *rwb = build_blob(rw, c, c);
                        free(rw);
                        NSString *mil = gen_fused_chain(c, s, d);
                        Kern k = compile_mil(mil, @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":rwb}});
                        if (!k.model) {
                            printf("%-6d %-6d %-8d       SKIP\n", c, s, d);
                            continue;
                        }
                        size_t bytes = c * s * 4;
                        IOSurfaceRef ioI = make_surface(bytes), ioO = make_surface(bytes);
                        IOSurfaceRef ins[] = {ioI}, outs[] = {ioO};
                        for (int i = 0; i < WARMUP; i++) ane_eval_io(&k, ins, 1, outs, 1);
                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < ITERS; i++) ane_eval_io(&k, ins, 1, outs, 1);
                        double ms = tb_ms(mach_absolute_time() - t0) / ITERS;
                        double gflops = 2.0*c*c*s*d / 1e9;
                        double tflops = gflops / ms;
                        double eff = tflops / 38.0 * 100.0;
                        printf("%-6d %-6d %-8d %9.3f  %7.2f  %6.1f%%\n", c, s, d, ms, tflops, eff);
                        if (d == 1 && tflops > peak_single) peak_single = tflops;
                        if (tflops > peak_any) peak_any = tflops;
                        CFRelease(ioI); CFRelease(ioO);
                        cleanup_kern(&k);
                    }
                }
            }
            printf("Peak single-dispatch: %.2f TFLOPS (%.1f%% of 38T)\n", peak_single, peak_single/38*100);
            printf("Peak sustained:       %.2f TFLOPS (%.1f%% of 38T)\n", peak_any, peak_any/38*100);
        }

        // ==================== TIER 3: Transformer Block ====================
        printf("\n=== Tier 3: Transformer Block (DIM=%d, SEQ=%d) ===\n", DIM, SEQ);

        srand48(42);
        float sc_d = 1.0f/sqrtf(DIM), sc_h = 1.0f/sqrtf(HIDDEN);
        float *Wq = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wq[i]=sc_d*(2*drand48()-1);
        float *Wk = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wk[i]=sc_d*(2*drand48()-1);
        float *Wv = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wv[i]=sc_d*(2*drand48()-1);
        float *Wo = (float*)malloc(DIM*DIM*4); for(int i=0;i<DIM*DIM;i++) Wo[i]=sc_d*(2*drand48()-1);
        float *W1 = (float*)malloc(HIDDEN*DIM*4); for(int i=0;i<HIDDEN*DIM;i++) W1[i]=sc_h*(2*drand48()-1);
        float *W2 = (float*)malloc(DIM*HIDDEN*4); for(int i=0;i<DIM*HIDDEN;i++) W2[i]=sc_d*(2*drand48()-1);
        float *W3 = (float*)malloc(HIDDEN*DIM*4); for(int i=0;i<HIDDEN*DIM;i++) W3[i]=sc_h*(2*drand48()-1);

        NSData *wq_blob = build_blob(Wq,DIM,DIM);
        NSData *wk_blob = build_blob(Wk,DIM,DIM);
        NSData *wv_blob = build_blob(Wv,DIM,DIM);
        NSData *wo_blob = build_blob(Wo,DIM,DIM);
        NSData *w1_blob = build_blob(W1,HIDDEN,DIM);
        NSData *w2_blob = build_blob(W2,DIM,HIDDEN);
        NSData *w3_blob = build_blob(W3,HIDDEN,DIM);

        _Float16 *mask_data = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
        for (int t = 0; t < SEQ; t++)
            for (int t2 = 0; t2 < SEQ; t2++)
                mask_data[t*SEQ+t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        NSData *mask_blob = build_blob_fp16(mask_data, SEQ*SEQ);
        free(mask_data);

        // Prepare input
        float *x_in = (float*)malloc(SEQ*DIM*4);
        for (int i = 0; i < SEQ*DIM; i++) x_in[i] = 0.1f*(2*drand48()-1);

        size_t io16 = DIM*SEQ*2;
        IOSurfaceRef surf_in = make_surface(io16);
        IOSurfaceLock(surf_in, 0, NULL);
        _Float16 *pin = (_Float16*)IOSurfaceGetBaseAddress(surf_in);
        for (int t = 0; t < SEQ; t++)
            for (int c = 0; c < DIM; c++)
                pin[c*SEQ+t] = (_Float16)x_in[t*DIM+c];
        IOSurfaceUnlock(surf_in, 0, NULL);

        int T3_WARMUP = 20, T3_ITERS = 500;
        double sep_qkv_ms=0, fused_qkv_ms=0;
        double sep_attn_ms=0, fused_attn_ms=0;
        double sep_ffn_ms=0, fused_ffn_ms=0;

        // --- Config A: Fused QKV (3→1) ---
        printf("\n--- Config A: Fused QKV (3 projections → 1 dispatch) ---\n");
        {
            // Fused
            NSString *mil = gen_fused_qkv(DIM, SEQ);
            NSDictionary *wd = @{
                @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":wq_blob},
                @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":wk_blob},
                @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":wv_blob},
            };
            Kern k = compile_mil(mil, wd);
            if (k.model) {
                IOSurfaceRef out = make_surface(3*io16);
                IOSurfaceRef ins[] = {surf_in}, outs[] = {out};
                for (int i=0;i<T3_WARMUP;i++) ane_eval_io(&k,ins,1,outs,1);
                uint64_t t0 = mach_absolute_time();
                for (int i=0;i<T3_ITERS;i++) ane_eval_io(&k,ins,1,outs,1);
                fused_qkv_ms = tb_ms(mach_absolute_time()-t0)/T3_ITERS;
                printf("  Fused QKV:    %.3f ms\n", fused_qkv_ms);
                CFRelease(out);
                cleanup_kern(&k);
            } else printf("  Fused QKV: COMPILE FAIL\n");

            // Separate: 3 individual conv kernels
            Kern kq = compile_mil(gen_single_conv_fp16(DIM,DIM,SEQ,@"wq"), @{@"@model_path/weights/wq.bin":@{@"offset":@0,@"data":wq_blob}});
            Kern kk = compile_mil(gen_single_conv_fp16(DIM,DIM,SEQ,@"wk"), @{@"@model_path/weights/wk.bin":@{@"offset":@0,@"data":wk_blob}});
            Kern kv = compile_mil(gen_single_conv_fp16(DIM,DIM,SEQ,@"wv"), @{@"@model_path/weights/wv.bin":@{@"offset":@0,@"data":wv_blob}});
            if (kq.model && kk.model && kv.model) {
                IOSurfaceRef oq = make_surface(io16), ok = make_surface(io16), ov = make_surface(io16);
                IOSurfaceRef iq[]={surf_in}, oqa[]={oq}, oka[]={ok}, ova[]={ov};
                for (int i=0;i<T3_WARMUP;i++) {
                    ane_eval_io(&kq,iq,1,oqa,1); ane_eval_io(&kk,iq,1,oka,1); ane_eval_io(&kv,iq,1,ova,1);
                }
                uint64_t t0 = mach_absolute_time();
                for (int i=0;i<T3_ITERS;i++) {
                    ane_eval_io(&kq,iq,1,oqa,1); ane_eval_io(&kk,iq,1,oka,1); ane_eval_io(&kv,iq,1,ova,1);
                }
                sep_qkv_ms = tb_ms(mach_absolute_time()-t0)/T3_ITERS;
                printf("  Separate QKV: %.3f ms\n", sep_qkv_ms);
                CFRelease(oq); CFRelease(ok); CFRelease(ov);
            } else printf("  Separate QKV: COMPILE FAIL\n");
            cleanup_kern(&kq); cleanup_kern(&kk); cleanup_kern(&kv);

            if (fused_qkv_ms > 0 && sep_qkv_ms > 0)
                printf("  Speedup:      %.1fx\n", sep_qkv_ms/fused_qkv_ms);
        }

        // --- Config B: Fused Full Attention (4→1) ---
        printf("\n--- Config B: Fused Full Attention (QKV+SDPA+Wo → 1 dispatch) ---\n");
        {
            int attn_seq = SEQ;
            NSString *mil = gen_fused_attention(DIM, HEADS, HD, attn_seq);
            NSDictionary *wd = @{
                @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":wq_blob},
                @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":wk_blob},
                @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":wv_blob},
                @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":wo_blob},
                @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":mask_blob},
            };
            Kern k = compile_mil(mil, wd);
            if (!k.model && attn_seq == 256) {
                printf("  SEQ=256 failed, falling back to SEQ=128\n");
                attn_seq = 128;
                _Float16 *m128 = (_Float16*)calloc(128*128, sizeof(_Float16));
                for (int t=0;t<128;t++) for (int t2=0;t2<128;t2++)
                    m128[t*128+t2] = (t2<=t)?(_Float16)0.0f:(_Float16)(-65504.0f);
                NSData *m128_blob = build_blob_fp16(m128, 128*128);
                free(m128);
                mil = gen_fused_attention(DIM, HEADS, HD, 128);
                wd = @{
                    @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":wq_blob},
                    @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":wk_blob},
                    @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":wv_blob},
                    @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":wo_blob},
                    @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":m128_blob},
                };
                k = compile_mil(mil, wd);
            }
            if (k.model) {
                size_t attn_io = DIM*attn_seq*2;
                IOSurfaceRef aIn = make_surface(attn_io), aOut = make_surface(attn_io);
                // Fill input
                IOSurfaceLock(aIn, 0, NULL);
                _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(aIn);
                for (int t=0;t<attn_seq;t++) for (int c=0;c<DIM;c++)
                    p[c*attn_seq+t] = (_Float16)(0.1f*(2*drand48()-1));
                IOSurfaceUnlock(aIn, 0, NULL);

                IOSurfaceRef ains[]={aIn}, aouts[]={aOut};
                for (int i=0;i<T3_WARMUP;i++) ane_eval_io(&k,ains,1,aouts,1);
                uint64_t t0 = mach_absolute_time();
                for (int i=0;i<T3_ITERS;i++) ane_eval_io(&k,ains,1,aouts,1);
                fused_attn_ms = tb_ms(mach_absolute_time()-t0)/T3_ITERS;

                // FLOPs: QKV=3*2*D*D*S + QKT=2*H*S*S*HD + SV=2*H*S*S*HD + Wo=2*D*D*S
                double flops = 4.0*2*DIM*DIM*attn_seq + 4.0*HEADS*attn_seq*attn_seq*HD;
                printf("  Fused attention (SEQ=%d): %.3f ms  %.1f GFLOPS\n", attn_seq, fused_attn_ms, flops/fused_attn_ms/1e6);

                CFRelease(aIn); CFRelease(aOut);
                cleanup_kern(&k);
            } else printf("  Fused attention: COMPILE FAIL\n");

            // Separate: 4 conv dispatches + CPU attention
            {
                Kern kq = compile_mil(gen_single_conv_fp16(DIM,DIM,attn_seq,@"wq"), @{@"@model_path/weights/wq.bin":@{@"offset":@0,@"data":wq_blob}});
                Kern kk2 = compile_mil(gen_single_conv_fp16(DIM,DIM,attn_seq,@"wk"), @{@"@model_path/weights/wk.bin":@{@"offset":@0,@"data":wk_blob}});
                Kern kv = compile_mil(gen_single_conv_fp16(DIM,DIM,attn_seq,@"wv"), @{@"@model_path/weights/wv.bin":@{@"offset":@0,@"data":wv_blob}});
                Kern ko = compile_mil(gen_single_conv_fp16(DIM,DIM,attn_seq,@"wo"), @{@"@model_path/weights/wo.bin":@{@"offset":@0,@"data":wo_blob}});

                if (kq.model && kk2.model && kv.model && ko.model) {
                    size_t attn_io = DIM*attn_seq*2;
                    IOSurfaceRef sIn = make_surface(attn_io);
                    IOSurfaceRef sOq = make_surface(attn_io), sOk = make_surface(attn_io);
                    IOSurfaceRef sOv = make_surface(attn_io), sOo = make_surface(attn_io);
                    IOSurfaceRef sAttnIn = make_surface(attn_io);

                    IOSurfaceLock(sIn, 0, NULL);
                    _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(sIn);
                    for (int t=0;t<attn_seq;t++) for (int c=0;c<DIM;c++)
                        p[c*attn_seq+t] = (_Float16)(0.1f*(2*drand48()-1));
                    IOSurfaceUnlock(sIn, 0, NULL);

                    IOSurfaceRef iq[]={sIn}, oqa[]={sOq}, oka[]={sOk}, ova[]={sOv};
                    IOSurfaceRef iAttn[]={sAttnIn}, ooa[]={sOo};

                    // CPU attention buffers
                    float *q_cpu = (float*)malloc(attn_seq*DIM*4);
                    float *k_cpu = (float*)malloc(attn_seq*DIM*4);
                    float *v_cpu = (float*)malloc(attn_seq*DIM*4);
                    float *attn_out = (float*)malloc(attn_seq*DIM*4);
                    float *scores = (float*)malloc(attn_seq*4);

                    for (int i=0;i<T3_WARMUP;i++) {
                        ane_eval_io(&kq,iq,1,oqa,1); ane_eval_io(&kk2,iq,1,oka,1); ane_eval_io(&kv,iq,1,ova,1);
                        // CPU attention (skip warmup detail)
                        ane_eval_io(&ko,iAttn,1,ooa,1);
                    }
                    uint64_t t0 = mach_absolute_time();
                    for (int i=0;i<T3_ITERS;i++) {
                        ane_eval_io(&kq,iq,1,oqa,1);
                        ane_eval_io(&kk2,iq,1,oka,1);
                        ane_eval_io(&kv,iq,1,ova,1);
                        // CPU attention
                        IOSurfaceLock(sOq, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceLock(sOk, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceLock(sOv, kIOSurfaceLockReadOnly, NULL);
                        _Float16 *pq=(_Float16*)IOSurfaceGetBaseAddress(sOq);
                        _Float16 *pk=(_Float16*)IOSurfaceGetBaseAddress(sOk);
                        _Float16 *pv=(_Float16*)IOSurfaceGetBaseAddress(sOv);
                        for (int t=0;t<attn_seq;t++) for (int c=0;c<DIM;c++) {
                            q_cpu[t*DIM+c]=(float)pq[c*attn_seq+t];
                            k_cpu[t*DIM+c]=(float)pk[c*attn_seq+t];
                            v_cpu[t*DIM+c]=(float)pv[c*attn_seq+t];
                        }
                        IOSurfaceUnlock(sOq, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceUnlock(sOk, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceUnlock(sOv, kIOSurfaceLockReadOnly, NULL);
                        float asc = 1.0f/sqrtf((float)HD);
                        for (int h=0;h<HEADS;h++) for (int t=0;t<attn_seq;t++) {
                            float maxs=-1e30f;
                            for (int t2=0;t2<=t;t2++) {
                                float s=0;
                                for (int d=0;d<HD;d++) s+=q_cpu[t*DIM+h*HD+d]*k_cpu[t2*DIM+h*HD+d];
                                s*=asc; scores[t2]=s; if(s>maxs) maxs=s;
                            }
                            float sum=0;
                            for (int t2=0;t2<=t;t2++){scores[t2]=expf(scores[t2]-maxs);sum+=scores[t2];}
                            for (int t2=0;t2<=t;t2++) scores[t2]/=sum;
                            for (int d=0;d<HD;d++){
                                float r=0;
                                for (int t2=0;t2<=t;t2++) r+=scores[t2]*v_cpu[t2*DIM+h*HD+d];
                                attn_out[t*DIM+h*HD+d]=r;
                            }
                        }
                        IOSurfaceLock(sAttnIn, 0, NULL);
                        _Float16 *pa=(_Float16*)IOSurfaceGetBaseAddress(sAttnIn);
                        for (int t=0;t<attn_seq;t++) for (int c=0;c<DIM;c++)
                            pa[c*attn_seq+t]=(_Float16)attn_out[t*DIM+c];
                        IOSurfaceUnlock(sAttnIn, 0, NULL);
                        ane_eval_io(&ko,iAttn,1,ooa,1);
                    }
                    sep_attn_ms = tb_ms(mach_absolute_time()-t0)/T3_ITERS;
                    printf("  Separate attention (SEQ=%d): %.3f ms\n", attn_seq, sep_attn_ms);

                    free(q_cpu); free(k_cpu); free(v_cpu); free(attn_out); free(scores);
                    CFRelease(sIn); CFRelease(sOq); CFRelease(sOk); CFRelease(sOv); CFRelease(sOo); CFRelease(sAttnIn);
                } else printf("  Separate attention: COMPILE FAIL\n");
                cleanup_kern(&kq); cleanup_kern(&kk2); cleanup_kern(&kv); cleanup_kern(&ko);
            }
            if (fused_attn_ms > 0 && sep_attn_ms > 0)
                printf("  Speedup:      %.1fx\n", sep_attn_ms/fused_attn_ms);
        }

        // --- Config C: Fused FFN (3→1) ---
        printf("\n--- Config C: Fused FFN (W1+W3+SiLU+W2 → 1 dispatch) ---\n");
        {
            NSString *mil = gen_fused_ffn(DIM, HIDDEN, SEQ);
            NSDictionary *wd = @{
                @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":w1_blob},
                @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":w3_blob},
                @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":w2_blob},
            };
            Kern k = compile_mil(mil, wd);
            if (k.model) {
                IOSurfaceRef fIn = make_surface(io16), fOut = make_surface(io16);
                // Copy input
                IOSurfaceLock(fIn, 0, NULL);
                _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(fIn);
                IOSurfaceLock(surf_in, kIOSurfaceLockReadOnly, NULL);
                memcpy(p, IOSurfaceGetBaseAddress(surf_in), io16);
                IOSurfaceUnlock(surf_in, kIOSurfaceLockReadOnly, NULL);
                IOSurfaceUnlock(fIn, 0, NULL);

                IOSurfaceRef fins[]={fIn}, fouts[]={fOut};
                for (int i=0;i<T3_WARMUP;i++) ane_eval_io(&k,fins,1,fouts,1);
                uint64_t t0 = mach_absolute_time();
                for (int i=0;i<T3_ITERS;i++) ane_eval_io(&k,fins,1,fouts,1);
                fused_ffn_ms = tb_ms(mach_absolute_time()-t0)/T3_ITERS;
                double flops = 2.0*(2*HIDDEN*DIM + DIM*HIDDEN)*(double)SEQ;
                printf("  Fused FFN:    %.3f ms  %.1f GFLOPS\n", fused_ffn_ms, flops/fused_ffn_ms/1e6);
                CFRelease(fIn); CFRelease(fOut);
                cleanup_kern(&k);
            } else printf("  Fused FFN: COMPILE FAIL\n");

            // Separate: 3 conv dispatches + CPU SiLU
            {
                Kern k1 = compile_mil(gen_single_conv_fp16(HIDDEN,DIM,SEQ,@"w1"), @{@"@model_path/weights/w1.bin":@{@"offset":@0,@"data":w1_blob}});
                Kern k3 = compile_mil(gen_single_conv_fp16(HIDDEN,DIM,SEQ,@"w3"), @{@"@model_path/weights/w3.bin":@{@"offset":@0,@"data":w3_blob}});
                Kern k2 = compile_mil(gen_single_conv_fp16(DIM,HIDDEN,SEQ,@"w2"), @{@"@model_path/weights/w2.bin":@{@"offset":@0,@"data":w2_blob}});
                if (k1.model && k3.model && k2.model) {
                    size_t hio = HIDDEN*SEQ*2;
                    IOSurfaceRef sIn2 = make_surface(io16);
                    IOSurfaceRef sO1 = make_surface(hio), sO3 = make_surface(hio);
                    IOSurfaceRef sGate = make_surface(hio), sOut2 = make_surface(io16);

                    IOSurfaceLock(sIn2, 0, NULL);
                    _Float16 *p = (_Float16*)IOSurfaceGetBaseAddress(sIn2);
                    IOSurfaceLock(surf_in, kIOSurfaceLockReadOnly, NULL);
                    memcpy(p, IOSurfaceGetBaseAddress(surf_in), io16);
                    IOSurfaceUnlock(surf_in, kIOSurfaceLockReadOnly, NULL);
                    IOSurfaceUnlock(sIn2, 0, NULL);

                    IOSurfaceRef i1[]={sIn2}, o1a[]={sO1}, o3a[]={sO3};
                    IOSurfaceRef iGate[]={sGate}, o2a[]={sOut2};
                    for (int i=0;i<T3_WARMUP;i++) {
                        ane_eval_io(&k1,i1,1,o1a,1); ane_eval_io(&k3,i1,1,o3a,1);
                        ane_eval_io(&k2,iGate,1,o2a,1);
                    }
                    uint64_t t0 = mach_absolute_time();
                    for (int i=0;i<T3_ITERS;i++) {
                        ane_eval_io(&k1,i1,1,o1a,1);
                        ane_eval_io(&k3,i1,1,o3a,1);
                        // CPU SiLU + gate
                        IOSurfaceLock(sO1, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceLock(sO3, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceLock(sGate, 0, NULL);
                        _Float16 *h1p = (_Float16*)IOSurfaceGetBaseAddress(sO1);
                        _Float16 *h3p = (_Float16*)IOSurfaceGetBaseAddress(sO3);
                        _Float16 *gp = (_Float16*)IOSurfaceGetBaseAddress(sGate);
                        for (int j = 0; j < HIDDEN*SEQ; j++) {
                            float h1v = (float)h1p[j];
                            float sig = 1.0f / (1.0f + expf(-h1v));
                            gp[j] = (_Float16)(h1v * sig * (float)h3p[j]);
                        }
                        IOSurfaceUnlock(sGate, 0, NULL);
                        IOSurfaceUnlock(sO3, kIOSurfaceLockReadOnly, NULL);
                        IOSurfaceUnlock(sO1, kIOSurfaceLockReadOnly, NULL);
                        ane_eval_io(&k2,iGate,1,o2a,1);
                    }
                    sep_ffn_ms = tb_ms(mach_absolute_time()-t0)/T3_ITERS;
                    printf("  Separate FFN: %.3f ms\n", sep_ffn_ms);
                    CFRelease(sIn2); CFRelease(sO1); CFRelease(sO3); CFRelease(sGate); CFRelease(sOut2);
                } else printf("  Separate FFN: COMPILE FAIL\n");
                cleanup_kern(&k1); cleanup_kern(&k3); cleanup_kern(&k2);
            }
            if (fused_ffn_ms > 0 && sep_ffn_ms > 0)
                printf("  Speedup:      %.1fx\n", sep_ffn_ms/fused_ffn_ms);
        }

        // --- Summary ---
        printf("\n=== Tier 3 Summary ===\n");
        printf("%-20s %10s %10s %8s\n", "Config", "Separate", "Fused", "Speedup");
        printf("------------------------------------------------------\n");
        if (sep_qkv_ms>0 && fused_qkv_ms>0)
            printf("%-20s %9.3fms %9.3fms %6.1fx\n", "QKV (3→1)", sep_qkv_ms, fused_qkv_ms, sep_qkv_ms/fused_qkv_ms);
        if (sep_attn_ms>0 && fused_attn_ms>0)
            printf("%-20s %9.3fms %9.3fms %6.1fx\n", "Attention (4→1)", sep_attn_ms, fused_attn_ms, sep_attn_ms/fused_attn_ms);
        if (sep_ffn_ms>0 && fused_ffn_ms>0)
            printf("%-20s %9.3fms %9.3fms %6.1fx\n", "FFN (3→1)", sep_ffn_ms, fused_ffn_ms, sep_ffn_ms/fused_ffn_ms);
        double full_sep = sep_attn_ms + sep_ffn_ms;
        double full_fused = fused_attn_ms + fused_ffn_ms;
        if (full_sep>0 && full_fused>0) {
            printf("%-20s %9.3fms %9.3fms %6.1fx\n", "Full Layer (2 disp)", full_sep, full_fused, full_sep/full_fused);
            printf("\nFull layer: %.0f tokens/sec (fused), %.0f tokens/sec (separate)\n",
                SEQ/(full_fused/1000.0), SEQ/(full_sep/1000.0));
        }

        CFRelease(surf_in);
        free(x_in);
        free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
        printf("\nDONE\n");
    }
    return 0;
}
