// test_ane_ops.h -- ANE MIL generators for individual Mistral ops under test
// Each function generates a MIL program for a single op that can be compiled
// and run on ANE, then compared against the CPU reference.
#pragma once
#import <Foundation/Foundation.h>
#include <math.h>

// Shared MIL header
#define TEST_MIL_HDR \
    @"program(1.3)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define TEST_CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// ---------- RMSNorm MIL ----------
// Input: x [1, D, 1, S] fp16 (ANE channel-first)
// Weights: rms_w [1, D, 1, 1] fp16 baked
// Output: [1, D, 1, S] fp16
static NSString *gen_test_rmsnorm(int D, int S) {
    float invd = 1.0f / (float)D;
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n"
        "        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"
        "        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"
        "        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n"
        "        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n"
        "        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n"
        "        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"
        "        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n"
        "        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"
        "        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n"
        "        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), "
        "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> out = mul(x=xr,y=rw)[name=string(\"out\")];\n"
        "    } -> (out);\n}\n",
        TEST_MIL_HDR, D, S,
        D, S,
        S, invd, S, S, S,
        D, S,
        D, D,
        D, S];
}

// ---------- SiLU MIL ----------
// Input: x [1, C, 1, S] fp16 (4D for ANE)
// Output: silu(x) = x * sigmoid(x)
// Note: ANE requires at least one baked const to compile. We bake a dummy
// scale=1.0 and multiply through to keep the compiler happy.
static NSString *gen_test_silu(int C, int S) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [1,1,1,1]> one = const()[name=string(\"one\"), "
        "val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> x2 = mul(x=x,y=one)[name=string(\"sc\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=x2)[name=string(\"sg\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> out = mul(x=x2,y=sig)[name=string(\"out\")];\n"
        "    } -> (out);\n}\n",
        TEST_MIL_HDR, C, S, C, S, C, S, C, S];
}

// ---------- Softmax MIL ----------
// Input: x [1, H, S, S] fp16 (attention scores)
// Output: softmax along last axis
// Bake a dummy weight (zeros added to input) to satisfy ANE compiler.
static NSString *gen_test_softmax(int H, int S) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> x) {\n"
        "        tensor<fp16, [1,1,1,1]> zero = const()[name=string(\"zero\"), "
        "val=tensor<fp16, [1,1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,%d,%d]> x2 = add(x=x,y=zero)[name=string(\"bias\")];\n"
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"
        "        tensor<fp16, [1,%d,%d,%d]> out = softmax(axis=sax,x=x2)[name=string(\"out\")];\n"
        "    } -> (out);\n}\n",
        TEST_MIL_HDR, H, S, S, H, S, S, H, S, S];
}

// ---------- Conv (matmul) MIL ----------
// Input: x [1, IC, 1, S] fp16
// Weight: W [OC, IC, 1, 1] fp16 baked
// Output: [1, OC, 1, S] fp16
static NSString *gen_test_conv(int IC, int OC, int S) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        TEST_CONV_CONST
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        "[name=string(\"out\")];\n"
        "    } -> (out);\n}\n",
        TEST_MIL_HDR, IC, S, OC, IC, OC, IC, OC, S];
}

// ---------- SDPA (scaled dot-product attention) MIL ----------
// Inputs: Q [1, H, S, HD], K [1, H, S, HD], V [1, H, S, HD] all fp16
// Causal mask baked as weight
// Output: [1, H, S, HD] fp16
static NSString *gen_test_sdpa(int H, int S, int HD) {
    float sc = 1.0f / sqrtf((float)HD);
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp16, [1, %d, %d, %d]> q, "
        "tensor<fp16, [1, %d, %d, %d]> k, "
        "tensor<fp16, [1, %d, %d, %d]> v) {\n"
        "        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"
        "        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"
        "        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n"
        "        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n"
        "        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n"
        "        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), "
        "val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n"
        "        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"
        "        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n"
        "        tensor<fp16, [1,%d,%d,%d]> out = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n"
        "    } -> (out);\n}\n",
        TEST_MIL_HDR,
        H, S, HD,
        H, S, HD,
        H, S, HD,
        H, S, S, sc,
        H, S, S,
        S, S, S, S,
        H, S, S,
        H, S, S,
        H, S, HD];
}

// ---------- SwiGLU FFN MIL ----------
// Input: x [1, D, 1, S] fp16
// Weights: W1, W3, W2 in separate blob files (matching training code pattern)
// out = W2 @ (silu(W1 @ x) * (W3 @ x))
static NSString *gen_test_ffn(int D, int H, int S) {
    return [NSString stringWithFormat:
        @"%@"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        TEST_CONV_CONST
        "        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=x)"
        "[name=string(\"c1\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=x)"
        "[name=string(\"c3\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)"
        "[name=string(\"c2\")];\n"
        "    } -> (out);\n}\n",
        TEST_MIL_HDR, D, S,
        H, D, H, D,
        H, D, H, D,
        D, H, D, H,
        H, S, H, S,
        H, S, H, S, H, S,
        D, S];
}
