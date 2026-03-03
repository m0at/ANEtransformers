// mistral_mil.h — MIL template generation for Mistral ANE kernels
// Uses 4D tensors [1, 1, rows, cols] with matmul for ANE compatibility.
// Weights passed as function inputs for multi-layer reuse.
#pragma once
#import <Foundation/Foundation.h>

#define MIL_HEADER \
    @"program(1.3)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n"

#define MIL_MATMUL_CONSTS \
    "        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n" \
    "        bool bT = const()[name = string(\"bT\"), val = bool(true)];\n"

// y = W @ x using 4D matmul
// x: [1, 1, in_ch, S] fp16 — activation columns
// W: [1, 1, out_ch, in_ch] fp16 — weight matrix
// y: [1, 1, out_ch, S] fp16 — output
// matmul(x=W, y=x) with transpose_x=false, transpose_y=false
//   W is [1,1,out,in] @ x is [1,1,in,S] => [1,1,out,S]
static NSString *mistral_mil_matmul(int in_ch, int out_ch, int S) {
    return [NSString stringWithFormat:
        @"%@{\n"
        "    func main<ios18>(tensor<fp16, [1, 1, %d, %d]> x, tensor<fp16, [1, 1, %d, %d]> W) {\n"
        MIL_MATMUL_CONSTS
        "        tensor<fp16, [1, 1, %d, %d]> y = matmul(transpose_x = bF, transpose_y = bF, x = W, y = x)[name = string(\"mm\")];\n"
        "    } -> (y);\n"
        "}\n",
        MIL_HEADER,
        in_ch, S, out_ch, in_ch,
        out_ch, S];
}

// Fused gate+up: h1 = W1 @ x, h3 = W3 @ x
// x: [1, 1, dim, S] fp16
// W1, W3: [1, 1, hidden, dim] fp16
// h1, h3: [1, 1, hidden, S] fp16
static NSString *mistral_mil_gate_up(int dim, int hidden, int S) {
    return [NSString stringWithFormat:
        @"%@{\n"
        "    func main<ios18>(tensor<fp16, [1, 1, %d, %d]> x, "
        "tensor<fp16, [1, 1, %d, %d]> W1, "
        "tensor<fp16, [1, 1, %d, %d]> W3) {\n"
        MIL_MATMUL_CONSTS
        "        tensor<fp16, [1, 1, %d, %d]> h1 = matmul(transpose_x = bF, transpose_y = bF, x = W1, y = x)[name = string(\"mm_gate\")];\n"
        "        tensor<fp16, [1, 1, %d, %d]> h3 = matmul(transpose_x = bF, transpose_y = bF, x = W3, y = x)[name = string(\"mm_up\")];\n"
        "    } -> (h1, h3);\n"
        "}\n",
        MIL_HEADER,
        dim, S, hidden, dim, hidden, dim,
        hidden, S, hidden, S];
}
