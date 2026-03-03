CC = xcrun clang
CFLAGS = -O2 -ffast-math -fobjc-arc -Wall -Wno-unused-function -Wno-unused-variable -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate -framework Metal -framework MetalPerformanceShaders
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS = gguf_loader.h dequant.h kv_cache.h tokenizer.h mistral_model.h mistral_mil.h ane_mil_gen_mistral.h mistral_ane_prefill.h metal_matvec.h

.PHONY: all clean ane_tests

all: mistral

mistral: mistral_infer.m $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

ane_probe: ../tests/ane_baked_probe.m ane_mil_gen_mistral.h ../training/ane_runtime.h dequant.h
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

matmul_probe: ../tests/ane_matmul_probe.m ../training/ane_runtime.h ../training/ane_mil_gen.h
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

# Metal shader compilation (requires Xcode, not just CLT)
# If available: make metallib
q4_0_metal.metallib: q4_0_metal.metal
	xcrun -sdk macosx metal -O2 -c -o q4_0_metal.air $<
	xcrun -sdk macosx metallib -o $@ q4_0_metal.air
	rm -f q4_0_metal.air

test_prefill: ../tests/test_blas_prefill.m $(HEADERS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

transpose_test: ../tests/ane_transpose_test.m ane_mil_gen_mistral.h ../training/ane_runtime.h
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

ane_layer_test: ../tests/ane_layer_test.m $(HEADERS) ../training/ane_runtime.h
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

ane_2phase_test: ../tests/ane_2phase_test.m ane_mil_gen_mistral.h ../training/ane_runtime.h
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

ane_prefill_compare: ../tests/ane_prefill_compare.m $(HEADERS) ../training/ane_runtime.h
	$(CC) $(CFLAGS) -I. -o $@ $< $(LDFLAGS)

ane_tests: transpose_test ane_layer_test ane_2phase_test ane_prefill_compare

clean:
	rm -f mistral q4_0_metal.metallib q4_0_metal.air test_prefill ane_probe matmul_probe transpose_test ane_layer_test ane_2phase_test ane_prefill_compare
