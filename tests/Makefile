CC = xcrun clang
CFLAGS = -O2 -fobjc-arc -Wall -Wno-unused-function -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate
LDFLAGS = -ldl

all: test_mistral transpose_test

test_mistral: test_mistral.m test_harness.h test_cpu_ref.h test_ane_ops.h
	$(CC) $(CFLAGS) -o $@ $< $(FRAMEWORKS) $(LDFLAGS)

transpose_test: ane_transpose_test.m ../training/ane_runtime.h ../mistral/ane_mil_gen_mistral.h
	$(CC) $(CFLAGS) -o $@ $< $(FRAMEWORKS) $(LDFLAGS)

run: test_mistral
	./test_mistral

run-unit: test_mistral
	./test_mistral --level 1

run-layer: test_mistral
	./test_mistral --level 2

run-integration: test_mistral
	./test_mistral --level 3

run-perf: test_mistral
	./test_mistral --level 4

run-correct: test_mistral
	./test_mistral --level 5

clean:
	rm -f test_mistral transpose_test

.PHONY: all run run-unit run-layer run-integration run-perf run-correct clean
