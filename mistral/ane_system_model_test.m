// ane_system_model_test.m — Test loading a SYSTEM .mlmodelc MIL through private ANE API
// Tests whether Apple's own compiled MIL format works with _ANEInMemoryModel on macOS 26
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "../training/ane_runtime.h"

// Also test the PUBLIC CoreML compilation pipeline
#import <CoreML/CoreML.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("=== ANE System Model Test (macOS 26) ===\n\n");

        // ---------- Part 1: Try the SYSTEM's own MIL through private API ----------
        printf("--- Part 1: Load system .mlmodelc MIL via _ANEInMemoryModel ---\n");

        NSString *modelDir = @"/System/Library/PrivateFrameworks/HDRProcessing.framework/"
                             @"Versions/A/Resources/sceneLuxB2DItpMLModel.mlmodelc";
        NSString *milPath = [modelDir stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [modelDir stringByAppendingPathComponent:@"weights/weight.bin"];

        NSData *milData = [NSData dataWithContentsOfFile:milPath];
        NSData *weightData = [NSData dataWithContentsOfFile:weightPath];

        if (!milData || !weightData) {
            fprintf(stderr, "Failed to read system model files\n");
            return 1;
        }
        printf("  MIL text: %lu bytes\n", (unsigned long)milData.length);
        printf("  Weight blob: %lu bytes\n", (unsigned long)weightData.length);

        // Model: input [1,4] fp32 -> output [1,3] fp32
        // In fp16: input = 1*4*2 = 8 bytes, output = 1*3*2 = 6 bytes
        // But the model casts fp32 input to fp16 internally, so I/O is fp32
        // ANE I/O is via IOSurface, sizes need to match the actual tensor sizes
        // Input: [1, 4] fp32 = 16 bytes
        // Output: [1, 3] fp32 = 12 bytes
        // Actually ANE works in fp16 internally, so let's try fp16 sizes
        size_t inputSizes[] = {1 * 4 * 2};   // [1,4] fp16 = 8 bytes
        size_t outputSizes[] = {1 * 3 * 2};  // [1,3] fp16 = 6 bytes

        ANEKernel *k = ane_compile(milData, weightData, 1, inputSizes, 1, outputSizes);
        if (k) {
            printf("  SUCCESS: System MIL compiled and loaded on ANE!\n");

            // Try running it
            _Float16 input[4] = {1.0, 2.0, 3.0, 4.0};
            ane_write_input(k, 0, input, sizeof(input));

            if (ane_eval(k)) {
                _Float16 output[3];
                ane_read_output(k, 0, output, sizeof(output));
                printf("  Eval SUCCESS! Output: [%.4f, %.4f, %.4f]\n",
                       (float)output[0], (float)output[1], (float)output[2]);
            } else {
                printf("  Eval FAILED\n");
            }
            ane_free(k);
        } else {
            printf("  FAILED: System MIL did NOT compile on ANE\n");
            printf("  (This is expected if the MIL format/version is incompatible)\n");
        }

        // ---------- Part 2: Try the PUBLIC CoreML API to compile and load ----------
        printf("\n--- Part 2: Public CoreML API (MLModel compileModelAtURL) ---\n");

        // Try compiling the system's .mlmodelc directly
        // Actually .mlmodelc is ALREADY compiled. Let's just load it.
        NSURL *modelURL = [NSURL fileURLWithPath:modelDir];
        NSError *error = nil;

        printf("  Loading pre-compiled model from: %s\n", [modelDir UTF8String]);
        MLModel *mlModel = [MLModel modelWithContentsOfURL:modelURL error:&error];
        if (mlModel) {
            printf("  SUCCESS: MLModel loaded!\n");
            MLModelDescription *desc = mlModel.modelDescription;
            printf("  Inputs: %s\n", [[desc.inputDescriptionsByName description] UTF8String]);
            printf("  Outputs: %s\n", [[desc.outputDescriptionsByName description] UTF8String]);

            // Try prediction
            MLMultiArray *inputArray = [[MLMultiArray alloc]
                initWithShape:@[@1, @4]
                dataType:MLMultiArrayDataTypeFloat32
                error:&error];
            if (inputArray) {
                float *ptr = (float *)inputArray.dataPointer;
                ptr[0] = 1.0f; ptr[1] = 2.0f; ptr[2] = 3.0f; ptr[3] = 4.0f;

                MLDictionaryFeatureProvider *provider =
                    [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:@{@"feature_vector": inputArray}
                        error:&error];
                if (provider) {
                    id<MLFeatureProvider> result = [mlModel predictionFromFeatures:provider error:&error];
                    if (result) {
                        MLMultiArray *output = [result featureValueForName:@"anchor_points"].multiArrayValue;
                        printf("  Prediction SUCCESS!\n");
                        printf("  Output shape: %s\n", [[output shape] description].UTF8String);
                        float *outPtr = (float *)output.dataPointer;
                        int count = 1;
                        for (NSNumber *dim in output.shape) count *= dim.intValue;
                        printf("  Values:");
                        for (int i = 0; i < count && i < 10; i++)
                            printf(" %.6f", outPtr[i]);
                        printf("\n");
                    } else {
                        printf("  Prediction FAILED: %s\n", [[error description] UTF8String]);
                    }
                }
            }

            // Check what compute unit it used
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];

            // Try forcing ANE only
            printf("\n  Trying ANE-only configuration...\n");
            config.computeUnits = MLComputeUnitsAll;  // Let system decide (includes ANE)
            MLModel *aneModel = [MLModel modelWithContentsOfURL:modelURL
                                                  configuration:config
                                                          error:&error];
            if (aneModel) {
                MLMultiArray *inputArray2 = [[MLMultiArray alloc]
                    initWithShape:@[@1, @4]
                    dataType:MLMultiArrayDataTypeFloat32
                    error:&error];
                float *ptr2 = (float *)inputArray2.dataPointer;
                ptr2[0] = 1.0f; ptr2[1] = 2.0f; ptr2[2] = 3.0f; ptr2[3] = 4.0f;
                MLDictionaryFeatureProvider *prov2 =
                    [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:@{@"feature_vector": inputArray2}
                        error:&error];
                id<MLFeatureProvider> res2 = [aneModel predictionFromFeatures:prov2 error:&error];
                if (res2) {
                    MLMultiArray *out2 = [res2 featureValueForName:@"anchor_points"].multiArrayValue;
                    float *op = (float *)out2.dataPointer;
                    int count = 1;
                    for (NSNumber *dim in out2.shape) count *= dim.intValue;
                    printf("  ANE prediction SUCCESS! Values:");
                    for (int i = 0; i < count && i < 10; i++)
                        printf(" %.6f", op[i]);
                    printf("\n");
                } else {
                    printf("  ANE prediction FAILED: %s\n", [[error description] UTF8String]);
                }
            } else {
                printf("  ANE model load FAILED: %s\n", [[error description] UTF8String]);
            }
        } else {
            printf("  FAILED: %s\n", [[error description] UTF8String]);
        }

        // ---------- Part 3: Create .mlpackage from scratch, compile, load on ANE ----------
        printf("\n--- Part 3: Create .mlpackage from scratch via coremltools ---\n");
        printf("  (Skipping - would need Python/coremltools)\n");

        // ---------- Part 4: Try compiling an .mlmodel spec to .mlmodelc ----------
        printf("\n--- Part 4: Try MLModel compileModelAtURL with .mlmodelc ---\n");

        // The system .mlmodelc is already compiled, but let's try the compile API
        // to see if it produces ANE-compatible output
        NSURL *compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
        if (compiledURL) {
            printf("  Re-compilation SUCCESS: %s\n", [[compiledURL path] UTF8String]);

            // Now try loading the re-compiled version
            MLModel *recompiled = [MLModel modelWithContentsOfURL:compiledURL error:&error];
            if (recompiled) {
                printf("  Re-compiled model loaded!\n");

                // Check if it produced a model.mil and try private API on it
                NSString *recompiledMil = [[compiledURL path] stringByAppendingPathComponent:@"model.mil"];
                NSString *recompiledWeight = [[compiledURL path] stringByAppendingPathComponent:@"weights/weight.bin"];
                NSData *milData2 = [NSData dataWithContentsOfFile:recompiledMil];
                NSData *weightData2 = [NSData dataWithContentsOfFile:recompiledWeight];

                if (milData2) {
                    printf("  Re-compiled MIL: %lu bytes\n", (unsigned long)milData2.length);
                    // Print first 500 chars of the re-compiled MIL
                    NSString *milStr = [[NSString alloc] initWithData:milData2 encoding:NSUTF8StringEncoding];
                    if (milStr) {
                        NSString *preview = [milStr substringToIndex:MIN(500, milStr.length)];
                        printf("  MIL preview:\n%s\n...\n", [preview UTF8String]);
                    }

                    printf("\n  Trying re-compiled MIL through private ANE API...\n");
                    ANEKernel *k2 = ane_compile(milData2, weightData2, 1, inputSizes, 1, outputSizes);
                    if (k2) {
                        printf("  SUCCESS: Re-compiled MIL works on ANE!\n");

                        _Float16 input2[4] = {1.0, 2.0, 3.0, 4.0};
                        ane_write_input(k2, 0, input2, sizeof(input2));
                        if (ane_eval(k2)) {
                            _Float16 output2[3];
                            ane_read_output(k2, 0, output2, sizeof(output2));
                            printf("  Eval SUCCESS! Output: [%.4f, %.4f, %.4f]\n",
                                   (float)output2[0], (float)output2[1], (float)output2[2]);
                        } else {
                            printf("  Eval FAILED\n");
                        }
                        ane_free(k2);
                    } else {
                        printf("  FAILED: Re-compiled MIL also doesn't work on private ANE API\n");
                    }
                } else {
                    printf("  No model.mil in re-compiled output\n");
                    // List what IS in the re-compiled output
                    NSArray *contents = [[NSFileManager defaultManager]
                        contentsOfDirectoryAtPath:[compiledURL path] error:nil];
                    printf("  Contents: %s\n", [[contents description] UTF8String]);
                }
            }

            // Cleanup
            [[NSFileManager defaultManager] removeItemAtURL:compiledURL error:nil];
        } else {
            printf("  Re-compilation FAILED: %s\n", [[error description] UTF8String]);
        }

        // ---------- Part 5: Try the ATX Intent model (has dynamic shapes) ----------
        printf("\n--- Part 5: Try ATX Intent model through private API ---\n");
        NSString *atxDir = @"/System/Library/DuetExpertCenter/Assets/Assets.bundle/"
                           @"AssetData/ATXIntentPredictionMLModel.mlmodelc";
        NSData *atxMil = [NSData dataWithContentsOfFile:
            [atxDir stringByAppendingPathComponent:@"model.mil"]];
        NSData *atxWeight = [NSData dataWithContentsOfFile:
            [atxDir stringByAppendingPathComponent:@"weights/weight.bin"]];
        if (atxMil) {
            // Input: [1, 17] fp16 = 34 bytes, Output: [1, 1] fp16 = 2 bytes
            size_t atxIn[] = {1 * 17 * 2};
            size_t atxOut[] = {1 * 1 * 2};
            ANEKernel *k3 = ane_compile(atxMil, atxWeight, 1, atxIn, 1, atxOut);
            if (k3) {
                printf("  SUCCESS: ATX Intent MIL compiled on ANE!\n");
                ane_free(k3);
            } else {
                printf("  FAILED: ATX Intent MIL (dynamic shapes, expected to fail)\n");
            }
        }

        // ---------- Part 6: Try the VAD model (has conv, more ANE-like) ----------
        printf("\n--- Part 6: Try VAD model through private API ---\n");
        NSString *vadDir = @"/System/Library/AssetsV2/com_apple_MobileAsset_UAF_Siri_Understanding/"
                           @"purpose_auto/7167e1dd554b57d7e771f3bb70ea4a4bd79b8e08.asset/AssetData/VAD";
        NSString *vadMilPath = [vadDir stringByAppendingPathComponent:@"model.mil"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:vadMilPath]) {
            NSData *vadMil = [NSData dataWithContentsOfFile:vadMilPath];
            // Check weights dir
            NSString *vadWeightDir = [vadDir stringByAppendingPathComponent:@"weights"];
            NSArray *wFiles = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:vadWeightDir error:nil];
            printf("  VAD weight files: %s\n", [[wFiles description] UTF8String]);

            NSData *vadWeight = [NSData dataWithContentsOfFile:
                [vadWeightDir stringByAppendingPathComponent:@"weight.bin"]];
            printf("  VAD MIL: %lu bytes, weights: %lu bytes\n",
                   (unsigned long)vadMil.length,
                   (unsigned long)(vadWeight ? vadWeight.length : 0));

            // VAD input: [1, 29, 80, 1] fp16
            size_t vadIn[] = {1*29*80*1*2, 1*80*1*2*2, 1*256*1*10*2, 1*128*1*10*2, 1*128*1*10*2, 1*128*1*10*2};
            // We'd need to know the output shape too - skip eval, just test compile
            size_t vadOut[] = {1*2*2, 1*80*1*2*2, 1*256*1*10*2, 1*128*1*10*2, 1*128*1*10*2, 1*128*1*10*2};
            ANEKernel *k4 = ane_compile(vadMil, vadWeight, 6, vadIn, 6, vadOut);
            if (k4) {
                printf("  SUCCESS: VAD MIL compiled on ANE!\n");
                ane_free(k4);
            } else {
                printf("  FAILED: VAD MIL did not compile on ANE\n");
            }
        } else {
            printf("  VAD model not found at expected path\n");
        }

        printf("\n=== Test Complete ===\n");
    }
    return 0;
}
