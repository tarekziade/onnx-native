#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>

// Make sure this matches your ONNX Runtime version
#include "onnxruntime_c_api.h"

int main() {
    // -------------------------------------------------------------------------
    // 1. Dynamically load the ONNX Runtime library
    // -------------------------------------------------------------------------
    const char* lib_path = "libonnxruntime.1.22.0.dylib";  // or full path
    void* handle = dlopen(lib_path, RTLD_NOW);
    if (!handle) {
        std::cerr << "Failed to load " << lib_path << ": " << dlerror() << std::endl;
        return 1;
    }

    // -------------------------------------------------------------------------
    // 2. Get the symbol OrtGetApiBase, retrieve the OrtApi
    // -------------------------------------------------------------------------
    const OrtApiBase* (*OrtGetApiBase)();
    *(void**)(&OrtGetApiBase) = dlsym(handle, "OrtGetApiBase");
    if (!OrtGetApiBase) {
        std::cerr << "Failed to locate OrtGetApiBase: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    const OrtApiBase* api_base = OrtGetApiBase();
    const OrtApi* ort_api = api_base->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        std::cerr << "Failed to retrieve OrtApi." << std::endl;
        dlclose(handle);
        return 1;
    }

    // -------------------------------------------------------------------------
    // 3. Create an environment
    // -------------------------------------------------------------------------
    OrtEnv* env = nullptr;
    {
        OrtStatus* status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sst2_env", &env);
        if (status != nullptr) {
            const char* msg = ort_api->GetErrorMessage(status);
            std::cerr << "CreateEnv error: " << msg << std::endl;
            ort_api->ReleaseStatus(status);
            dlclose(handle);
            return 1;
        }
        std::cout << "Successfully created OrtEnv." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 4. Create a session for the DistilBERT SST-2 model (model.onnx)
    // -------------------------------------------------------------------------
    const char* model_path = "model.onnx";  // rename if your file differs
    OrtSessionOptions* session_options = nullptr;
    {
        OrtStatus* status = ort_api->CreateSessionOptions(&session_options);
        if (status != nullptr) {
            std::cerr << "CreateSessionOptions error: "
                      << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            ort_api->ReleaseEnv(env);
            dlclose(handle);
            return 1;
        }
    }

    // Optionally set optimizations, threading, etc.

    OrtSession* session = nullptr;
    {
        OrtStatus* status = ort_api->CreateSession(env, model_path, session_options, &session);
        if (status != nullptr) {
            std::cerr << "CreateSession error: "
                      << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            ort_api->ReleaseSessionOptions(session_options);
            ort_api->ReleaseEnv(env);
            dlclose(handle);
            return 1;
        }
        std::cout << "Successfully created ONNX Runtime session." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 5. Hardcode the tokens for "I think this is wonderful"
    //    input_ids = [101, 1045, 2228, 2023, 2003, 6919, 102]
    //    attention_mask = [1, 1, 1, 1, 1, 1, 1]
    // -------------------------------------------------------------------------
    std::vector<int64_t> input_ids       = {101, 1045, 2228, 2023, 2003, 6919, 102};
    std::vector<int64_t> attention_mask  = {   1,    1,    1,    1,    1,    1,   1};
    std::vector<int64_t> input_shape     = {1, (int64_t)input_ids.size()}; // [1,7]

    // Create a CPU memory info
    OrtMemoryInfo* memory_info = nullptr;
    {
        OrtStatus* status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            std::cerr << "CreateCpuMemoryInfo failed: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            // Cleanup
            ort_api->ReleaseSession(session);
            ort_api->ReleaseSessionOptions(session_options);
            ort_api->ReleaseEnv(env);
            dlclose(handle);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // 6. Create OrtValues for input_ids and attention_mask
    //    We use CreateTensorWithDataAsOrtValue since we have our own buffers.
    // -------------------------------------------------------------------------
    OrtValue* input_ids_ort = nullptr;
    {
        OrtStatus* status = ort_api->CreateTensorWithDataAsOrtValue(
            memory_info,
            input_ids.data(),
            input_ids.size() * sizeof(int64_t),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &input_ids_ort
        );
        if (status != nullptr) {
            std::cerr << "CreateTensorWithDataAsOrtValue for input_ids failed: "
                      << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            // Cleanup
            ort_api->ReleaseMemoryInfo(memory_info);
            ort_api->ReleaseSession(session);
            ort_api->ReleaseSessionOptions(session_options);
            ort_api->ReleaseEnv(env);
            dlclose(handle);
            return 1;
        }
    }

    OrtValue* attention_mask_ort = nullptr;
    {
        OrtStatus* status = ort_api->CreateTensorWithDataAsOrtValue(
            memory_info,
            attention_mask.data(),
            attention_mask.size() * sizeof(int64_t),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &attention_mask_ort
        );
        if (status != nullptr) {
            std::cerr << "CreateTensorWithDataAsOrtValue for attention_mask failed: "
                      << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            // Cleanup
            ort_api->ReleaseValue(input_ids_ort);
            ort_api->ReleaseMemoryInfo(memory_info);
            ort_api->ReleaseSession(session);
            ort_api->ReleaseSessionOptions(session_options);
            ort_api->ReleaseEnv(env);
            dlclose(handle);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // 7. Run the session (common DistilBERT input names: "input_ids", "attention_mask")
    //    Check your ONNX model's actual input node names if these differ.
    // -------------------------------------------------------------------------
    const char* input_names[] = {"input_ids", "attention_mask"};
    const OrtValue* input_values[] = {input_ids_ort, attention_mask_ort};

    // Usually the DistilBertForSequenceClassification output is named "logits"
    const char* output_names[] = {"logits"};
    OrtValue* output_tensor = nullptr;

    {
        OrtStatus* status = ort_api->Run(
            session,
            nullptr,        // Run options
            input_names,
            input_values,
            2,              // number of inputs
            output_names,
            1,              // number of outputs
            &output_tensor
        );
        if (status != nullptr) {
            std::cerr << "Session Run failed: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            // Cleanup
            ort_api->ReleaseValue(attention_mask_ort);
            ort_api->ReleaseValue(input_ids_ort);
            ort_api->ReleaseMemoryInfo(memory_info);
            ort_api->ReleaseSession(session);
            ort_api->ReleaseSessionOptions(session_options);
            ort_api->ReleaseEnv(env);
            dlclose(handle);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // 8. Interpret output (DistilBERT for SST-2 yields [batch_size, 2] logits)
    //    Index 0 -> NEGATIVE, Index 1 -> POSITIVE
    // -------------------------------------------------------------------------
    float* logits_data = nullptr;
    {
        OrtStatus* status = ort_api->GetTensorMutableData(output_tensor, (void**)&logits_data);
        if (status != nullptr) {
            std::cerr << "GetTensorMutableData failed: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
        } else {
            // Because we only ran a batch of size 1, we expect 2 logits: [logit_neg, logit_pos]
            float neg_logit = logits_data[0];
            float pos_logit = logits_data[1];
            std::cout << "Logits: NEG=" << neg_logit << " | POS=" << pos_logit << std::endl;

            // Decide sentiment by whichever logit is larger
            if (pos_logit > neg_logit) {
                std::cout << "Predicted sentiment: POSITIVE" << std::endl;
            } else {
                std::cout << "Predicted sentiment: NEGATIVE" << std::endl;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 9. Cleanup
    // -------------------------------------------------------------------------
    ort_api->ReleaseValue(output_tensor);
    ort_api->ReleaseValue(attention_mask_ort);
    ort_api->ReleaseValue(input_ids_ort);
    ort_api->ReleaseMemoryInfo(memory_info);

    ort_api->ReleaseSession(session);
    ort_api->ReleaseSessionOptions(session_options);
    ort_api->ReleaseEnv(env);

    dlclose(handle);

    std::cout << "Done." << std::endl;
    return 0;
}
