#include <dlfcn.h>
#include <iostream>
#include "onnxruntime_c_api.h"

// Global variables
static void* g_handle = nullptr;
static const OrtApi* g_ort_api = nullptr;
static OrtEnv* g_env = nullptr;

/**
 * @brief Initializes the ONNX Runtime API and environment by loading the shared library
 *        and creating a global OrtEnv if necessary.
 *
 * @param lib_path Path to the ONNX Runtime shared library (e.g. "libonnxruntime.dylib").
 * @return True on success; false if any step fails.
 */
bool initRuntime(const char* lib_path) {
    if (!g_handle) {
        g_handle = dlopen(lib_path, RTLD_NOW);
        if (!g_handle) {
            std::cerr << "Failed to load " << lib_path << ": " << dlerror() << std::endl;
            return false;
        }
    }

    // Retrieve OrtGetApiBase
    if (!g_ort_api) {
        const OrtApiBase* (*OrtGetApiBase)() = nullptr;
        *(void**)(&OrtGetApiBase) = dlsym(g_handle, "OrtGetApiBase");
        if (!OrtGetApiBase) {
            std::cerr << "Failed to locate OrtGetApiBase: " << dlerror() << std::endl;
            return false;
        }
        const OrtApiBase* api_base = OrtGetApiBase();
        g_ort_api = api_base->GetApi(ORT_API_VERSION);
        if (!g_ort_api) {
            std::cerr << "Failed to retrieve OrtApi." << std::endl;
            return false;
        }
    }

    // Create an environment if needed
    if (!g_env) {
        OrtStatus* status = g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "my_env", &g_env);
        if (status != nullptr) {
            std::cerr << "CreateEnv error: " << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            return false;
        }
        std::cout << "Successfully created OrtEnv." << std::endl;
    }

    return true; // success
}

/**
 * @brief Creates a new ONNX Runtime session options object with graph optimizations
 *        enabled.
 *
 * @return A pointer to the newly created OrtSessionOptions on success, or nullptr on failure.
 */
OrtSessionOptions* createSessionOptions() {
    if (!g_ort_api) {
        std::cerr << "createSessionOptions: g_ort_api is not initialized yet." << std::endl;
        return nullptr;
    }

    OrtSessionOptions* session_options = nullptr;
    OrtStatus* status = g_ort_api->CreateSessionOptions(&session_options);
    if (status != nullptr) {
        std::cerr << "CreateSessionOptions error: "
                  << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return nullptr;
    }

    status = g_ort_api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
    if (status != nullptr) {
        std::cerr << "SetSessionGraphOptimizationLevel error: "
                  << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return nullptr;
    }

    return session_options;
}

/**
 * @brief Creates a new ONNX Runtime session with the given model path and session options.
 *
 * @param model_path The path to the ONNX model file.
 * @param session_options A pointer to an existing OrtSessionOptions instance to configure the session.
 * @return A pointer to the newly created OrtSession on success, or nullptr on failure.
 */
OrtSession* createSession(const char* model_path, OrtSessionOptions* session_options) {
    if (!g_ort_api || !g_env) {
        std::cerr << "createSession: g_ort_api/g_env not initialized." << std::endl;
        return nullptr;
    }

    OrtSession* session = nullptr;
    OrtStatus* status = g_ort_api->CreateSession(g_env, model_path, session_options, &session);
    if (status != nullptr) {
        std::cerr << "CreateSession error: " << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return nullptr;
    }
    std::cout << "Successfully created ONNX Runtime session." << std::endl;
    return session;
}

/**
 * @brief Runs inference on a given session, producing a vector of floats (logits) from
 *        the provided input tensors.
 *
 * @param session Pointer to a valid OrtSession.
 * @param input_ids A vector representing the token IDs of the input text (shape: [1, sequence_length]).
 * @param attention_mask A vector representing the attention mask for the input (shape: [1, sequence_length]).
 * @return A vector of floats containing the logits on success, or an empty vector if any error occurs.
 */
std::vector<float> runInference(
    OrtSession* session,
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask)
{
    std::vector<float> logits;  // will return this

    if (!session) {
        std::cerr << "runInference: session pointer is null." << std::endl;
        return logits; // empty
    }
    if (!g_ort_api || !g_env) {
        std::cerr << "runInference: global OrtApi or OrtEnv is invalid." << std::endl;
        return logits; // empty
    }

    // Shape: [1, sequence_length]
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

    // 1. Create CPU memory info
    OrtMemoryInfo* memory_info = nullptr;
    {
        OrtStatus* status = g_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            std::cerr << "CreateCpuMemoryInfo failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            return logits;
        }
    }

    // 2. Create OrtValues for input_ids
    OrtValue* input_ids_ort = nullptr;
    {
        OrtStatus* status = g_ort_api->CreateTensorWithDataAsOrtValue(
            memory_info,
            (void*)input_ids.data(),
            input_ids.size() * sizeof(int64_t),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &input_ids_ort
        );
        if (status != nullptr) {
            std::cerr << "CreateTensorWithDataAsOrtValue for input_ids failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            g_ort_api->ReleaseMemoryInfo(memory_info);
            return logits;
        }
    }

    // 3. Create OrtValues for attention_mask
    OrtValue* attention_mask_ort = nullptr;
    {
        OrtStatus* status = g_ort_api->CreateTensorWithDataAsOrtValue(
            memory_info,
            (void*)attention_mask.data(),
            attention_mask.size() * sizeof(int64_t),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &attention_mask_ort
        );
        if (status != nullptr) {
            std::cerr << "CreateTensorWithDataAsOrtValue for attention_mask failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            g_ort_api->ReleaseValue(input_ids_ort);
            g_ort_api->ReleaseMemoryInfo(memory_info);
            return logits;
        }
    }

    // 4. Run the session
    const char* input_names[] = {"input_ids", "attention_mask"};
    const OrtValue* input_values[] = {input_ids_ort, attention_mask_ort};
    const char* output_names[] = {"logits"};
    OrtValue* output_tensor = nullptr;

    {
        OrtStatus* status = g_ort_api->Run(
            session,
            nullptr,  // Run options
            input_names,
            input_values,
            2,        // number of inputs
            output_names,
            1,        // number of outputs
            &output_tensor
        );
        if (status != nullptr) {
            std::cerr << "Session Run failed: " << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            // Cleanup
            g_ort_api->ReleaseValue(attention_mask_ort);
            g_ort_api->ReleaseValue(input_ids_ort);
            g_ort_api->ReleaseMemoryInfo(memory_info);
            return logits; // empty
        }
    }

    // 5. Extract the logits from the output
    {
        float* logits_data = nullptr;
        OrtStatus* status = g_ort_api->GetTensorMutableData(output_tensor, (void**)&logits_data);
        if (status != nullptr) {
            std::cerr << "GetTensorMutableData failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
        } else {
            // For DistilBERT on SST-2, shape is [1,2]: [neg_logit, pos_logit]
            // Copy them into a vector
            logits.resize(2);
            logits[0] = logits_data[0];  // negative
            logits[1] = logits_data[1];  // positive
        }
    }

    // Cleanup
    g_ort_api->ReleaseValue(output_tensor);
    g_ort_api->ReleaseValue(attention_mask_ort);
    g_ort_api->ReleaseValue(input_ids_ort);
    g_ort_api->ReleaseMemoryInfo(memory_info);

    return logits; // either empty if error or 2-element vector
}

int main() {
    // Initialize
    if (!initRuntime("libonnxruntime.1.22.0.dylib")) {
        std::cerr << "initRuntime failed." << std::endl;
        return 1;
    }

    // Session options
    OrtSessionOptions* session_options = createSessionOptions();
    if (!session_options) {
        std::cerr << "Failed to create session options." << std::endl;
        return 1;
    }

    // Create session
    OrtSession* session = createSession("model.onnx", session_options);
    if (!session) {
        std::cerr << "Failed to create session." << std::endl;
        g_ort_api->ReleaseSessionOptions(session_options);
        return 1;
    }

    // Prepare sample input data for "I think this is wonderful"
    std::vector<int64_t> input_ids       = {101, 1045, 2228, 2023, 2003, 6919, 102};
    std::vector<int64_t> attention_mask  = {   1,    1,    1,    1,    1,    1,   1};

    // Run inference
    std::vector<float> logits = runInference(session, input_ids, attention_mask);
    if (logits.empty()) {
        std::cerr << "runInference returned empty logits (error)!\n";
    } else {
        float neg_logit = logits[0];
        float pos_logit = logits[1];
        std::cout << "Logits: NEG=" << neg_logit << " | POS=" << pos_logit << std::endl;

        // Choose sentiment
        std::string sentiment = (pos_logit > neg_logit) ? "POSITIVE" : "NEGATIVE";
        std::cout << "Predicted sentiment: " << sentiment << std::endl;
    }

    // Cleanup
    g_ort_api->ReleaseSession(session);
    g_ort_api->ReleaseSessionOptions(session_options);
    g_ort_api->ReleaseEnv(g_env);
    dlclose(g_handle);

    std::cout << "Done." << std::endl;
    return 0;
}
