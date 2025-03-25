
#include <iostream>
#include <vector>
#include <string>
#include <dlfcn.h>
#include "onnxruntime_c_api.h"

// Global variables (as in your example)
static void* g_handle = nullptr;
static const OrtApi* g_ort_api = nullptr;
static OrtEnv* g_env = nullptr;

//------------------------------------------------------------------------------
// 1) Initialize the runtime by dynamically loading the ONNX Runtime library
//------------------------------------------------------------------------------
bool initRuntime(const char* lib_path) {
    if (!g_handle) {
        g_handle = dlopen(lib_path, RTLD_NOW);
        if (!g_handle) {
            std::cerr << "Failed to load " << lib_path << ": " << dlerror() << std::endl;
            return false;
        }
    }

    // Retrieve OrtGetApiBase symbol
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
        std::cout << "Successfully created OrtEnv.\n";
    }

    return true; // success
}

//------------------------------------------------------------------------------
// 2) Create a session options object
//------------------------------------------------------------------------------
OrtSessionOptions* createSessionOptions() {
    if (!g_ort_api) {
        std::cerr << "createSessionOptions: g_ort_api is not initialized.\n";
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

    // Enable all graph optimizations
    status = g_ort_api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
    if (status != nullptr) {
        std::cerr << "SetSessionGraphOptimizationLevel error: "
                  << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        g_ort_api->ReleaseSessionOptions(session_options);
        return nullptr;
    }

    return session_options;
}

//------------------------------------------------------------------------------
// 3) Create a session with a given model path
//------------------------------------------------------------------------------
OrtSession* createSession(const char* model_path, OrtSessionOptions* session_options) {
    if (!g_ort_api || !g_env) {
        std::cerr << "createSession: g_ort_api/g_env not initialized.\n";
        return nullptr;
    }

    OrtSession* session = nullptr;
    OrtStatus* status = g_ort_api->CreateSession(g_env, model_path, session_options, &session);
    if (status != nullptr) {
        std::cerr << "CreateSession error: " << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return nullptr;
    }
    std::cout << "Successfully created ONNX Runtime session.\n";
    return session;
}

//------------------------------------------------------------------------------
// Helper to retrieve all input names
//------------------------------------------------------------------------------
std::vector<std::string> getInputNames(OrtSession* session) {
    std::vector<std::string> names;
    if (!session) return names;

    size_t num_input_nodes = 0;
    OrtStatus* status = g_ort_api->SessionGetInputCount(session, &num_input_nodes);
    if (status != nullptr) {
        std::cerr << "SessionGetInputCount failed: "
                  << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return names;
    }

    // Create default allocator
    OrtAllocator* allocator = nullptr;
    status = g_ort_api->GetAllocatorWithDefaultOptions(&allocator);
    if (status != nullptr) {
        std::cerr << "GetAllocatorWithDefaultOptions failed: "
                  << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return names;
    }

    // Retrieve each input name
    names.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        char* input_name = nullptr;
        status = g_ort_api->SessionGetInputName(session, i, allocator, &input_name);
        if (status != nullptr) {
            std::cerr << "SessionGetInputName failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            continue;
        }
        // Copy into a std::string
        names.emplace_back(input_name);

        // Free the char* allocated by ORT
        g_ort_api->AllocatorFree(allocator, input_name);
    }

    return names;
}

//------------------------------------------------------------------------------
// Helper to retrieve all output names
//------------------------------------------------------------------------------
std::vector<std::string> getOutputNames(OrtSession* session) {
    std::vector<std::string> names;
    if (!session) return names;

    size_t num_output_nodes = 0;
    OrtStatus* status = g_ort_api->SessionGetOutputCount(session, &num_output_nodes);
    if (status != nullptr) {
        std::cerr << "SessionGetOutputCount failed: "
                  << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return names;
    }

    // Create default allocator
    OrtAllocator* allocator = nullptr;
    status = g_ort_api->GetAllocatorWithDefaultOptions(&allocator);
    if (status != nullptr) {
        std::cerr << "GetAllocatorWithDefaultOptions failed: "
                  << g_ort_api->GetErrorMessage(status) << std::endl;
        g_ort_api->ReleaseStatus(status);
        return names;
    }

    // Retrieve each output name
    names.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        char* output_name = nullptr;
        status = g_ort_api->SessionGetOutputName(session, i, allocator, &output_name);
        if (status != nullptr) {
            std::cerr << "SessionGetOutputName failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            continue;
        }
        // Copy into a std::string
        names.emplace_back(output_name);

        // Free the char* allocated by ORT
        g_ort_api->AllocatorFree(allocator, output_name);
    }

    return names;
}

//------------------------------------------------------------------------------
// 4) Runs inference given a session, discovered input/output names,
//    and sample data (two inputs).
//------------------------------------------------------------------------------
std::vector<float> runInference(
    OrtSession* session,
    const std::vector<std::string>& input_names,  // e.g. [ "input_ids", "attention_mask" ]
    const std::vector<std::string>& output_names, // e.g. [ "logits" ]
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask)
{
    std::vector<float> logits;
    if (!session) {
        std::cerr << "runInference: session is null.\n";
        return logits;
    }
    if (input_names.size() < 2) {
        std::cerr << "runInference: expected at least 2 input names.\n";
        return logits;
    }
    if (output_names.empty()) {
        std::cerr << "runInference: expected at least 1 output name.\n";
        return logits;
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

    // 2. Create OrtValue for input_ids
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
            std::cerr << "CreateTensorWithDataAsOrtValue(input_ids) failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            g_ort_api->ReleaseMemoryInfo(memory_info);
            return logits;
        }
    }

    // 3. Create OrtValue for attention_mask
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
            std::cerr << "CreateTensorWithDataAsOrtValue(attention_mask) failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            g_ort_api->ReleaseValue(input_ids_ort);
            g_ort_api->ReleaseMemoryInfo(memory_info);
            return logits;
        }
    }

    // 4. Run the session
    // Convert std::string -> const char* for the two input names
    const char* input_name_array[2] = { input_names[0].c_str(), input_names[1].c_str() };
    const OrtValue* input_values[2] = { input_ids_ort, attention_mask_ort };

    // We'll assume there's exactly 1 output; you can adapt if there's more
    const char* output_name_array[1] = { output_names[0].c_str() };

    OrtValue* output_tensor = nullptr;
    {
        OrtStatus* status = g_ort_api->Run(
            session,
            nullptr,                      // run options
            input_name_array,             // input names
            input_values,                 // input OrtValues
            2,                            // number of inputs
            output_name_array,            // output name
            1,                            // number of outputs
            &output_tensor
        );
        if (status != nullptr) {
            std::cerr << "Session Run failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);

            // Cleanup
            g_ort_api->ReleaseValue(attention_mask_ort);
            g_ort_api->ReleaseValue(input_ids_ort);
            g_ort_api->ReleaseMemoryInfo(memory_info);
            return logits;
        }
    }

    // 5. Extract the logits from output_tensor
    {
        float* output_data = nullptr;
        OrtStatus* status = g_ort_api->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            std::cerr << "GetTensorMutableData failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
        } else {
            // For DistilBERT on SST-2, shape [1, 2]: [neg_logit, pos_logit]
            // We'll just copy the first 2 entries (or adapt to the correct shape in a real scenario).
            logits.resize(2);
            logits[0] = output_data[0];
            logits[1] = output_data[1];
        }
    }

    // Cleanup
    g_ort_api->ReleaseValue(output_tensor);
    g_ort_api->ReleaseValue(attention_mask_ort);
    g_ort_api->ReleaseValue(input_ids_ort);
    g_ort_api->ReleaseMemoryInfo(memory_info);

    return logits; // either empty or [neg_logit, pos_logit]
}

//------------------------------------------------------------------------------
// main
//------------------------------------------------------------------------------
int main() {
    // 1) Load the ONNX Runtime library and create global Env
    if (!initRuntime("libonnxruntime.1.22.0.dylib")) {
        std::cerr << "initRuntime failed.\n";
        return 1;
    }

    // 2) Create session options
    OrtSessionOptions* session_options = createSessionOptions();
    if (!session_options) {
        std::cerr << "Failed to create session options.\n";
        return 1;
    }

    // 3) Create the session
    OrtSession* session = createSession("model.onnx", session_options);
    if (!session) {
        std::cerr << "Failed to create session.\n";
        g_ort_api->ReleaseSessionOptions(session_options);
        return 1;
    }

    // 4) Discover the input/output names from the model
    std::vector<std::string> input_names  = getInputNames(session);
    std::vector<std::string> output_names = getOutputNames(session);

    std::cout << "Discovered " << input_names.size() << " input(s):\n";
    for (auto& nm : input_names)
        std::cout << "  " << nm << "\n";

    std::cout << "Discovered " << output_names.size() << " output(s):\n";
    for (auto& nm : output_names)
        std::cout << "  " << nm << "\n";

    // Prepare sample input data: "I think this is wonderful"
    // (IDs correspond to a DistilBERT tokenizer for demonstration)
    std::vector<int64_t> input_ids      = {101, 1045, 2228, 2023, 2003, 6919, 102};
    std::vector<int64_t> attention_mask = {   1,    1,    1,    1,    1,    1,   1};

    // 5) Run inference using the discovered names
    std::vector<float> logits = runInference(session, input_names, output_names, input_ids, attention_mask);
    if (logits.empty()) {
        std::cerr << "runInference returned empty logits (error).\n";
    } else {
        float neg_logit = logits[0];
        float pos_logit = logits[1];
        std::cout << "Logits: NEG=" << neg_logit << " | POS=" << pos_logit << std::endl;

        // Choose sentiment
        std::string sentiment = (pos_logit > neg_logit) ? "POSITIVE" : "NEGATIVE";
        std::cout << "Predicted sentiment: " << sentiment << std::endl;
    }

    // Cleanup session, options, env, and dynamic library
    g_ort_api->ReleaseSession(session);
    g_ort_api->ReleaseSessionOptions(session_options);
    g_ort_api->ReleaseEnv(g_env);
    dlclose(g_handle);

    std::cout << "Done.\n";
    return 0;
}


