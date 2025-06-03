#include <iostream>
#include <vector>
#include <string>
#include <numeric>   // for std::accumulate
#include <chrono>    // for timing
#include <dlfcn.h>
#include <algorithm>
#include "onnxruntime_c_api.h"
#include <iostream>
#include <fstream>

#include "onnx.pb.h"
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

// Global variables
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
// 4) Retrieve model input/output names
//------------------------------------------------------------------------------
std::pair<std::vector<std::string>, std::vector<std::string>>
getModelInputOutputNames(OrtSession* session)
{
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    if (!session) {
        std::cerr << "[getModelInputOutputNames] session is null.\n";
        return {input_names, output_names};
    }

    // 1) Get input count
    size_t num_input_nodes = 0;
    {
        OrtStatus* status = g_ort_api->SessionGetInputCount(session, &num_input_nodes);
        if (status != nullptr) {
            std::cerr << "SessionGetInputCount failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            return {input_names, output_names};
        }
    }

    // 2) Get output count
    size_t num_output_nodes = 0;
    {
        OrtStatus* status = g_ort_api->SessionGetOutputCount(session, &num_output_nodes);
        if (status != nullptr) {
            std::cerr << "SessionGetOutputCount failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            return {input_names, output_names};
        }
    }

    // 3) Create default allocator
    OrtAllocator* allocator = nullptr;
    {
        OrtStatus* status = g_ort_api->GetAllocatorWithDefaultOptions(&allocator);
        if (status != nullptr) {
            std::cerr << "GetAllocatorWithDefaultOptions failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            return {input_names, output_names};
        }
    }

    // 4) Retrieve all input names
    input_names.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        char* name = nullptr;
        OrtStatus* status = g_ort_api->SessionGetInputName(session, i, allocator, &name);
        if (status != nullptr) {
            std::cerr << "SessionGetInputName failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            continue;
        }
        input_names.emplace_back(name);
        (void)g_ort_api->AllocatorFree(allocator, name);
    }

    // 5) Retrieve all output names
    output_names.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        char* name = nullptr;
        OrtStatus* status = g_ort_api->SessionGetOutputName(session, i, allocator, &name);
        if (status != nullptr) {
            std::cerr << "SessionGetOutputName failed: "
                      << g_ort_api->GetErrorMessage(status) << std::endl;
            g_ort_api->ReleaseStatus(status);
            continue;
        }
        output_names.emplace_back(name);
        (void)g_ort_api->AllocatorFree(allocator, name);
    }

    return {input_names, output_names};
}

//------------------------------------------------------------------------------
// 5) Run inference
//------------------------------------------------------------------------------
std::vector<float> runInference(
    OrtSession* session,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
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
    const char* input_name_array[2] = { input_names[0].c_str(), input_names[1].c_str() };
    const OrtValue* input_values[2] = { input_ids_ort, attention_mask_ort };
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

struct AutoTime {
  AutoTime(const char* str)
  :start(std::chrono::high_resolution_clock::now())
  , str(str) { }
  ~AutoTime() {
    printf("%s: %0.02lfms\n", str, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-start).count());
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  const char* str;
};

bool loadFileToBuffer(const std::string& path, std::vector<char>& buffer) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size);
    return file.read(buffer.data(), size).good();
}

bool load_external_data_for_model(onnx::ModelProto& model, const std::string& base_dir) {
    for (auto& tensor : *model.mutable_graph()->mutable_initializer()) {
        if (tensor.data_location() != onnx::TensorProto_DataLocation_EXTERNAL) continue;

        std::string location;
        size_t offset = 0, length = 0;

        for (const auto& entry : tensor.external_data()) {
            if (entry.key() == "location") location = entry.value();
            else if (entry.key() == "offset") offset = std::stoull(entry.value());
            else if (entry.key() == "length") length = std::stoull(entry.value());
        }

        std::ifstream in(base_dir + "/" + location, std::ios::binary);
        if (!in) {
            std::cerr << "Failed to open external data file: " << location << "\n";
            return false;
        }

        in.seekg(offset);
        std::string raw_data(length, '\0');
        in.read(&raw_data[0], length);
        tensor.set_raw_data(raw_data);
        tensor.set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
        tensor.clear_external_data();
    }
    return true;
}

int main() {
  {
    AutoTime t("dlopen(libonnxruntime)");
    if (!initRuntime("libonnxruntime.so")) return 1;
  }

  std::vector<char> graph_buf;

    // Step 1: Load graph.onnx
    onnx::ModelProto model;
    {
      AutoTime t("stream loading graph");
      std::ifstream in("graph.onnx", std::ios::binary);
      if (!in || !model.ParseFromIstream(&in)) {
          std::cerr << "Failed to load graph.onnx\n";
          return 1;
      }
    }

    {
      AutoTime t("loading weights");
      // Step 2: Load external weights from weights.data
      if (!load_external_data_for_model(model, ".")) {
          std::cerr << "Failed to load external weights\n";
          return 1;
      }
    }

    std::string model_buf;
    {
      AutoTime t("serializing into mem");
      // Step 3: Serialize full model to memory
      model_buf = model.SerializeAsString();
    }

    OrtMemoryInfo* mem_info = nullptr;
    (void)g_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

    // Step 4: Create session
    OrtSessionOptions* session_options = createSessionOptions();
    OrtSession* session = nullptr;
    OrtStatus* status = g_ort_api->CreateSessionFromArray(
        g_env, model_buf.data(), model_buf.size(), session_options, &session);
    if (status) {
        std::cerr << "Session creation failed: " << g_ort_api->GetErrorMessage(status) << "\n";
        g_ort_api->ReleaseStatus(status);
        return 1;
    }

    // 4) Discover the input/output names from the model
    auto [input_names, output_names] = getModelInputOutputNames(session);

    std::cout << "Discovered " << input_names.size() << " input(s):\n";
    for (auto& nm : input_names) {
        std::cout << "  " << nm << "\n";
    }

    std::cout << "Discovered " << output_names.size() << " output(s):\n";
    for (auto& nm : output_names) {
        std::cout << "  " << nm << "\n";
    }

    // Prepare sample input data: "I think this is wonderful"
    // (IDs correspond to a DistilBERT tokenizer, for demonstration)
    std::vector<int64_t> input_ids      = {101, 1045, 2228, 2023, 2003, 6919, 102};
    std::vector<int64_t> attention_mask = {   1,    1,    1,    1,    1,    1,   1};

    // We'll run the inference 25 times and measure durations
    const size_t NUM_RUNS = 25;
    std::vector<double> timings(NUM_RUNS, 0.0);

    // Loop 25 times
    for (size_t i = 0; i < NUM_RUNS; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<float> logits = runInference(session, input_names, output_names, input_ids, attention_mask);

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        timings[i] = elapsed_ms;

        // For debugging, you might print an example result each time:
        if (!logits.empty()) {
            float neg_logit = logits[0];
            float pos_logit = logits[1];
            std::string sentiment = (pos_logit > neg_logit) ? "POSITIVE" : "NEGATIVE";
            std::cout << "Run #" << (i + 1) << ": NEG=" << neg_logit
                      << ", POS=" << pos_logit
                      << ", sentiment=" << sentiment
                      << ", time=" << elapsed_ms << " ms\n";
        } else {
            std::cerr << "Run #" << (i + 1) << ": runInference returned empty logits.\n";
        }
    }

    // Compute some performance statistics
    double sum_time = std::accumulate(timings.begin(), timings.end(), 0.0);
    double avg_time = sum_time / static_cast<double>(NUM_RUNS);

    // Find min and max
    auto minmax = std::minmax_element(timings.begin(), timings.end());
    double min_time = *minmax.first;
    double max_time = *minmax.second;

    std::cout << "\nPerformance over " << NUM_RUNS << " runs:\n";
    std::cout << "  Average time: " << avg_time << " ms\n";
    std::cout << "  Min time:     " << min_time << " ms\n";
    std::cout << "  Max time:     " << max_time << " ms\n";

    // Cleanup
    g_ort_api->ReleaseSession(session);
    g_ort_api->ReleaseSessionOptions(session_options);
    g_ort_api->ReleaseEnv(g_env);
    dlclose(g_handle);

    std::cout << "Done.\n";
    return 0;
}
