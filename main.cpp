#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include onnxruntime_c_api.h for definitions of Ort* functions.
#include "onnxruntime_c_api.h"

int main() {
    // 1. Open the ONNX Runtime library
    //    Replace with a full path if needed, e.g. "/path/to/libonnxruntime.1.22.0.dylib"
    void* handle = dlopen("libonnxruntime.1.22.0.dylib", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Failed to load libonnxruntime.1.22.0.dylib: %s\n", dlerror());
        return 1;
    }

    // 2. Look up the OrtGetApiBase symbol
    const OrtApiBase* (*OrtGetApiBase)();
    *(void**)(&OrtGetApiBase) = dlsym(handle, "OrtGetApiBase");
    if (!OrtGetApiBase) {
        fprintf(stderr, "Failed to locate symbol OrtGetApiBase: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    // 3. Obtain the API base, then the current API
    const OrtApiBase* api_base = OrtGetApiBase();
    const OrtApi* ort_api = api_base->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        fprintf(stderr, "Failed to retrieve OrtApi.\n");
        dlclose(handle);
        return 1;
    }

    // 4. Use the ONNX Runtime API
    //    Example: create and destroy an OrtEnv
    OrtEnv* env = nullptr;
    OrtStatus* status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test_env", &env);
    if (status != nullptr) {
        const char* msg = ort_api->GetErrorMessage(status);
        fprintf(stderr, "CreateEnv error: %s\n", msg);
        ort_api->ReleaseStatus(status);
        dlclose(handle);
        return 1;
    } else {
        printf("Successfully created OrtEnv!\n");
    }

    // 5. Destroy the environment
    ort_api->ReleaseEnv(env);

    // 6. Close the dynamically loaded library
    dlclose(handle);

    printf("Successfully closed libonnxruntime.1.22.0.dylib.\n");
    return 0;
}

