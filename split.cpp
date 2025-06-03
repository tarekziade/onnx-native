#include <fstream>
#include <string>
#include <unordered_set>
#include "onnx.pb.h"

void switch_tensor_to_external_data(onnx::TensorProto& tensor, const std::string& location) {
    std::ofstream out(location, std::ios::binary | std::ios::app);
    if (!out) throw std::runtime_error("Failed to open file: " + location);

    // Get offset by seeking to the end
    out.seekp(0, std::ios::end);
    size_t offset = out.tellp();
    size_t length = tensor.raw_data().size();
    out.write(tensor.raw_data().data(), length);
    out.close();

    // set this tensor to have external data, loaded separately
    tensor.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);

    // and say where this data is
    auto add_entry = [&](const std::string& key, const std::string& value) {
        auto* entry = tensor.add_external_data();
        entry->set_key(key);
        entry->set_value(value);
    };

    add_entry("location", location);
    add_entry("offset", std::to_string(offset));
    add_entry("length", std::to_string(length));

    // clear embedded payload from the tensor, making it minuscule
    tensor.clear_raw_data();
}

void convert_model_to_use_external_data(
    onnx::ModelProto& model,
    const std::string& location,
    size_t size_threshold = 1024
) {
    auto* graph = model.mutable_graph();

    // loop over all initializers, and switch them to external if they are big
    // enough
    for (auto& tensor : *graph->mutable_initializer()) {
        if (!tensor.has_raw_data()) {
          continue;
        }
        if (tensor.raw_data().size() < size_threshold) {
          continue;
        }
        switch_tensor_to_external_data(tensor, location);
    }
}

int main() {
    onnx::ModelProto model;
    std::ifstream in("model.onnx", std::ios::binary);
    model.ParseFromIstream(&in);
    in.close();

    // remove previously created file
    std::remove("graph.onnx");
    std::remove("weights.data");

    // this modifies `model` to save the data into `weights.data` when it is
    // over 1024 bytes.
    convert_model_to_use_external_data(model, "weights.data", 1024);

    // And we can serialize the model back, it will be very small
    std::ofstream out("graph.onnx", std::ios::binary);
    model.SerializeToOstream(&out);
    out.close();

    return 0;
}
