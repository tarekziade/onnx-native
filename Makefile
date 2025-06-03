# Makefile for building and testing ONNX model

# Default target
.PHONY: all
all: run

# Rule to download the ONNX model if it does not exist
model.onnx:
	@echo "Downloading ONNX model..."
	@curl -L -o model.onnx https://huggingface.co/onnx-community/distilbert-base-uncased-finetuned-sst-2-english-ONNX/resolve/main/onnx/model.onnx?download=true

# Rule to build the ONNX runtime (libonnxruntime.1.22.0.dylib)
libonnxruntime.1.22.0.dylib:
	@echo "Building ONNX Runtime library..."
	@./build.sh

# Rule to build the ONNX test executable
onnx_test: main.cpp libonnxruntime.1.22.0.dylib
	@echo "Building ONNX test..."
	@clang++ -std=c++17 -o onnx_test main.cpp onnx.pb.cc -ldl -lprotobuf

# Rule to run the test with the downloaded ONNX model
run: model.onnx onnx_test
	@echo "Running ONNX test..."
	@./onnx_test

# Clean up generated files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -f onnx_test model.onnx libonnxruntime.*

# build for Ubuntu 18.04
.PHONY: docker
docker:
	docker buildx build --platform=linux/amd64 -t onnxruntime-linux .
	docker run --rm -it --platform=linux/amd64 -v $(shell pwd):/workspace onnxruntime-linux ./build.sh
