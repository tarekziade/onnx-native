
build_onnx:
	./build.sh

build:
	clang++ -std=c++17 -o onnx_test main.cpp -ldl

model.onnx:
	https://huggingface.co/onnx-community/distilbert-base-uncased-finetuned-sst-2-english-ONNX/resolve/main/onnx/model.onnx?download=true



