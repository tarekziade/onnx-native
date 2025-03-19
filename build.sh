git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
python3 -m venv .
bin/pip install -r requirements-dev.txt
bin/python onnxruntime/core/flatbuffers/schema/compile_schema.py --flatc "$(which flatc)"
bin/python onnxruntime/lora/adapter_format/compile_schema.py --flatc "$(which flatc)"
bin/python tools/ci_build/build.py --build_dir build/MacOS --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --disable_exceptions --disable_rtti --skip_tests --path_to_protoc_exe /opt/homebrew/bin/protoc
