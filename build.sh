#!/bin/bash -xe

[ -d onnxruntime/ ] || git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
python3 -m venv .
. bin/activate
pip install -r requirements-dev.txt
BUILD_TYPE=Release
# CMAKE_OSX_ARCHITECTURES is ignored on non-macOS
python tools/ci_build/build.py --build_dir build/ --config $BUILD_TYPE --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --disable_exceptions --disable_rtti --skip_tests
cp build/$BUILD_TYPE/libonnxruntime.so.1.22.0 ../libonnxruntime.so
