git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
python3 -m venv .
. bin/activate
pip install -r requirements-dev.txt
python tools/ci_build/build.py --build_dir build/MacOS --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --disable_exceptions --disable_rtti --skip_tests --path_to_protoc_exe /opt/homebrew/bin/protoc
