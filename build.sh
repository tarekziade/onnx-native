#!/usr/bin/env bash
set -e

#######################################
# 1) Detect platform & set variables
#######################################
OS_TYPE="$(uname -s)"
if [[ "$OS_TYPE" == "Darwin" ]]; then
  BUILD_DIR="build/MacOS"
  LIB_EXT="dylib"
  PROTOC_PATH="/usr/local/bin/protoc" # Example path for macOS (adjust if needed)
else
  BUILD_DIR="build/Linux"
  LIB_EXT="so"
  PROTOC_PATH="/usr/bin/protoc"
fi

BUILD_CONFIG="MinSizeRel"

#######################################
# 2) Clone the repo if missing
#######################################
if [ ! -d onnxruntime ]; then
  echo "No local onnxruntime directory found; cloning..."
  git clone --recursive https://github.com/Microsoft/onnxruntime.git
fi

cd onnxruntime
git checkout 89f8206ba4
mkdir -p "$BUILD_DIR"

#######################################
# 3) Decide: create venv or use system
#######################################
if [ -z "$USE_SYSTEM_PYTHON" ]; then
  # Default: create/use local venv for building outside Docker
  echo "Creating venv in .env folder..."
  if [ ! -d .env ]; then
    python3 -m venv .env
  fi
  source .env/bin/activate

  # Optionally upgrade pip
  pip install --upgrade pip
  pip install -r ../requirements-dev.txt
  PYTHON="python"
else
  # Docker path: skip venv, use system python
  echo "Using system Python instead of venv..."
  $USE_SYSTEM_PYTHON -m pip install --upgrade pip
  $USE_SYSTEM_PYTHON -m pip install -r ../requirements-dev.txt
  PYTHON="$USE_SYSTEM_PYTHON"
fi

#######################################
# 4) Build onnxruntime
#    (Pass additional flags from script args: "$@")
#######################################
$PYTHON tools/ci_build/build.py \
  --build_dir "$BUILD_DIR" \
  --update \
  --build \
  --build_shared_lib \
  --parallel \
  --skip_submodule_sync \
  --disable_rtti \
  --enable_lto \
  --use_lock_free_queue --skip_tests \
  --config "$BUILD_CONFIG" \
  --compile_no_warning_as_error \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines ONNX_USE_LITE_PROTO=ON \
  --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON
"$@"

#--path_to_protoc_exe /opt/homebrew/bin/protoc

#######################################
# 5) Copy out the final library
#    (macOS => .dylib, Linux => .so)
#######################################
cp "$BUILD_DIR"/"$BUILD_CONFIG"/libonnxruntime.*"$LIB_EXT"* ../

strip -x ../libonnxruntime.*
