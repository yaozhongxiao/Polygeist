#!/bin/bash

script_dir="$(cd "$(dirname "$0")"; pwd -P)"
polygeist_dir=${script_dir}

mkdir build-tools
cd build-tools

cmake -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=${polygeist_dir}/tools-install \
  -DLLVM_DIR=${polygeist_dir}/build/lib/cmake/llvm \
  -DMLIR_DIR=${polygeist_dir}/build/lib/cmake/mlir \
  -DCLANG_DIR=${polygeist_dir}/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host;Native;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
