#!/bin/bash

script_dir="$(cd "$(dirname "$0")"; pwd -P)"
polygeist_dir=${script_dir}

mkdir build
cd build

cmake -G Ninja ${polygeist_dir}/llvm-project/llvm \
  -DCMAKE_INSTALL_PREFIX=${polygeist_dir}/tools-install \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host;Native;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
