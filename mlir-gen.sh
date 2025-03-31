#!/bin/bash

script_dir=$(dirname $0)
polygeist_dir=${script_dir}
test_dir=${polygeist_dir}/test/c2mlir
bin_dir=${polygeist_dir}/build/bin

set -x

${bin_dir}/cgeist ${test_dir}/input.c --function=* -S -o ${test_dir}/input.scf.mlir
${bin_dir}/polygeist-opt --raise-scf-to-affine ${test_dir}/input.scf.mlir -o ${test_dir}/affine_output.mlir
