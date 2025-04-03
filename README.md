# Build instructions

## Requirements 
- Working C and C++ toolchains(compiler, linker)
- cmake
- make or ninja

## 1. Clone Polygeist
```sh
git clone --recursive https://github.com/llvm/Polygeist
cd Polygeist
```

## 2. Install LLVM, MLIR, Clang, and Polygeist

### Option 1: Using pre-built LLVM, MLIR, and Clang

Polygeist can be built by providing paths to a pre-built MLIR and Clang toolchain.

#### 1. Build LLVM, MLIR, and Clang:
```sh
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```

To enable compilation to cuda add `-DMLIR_ENABLE_CUDA_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `CUDACXX`, `CUDA_PATH`, and/or `-DCMAKE_CUDA_COMPILER`)

To enable the ROCM backend add `-DMLIR_ENABLE_ROCM_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `-DHIP_CLANG_INCLUDE_PATH`, and/or `ROCM_PATH`)

For ISL-enabled polymer, `polly` must be added to the `LLVM_ENABLE_PROJECTS` variable.

For faster compilation we recommend using `-DLLVM_USE_LINKER=lld`.

#### 2. Build Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

For faster compilation we recommend using `-DPOLYGEIST_USE_LINKER=lld`.

##### GPU backends

To enable the CUDA backend add `-DPOLYGEIST_ENABLE_CUDA=1`

To enable the ROCM backend add `-DPOLYGEIST_ENABLE_ROCM=1`

##### Polymer

To enable polymer, add `-DPOLYGEIST_ENABLE_POLYMER=1`

There are two configurations of polymer that can be built - one with Pluto and one with ISL. 

###### Pluto
Add `-DPOLYGEIST_POLYMER_ENABLE_PLUTO=1`
This will cause the cmake invokation to pull and build the dependencies for polymer. To specify a custom directory for the dependencies, specify `-DPOLYMER_DEP_DIR=<absolute-dir>`. The dependencies will be build using the `tools/polymer/build_polymer_deps.sh`.

To run the polymer pluto tests, use `ninja check-polymer`.

###### ISL

Add `-DPOLYGEIST_POLYMER_ENABLE_ISL=1`
This requires an `llvm-project` build with `polly` enabled as a subproject.


### Option 2: Using unified LLVM, MLIR, Clang, and Polygeist build

Polygeist can also be built as an external LLVM project using [LLVM_EXTERNAL_PROJECTS](https://llvm.org/docs/CMake.html#llvm-related-variables).

1. Build LLVM, MLIR, Clang, and Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

`ninja check-polygeist-opt` runs the tests in `Polygeist/test/polygeist-opt`
`ninja check-cgeist` runs the tests in `Polygeist/tools/cgeist/Test`

# MLIR Docs Generation
```
cd build

ninja mlir-doc

docs
├── HLIDialect.md
├── HLIOps.md
├── Polygeist
│   ├── PolygeistDialect.md
│   └── PolygeistOps.md
└── PolygeistPasses.md
```

# hli --convert-affine-to-hli pass
```
./tools/mlir-toy/toy-opt ../test/mlir-toy/hli.ops.mlir --convert-affine-to-hli

Run ConvertAffineToHLIPass On Operation

module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = hli.add %arg0, %arg1 : i32, i32 -> i32
    %2 = hli.sub %0, %1 : i32, i32 -> i32
    return %2 : i32
  }
}
```

# hli --hli-dce pass
```
./tools/mlir-toy/toy-opt ../test/mlir-toy/hli.ops.mlir --hli-dce -debug-only="hli-dce"

// hli.ops.mlir
func.func @test(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  %d = hli.add %a, %b : i32, i32 -> i32
  %f = hli.vadd %a, %b : i32, i32 -> i32
  %e = hli.sub %c, %d : i32, i32 -> i32
  hli.return %e : i32
}

Run HLIDECPass On Operation

Used Operations:
[1]hli.return %3 : i32
[2]%0 = arith.addi %arg0, %arg1 : i32
[3]%3 = hli.sub %0, %1 : i32, i32 -> i32
[4]%1 = hli.add %arg0, %arg1 : i32, i32 -> i32
----------------

[1] dead operation: %2 = hli.vadd %arg0, %arg1 : i32, i32 -> i32
dead Operations:
[1]%2 = hli.vadd %arg0, %arg1 : i32, i32 -> i32
----------------

module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = hli.add %arg0, %arg1 : i32, i32 -> i32
    %2 = hli.sub %0, %1 : i32, i32 -> i32
    hli.return %2 : i32
  }
}

```

# hli --convert-arith-to-hli
```
./tools/mlir-toy/toy-opt ../test/mlir-toy/hli.ops.mlir -convert-arith-to-hli

// hli.ops.mlir
func.func @test(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  %d = hli.add %a, %b : i32, i32 -> i32
  %f = hli.vadd %a, %b : i32, i32 -> i32
  %e = hli.sub %c, %d : i32, i32 -> i32
  hli.return %e : i32
}

Run ConvertArithToHLIPass On Operation

module {
  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = hli.add %arg0, %arg1 : i32, i32 -> i32
    %1 = hli.add %arg0, %arg1 : i32, i32 -> i32
    %2 = hli.vadd %arg0, %arg1 : i32, i32 -> i32
    %3 = hli.sub %0, %1 : i32, i32 -> i32
    hli.return %3 : i32
  }
}

```