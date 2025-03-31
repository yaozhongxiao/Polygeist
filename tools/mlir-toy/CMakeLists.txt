cmake_minimum_required(VERSION 3.20.0)

project(mlir-toy VERSION 0.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json 便于代码高亮
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_executable(mlir-printer mlir-printer.cpp)
target_link_libraries(mlir-printer
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)
install(TARGETS mlir-printer RUNTIME DESTINATION bin)

add_mlir_tool(toy-opt toy-opt.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(toy-opt PUBLIC HLI
  MLIRIR MLIRParser MLIRSupport
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
  MLIRArithDialect
)

# add_executable(toy-opt toy-opt.cpp)
# target_link_libraries(toy-opt HLI
#   MLIRIR MLIRParser MLIRSupport
#   MLIROptLib
#   MLIRFuncDialect
#   MLIRTransforms
#   MLIRArithDialect
# )
install(TARGETS toy-opt RUNTIME DESTINATION bin)