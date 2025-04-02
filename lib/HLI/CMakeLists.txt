add_subdirectory(Transforms)

add_mlir_library(HLI HLIDialect.cpp HLIOps.cpp
  DEPENDS mlir-headers
)
target_link_directories(HLI PUBLIC HLITransforms)
target_compile_options(HLI PUBLIC "-Wno-unused-but-set-parameter")

