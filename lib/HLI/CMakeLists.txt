
add_mlir_library(HLI HLIDialect.cpp HLIOps.cpp
  DEPENDS MLIRHLIIncGen mlir-doc
)
target_compile_options(HLI PUBLIC "-Wno-unused-but-set-parameter")