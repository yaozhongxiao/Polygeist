set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(HLIPasses.h.inc -gen-pass-decls -name HLI)
add_public_tablegen_target(MLIRHLIPassIncGen)

add_mlir_doc(Passes HLIPasses ./ -gen-pass-doc)

add_dependencies(mlir-headers MLIRHLIPassIncGen)