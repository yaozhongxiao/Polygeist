add_subdirectory(Passes)

add_mlir_dialect(HLI hli)
add_mlir_doc(HLIDialect HLIDialect ./ -gen-dialect-doc)
add_mlir_doc(HLIOps HLIOps ./ -gen-op-doc)

# add_custom_target(hli-header DEPENDS MLIRHLIIncGen MLIRHLIPassIncGen)
# mlir-headers is a target always built defined in the root CMakeLists.txt 
add_dependencies(mlir-headers MLIRHLIIncGen mlir-doc)