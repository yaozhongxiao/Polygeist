add_mlir_dialect(PolygeistOps polygeist)
add_mlir_doc(Dialect PolygeistDialect Polygeist/ -gen-dialect-doc )
add_mlir_doc(PolygeistOps PolygeistOps Polygeist/ -gen-op-doc)

add_subdirectory(Passes)