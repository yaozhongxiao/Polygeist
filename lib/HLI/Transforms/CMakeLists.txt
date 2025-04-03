
add_mlir_library(HLITransforms
  ConvertAffineToHLI.cpp
  DCE.cpp
  DEPENDS mlir-headers
)
