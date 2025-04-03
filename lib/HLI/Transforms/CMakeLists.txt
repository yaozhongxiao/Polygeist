
add_mlir_library(HLITransforms
  ConvertAffineToHLI.cpp
  DCE.cpp
  ConvertArithToHLI.cpp
  DEPENDS mlir-headers
)
