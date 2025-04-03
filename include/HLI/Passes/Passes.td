#ifndef HLI_PASSES
#define HLI_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def ConvertAffineToHLI : Pass<"convert-affine-to-hli"> {
  let summary = "Convert Affine To HLI";
  let constructor = "hli::createConvertAffineToHLIPass()";
}

def HLIDCE : Pass<"hli-dce", "::mlir::func::FuncOp"> {
  let summary = "HLI Dead Code Elimination";
  let constructor = "hli::createHLIDCEPass()";
}

#endif // HLI_PASSES