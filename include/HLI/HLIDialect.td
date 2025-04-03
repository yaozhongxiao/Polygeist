#ifndef HLI_DIALECT
#define HLI_DIALECT

include "mlir/IR/OpBase.td"

def HLIDialect : Dialect {
  let name = "hli";
  let cppNamespace = "::hli";
  let summary = "HLI Dialect";
  // let dependentDialects = [
  //   "::mlir::arith::ArithDialect"
  // ];
  let useDefaultTypePrinterParser = true;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // HLI_DIALECT