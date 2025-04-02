#ifndef HLI_DIALECT
#define HLI_DIALECT

include "mlir/IR/OpBase.td"

def HLIDialect : Dialect {
  let name = "hli";
  let cppNamespace = "::hli";
  let summary = "HLI Dialect";
}

#endif // HLI_DIALECT