include "mlir/IR/OpBase.td"

#ifndef HLI_DIALECT
#define HLI_DIALECT

def HLIDialect : Dialect {
  let name = "HLIDialect";
  let cppNamespace = "::hli";
  let summary = "HLI Dialect";
}

#endif // HLI_DIALECT