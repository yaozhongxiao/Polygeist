#ifndef HLI_TYPES
#define HLI_TYPES

include "mlir/IR/AttrTypeBase.td"
include "HLI/HLIDialect.td"

class HLIType<string name, list<Trait> traits=[]>: TypeDef<HLIDialect, name, traits>;

def HLIInteger: HLIType<"HLIInteger"> {
  let mnemonic = "int";
  let parameters = (ins "uint64_t":$width);
  let assemblyFormat = "`<` $width `>`";
  // let useDefaultTypePrinterParser = true;
}

#endif // HLI_TYPES