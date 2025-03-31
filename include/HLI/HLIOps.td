include "mlir/IR/OpBase.td"
include "HLI/HLIDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

#ifndef HLI_OPS
#define HLI_OPS

// mnemonic op-name
class HliOp<string mnemonic, list<Trait> traits = []> :
  Op<HLIDialect, mnemonic, traits>;

// Pure 是 Trait，表示没有 SideEffect 的纯函数
def AddOp : HliOp<"add", [Pure]> {
  let summary = "add operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
}

def ConstantOp : HliOp<"const", [Pure]> {
  let summary = "const operation";
  let arguments = (ins APIntAttr:$value);
  let results = (outs AnyInteger:$result);
}

def SubOp : HliOp<"sub", [Pure]> {
  let summary = "sub operation";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  let hasVerifier = true;
}

#endif // HLI_OPS