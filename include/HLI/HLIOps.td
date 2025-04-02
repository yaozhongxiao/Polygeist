include "mlir/IR/OpBase.td"
include "HLI/HLIDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

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
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
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
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = true;
}

def VAddOp : HliOp<"vadd", [Pure]> {
  let summary = "vadd operation";
  let arguments = (ins Variadic<AnyInteger>:$inputs);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
}

def ReturnOp : HliOp<"return", [Terminator]> {
  let summary = "return operation";
  let arguments = (ins Optional<AnyType>:$data);
  let assemblyFormat = "$data attr-dict `:` type($data)";
}

#endif // HLI_OPS