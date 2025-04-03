#include "HLI/HLIOps.h"

#define GET_OP_CLASSES
#include "HLI/HLI.cpp.inc"
namespace hli {

::mlir::LogicalResult SubOp::verify() {
  // this->emitWarning("SubOp::verify() is unimplemented");
  // llvm::errs() << "SubOp::verify() is unimplemented\n";
  return ::mlir::success();
}
} // namespace hli