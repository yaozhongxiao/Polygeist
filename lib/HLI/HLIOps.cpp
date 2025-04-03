#include "HLI/HLIOps.h"

namespace hli {
::mlir::LogicalResult SubOp::verify() {
  // this->emitWarning("SubOp::verify() is unimplemented");
  // llvm::errs() << "SubOp::verify() is unimplemented\n";
  return ::mlir::success();
}
} // namespace hli