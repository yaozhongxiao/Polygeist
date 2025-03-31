#include "HLI/HLIOps.h"

namespace hli {
::mlir::LogicalResult SubOp::verify() {
  this->emitWarning("SubOp::verify() is unimplemented");
  return ::mlir::success();
}
} // namespace hli