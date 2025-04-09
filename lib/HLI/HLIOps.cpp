#include "HLI/HLIOps.h"
#include "llvm/Support/Debug.h"

#define GET_OP_CLASSES
#include "HLI/HLI.cpp.inc"
namespace hli {

#define DEBUG_TYPE "hli-ops"

::mlir::LogicalResult SubOp::verify() {
  // this->emitWarning("SubOp::verify() is unimplemented");
  // llvm::errs() << "SubOp::verify() is unimplemented\n";
  return ::mlir::success();
}

void AddOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context) {
  LLVM_DEBUG(llvm::dbgs() << "AddOp::getCanonicalizationPatterns()\n");
}

void VAddOp::nonStaticMethod() {
  llvm::errs() << "VAddOp::nonStaticMethod()\n";
}
// mlir::Value VAddOp::nonStaticMethodWithParams(unsigned i) {}
void VAddOp::staticMethod() {
  llvm::errs() << "VAddOp::staticMethod()\n";
}

} // namespace hli