#include "HLI/Passes/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

#include "HLI/HLIOps.h"

#define GEN_PASS_DEF_HLIDCE
#include "HLI/Passes/HLIPasses.h.inc"

#define DEBUG_TYPE "hli-dce"

namespace hli {

struct HLIDCEPass : ::impl::HLIDCEBase<HLIDCEPass> {
  using ::impl::HLIDCEBase<HLIDCEPass>::HLIDCEBase;

  llvm::DenseSet<::mlir::Operation *> usedOps;

  template <typename ContainerType> void PrintOps(ContainerType &ops) {
    int index = 1;
    for (auto op : ops) {
      LLVM_DEBUG(llvm::dbgs() << "[" << index++ << "]" << *op << "\n");
    }
  }
  void RecursiveOpVisit(::mlir::Operation *op) {
    if (usedOps.contains(op))
      return;
    usedOps.insert(op);
    for (auto operand : op->getOperands())
      if (auto def = operand.getDefiningOp())
        RecursiveOpVisit(def);
  }

  void runOnOperation() {
    llvm::errs() << "Run HLIDECPass On Operation" << "\n";
    usedOps.clear();
    // scan used operations
    getOperation()->walk([&](hli::ReturnOp op) { RecursiveOpVisit(op); });

    LLVM_DEBUG(llvm::dbgs() << "Used Operations:\n");
    PrintOps<decltype(usedOps)>(usedOps);
    LLVM_DEBUG(llvm::dbgs() << "----------------\n");

    // remove dead operations
    int index = 1;
    llvm::SmallVector<::mlir::Operation *> opToRemove;
    getOperation()->walk([&](::mlir::Operation *op) {
      if (op == getOperation())
        return;
      if (!usedOps.contains(op)) {
        opToRemove.push_back(op);
        LLVM_DEBUG(llvm::errs() << "[" << index++ << "] dead operation: " << *op << "\n");
      }
    });
    LLVM_DEBUG(llvm::dbgs() << "dead Operations:\n");
    PrintOps<decltype(opToRemove)>(opToRemove);
    LLVM_DEBUG(llvm::dbgs() << "----------------\n");

    // reverse remove dead operations
    for (auto v : reverse(opToRemove)) {
      v->erase();
    }
  }
};

std::unique_ptr<::mlir::Pass> createHLIDCEPass() {
  return std::make_unique<HLIDCEPass>();
}

} // namespace hli