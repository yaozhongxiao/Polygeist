#include "HLI/Passes/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace hli {
#define GEN_PASS_DEF_CONVERTAFFINETOHLI
#include "HLI/Passes/HLIPasses.h.inc"

struct ConvertAffineToHLIPass
    : impl::ConvertAffineToHLIBase<ConvertAffineToHLIPass> {
  using impl::ConvertAffineToHLIBase<
      ConvertAffineToHLIPass>::ConvertAffineToHLIBase;
  void runOnOperation() {
    llvm::errs() << "Run ConvertAffineToHLIPass On Operation" << "\n";
    getOperation()->print(llvm::errs());
  }
};

std::unique_ptr<::mlir::Pass> createConvertAffineToHLIPass() {
  return std::make_unique<ConvertAffineToHLIPass>();
}

} // namespace hli