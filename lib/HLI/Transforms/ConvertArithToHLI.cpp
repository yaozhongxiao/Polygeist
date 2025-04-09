#include "HLI/HLIOps.h"
#include "HLI/Passes/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "HLI/Interfaces/TypeInferInterface.h"
#include <memory>

namespace hli {
#define GEN_PASS_DEF_CONVERTARITHTOHLI
#include "HLI/Passes/HLIPasses.h.inc"

struct AddOpPat : ::mlir::OpRewritePattern<::mlir::arith::AddIOp> {
  using ::mlir::OpRewritePattern<::mlir::arith::AddIOp>::OpRewritePattern;
  ::mlir::LogicalResult matchAndRewrite(::mlir::arith::AddIOp op,
                                        mlir::PatternRewriter &rewriter) const {
    auto inputs = llvm::to_vector(op.getOperands());
    auto result = rewriter.create<hli::AddOp>(
        op->getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs());

    rewriter.replaceOp(op, ::mlir::ValueRange(result));
    return mlir::success();
  }
};

struct ConvertArithToHLIPass
    : impl::ConvertArithToHLIBase<ConvertArithToHLIPass> {
  using impl::ConvertArithToHLIBase<
      ConvertArithToHLIPass>::ConvertArithToHLIBase;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<::mlir::arith::ArithDialect>();
  }

  void runOnOperation() {
    llvm::dbgs() << "runOnOperation" << "\n";
    getOperation()->dump();
    getOperation()->walk([&](hli::VAddOp op) { 
      llvm::dbgs() << "VAddOp: " << *op << "\n";
      mlir::Operation *addOperation = op.getOperation();
      op.nonStaticMethod();
      op.staticMethod();

      if(auto typeInfer = llvm::dyn_cast<TypeInferInterface>(addOperation)) {
        llvm::dbgs() << "typeInfer: " << *typeInfer << "\n";
        typeInfer.nonStaticMethod();
        typeInfer.staticMethod();
      } else {
        llvm::errs() << "VAddOp: " << *op << " does not implement TypeInferInterface\n";
      }
     });

    llvm::errs() << "Run ConvertArithToHLIPass On Operation" << "\n";
    ::mlir::ConversionTarget target(getContext());
    target.addLegalDialect<hli::HLIDialect>();
    ::mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AddOpPat>(&getContext());
    if (::mlir::failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<::mlir::Pass> createConvertArithToHLIPass() {
  return std::make_unique<ConvertArithToHLIPass>();
}

} // namespace hli