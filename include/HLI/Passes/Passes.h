#ifndef HLI_DIALECT_PASSES_H
#define HLI_DIALECT_PASSES_H

#include "mlir/Pass/Pass.h"
namespace hli {
#define GEN_PASS_DECL
#include "HLI/Passes/HLIPasses.h.inc"

std::unique_ptr<::mlir::Pass> createConvertAffineToHLIPass();
std::unique_ptr<::mlir::Pass> createHLIDCEPass();

#define GEN_PASS_REGISTRATION
#include "HLI/Passes/HLIPasses.h.inc"
}

#endif // HLI_DIALECT_PASSES_H