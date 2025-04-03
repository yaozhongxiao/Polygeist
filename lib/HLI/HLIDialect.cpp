#include "HLI/HLIDialect.h"
#include "HLI/HLIOps.h"
#include "HLI/HLIDialect.cpp.inc"
#include "HLI/HLITypes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "HLI/HLITypes.cpp.inc"

namespace hli {

void HLIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HLI/HLI.cpp.inc"
  >();
  registerTypes();
}

void HLIDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "HLI/HLITypes.cpp.inc"
  >();
}

}