#include "HLI/HLIDialect.h"
#include "HLI/HLIOps.h"
#include "HLI/HLIDialect.cpp.inc"

#define GET_OP_CLASSES
#include "HLI/HLI.cpp.inc"

using namespace hli;

void HLIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HLI/HLI.cpp.inc"
  >();
}