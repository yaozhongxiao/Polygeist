#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "HLI/HLIDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define GET_OP_CLASSES
#include "HLI/HLI.h.inc"