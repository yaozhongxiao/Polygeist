//===- polygeist-opt.cpp - The polygeist-opt driver -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'polygeist-opt' tool, which is the polygeist analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "HLI/HLIDialect.h"
#include "HLI/Passes/Passes.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char **argv) {
  DialectRegistry registry;
  // register HLI Dialect
  registry.insert<hli::HLIDialect, func::FuncDialect, arith::ArithDialect>();
  // register optimization Pass
  registerCSEPass();
  registerCanonicalizerPass();

  hli::registerHLIPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "toy-opt", registry));
}
