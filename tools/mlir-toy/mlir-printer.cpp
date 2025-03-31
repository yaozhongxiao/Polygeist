//===- polygeist-opt.cpp - The polygeist-opt driver -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'mlir-toy' tool, which is the toy analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char ** argv) {
  MLIRContext ctx;
  // load dialect into mlir context
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
  // load mlir source file
  auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
  // print the module
  src->print(llvm::outs());
  // dump the module
  // src->dump();
  return 0;
}