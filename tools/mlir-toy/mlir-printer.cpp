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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

ModuleOp moduleOpCreate(MLIRContext *ctx) {
  // 创建 OpBuilder
  OpBuilder builder(ctx);
  ModuleOp mod = builder.create<ModuleOp>(builder.getUnknownLoc());

  // 设置插入点
  builder.setInsertionPointToEnd(mod.getBody());

  // 创建 func
  auto i32 = builder.getI32Type();
  auto funcType = builder.getFunctionType({i32, i32}, {i32});
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                           "test_creator", funcType);

  // 添加基本块
  auto entry = func.addEntryBlock();
  auto args = entry->getArguments();

  // 设置插入点
  builder.setInsertionPointToEnd(entry);

  // 创建 arith.addi
  auto addi =
      builder.create<arith::AddIOp>(builder.getUnknownLoc(), args[0], args[1]);

  // 创建 func.return
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange({addi}));
  // mod->print(llvm::outs());
  return mod;
}

OwningOpRef<ModuleOp> loadModule(char *mlirFile, MLIRContext *ctx) {
  // load mlir source file
  auto mod = parseSourceFile<ModuleOp>(mlirFile, ctx);
  // mod->print(llvm::outs());
  return mod;
}

int main(int argc, char **argv) {
  MLIRContext ctx;
  // load dialect into mlir context
  ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();

  auto modOp = loadModule(argv[1], &ctx);
  modOp->print(llvm::outs());

  modOp = moduleOpCreate(&ctx);
  modOp->dump();
  return 0;
}