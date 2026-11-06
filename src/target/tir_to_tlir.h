// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// TIR -> TLIR importer, memref form only.
// Supported: alloc (tir::Allocate), copy (tl.copy), add (tl.npuir_add).
// TLIR here is a convention over upstream dialects (func/memref/linalg/arith);
// no custom tlir ops are emitted.

#pragma once

#include <string>
#include <unordered_map>

#include <tvm/ir/module.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace tvm {
namespace codegen {

class CodeGenTIRToTLIR : public tir::StmtExprVisitor {
public:
  CodeGenTIRToTLIR();

  // Translate one PrimFunc into a func.func inside the module.
  void AddFunction(const GlobalVar &gvar, const tir::PrimFunc &f);

  // Verify and print the whole module as MLIR text.
  std::string Finish();

protected:
  // tir::Allocate -> memref.alloc
  void VisitStmt_(const tir::AllocateNode *op) override;
  // tl.copy -> memref.copy ; tl.npuir_add -> linalg.generic { arith.add }
  void VisitExpr_(const tir::CallNode *op) override;

private:
  // Storage scope string ("global", "shared", "ub", ...) -> memref memory space.
  mlir::Attribute AddressSpaceFor(const std::string &scope);
  mlir::MemRefType MakeMemRefType(llvm::ArrayRef<int64_t> shape, DataType dtype,
                                  const std::string &scope);
  // T.region(BufferLoad, rank, extents...) -> the memref bound to that buffer.
  // Only whole-buffer regions are supported.
  mlir::Value ResolveWholeBufferRegionArg(const PrimExpr &arg);

  mlir::MLIRContext context; // must be declared before `builder`
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
  std::unordered_map<const tir::VarNode *, mlir::Value> var_map_;
};

} // namespace codegen
} // namespace tvm
