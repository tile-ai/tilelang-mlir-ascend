// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_tlir.h
 * \brief TVM TIR -> TileLang TLIR (tl dialect) lowering.
 *
 * Expert-mode subset for the add case study:
 *   allocate -> tl.alloc
 *   tl.copy  -> tl.copy
 *   tl.npuir_add -> tl.add
 *
 * HIVM/runtime ABI attrs are intentionally omitted; this is the middle IR.
 */
#ifndef TILELANG_TARGET_CODEGEN_TLIR_H_
#define TILELANG_TARGET_CODEGEN_TLIR_H_

#include <string>
#include <unordered_map>

#include "tilelangir/Dialect/TL/TLOps.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include <tvm/ir/module.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../op/op.h"

namespace tvm {
namespace codegen {

using namespace tir;
using tl::BufferMap;

class CodeGenTileLangTLIR
    : public ExprFunctor<mlir::Value(const PrimExpr &)>,
      public StmtFunctor<void(const Stmt &)> {
public:
  CodeGenTileLangTLIR();
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  std::string Finish();
  String GetCurrentFunctionName() const { return current_function_name_; }

  mlir::Value VisitExpr_(const VarNode *op) final;
  mlir::Value VisitExpr_(const IntImmNode *op) final;
  mlir::Value VisitExpr_(const FloatImmNode *op) final;
  mlir::Value VisitExpr_(const AddNode *op) final;
  mlir::Value VisitExpr_(const SubNode *op) final;
  mlir::Value VisitExpr_(const MulNode *op) final;
  mlir::Value VisitExpr_(const MinNode *op) final;
  mlir::Value VisitExpr_(const CallNode *op) final;

  void VisitStmt_(const SeqStmtNode *op) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitStmt_(const IfThenElseNode *op) final;
  void VisitStmt_(const DeclBufferNode *op) final;

private:
  void InitFuncState();
  mlir::Type DTypeToMLIRType(DataType dtype);
  mlir::tl::AddressSpace ScopeToAddressSpace(const std::string &scope);
  mlir::MemRefType BufferToMemRefType(const Buffer &buffer);
  mlir::MemRefType AllocToMemRefType(const AllocateNode *op);
  mlir::Value GetBufferValue(const Buffer &buffer);
  mlir::Attribute AddressSpaceAttr(mlir::tl::AddressSpace space);
  void EmitCopy(const CallNode *op);
  void EmitAdd(const CallNode *op);

  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  BufferMap vmap_;
  std::unordered_map<const VarNode *, mlir::Value> var_map_;
  String current_function_name_;
};

} // namespace codegen
} // namespace tvm

#endif // TILELANG_TARGET_CODEGEN_TLIR_H_
