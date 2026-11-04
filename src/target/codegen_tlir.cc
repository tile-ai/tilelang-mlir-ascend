// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file target/codegen_tlir.cc
 * \brief TVM TIR -> TLIR lowering (expert-mode add subset).
 */

#include "codegen_tlir.h"

#include "../op/ascend.h"
#include "../op/builtin.h"
#include "target/source/codegen_c.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>

#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace codegen {

CodeGenTileLangTLIR::CodeGenTileLangTLIR() : builder_(&context_) {
  context_.loadDialect<::mlir::func::FuncDialect, ::mlir::memref::MemRefDialect,
                       ::mlir::arith::ArithDialect, ::mlir::tl::TLDialect>();
  module_ = ::mlir::ModuleOp::create(builder_.getUnknownLoc());
}

void CodeGenTileLangTLIR::InitFuncState() {
  var_map_.clear();
  current_function_name_ = "";
}

mlir::Attribute
CodeGenTileLangTLIR::AddressSpaceAttr(mlir::tl::AddressSpace space) {
  return ::mlir::tl::AddressSpaceAttr::get(&context_, space);
}

mlir::tl::AddressSpace
CodeGenTileLangTLIR::ScopeToAddressSpace(const std::string &scope) {
  if (scope == "global" || scope.empty())
    return ::mlir::tl::AddressSpace::GM;
  if (scope == "shared")
    return ::mlir::tl::AddressSpace::UB;
  if (scope == "shared.dyn")
    return ::mlir::tl::AddressSpace::L1;
  if (scope == "wmma.accumulator")
    return ::mlir::tl::AddressSpace::L0;
  return ::mlir::tl::AddressSpace::UB;
}

mlir::Type CodeGenTileLangTLIR::DTypeToMLIRType(DataType dtype) {
  if (dtype.is_float()) {
    if (dtype.bits() == 16)
      return builder_.getF16Type();
    if (dtype.bits() == 32)
      return builder_.getF32Type();
    if (dtype.bits() == 64)
      return builder_.getF64Type();
  } else if (dtype.is_bfloat16()) {
    return builder_.getBF16Type();
  } else if (dtype.is_bool()) {
    return builder_.getI1Type();
  } else if (dtype.is_int() || dtype.is_uint()) {
    return builder_.getIntegerType(dtype.bits());
  }
  LOG(FATAL) << "CodeGenTileLangTLIR: unsupported dtype " << dtype;
  return {};
}

static std::vector<int64_t> ShapeFromExprs(const Array<PrimExpr> &shape) {
  std::vector<int64_t> out;
  out.reserve(shape.size());
  for (const PrimExpr &dim : shape) {
    if (auto *imm = as_const_int(dim))
      out.push_back(*imm);
    else
      out.push_back(::mlir::ShapedType::kDynamic);
  }
  return out;
}

mlir::MemRefType CodeGenTileLangTLIR::BufferToMemRefType(const Buffer &buffer) {
  auto space = ScopeToAddressSpace(GetPtrStorageScope(buffer->data));
  return ::mlir::MemRefType::get(ShapeFromExprs(buffer->shape),
                                 DTypeToMLIRType(buffer->dtype),
                                 /*layout=*/::mlir::MemRefLayoutAttrInterface{},
                                 AddressSpaceAttr(space));
}

mlir::MemRefType
CodeGenTileLangTLIR::AllocToMemRefType(const AllocateNode *op) {
  auto space = ScopeToAddressSpace(GetPtrStorageScope(op->buffer_var));
  return ::mlir::MemRefType::get(ShapeFromExprs(op->extents),
                                 DTypeToMLIRType(op->dtype),
                                 /*layout=*/::mlir::MemRefLayoutAttrInterface{},
                                 AddressSpaceAttr(space));
}

mlir::Value CodeGenTileLangTLIR::GetBufferValue(const Buffer &buffer) {
  auto it = var_map_.find(buffer->data.get());
  ICHECK(it != var_map_.end()) << "CodeGenTileLangTLIR: unknown buffer "
                               << buffer->name;
  return it->second;
}

void CodeGenTileLangTLIR::EmitCopy(const CallNode *op) {
  tvm::tl::AscendCopy copy(op->args, vmap_);
  builder_.create<::mlir::tl::CopyOp>(builder_.getUnknownLoc(),
                                      GetBufferValue(copy.src),
                                      GetBufferValue(copy.dst));
}

void CodeGenTileLangTLIR::EmitAdd(const CallNode *op) {
  tvm::tl::NpuirAdd add(op->args, vmap_);
  ICHECK(add.Src0().IsTensor() && add.Src1().IsTensor() && add.Dst().IsTensor())
      << "CodeGenTileLangTLIR: tl.npuir_add currently requires tensor operands";
  builder_.create<::mlir::tl::AddOp>(builder_.getUnknownLoc(),
                                     GetBufferValue(add.Src0().GetBuffer()),
                                     GetBufferValue(add.Src1().GetBuffer()),
                                     GetBufferValue(add.Dst().GetBuffer()));
}

void CodeGenTileLangTLIR::AddFunction(const GlobalVar &gvar, const PrimFunc &f) {
  InitFuncState();
  vmap_ = f->buffer_map;

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenTileLangTLIR: PrimFunc needs kGlobalSymbol";
  current_function_name_ = global_symbol.value();

  llvm::SmallVector<::mlir::Type> arg_types;
  for (size_t i = 0; i < f->params.size(); ++i) {
    Var v = f->params[i];
    if (v.dtype().is_handle()) {
      ICHECK(f->buffer_map.count(v));
      arg_types.push_back(BufferToMemRefType(f->buffer_map[v]));
    } else {
      arg_types.push_back(DTypeToMLIRType(v.dtype()));
    }
  }

  builder_.setInsertionPointToEnd(module_.getBody());
  auto func_type = builder_.getFunctionType(arg_types, {});
  std::string func_name(current_function_name_);
  auto func = builder_.create<::mlir::func::FuncOp>(
      builder_.getUnknownLoc(), func_name, func_type);
  func->setAttr("tl.core_type", ::mlir::tl::CoreTypeAttr::get(
                                    &context_, ::mlir::tl::CoreType::AIV));
  ::mlir::Block *entry = func.addEntryBlock();
  builder_.setInsertionPointToStart(entry);

  for (size_t i = 0; i < f->params.size(); ++i) {
    Var v = f->params[i];
    Var real = v.dtype().is_handle() ? f->buffer_map[v]->data : v;
    var_map_[real.get()] = func.getArgument(i);
  }

  VisitStmt(f->body);
  builder_.create<::mlir::func::ReturnOp>(builder_.getUnknownLoc());
}

std::string CodeGenTileLangTLIR::Finish() {
  std::string code;
  llvm::raw_string_ostream os(code);
  module_.print(os);
  os.flush();
  if (::mlir::failed(::mlir::verify(module_))) {
    LOG(FATAL) << "CodeGenTileLangTLIR: module failed verification:\n" << code;
  }
  return code;
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const VarNode *op) {
  auto it = var_map_.find(op);
  ICHECK(it != var_map_.end()) << "CodeGenTileLangTLIR: unknown var "
                               << op->name_hint;
  return it->second;
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const IntImmNode *op) {
  return builder_.create<::mlir::arith::ConstantOp>(
      builder_.getUnknownLoc(),
      builder_.getIntegerAttr(builder_.getIntegerType(op->dtype.bits()),
                              op->value));
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const FloatImmNode *op) {
  return builder_.create<::mlir::arith::ConstantOp>(
      builder_.getUnknownLoc(), builder_.getF32FloatAttr(op->value));
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const AddNode * /*op*/) {
  LOG(FATAL) << "CodeGenTileLangTLIR: scalar Add is not part of the add "
                "case study; use tl.npuir_add";
  return {};
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const SubNode * /*op*/) {
  LOG(FATAL) << "CodeGenTileLangTLIR: scalar Sub is not supported yet";
  return {};
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const MulNode * /*op*/) {
  LOG(FATAL) << "CodeGenTileLangTLIR: scalar Mul is not supported yet";
  return {};
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const MinNode * /*op*/) {
  LOG(FATAL) << "CodeGenTileLangTLIR: Min is not supported yet";
  return {};
}

mlir::Value CodeGenTileLangTLIR::VisitExpr_(const CallNode *op) {
  if (op->op.same_as(Op::Get("tl.copy"))) {
    EmitCopy(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_add"))) {
    EmitAdd(op);
    return {};
  }
  LOG(FATAL) << "CodeGenTileLangTLIR: unsupported call " << op->op;
  return {};
}

void CodeGenTileLangTLIR::VisitStmt_(const SeqStmtNode *op) {
  for (const Stmt &stmt : op->seq)
    VisitStmt(stmt);
}

void CodeGenTileLangTLIR::VisitStmt_(const EvaluateNode *op) {
  VisitExpr(op->value);
}

void CodeGenTileLangTLIR::VisitStmt_(const AllocateNode *op) {
  auto alloc = builder_.create<::mlir::tl::AllocOp>(builder_.getUnknownLoc(),
                                                    AllocToMemRefType(op));
  ICHECK(!var_map_.count(op->buffer_var.get()));
  var_map_[op->buffer_var.get()] = alloc.getResult();
  VisitStmt(op->body);
}

void CodeGenTileLangTLIR::VisitStmt_(const AttrStmtNode *op) {
  VisitStmt(op->body);
}

void CodeGenTileLangTLIR::VisitStmt_(const ForNode *op) {
  // Launch / tail loops are not modeled in TLIR v0.1; walk the body.
  VisitStmt(op->body);
}

void CodeGenTileLangTLIR::VisitStmt_(const LetStmtNode *op) {
  VisitStmt(op->body);
}

void CodeGenTileLangTLIR::VisitStmt_(const IfThenElseNode *op) {
  VisitStmt(op->then_case);
  if (op->else_case.defined())
    VisitStmt(op->else_case.value());
}

void CodeGenTileLangTLIR::VisitStmt_(const DeclBufferNode *op) {
  VisitStmt(op->body);
}

runtime::Module BuildTileLangTLIR(IRModule mod, Target) {
  CodeGenTileLangTLIR cg;
  Array<String> function_names;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangTLIR: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    bool has_handle = false;
    for (const Var &p : f->params) {
      if (p.dtype().is_handle()) {
        has_handle = true;
        break;
      }
    }
    if (!has_handle)
      continue;
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetCurrentFunctionName());
  }
  ICHECK(!function_names.empty())
      << "CodeGenTileLangTLIR: no device PrimFunc with handle params";
  return CSourceModuleCreate(cg.Finish(), "c", function_names);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_tlir")
    .set_body_typed(BuildTileLangTLIR);

} // namespace codegen
} // namespace tvm
