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
  return ::mlir::MemRefType::get(
      ShapeFromExprs(buffer->shape), DTypeToMLIRType(buffer->dtype),
      /*layout=*/::mlir::MemRefLayoutAttrInterface{}, AddressSpaceAttr(space));
}

mlir::MemRefType
CodeGenTileLangTLIR::AllocToMemRefType(const AllocateNode *op) {
  auto space = ScopeToAddressSpace(GetPtrStorageScope(op->buffer_var));
  return ::mlir::MemRefType::get(
      ShapeFromExprs(op->extents), DTypeToMLIRType(op->dtype),
      /*layout=*/::mlir::MemRefLayoutAttrInterface{}, AddressSpaceAttr(space));
}

mlir::Value CodeGenTileLangTLIR::GetBufferValue(const Buffer &buffer) {
  auto it = var_map_.find(buffer->data.get());
  ICHECK(it != var_map_.end())
      << "CodeGenTileLangTLIR: unknown buffer " << buffer->name;
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

void CodeGenTileLangTLIR::EmitSort(const CallNode *op) {
  tvm::tl::NpuirSort sort(op->args, vmap_);
  builder_.create<::mlir::tl::SortOp>(
      builder_.getUnknownLoc(), GetBufferValue(sort.src),
      GetBufferValue(sort.dst_value), GetBufferValue(sort.dst_index),
      sort.descending, sort.sort_axis);
      }
void CodeGenTileLangTLIR::EmitCast(const CallNode *op) {
  tvm::tl::NpuirCast cast(op->args, vmap_);
  builder_.create<::mlir::tl::CastOp>(
      builder_.getUnknownLoc(), GetBufferValue(cast.src),
      GetBufferValue(cast.dst), builder_.getStringAttr(cast.round_mode));
      }
void CodeGenTileLangTLIR::EmitShr(const CallNode *op) {
  tvm::tl::NpuirShr shr(op->args, vmap_);
  auto round_attr = builder_.getBoolAttr(shr.round);
  builder_.create<::mlir::tl::ShrOp>(
      builder_.getUnknownLoc(), GetBufferValue(shr.src0),
      GetBufferValue(shr.src1), GetBufferValue(shr.dst), round_attr);
      }
void CodeGenTileLangTLIR::EmitShl(const CallNode *op) {
  tvm::tl::NpuirShl shl(op->args, vmap_);
  ICHECK(shl.Src0().IsTensor() && shl.Dst().IsTensor())
      << "CodeGenTileLangTLIR: tl.npuir_shl requires tensor lhs/out";
  ICHECK(shl.Src1().IsScalar()) << "CodeGenTileLangTLIR: tl.npuir_shl requires "
                                   "a scalar rhs (shift amount)";
  mlir::Value rhs_val = VisitExpr(shl.Src1().GetExpr());
  builder_.create<::mlir::tl::ShlOp>(
      builder_.getUnknownLoc(), GetBufferValue(shl.Src0().GetBuffer()), rhs_val,
      GetBufferValue(shl.Dst().GetBuffer()));
      }
void CodeGenTileLangTLIR::EmitPrintVar(const CallNode *op) {
  tvm::tl::NpuirDevicePrintVar print(op->args, vmap_);
  builder_.create<::mlir::tl::PrintOp>(
      builder_.getUnknownLoc(), VisitExpr(print.src),
      builder_.getStringAttr(print.prefix), builder_.getBoolAttr(print.hex));
}

void CodeGenTileLangTLIR::EmitPrintBuf(const CallNode *op) {
  tvm::tl::NpuirDevicePrintBuf print(op->args, vmap_);
  builder_.create<::mlir::tl::PrintOp>(
      builder_.getUnknownLoc(), GetBufferValue(print.src),
      builder_.getStringAttr(print.prefix), builder_.getBoolAttr(print.hex));
      }
void CodeGenTileLangTLIR::EmitSub(const CallNode *op) {
  tvm::tl::NpuirSub sub(op->args, vmap_);
  ICHECK(sub.Src0().IsTensor() && sub.Src1().IsTensor() && sub.Dst().IsTensor())
      << "CodeGenTileLangTLIR: tl.npuir_sub currently requires tensor operands";
  builder_.create<::mlir::tl::SubOp>(builder_.getUnknownLoc(),
                                     GetBufferValue(sub.Src0().GetBuffer()),
                                     GetBufferValue(sub.Src1().GetBuffer()),
                                     GetBufferValue(sub.Dst().GetBuffer()));
}

void CodeGenTileLangTLIR::EmitVErf(const CallNode *op) {
  tvm::tl::NpuirVErf verf(op->args, vmap_);
  ICHECK_EQ(verf.srcs.size(), 1u)
      << "CodeGenTileLangTLIR: tl.npuir_verf currently requires exactly one "
         "source operand";
  builder_.create<::mlir::tl::VErfOp>(builder_.getUnknownLoc(),
                                      GetBufferValue(verf.srcs[0]),
                                      GetBufferValue(verf.dst));
}
void CodeGenTileLangTLIR::EmitPad(const CallNode *op) {
  tvm::tl::NpuirPad pad(op->args, vmap_);
  mlir::Location loc = builder_.getUnknownLoc();

  mlir::Value src = GetBufferValue(pad.src);
  mlir::Value dst = GetBufferValue(pad.dst);
  mlir::Value pad_value = VisitExpr(pad.pad_value);

  llvm::SmallVector<mlir::Value> low, high;
  for (const PrimExpr &l : pad.low) {
    mlir::Value v = VisitExpr(l);
    if (!v.getType().isa<mlir::IndexType>())
      v = builder_.create<mlir::arith::IndexCastOp>(loc,
                                                    builder_.getIndexType(), v);
    low.push_back(v);
  }
  for (const PrimExpr &h : pad.high) {
    mlir::Value v = VisitExpr(h);
    if (!v.getType().isa<mlir::IndexType>())
      v = builder_.create<mlir::arith::IndexCastOp>(loc,
                                                    builder_.getIndexType(), v);
    high.push_back(v);
  }

  // Mirror CodeGenTileLangNPUIRAPI::VpadCodegen: when a dimension's
  // low/high padding amount is dynamic, its static_low/static_high entry
  // must carry the dynamic-size sentinel rather than a stale static value.
  if (!low.empty()) {
    pad.s_low[pad.pad_dim] = mlir::ShapedType::kDynamic;
  }
  if (!high.empty()) {
    pad.s_high[pad.pad_dim] = mlir::ShapedType::kDynamic;
  }

  builder_.create<::mlir::tl::PadOp>(loc, src, dst, pad_value, low, high,
                                     builder_.getDenseI64ArrayAttr(pad.s_low),
                                     builder_.getDenseI64ArrayAttr(pad.s_high));
 }
void CodeGenTileLangTLIR::EmitFlip(const CallNode *op) {
  tvm::tl::NpuirFlip flip(op->args, vmap_);
  builder_.create<::mlir::tl::FlipOp>(builder_.getUnknownLoc(),
                                      GetBufferValue(flip.src),
                                      GetBufferValue(flip.dst), flip.axis);
}

template <typename TLOpT, typename NpuirOpT>
void CodeGenTileLangTLIR::EmitBinary(const CallNode *op) {
  NpuirOpT bin(op->args, vmap_);
  ICHECK(bin.Src0().IsTensor() && bin.Src1().IsTensor() && bin.Dst().IsTensor())
      << "CodeGenTileLangTLIR: binary npuir op currently requires tensor "
         "operands";
  builder_.create<TLOpT>(builder_.getUnknownLoc(),
                         GetBufferValue(bin.Src0().GetBuffer()),
                         GetBufferValue(bin.Src1().GetBuffer()),
                         GetBufferValue(bin.Dst().GetBuffer()));
}

void CodeGenTileLangTLIR::EmitAtomicAdd(const CallNode *op) {
  tvm::tl::NpuirAtomicAdd atomic_add(op->args, vmap_);
  builder_.create<::mlir::tl::AtomicAddOp>(builder_.getUnknownLoc(),
                                           GetBufferValue(atomic_add.src),
                                           GetBufferValue(atomic_add.dst));
}
template <typename TLOpT, typename NpuirOpT>
void CodeGenTileLangTLIR::EmitUnary(const CallNode *op) {
  NpuirOpT un(op->args, vmap_);
  builder_.create<TLOpT>(builder_.getUnknownLoc(), GetBufferValue(un.src),
                         GetBufferValue(un.dst));
}

void CodeGenTileLangTLIR::EmitConcat(const CallNode *op) {
  tvm::tl::NpuirConcat concat(op->args, vmap_);

  llvm::SmallVector<::mlir::Value> srcs;
  srcs.reserve(concat.srcs.size());
  for (const Buffer &src : concat.srcs)
    srcs.push_back(GetBufferValue(src));

  builder_.create<::mlir::tl::ConcatOp>(builder_.getUnknownLoc(), srcs,
                                        GetBufferValue(concat.dst),
                                        builder_.getI64IntegerAttr(concat.dim));
}

mlir::Value CodeGenTileLangTLIR::ToIndex(mlir::Value v) {
  if (v.getType().isIndex())
    return v;
  return builder_.create<::mlir::arith::IndexCastOp>(
      builder_.getUnknownLoc(), builder_.getIndexType(), v);
}

void CodeGenTileLangTLIR::EmitArange(const CallNode *op) {
  tvm::tl::NpuirArange arange(op->args, vmap_);
  mlir::Value dst = GetBufferValue(arange.dst);
  mlir::Value offset = ToIndex(VisitExpr(arange.offset));

  llvm::SmallVector<::mlir::Value> strides;
  strides.reserve(arange.strides.size());
  for (const PrimExpr &stride : arange.strides)
    strides.push_back(ToIndex(VisitExpr(stride)));

  builder_.create<::mlir::tl::ArangeOp>(builder_.getUnknownLoc(), dst, offset,
                                        strides);
}

void CodeGenTileLangTLIR::EmitReshape(const CallNode *op) {
  tvm::tl::NpuirReshape reshape(op->args, vmap_);
  mlir::Value result = builder_.create<::mlir::tl::ReshapeOp>(
      builder_.getUnknownLoc(), BufferToMemRefType(reshape.dst),
      GetBufferValue(reshape.src));
  var_map_[reshape.dst->data.get()] = result;
}

void CodeGenTileLangTLIR::AddFunction(const GlobalVar &gvar,
                                      const PrimFunc &f) {
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
  auto func = builder_.create<::mlir::func::FuncOp>(builder_.getUnknownLoc(),
                                                    func_name, func_type);
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
  ICHECK(it != var_map_.end())
      << "CodeGenTileLangTLIR: unknown var " << op->name_hint;
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
  if (op->op.same_as(Op::Get("tl.npuir_sort"))) {
    EmitSort(op);
      return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_cast"))) {
    EmitCast(op);
      return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_shr"))) {
    EmitShr(op);
      return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_shl"))) {
    EmitShl(op);
      return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_debug_print_var"))) { 
    EmitPrintVar(op);
    return {};
  }
  if (op->op.same_as(
          Op::Get("tl.npuir_debug_print_buffer_value"))) { 
    EmitPrintBuf(op);
      return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_sub"))) {
    EmitSub(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_verf"))) {
    EmitVErf(op);
     return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_pad"))) {
    EmitPad(op);
     return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_flip"))) {
    EmitFlip(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_mul"))) {
    EmitBinary<::mlir::tl::MulOp, tvm::tl::NpuirMul>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_div"))) {
    EmitBinary<::mlir::tl::DivOp, tvm::tl::NpuirDiv>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_max"))) {
    EmitBinary<::mlir::tl::MaxOp, tvm::tl::NpuirMax>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_min"))) {
    EmitBinary<::mlir::tl::MinOp, tvm::tl::NpuirMin>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_rec"))) {
    EmitUnary<::mlir::tl::RecOp, tvm::tl::NpuirRec>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_atomic_add"))) {
    EmitAtomicAdd(op);
     return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_rsqrt"))) {
    EmitUnary<::mlir::tl::RsqrtOp, tvm::tl::NpuirRsqrt>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_sqrt"))) {
    EmitUnary<::mlir::tl::SqrtOp, tvm::tl::NpuirSqrt>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_abs"))) {
    EmitUnary<::mlir::tl::AbsOp, tvm::tl::NpuirAbs>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_not"))) {
    EmitUnary<::mlir::tl::NotOp, tvm::tl::NpuirNot>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_concat"))) {
    EmitConcat(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_arange"))) {
    EmitArange(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_reshape"))) {
    EmitReshape(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_pow"))) {
    EmitBinary<::mlir::tl::PowOp, tvm::tl::NpuirPow>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_xor"))) {
    EmitBinary<::mlir::tl::XorOp, tvm::tl::NpuirXor>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_or"))) {
    EmitBinary<::mlir::tl::OrOp, tvm::tl::NpuirOr>(op);
    return {};
  }
  if (op->op.same_as(Op::Get("tl.npuir_and"))) {
    EmitBinary<::mlir::tl::AndOp, tvm::tl::NpuirAnd>(op);
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
