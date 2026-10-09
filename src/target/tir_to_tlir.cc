// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tir_to_tlir.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"

namespace tvm {
namespace codegen {

using namespace tir;

namespace {

mlir::Type DTypeToMLIRType(mlir::OpBuilder &builder, DataType dtype) {
  if (dtype.is_float() && dtype.bits() == 32)
    return builder.getF32Type();
  if (dtype.is_float() && dtype.bits() == 16)
    return builder.getF16Type();
  if (dtype.is_int())
    return builder.getIntegerType(dtype.bits());
  if (dtype.is_uint())
    return builder.getIntegerType(dtype.bits(), /*isSigned=*/false);
  LOG(FATAL) << "tir_to_tlir v1: unsupported dtype " << dtype
             << " (extend DTypeToMLIRType before using a kernel that needs "
                "it)";
  return nullptr;
}

} // namespace

CodeGenTIRToTLIR::CodeGenTIRToTLIR() : builder(&context) {
  context.getOrLoadDialect<mlir::tlir::TLIRDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  module = mlir::ModuleOp::create(builder.getUnknownLoc());
}

void CodeGenTIRToTLIR::AddFunction(const GlobalVar &gvar, const tir::PrimFunc &f) {
  var_map_.clear();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "tir_to_tlir: expect PrimFunc to have the global_symbol attribute";
  std::string fname = static_cast<std::string>(global_symbol.value());

  // v1: real buffer/scalar params only -- no FFTS/sync-lock/workspace/
  // grid-info padding (see file header).
  llvm::SmallVector<mlir::Type> funcArgs;
  for (const tir::Var &v : f->params) {
    if (v.dtype().is_handle()) {
      Buffer buf = f->buffer_map[v];
      std::vector<int64_t> shape;
      for (const PrimExpr &s : buf->shape) {
        auto s_int = as_const_int(s);
        ICHECK(s_int) << "tir_to_tlir v1: only static buffer shapes are "
                         "supported";
        shape.push_back(*s_int);
      }
      funcArgs.push_back(
          mlir::MemRefType::get(shape, DTypeToMLIRType(builder, buf->dtype)));
    } else {
      funcArgs.push_back(DTypeToMLIRType(builder, v.dtype()));
    }
  }
  auto funcType = builder.getFunctionType(funcArgs, {});

  builder.setInsertionPointToEnd(module.getBody());
  auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                     fname, funcType);
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    tir::Var real_v = v.dtype().is_handle() ? f->buffer_map[v]->data : v;
    var_map_[real_v.get()] = funcOp.getArgument(i);
  }

  this->VisitStmt(f->body);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
}

void CodeGenTIRToTLIR::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::vector<int64_t> shape;
  for (const PrimExpr &e : op->extents) {
    auto s_int = as_const_int(e);
    ICHECK(s_int) << "tir_to_tlir v1: only static allocate extents are "
                     "supported";
    shape.push_back(*s_int);
  }
  auto memrefType =
      mlir::MemRefType::get(shape, DTypeToMLIRType(builder, op->dtype));
  auto allocOp =
      builder.create<mlir::tlir::AllocOp>(builder.getUnknownLoc(), memrefType);
  ICHECK(!var_map_.count(op->buffer_var.get()))
      << "tir_to_tlir v1: buffer var allocated twice";
  var_map_[op->buffer_var.get()] = allocOp.getResult();
  this->VisitStmt(op->body);
}

void CodeGenTIRToTLIR::VisitStmt_(const EvaluateNode *op) {
  this->VisitExpr(op->value);
}

void CodeGenTIRToTLIR::VisitStmt_(const SeqStmtNode *op) {
  for (const Stmt &s : op->seq)
    this->VisitStmt(s);
}

void CodeGenTIRToTLIR::VisitStmt_(const DeclBufferNode *op) {
  this->VisitStmt(op->body);
}

void CodeGenTIRToTLIR::VisitStmt_(const AttrStmtNode *op) {
  // v1: sync/scope annotations are not modeled yet (see proposal section
  // 4.3 -- tl.sync is future work); pass through to the body.
  this->VisitStmt(op->body);
}

mlir::Value CodeGenTIRToTLIR::ResolveWholeBufferRegionArg(const PrimExpr &arg) {
  const CallNode *region = arg.as<CallNode>();
  ICHECK(region) << "tir_to_tlir v1: expected a T.region(...) call argument, "
                    "got: "
                 << arg;
  const BufferLoadNode *bufferLoad = region->args[0].as<BufferLoadNode>();
  ICHECK(bufferLoad) << "tir_to_tlir v1: expected region arg0 to be a "
                         "BufferLoad";
  Buffer buf = bufferLoad->buffer;

  // region->args layout (matches CreateHIVMBinaryVectorOp's processImm):
  // args[0] = BufferLoad, args[1] = rank, args[2..] = per-dim extents.
  // v1 requires the region to cover the whole buffer (offset 0, full
  // extent) -- this matches tl.add's existing no-broadcast constraint.
  for (size_t i = 0; i < buf->shape.size(); ++i) {
    auto extent = region->args[2 + i].as<IntImmNode>();
    auto full = as_const_int(buf->shape[i]);
    ICHECK(extent && full && extent->value == *full)
        << "tir_to_tlir v1: only whole-buffer regions are supported "
           "(partial/subview regions are future work, matching tl.add's "
           "current no-broadcast verifier restriction)";
  }

  auto it = var_map_.find(buf->data.get());
  ICHECK(it != var_map_.end())
      << "tir_to_tlir v1: reference to buffer '" << buf->name
      << "' before its tl.alloc / function argument was seen";
  return it->second;
}

void CodeGenTIRToTLIR::VisitExpr_(const CallNode *op) {
  if (op->op.same_as(Op::Get("tl.copy"))) {
    ICHECK_EQ(op->args.size(), 2u);
    mlir::Value src = ResolveWholeBufferRegionArg(op->args[0]);
    mlir::Value dst = ResolveWholeBufferRegionArg(op->args[1]);
    builder.create<mlir::tlir::CopyOp>(builder.getUnknownLoc(), src, dst);
  } else if (op->op.same_as(Op::Get("tl.npuir_add"))) {
    ICHECK_EQ(op->args.size(), 3u);
    mlir::Value lhs = ResolveWholeBufferRegionArg(op->args[0]);
    mlir::Value rhs = ResolveWholeBufferRegionArg(op->args[1]);
    mlir::Value out = ResolveWholeBufferRegionArg(op->args[2]);
    builder.create<mlir::tlir::AddOp>(builder.getUnknownLoc(), lhs, rhs, out);
  } else {
    LOG(FATAL) << "tir_to_tlir v1: unsupported op " << op->op
               << " (documented unsupported path, not silent miscompilation "
                  "-- extend VisitExpr_(CallNode) before using a kernel "
                  "that needs this op)";
  }
}

std::string CodeGenTIRToTLIR::Finish() {
  if (failed(mlir::verify(module))) {
    module.emitError("tir_to_tlir v1: produced module failed verification");
  }
  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  return out;
}

} // namespace codegen
} // namespace tvm
