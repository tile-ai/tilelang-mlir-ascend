// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// TIR -> TLIR importer, memref form only, for: alloc, copy, add.
//
//   tir::Allocate        -> memref.alloc  (memory space from storage scope)
//   tl.copy              -> memref.copy
//   tl.npuir_add         -> linalg.elemwise_binary {fun = add}
//   buffer params        -> memref args in the global ("gm") address space
//
// Everything is destination-passing on memrefs: no tensors, no function
// results, no bufferization ops. Any other TIR call is a hard error so a
// kernel that needs more never miscompiles silently.

#include "tir_to_tlir.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <tvm/tir/op.h>

namespace tvm {
namespace codegen {

using namespace tir;

namespace {

mlir::Type DTypeToMLIRType(mlir::OpBuilder &builder, DataType dtype) {
  if (dtype.is_bfloat16())
    return builder.getBF16Type();
  if (dtype.is_float() && dtype.bits() == 32)
    return builder.getF32Type();
  if (dtype.is_float() && dtype.bits() == 16)
    return builder.getF16Type();
  // arith ops need signless integers, so uint maps to signless too.
  if (dtype.is_int() || dtype.is_uint())
    return builder.getIntegerType(dtype.bits());
  LOG(FATAL) << "tir_to_tlir: unsupported dtype " << dtype
             << " (extend DTypeToMLIRType)";
  return nullptr;
}

std::vector<int64_t> StaticShape(const Array<PrimExpr> &shape) {
  std::vector<int64_t> out;
  for (const PrimExpr &s : shape) {
    auto v = as_const_int(s);
    ICHECK(v) << "tir_to_tlir: only static buffer shapes are supported";
    out.push_back(*v);
  }
  return out;
}

} // namespace

CodeGenTIRToTLIR::CodeGenTIRToTLIR() : builder(&context) {
  // Lets us build #hivm.address_space<...> as an opaque attribute even though
  // the hivm dialect is not linked into this library. bishengir-opt parses it
  // properly later.
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  module = mlir::ModuleOp::create(builder.getUnknownLoc());
}

mlir::Attribute CodeGenTIRToTLIR::AddressSpaceFor(const std::string &scope) {
  // TIR storage scope -> Ascend address space name. Adjust if your kernels
  // use other scope strings (the FATAL below prints the one it did not know).
  static const std::unordered_map<std::string, std::string> kMap = {
      {"", "gm"},          {"global", "gm"},   {"shared", "ub"},
      {"shared.dyn", "ub"}, {"local", "ub"},    {"ub", "ub"},
      {"fragment", "ub"},  {"l1", "l1"},       {"l0a", "l0a"},
      {"l0b", "l0b"},      {"l0c", "l0c"}};
  auto it = kMap.find(scope);
  ICHECK(it != kMap.end()) << "tir_to_tlir: unknown storage scope '" << scope
                           << "' (add it to AddressSpaceFor)";
  mlir::Attribute attr = mlir::parseAttribute(
      "#hivm.address_space<" + it->second + ">", &context);
  ICHECK(attr) << "tir_to_tlir: could not build address space attribute";
  return attr;
}

mlir::MemRefType CodeGenTIRToTLIR::MakeMemRefType(llvm::ArrayRef<int64_t> shape,
                                                  DataType dtype,
                                                  const std::string &scope) {
  return mlir::MemRefType::get(shape, DTypeToMLIRType(builder, dtype),
                               mlir::MemRefLayoutAttrInterface{},
                               AddressSpaceFor(scope));
}

void CodeGenTIRToTLIR::AddFunction(const GlobalVar &gvar,
                                   const tir::PrimFunc &f) {
  var_map_.clear();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "tir_to_tlir: expect PrimFunc to have the global_symbol attribute";
  std::string fname = static_cast<std::string>(global_symbol.value());

  // Every buffer param is a memref in global memory (inputs AND outputs);
  // scalar params stay scalars. The function returns nothing.
  llvm::SmallVector<mlir::Type> funcArgs;
  for (const tir::Var &v : f->params) {
    if (v.dtype().is_handle()) {
      Buffer buf = f->buffer_map[v];
      funcArgs.push_back(
          MakeMemRefType(StaticShape(buf->shape), buf->dtype, "global"));
    } else {
      funcArgs.push_back(DTypeToMLIRType(builder, v.dtype()));
    }
  }
  auto funcType = builder.getFunctionType(funcArgs, {});

  builder.setInsertionPointToEnd(module.getBody());
  auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                   fname, funcType);

  // Mark the function as a device kernel entry. Without these attributes,
  // hivmc's HIVM->LLVM stage refuses to lower the hivm ops inside it.
  funcOp->setAttr("hacc.entry", builder.getUnitAttr());
  funcOp->setAttr("hacc.function_kind",
                  mlir::parseAttribute("#hacc.function_kind<DEVICE>",
                                       builder.getContext()));
  mlir::Block *entry = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  size_t idx = 0;
  for (const tir::Var &v : f->params) {
    tir::Var real_v = v.dtype().is_handle() ? f->buffer_map[v]->data : v;
    var_map_[real_v.get()] = funcOp.getArgument(idx++);
  }

  this->VisitStmt(f->body);

  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
}

void CodeGenTIRToTLIR::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  ICHECK(!var_map_.count(op->buffer_var.get()))
      << "tir_to_tlir: buffer var allocated twice";

  const auto *ptr = op->buffer_var->type_annotation.as<PointerTypeNode>();
  std::string scope = ptr ? std::string(ptr->storage_scope) : "";

  std::vector<int64_t> shape = StaticShape(op->extents);
  auto ty = MakeMemRefType(shape, op->dtype, scope);
  var_map_[op->buffer_var.get()] =
      builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), ty);

  this->VisitStmt(op->body);
}

mlir::Value CodeGenTIRToTLIR::ResolveWholeBufferRegionArg(const PrimExpr &arg) {
  const CallNode *region = arg.as<CallNode>();
  ICHECK(region) << "tir_to_tlir: expected a T.region(...) call argument, got: "
                 << arg;
  const BufferLoadNode *bufferLoad = region->args[0].as<BufferLoadNode>();
  ICHECK(bufferLoad) << "tir_to_tlir: expected region arg0 to be a BufferLoad";
  Buffer buf = bufferLoad->buffer;

  // region->args = [BufferLoad, rank, extent0, extent1, ...]
  ICHECK_GE(region->args.size(), 2 + buf->shape.size())
      << "tir_to_tlir: malformed region call for buffer '" << buf->name << "'";
  for (size_t i = 0; i < buf->shape.size(); ++i) {
    auto extent = region->args[2 + i].as<IntImmNode>();
    auto full = as_const_int(buf->shape[i]);
    ICHECK(extent && full && extent->value == *full)
        << "tir_to_tlir: only whole-buffer regions are supported "
           "(subviews are future work)";
  }

  auto it = var_map_.find(buf->data.get());
  ICHECK(it != var_map_.end())
      << "tir_to_tlir: buffer '" << buf->name
      << "' used before its allocation / function argument was seen";
  return it->second;
}

void CodeGenTIRToTLIR::VisitExpr_(const CallNode *op) {
  auto loc = builder.getUnknownLoc();

  if (op->op.same_as(Op::Get("tl.copy"))) {
    ICHECK_EQ(op->args.size(), 2u);
    mlir::Value src = ResolveWholeBufferRegionArg(op->args[0]);
    mlir::Value dst = ResolveWholeBufferRegionArg(op->args[1]);
    builder.create<mlir::memref::CopyOp>(loc, src, dst);

  } else if (op->op.same_as(Op::Get("tl.npuir_add"))) {
    ICHECK_EQ(op->args.size(), 3u);
    mlir::Value lhs = ResolveWholeBufferRegionArg(op->args[0]);
    mlir::Value rhs = ResolveWholeBufferRegionArg(op->args[1]);
    mlir::Value out = ResolveWholeBufferRegionArg(op->args[2]);

    auto outTy = mlir::cast<mlir::MemRefType>(out.getType());
    ICHECK(mlir::cast<mlir::MemRefType>(lhs.getType()).getShape() ==
               outTy.getShape() &&
           mlir::cast<mlir::MemRefType>(rhs.getType()).getShape() ==
               outTy.getShape())
        << "tir_to_tlir: add requires equal shapes (no broadcast in v1)";

    // NOTE: emitted as the *named* op linalg.elemwise_binary{fun=add}. The
    // prebuilt convert-hfusion-to-hivm pass rejects linalg.generic and
    // linalg.add, but lowers this form to hivm.hir.vadd. binary_fn<add> covers
    // both float and integer element types.
    auto funAttr =
        mlir::linalg::BinaryFnAttr::get(&context, mlir::linalg::BinaryFn::add);
    builder.create<mlir::linalg::ElemwiseBinaryOp>(
        loc, mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
        mlir::ValueRange{out},
        llvm::ArrayRef<mlir::NamedAttribute>{
            builder.getNamedAttr("fun", funAttr)});

  } else {
    LOG(FATAL) << "tir_to_tlir: unsupported op " << op->op
               << " (only tl.copy and tl.npuir_add are implemented; extend "
                  "VisitExpr_(CallNode) for more)";
  }
}

std::string CodeGenTIRToTLIR::Finish() {
  std::string out;
  {
    llvm::raw_string_ostream os(out);
    module.print(os);
  }
  if (failed(mlir::verify(module))) {
    LOG(FATAL) << "tir_to_tlir: produced module failed verification:\n" << out;
  }
  return out;
}

} // namespace codegen
} // namespace tvm
