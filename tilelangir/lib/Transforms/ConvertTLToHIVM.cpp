// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/ConvertTLToHIVM.cpp
 * \brief Convert TL dialect (TLIR) to HIVM for the expert-mode add subset.
 */

#include "tilelangir/Dialect/TL/TLOps.h"
#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_CONVERTTLTOHIVM
#include "tilelangir/Transforms/Passes.h.inc"

namespace {

hivm::AddressSpace convertAddressSpace(tl::AddressSpace space) {
  switch (space) {
  case tl::AddressSpace::GM:
    return hivm::AddressSpace::GM;
  case tl::AddressSpace::UB:
    return hivm::AddressSpace::UB;
  case tl::AddressSpace::L1:
    return hivm::AddressSpace::L1;
  case tl::AddressSpace::L0:
    return hivm::AddressSpace::L0C;
  case tl::AddressSpace::L0A:
    return hivm::AddressSpace::L0A;
  case tl::AddressSpace::L0B:
    return hivm::AddressSpace::L0B;
  case tl::AddressSpace::L0C:
    return hivm::AddressSpace::L0C;
  }
  return hivm::AddressSpace::UB;
}

hivm::TFuncCoreType convertFuncCoreType(tl::CoreType core) {
  switch (core) {
  case tl::CoreType::AIC:
    return hivm::TFuncCoreType::AIC;
  case tl::CoreType::AIV:
    return hivm::TFuncCoreType::AIV;
  case tl::CoreType::MIX:
    return hivm::TFuncCoreType::MIX;
  }
  return hivm::TFuncCoreType::AIV;
}

hivm::ReduceOperation convertReduceMode(tl::ReduceMode mode) {
  switch (mode) {
  case tl::ReduceMode::Sum:
    return hivm::ReduceOperation::sum;
  case tl::ReduceMode::Max:
    return hivm::ReduceOperation::max;
  case tl::ReduceMode::Min:
    return hivm::ReduceOperation::min;
  }
  return hivm::ReduceOperation::sum;
}
  
hivm::PIPE convertPipe(tl::Pipe pipe) {
  switch (pipe) {
  case tl::Pipe::PIPE_S:
    return hivm::PIPE::PIPE_S;
  case tl::Pipe::PIPE_V:
    return hivm::PIPE::PIPE_V;
  case tl::Pipe::PIPE_M:
    return hivm::PIPE::PIPE_M;
  case tl::Pipe::PIPE_MTE1:
    return hivm::PIPE::PIPE_MTE1;
  case tl::Pipe::PIPE_MTE2:
    return hivm::PIPE::PIPE_MTE2;
  case tl::Pipe::PIPE_MTE3:
    return hivm::PIPE::PIPE_MTE3;
  case tl::Pipe::PIPE_ALL:
    return hivm::PIPE::PIPE_ALL;
  case tl::Pipe::PIPE_MTE4:
    return hivm::PIPE::PIPE_MTE4;
  case tl::Pipe::PIPE_MTE5:
    return hivm::PIPE::PIPE_MTE5;
  case tl::Pipe::PIPE_V2:
    return hivm::PIPE::PIPE_V2;
  case tl::Pipe::PIPE_FIX:
    return hivm::PIPE::PIPE_FIX;
  }
  return hivm::PIPE::PIPE_ALL;
}

hivm::TModuleCoreType convertModuleCoreType(tl::CoreType core) {
  switch (core) {
  case tl::CoreType::AIC:
    return hivm::TModuleCoreType::AIC;
  case tl::CoreType::AIV:
    return hivm::TModuleCoreType::AIV;
  case tl::CoreType::MIX:
    return hivm::TModuleCoreType::MIX;
  }
  return hivm::TModuleCoreType::AIV;
}

class TLTypeConverter : public TypeConverter {
public:
  TLTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](MemRefType type) -> Type {
      Attribute space = type.getMemorySpace();
      auto tlSpace = dyn_cast_if_present<tl::AddressSpaceAttr>(space);
      if (!tlSpace)
        return type;
      auto hivmSpace = hivm::AddressSpaceAttr::get(
          type.getContext(), convertAddressSpace(tlSpace.getValue()));
      return MemRefType::get(type.getShape(), type.getElementType(),
                             type.getLayout(), hivmSpace);
    });
  }
};

struct ConvertAlloc : public OpConversionPattern<tl::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto newTy = dyn_cast_if_present<MemRefType>(
        getTypeConverter()->convertType(op.getType()));
    if (!newTy)
      return rewriter.notifyMatchFailure(op, "failed to convert alloc type");
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newTy);
    return success();
  }
};

struct ConvertVBitcast : public OpConversionPattern<tl::VBitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::VBitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newTy = dyn_cast_if_present<MemRefType>(
        getTypeConverter()->convertType(op.getType()));
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert vbitcast result type");
    rewriter.replaceOpWithNewOp<hivm::BitcastOp>(op, newTy,
                                                 adaptor.getSource());
    return success();
  }
};

struct ConvertCopy : public OpConversionPattern<tl::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, adaptor.getSource(),
                                                adaptor.getDestination());
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<tl::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VAddOp>(
        op, TypeRange{}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{adaptor.getOut()});
    return success();
  }
};

struct ConvertCast : public OpConversionPattern<tl::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundModeOpt = hivm::symbolizeRoundMode(adaptor.getRoundMode());
    if (!roundModeOpt)
      return rewriter.notifyMatchFailure(op, "unknown round_mode: " +
                                                 adaptor.getRoundMode());
    auto roundModeAttr = hivm::RoundModeAttr::get(getContext(), *roundModeOpt);
    auto castAttr =
        hivm::TypeFnAttr::get(getContext(), hivm::TypeFn::cast_signed);
    rewriter.replaceOpWithNewOp<hivm::VCastOp>(
        op, TypeRange{}, ValueRange{adaptor.getSrc()},
        ValueRange{adaptor.getDst()}, roundModeAttr, castAttr);
     return success();
  }
};

struct ConvertShr : public OpConversionPattern<tl::ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VShROp>(
        op, TypeRange{}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{adaptor.getOut()}, adaptor.getRoundAttr());
     return success();
  }
};

struct ConvertShl : public OpConversionPattern<tl::ShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VShLOp>(
        op, TypeRange{}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{adaptor.getOut()});
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<tl::SubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VSubOp>(
        op, TypeRange{}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{adaptor.getOut()});
    return success();
  }
};

struct ConvertReduce : public OpConversionPattern<tl::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arith = hivm::ReduceOpAttr::get(getContext(),
                                         convertReduceMode(adaptor.getMode()));
    rewriter.replaceOpWithNewOp<hivm::VReduceOp>(
        op, TypeRange{}, adaptor.getSrc(), ValueRange{adaptor.getDst()}, arith,
        adaptor.getReduceDimsAttr());
        return success();
  }
};
  
struct ConvertVErf : public OpConversionPattern<tl::VErfOp> {
  using OpConversionPattern::OpConversionPattern;

  // Expands erf(x) via the same polynomial approximation used by the
  // existing NPUIR API codegen path (src/target/codegen_npuir_api.cc,
  // VerfCodegen):
  //   erf(x) ~= (2/sqrt(pi)) * (x - x^3/3 + x^5/10 - x^7/42)
  LogicalResult
  matchAndRewrite(tl::VErfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    auto memTy = cast<MemRefType>(src.getType());
    Type elementType = memTy.getElementType();

    Value two = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 2.0));
    Value sqrtPi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 1.7724538509055160));
    Value minusOne = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, -1.0));
    Value three = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 3.0));
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 1.0));
    Value ten = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 10.0));
    Value fortyTwo = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, 42.0));

    Value twoOverSqrtPi = rewriter.create<arith::DivFOp>(loc, two, sqrtPi);
    Value minusOneOver3 = rewriter.create<arith::DivFOp>(loc, minusOne, three);
    Value oneOver10 = rewriter.create<arith::DivFOp>(loc, one, ten);
    Value minusOneOver42 =
        rewriter.create<arith::DivFOp>(loc, minusOne, fortyTwo);

    Value x2 = rewriter.create<memref::AllocOp>(loc, memTy);
    Value x3 = rewriter.create<memref::AllocOp>(loc, memTy);
    Value x5 = rewriter.create<memref::AllocOp>(loc, memTy);
    Value x7 = rewriter.create<memref::AllocOp>(loc, memTy);
    Value tmp = rewriter.create<memref::AllocOp>(loc, memTy);

    rewriter.create<hivm::VMulOp>(loc, TypeRange{}, ValueRange{src, src},
                                  ValueRange{x2});
    rewriter.create<hivm::VMulOp>(loc, TypeRange{}, ValueRange{x2, src},
                                  ValueRange{x3});
    rewriter.create<hivm::VMulOp>(loc, TypeRange{}, ValueRange{x3, x2},
                                  ValueRange{x5});
    rewriter.create<hivm::VMulOp>(loc, TypeRange{}, ValueRange{x5, x2},
                                  ValueRange{x7});

    rewriter.create<hivm::VMulOp>(
        loc, TypeRange{}, ValueRange{x3, minusOneOver3}, ValueRange{x3});
    rewriter.create<hivm::VMulOp>(loc, TypeRange{}, ValueRange{x5, oneOver10},
                                  ValueRange{x5});
    rewriter.create<hivm::VMulOp>(
        loc, TypeRange{}, ValueRange{x7, minusOneOver42}, ValueRange{x7});

    rewriter.create<hivm::VAddOp>(loc, TypeRange{}, ValueRange{src, x3},
                                  ValueRange{tmp});
    rewriter.create<hivm::VAddOp>(loc, TypeRange{}, ValueRange{x5, tmp},
                                  ValueRange{tmp});
    rewriter.create<hivm::VAddOp>(loc, TypeRange{}, ValueRange{x7, tmp},
                                  ValueRange{tmp});

    rewriter.create<hivm::VMulOp>(
        loc, TypeRange{}, ValueRange{tmp, twoOverSqrtPi}, ValueRange{dst});

    rewriter.eraseOp(op);
       return success();
  }
};
struct ConvertPad : public OpConversionPattern<tl::PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VPadOp>(
        op, TypeRange{}, adaptor.getSrc(), adaptor.getDst(),
        adaptor.getPadValue(), adaptor.getLow(), adaptor.getHigh(),
        adaptor.getStaticLow(), adaptor.getStaticHigh());
      return success();
  }
};
struct ConvertFlip : public OpConversionPattern<tl::FlipOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::FlipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VFlipOp>(
        op, TypeRange{}, adaptor.getSrc(), adaptor.getDst(), adaptor.getAxis());
    return success();
  }
};

template <typename TLOpT, typename HIVMOpT>
struct ConvertBinary : public OpConversionPattern<TLOpT> {
  using OpConversionPattern<TLOpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TLOpT op,
                  typename OpConversionPattern<TLOpT>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<HIVMOpT>(
        op, TypeRange{}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{adaptor.getOut()});
    return success();
  }
};

struct ConvertSetFlag : public OpConversionPattern<tl::SetFlagOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::SetFlagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto setPipe = hivm::PipeAttr::get(rewriter.getContext(),
                                       convertPipe(op.getSetPipe()));
    auto waitPipe = hivm::PipeAttr::get(rewriter.getContext(),
                                        convertPipe(op.getWaitPipe()));
    rewriter.replaceOpWithNewOp<hivm::SetFlagOp>(
        op, setPipe, waitPipe, hivm::EventAttr{}, adaptor.getEventId());
    return success();
  }
};
  
struct ConvertAtomicAdd : public OpConversionPattern<tl::AtomicAddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::AtomicAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeOp = rewriter.replaceOpWithNewOp<hivm::StoreOp>(
        op, TypeRange{}, adaptor.getSrc(), adaptor.getDst());
    storeOp.setAtomicKind(hivm::AtomicKind::ADD);
    return success();
  }
};
  
template <typename TLOpT, typename HIVMOpT>
struct ConvertUnary : public OpConversionPattern<TLOpT> {
  using OpConversionPattern<TLOpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TLOpT op,
                  typename OpConversionPattern<TLOpT>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<HIVMOpT>(op, TypeRange{},
                                                  ValueRange{adaptor.getSrc()},
                                                  ValueRange{adaptor.getOut()});
    return success();
  }
};

struct ConvertWaitFlag : public OpConversionPattern<tl::WaitFlagOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::WaitFlagOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto setPipe = hivm::PipeAttr::get(rewriter.getContext(),
                                       convertPipe(op.getSetPipe()));
    auto waitPipe = hivm::PipeAttr::get(rewriter.getContext(),
                                        convertPipe(op.getWaitPipe()));
    rewriter.replaceOpWithNewOp<hivm::WaitFlagOp>(
        op, setPipe, waitPipe, hivm::EventAttr{}, adaptor.getEventId());
    return success();
  }
};
  
struct ConvertConcat : public OpConversionPattern<tl::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // NOTE: confirm the exact HIVM op name/signature for concat
    // (likely hivm.hir.vconcat or similar) and adjust accordingly.
    rewriter.replaceOpWithNewOp<hivm::VConcatOp>(
        op, TypeRange{}, rewriter.getI64IntegerAttr(adaptor.getAxis()),
        adaptor.getSrcs(), adaptor.getDst());
    return success();
  }
};
struct ConvertArange : public OpConversionPattern<tl::ArangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VArangeOp>(
        op, TypeRange{}, adaptor.getDst(), adaptor.getOffset(),
        adaptor.getStrides());
    return success();
  }
};
struct ConvertReshape : public OpConversionPattern<tl::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newTy = dyn_cast_if_present<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!newTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert reshape result type");

    ArrayRef<int64_t> shape = newTy.getShape();
    SmallVector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];

    // newTy already has no explicit layout (the type converter only rewrites
    // the memory space), which MLIR treats as the canonical contiguous,
    // offset-0 layout -- exactly what we compute below. Reusing it directly
    // (instead of attaching an explicit StridedLayoutAttr) keeps this op's
    // result type identical to what downstream ops like tl.copy's lowering
    // expect, avoiding an unresolved materialization between an explicitly-
    // strided type and the plain one.
    SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes, stridesFold;
    for (int64_t s : shape)
      sizes.push_back(rewriter.getIndexAttr(s));
    for (int64_t s : strides)
      stridesFold.push_back(rewriter.getIndexAttr(s));

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, newTy, adaptor.getSrc(), offsets, sizes, stridesFold);
    return success();
  }
};
struct ConvertSort : public OpConversionPattern<tl::SortOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::SortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hivm::VSortOp>(
        op, TypeRange{}, adaptor.getSrc(),
        ValueRange{adaptor.getDstValue(), adaptor.getDstIndex()},
        op.getDescending(), op.getSortAxis());
    return success();
  }
};

struct ConvertPrint : public OpConversionPattern<tl::PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // VERIFY operand/attr order against HIVM_DebugOp in HIVMOps.td before
    // relying on this — inferred from the builder.create<mlir::hivm::DebugOp>
    // call in codegen_npuir_api.cc::DebugPrintCodegen.
    rewriter.replaceOpWithNewOp<hivm::DebugOp>(op, "print", op.getPrefix(),
                                               op.getHex(), adaptor.getSrc(),
                                               hivm::TCoreTypeAttr{});
    return success();
  }
};

void rewriteCoreTypeAttrs(ModuleOp module) {
  tl::CoreType moduleCore = tl::CoreType::AIV;
  bool sawCore = false;

  module.walk([&](func::FuncOp func) {
    auto tlCore = func->getAttrOfType<tl::CoreTypeAttr>("tl.core_type");
    if (!tlCore)
      return;
    if (!sawCore) {
      moduleCore = tlCore.getValue();
      sawCore = true;
    }
    func->removeAttr("tl.core_type");
    func->setAttr(
        hivm::TFuncCoreTypeAttr::name,
        hivm::TFuncCoreTypeAttr::get(func.getContext(),
                                     convertFuncCoreType(tlCore.getValue())));
  });

  module->setAttr(hivm::TModuleCoreTypeAttr::name,
                  hivm::TModuleCoreTypeAttr::get(
                      module.getContext(), convertModuleCoreType(moduleCore)));
}
struct ConvertTLToHIVMPass : impl::ConvertTLToHIVMBase<ConvertTLToHIVMPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    TLTypeConverter converter;

    ConversionTarget target(*ctx);
    target.addLegalDialect<hivm::HIVMDialect, memref::MemRefDialect,
                           func::FuncDialect, arith::ArithDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<tl::TLDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertVErf>(converter,
                                                                     ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertPad>(converter,
                                                                     ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertFlip>(converter,
                                                                     ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertSort>(converter,
                                                                     ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertCast>(converter,
                                                                     ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertShr>(converter, ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertShl>(converter,
                                                                    ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertPrint>(converter,
                                                                      ctx);
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertSub,
                 ConvertReduce, 
                 ConvertVBitcast,ConvertConcat, ConvertArange,
                 ConvertBinary<tl::MulOp, hivm::VMulOp>,
                 ConvertBinary<tl::DivOp, hivm::VDivOp>,
                 ConvertBinary<tl::MaxOp, hivm::VMaxOp>,
                 ConvertBinary<tl::MinOp, hivm::VMinOp>,
                 ConvertUnary<tl::CosOp, hivm::VCosOp>,
                 ConvertUnary<tl::LnOp, hivm::VLnOp>,
                 ConvertUnary<tl::ExpOp, hivm::VExpOp>,
                 ConvertUnary<tl::RecOp, hivm::VRecOp>,
                 ConvertUnary<tl::RsqrtOp, hivm::VRsqrtOp>,
                 ConvertUnary<tl::SqrtOp, hivm::VSqrtOp>,
                 ConvertUnary<tl::AbsOp, hivm::VAbsOp>,
                 ConvertUnary<tl::NotOp, hivm::VNotOp>,
                 ConvertBinary<tl::PowOp, hivm::VPowOp>,
                 ConvertBinary<tl::XorOp, hivm::VXorOp>,
                 ConvertBinary<tl::OrOp, hivm::VOrOp>,
                 ConvertBinary<tl::AndOp, hivm::VAndOp>,
                 ConvertReshape,ConvertAtomicAdd,ConvertSetFlag, ConvertWaitFlag>(converter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    rewriteCoreTypeAttrs(getOperation());
  }
};

   

} // namespace

} // namespace tilelangir
} // namespace mlir

namespace tilelangir {

void buildTileLangIRCompilePipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::tilelangir::createConvertTLToHIVM());
}

} // namespace tilelangir
