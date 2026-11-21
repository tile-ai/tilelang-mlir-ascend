// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/ConvertTLToHIVM.cpp
 * \brief Convert TL dialect (TLIR) to HIVM for the expert-mode add subset.
 */

#include "tilelangir/Dialect/TL/TLOps.h"
#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

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

hivm::PIPE stringToHIVMPipe(StringRef pipeType) {
  // Verified: TIR-level pipe_type strings match HIVM_PIPE_* case names
  // without the "PIPE_" prefix (confirmed via pipe_barrier()'s docstring
  // in tilelang/language/customize_npuir.py).
  if (pipeType == "S")
    return hivm::PIPE::PIPE_S;
  if (pipeType == "V")
    return hivm::PIPE::PIPE_V;
  if (pipeType == "M")
    return hivm::PIPE::PIPE_M;
  if (pipeType == "MTE1")
    return hivm::PIPE::PIPE_MTE1;
  if (pipeType == "MTE2")
    return hivm::PIPE::PIPE_MTE2;
  if (pipeType == "MTE3")
    return hivm::PIPE::PIPE_MTE3;
  if (pipeType == "ALL")
    return hivm::PIPE::PIPE_ALL;
  if (pipeType == "MTE4")
    return hivm::PIPE::PIPE_MTE4;
  if (pipeType == "MTE5")
    return hivm::PIPE::PIPE_MTE5;
  if (pipeType == "V2")
    return hivm::PIPE::PIPE_V2;
  if (pipeType == "FIX")
    return hivm::PIPE::PIPE_FIX;
  llvm_unreachable("unknown pipe type string in tl.pipe_barrier");
}

hivm::TCoreType inferTCoreType(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return hivm::TCoreType::VECTOR;
  auto tlCore = func->getAttrOfType<tl::CoreTypeAttr>("tl.core_type");
  if (!tlCore)
    return hivm::TCoreType::VECTOR;
  switch (tlCore.getValue()) {
  case tl::CoreType::AIC:
    return hivm::TCoreType::CUBE;
  case tl::CoreType::AIV:
    return hivm::TCoreType::VECTOR;
  case tl::CoreType::MIX:
    // Not modeled yet in this TLIR subset; default to VECTOR.
    return hivm::TCoreType::VECTOR;
  }
  return hivm::TCoreType::VECTOR;
}

hivm::SyncBlockInstrMode convertSyncInstrMode(tl::SyncInstrMode mode) {
  switch (mode) {
  case tl::SyncInstrMode::INTER_BLOCK:
    return hivm::SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION;
  case tl::SyncInstrMode::INTER_SUBBLOCK:
    return hivm::SyncBlockInstrMode::INTER_SUBBLOCK_SYNCHRONIZATION;
  case tl::SyncInstrMode::INTRA_BLOCK:
    return hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION;
  }
  return hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION;
}

struct ConvertPipeBarrier : public OpConversionPattern<tl::PipeBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::PipeBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pipeAttr =
        hivm::PipeAttr::get(op.getContext(), stringToHIVMPipe(op.getPipe()));
    rewriter.replaceOpWithNewOp<hivm::PipeBarrierOp>(op, pipeAttr);
    return success();
  }
};

struct ConvertSyncBlockSet : public OpConversionPattern<tl::SyncBlockSetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::SyncBlockSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto tcoreTypeAttr = hivm::TCoreTypeAttr::get(ctx, inferTCoreType(op));
    auto tpipeAttr = hivm::PipeAttr::get(ctx, stringToHIVMPipe(op.getPipe()));
    auto pipeAttr = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_S);
    auto flagIdAttr = rewriter.getI64IntegerAttr(op.getFlagId());
    auto instrModeAttr = hivm::SyncBlockInstrModeAttr::get(
        ctx, convertSyncInstrMode(op.getSyncInstrMode()));
    rewriter.replaceOpWithNewOp<hivm::SyncBlockSetOp>(
        op, tcoreTypeAttr, tpipeAttr, pipeAttr,
        /*static_flag_id=*/flagIdAttr, /*dynamic_flag_id=*/Value(),
        /*ffts_base_addr=*/Value(), instrModeAttr);
    return success();
  }
};

struct ConvertSyncBlockWait : public OpConversionPattern<tl::SyncBlockWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tl::SyncBlockWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = op.getContext();
    auto tcoreTypeAttr = hivm::TCoreTypeAttr::get(ctx, inferTCoreType(op));
    auto tpipeAttr = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_S);
    auto pipeAttr = hivm::PipeAttr::get(ctx, stringToHIVMPipe(op.getPipe()));
    auto flagIdAttr = rewriter.getI64IntegerAttr(op.getFlagId());
    rewriter.replaceOpWithNewOp<hivm::SyncBlockWaitOp>(
        op, tcoreTypeAttr, tpipeAttr, pipeAttr,
        /*static_flag_id=*/flagIdAttr, /*dynamic_flag_id=*/Value());
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
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd, ConvertSub,
                 ConvertBinary<tl::MulOp, hivm::VMulOp>,
                 ConvertBinary<tl::DivOp, hivm::VDivOp>,
                 ConvertBinary<tl::MaxOp, hivm::VMaxOp>,
                 ConvertBinary<tl::MinOp, hivm::VMinOp>, ConvertPipeBarrier,
                 ConvertSyncBlockSet, ConvertSyncBlockWait>(converter, ctx);
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
