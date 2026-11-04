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
      auto hivmSpace =
          hivm::AddressSpaceAttr::get(type.getContext(),
                                      convertAddressSpace(tlSpace.getValue()));
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
    auto newTy =
        dyn_cast_if_present<MemRefType>(getTypeConverter()->convertType(op.getType()));
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
        op, TypeRange{},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{adaptor.getOut()});
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
    func->setAttr(hivm::TFuncCoreTypeAttr::name,
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
    patterns.add<ConvertAlloc, ConvertCopy, ConvertAdd>(converter, ctx);
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
