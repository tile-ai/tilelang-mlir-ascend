//===- TLIROps.cpp - TLIR op verifiers ----------------------------------===//

#include "tilelangir/Dialect/TL/TLIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tlir;

LogicalResult CopyOp::verify() {
  auto srcTy = llvm::cast<MemRefType>(getSource().getType());
  auto dstTy = llvm::cast<MemRefType>(getTarget().getType());
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("source and target shapes must match, got ")
           << srcTy << " and " << dstTy;
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("source and target element types must match");
  return success();
}

LogicalResult AddOp::verify() {
  auto lhsTy = llvm::cast<MemRefType>(getLhs().getType());
  auto rhsTy = llvm::cast<MemRefType>(getRhs().getType());
  auto outTy = llvm::cast<MemRefType>(getOut().getType());

  if (lhsTy.getShape() != rhsTy.getShape() ||
      lhsTy.getShape() != outTy.getShape())
    return emitOpError(
        "lhs, rhs and out must have identical shapes (v1: no broadcast; "
        "extend before adding a kernel that needs it)");

  if (lhsTy.getElementType() != rhsTy.getElementType() ||
      lhsTy.getElementType() != outTy.getElementType())
    return emitOpError("lhs, rhs and out must have identical element type");

  if (!lhsTy.getElementType().isIntOrFloat())
    return emitOpError("tl.add currently only supports int/float element "
                        "types");

  return success();
}
