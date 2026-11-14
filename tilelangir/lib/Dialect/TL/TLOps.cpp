// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Dialect/TL/TLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tl;

#define GET_OP_CLASSES
#include "tilelangir/Dialect/TL/TLOps.cpp.inc"

LogicalResult AllocOp::verify() {
  if (!llvm::isa<MemRefType>(getResult().getType()))
    return emitOpError("result must be a memref");
  return success();
}

LogicalResult CopyOp::verify() {
  auto src = llvm::cast<MemRefType>(getSource().getType());
  auto dst = llvm::cast<MemRefType>(getDestination().getType());
  if (src.getElementType() != dst.getElementType())
    return emitOpError("source and destination element types must match");
  if (src.getShape() != dst.getShape())
    return emitOpError("source and destination shapes must match");
  return success();
}

LogicalResult AddOp::verify() {
  auto lhs = llvm::cast<MemRefType>(getLhs().getType());
  auto rhs = llvm::cast<MemRefType>(getRhs().getType());
  auto out = llvm::cast<MemRefType>(getOut().getType());
  if (lhs.getElementType() != rhs.getElementType() ||
      lhs.getElementType() != out.getElementType())
    return emitOpError("operand element types must match");
  if (lhs.getShape() != rhs.getShape() || lhs.getShape() != out.getShape())
    return emitOpError("operand shapes must match");
  return success();
}

LogicalResult SubOp::verify() {
  auto lhs = llvm::cast<MemRefType>(getLhs().getType());
  auto rhs = llvm::cast<MemRefType>(getRhs().getType());
  auto out = llvm::cast<MemRefType>(getOut().getType());
  if (lhs.getElementType() != rhs.getElementType() ||
      lhs.getElementType() != out.getElementType())
    return emitOpError("operand element types must match");
  if (lhs.getShape() != rhs.getShape() || lhs.getShape() != out.getShape())
    return emitOpError("operand shapes must match");
  return success();
}

static LogicalResult verifyElementwiseBinary(Operation *op, Value lhsV,
                                             Value rhsV, Value outV) {
  auto lhs = llvm::cast<MemRefType>(lhsV.getType());
  auto rhs = llvm::cast<MemRefType>(rhsV.getType());
  auto out = llvm::cast<MemRefType>(outV.getType());
  if (lhs.getElementType() != rhs.getElementType() ||
      lhs.getElementType() != out.getElementType())
    return op->emitOpError("operand element types must match");
  if (lhs.getShape() != rhs.getShape() || lhs.getShape() != out.getShape())
    return op->emitOpError("operand shapes must match");
  return success();
}

LogicalResult MulOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult DivOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult MaxOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult MinOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}

LogicalResult PadOp::verify() {
  auto src = llvm::cast<MemRefType>(getSrc().getType());
  auto dst = llvm::cast<MemRefType>(getDst().getType());
  if (src.getElementType() != dst.getElementType())
    return emitOpError("source and destination element types must match");
  if (src.getRank() != dst.getRank())
    return emitOpError("source and destination ranks must match");

  int64_t rank = src.getRank();
  llvm::ArrayRef<int64_t> staticLow = getStaticLow();
  llvm::ArrayRef<int64_t> staticHigh = getStaticHigh();
  if (static_cast<int64_t>(staticLow.size()) != rank)
    return emitOpError("static_low must have one entry per source rank");
  if (static_cast<int64_t>(staticHigh.size()) != rank)
    return emitOpError("static_high must have one entry per source rank");

  int64_t numDynamicLow = llvm::count(staticLow, ShapedType::kDynamic);
  int64_t numDynamicHigh = llvm::count(staticHigh, ShapedType::kDynamic);
  if (static_cast<int64_t>(getLow().size()) != numDynamicLow)
    return emitOpError("number of dynamic low operands must match the "
                       "number of dynamic entries in static_low");
  if (static_cast<int64_t>(getHigh().size()) != numDynamicHigh)
    return emitOpError("number of dynamic high operands must match the "
                       "number of dynamic entries in static_high");
  return success();
}
