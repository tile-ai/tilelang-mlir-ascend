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

LogicalResult ConcatOp::verify() {
  auto dstTy = llvm::cast<MemRefType>(getDst().getType());
  int64_t axis = getAxis();
  int64_t rank = dstTy.getRank();

  if (axis < 0 || axis >= rank)
    return emitOpError("axis (") << axis << ") out of bounds for rank " << rank;

  if (getSrcs().empty())
    return emitOpError("expected at least one source operand");

  int64_t axisSum = 0;
  for (auto src : getSrcs()) {
    auto srcTy = llvm::cast<MemRefType>(src.getType());
    if (srcTy.getRank() != rank)
      return emitOpError("all operands must have the same rank as dst");
    if (srcTy.getElementType() != dstTy.getElementType())
      return emitOpError("all operands must have the same element type as dst");
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis)
        continue;
      if (srcTy.getDimSize(d) != dstTy.getDimSize(d))
        return emitOpError("dimension ")
               << d << " must match dst outside the concat axis";
    }
    axisSum += srcTy.getDimSize(axis);
  }

  if (axisSum != dstTy.getDimSize(axis))
    return emitOpError("sum of source sizes along axis (")
           << axisSum << ") does not match dst axis size ("
           << dstTy.getDimSize(axis) << ")";

  return success();
}
