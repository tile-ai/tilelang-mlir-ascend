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

LogicalResult SortOp::verify() {
  auto src = llvm::cast<MemRefType>(getSrc().getType());
  auto dstValue = llvm::cast<MemRefType>(getDstValue().getType());
  auto dstIndex = llvm::cast<MemRefType>(getDstIndex().getType());

  if (src.getElementType() != dstValue.getElementType())
    return emitOpError("src and dst_value element types must match");
  if (src.getShape() != dstValue.getShape())
    return emitOpError("src and dst_value shapes must match");
  if (src.getShape() != dstIndex.getShape())
    return emitOpError("src and dst_index shapes must match");
  if (!dstIndex.getElementType().isSignlessInteger() &&
      !dstIndex.getElementType().isSignedInteger() &&
      !dstIndex.getElementType().isUnsignedInteger())
    return emitOpError("dst_index must have an integer element type");

  // HIVM's vsort restricts src/dst_value to f16 or f32 (see
  // OperElemTypeConstraints on hivm::VSortOp). Enforce the same restriction
  // here so invalid element types are rejected at the tl.sort level with a
  // clear message, rather than surfacing later as a confusing error on the
  // lowered hivm.hir.vsort op.
  if (!src.getElementType().isF16() && !src.getElementType().isF32())
    return emitOpError("src and dst_value element type must be f16 or f32");

  // HIVM's vsort currently only supports tail-axis sorting: sort_axis must
  // be exactly rank - 1, or -1 (which also means tail axis). See
  // VSortOp::verify() in HIVMVectorOps.cpp. Any other axis value verifies
  // fine here but would be rejected downstream by HIVM with a less legible
  // error, so reject it at this level instead.
  int64_t rank = src.getRank();
  int64_t axis = getSortAxis();
  if (axis != -1 && axis != rank - 1)
    return emitOpError("sort_axis must be the tail axis (rank - 1) or -1; "
                       "only tail-axis sorting is currently supported")
           << ", got " << axis << " for rank " << rank;
      return success();
   }

LogicalResult CastOp::verify() {
  auto src = llvm::cast<MemRefType>(getSrc().getType());
  auto dst = llvm::cast<MemRefType>(getDst().getType());
  if (src.getShape() != dst.getShape())
    return emitOpError("src/dst shapes must match");
  // Element types are intentionally allowed to differ: that is the point
  // of a cast. Legal src/dst dtype pairs are constrained by the HIVM vcast
  // hardware op, not here.
  llvm::StringRef mode = getRoundMode();
  static constexpr llvm::StringRef kValidModes[] = {"round", "rint",  "floor",
                                                    "ceil",  "trunc", "odd"};
  if (!llvm::is_contained(kValidModes, mode))
    return emitOpError("round_mode must be one of round/rint/floor/ceil/"
                       "trunc/odd, got '")
           << mode << "'";
  return success();
}
LogicalResult ShrOp::verify() {
  auto lhs = llvm::cast<MemRefType>(getLhs().getType());
  auto rhs = llvm::cast<MemRefType>(getRhs().getType());
  auto out = llvm::cast<MemRefType>(getOut().getType());

  // AllTypesMatch<["lhs", "rhs", "out"]> already guarantees identical
  // shape and element type across all three operands, so only the
  // element-type restriction needs checking here.
  Type elemTy = lhs.getElementType();
  unsigned width = elemTy.isInteger() ? elemTy.getIntOrFloatBitWidth() : 0;
  if (!elemTy.isSignlessInteger() || (width != 16 && width != 32 && width != 64))
    return emitOpError("operand element type must be i16, i32, or i64, "
                       "matching HIVM vshr's integer-only constraint");

  // rhs is modeled as a tensor operand here, not scalar — a known mismatch
  // with HIVM vshr's ScalarOnlyHWTrait<1>. Conversion to HIVM is only
  // hardware-correct when rhs is, in practice, a single-element buffer;
  // that is not checked here and is tracked as follow-up work.
  (void)rhs;
  (void)out;

  return success();
}

LogicalResult ShlOp::verify() {
  auto lhs = llvm::cast<MemRefType>(getLhs().getType());
  auto out = llvm::cast<MemRefType>(getOut().getType());
  if (lhs.getElementType() != out.getElementType())
    return emitOpError("lhs/out element types must match");
  if (lhs.getShape() != out.getShape())
    return emitOpError("lhs/out shapes must match");
  // rhs is intentionally unconstrained here (scalar shift amount).
  return success();
}
LogicalResult PrintOp::verify() {
  Type ty = getSrc().getType();
  if (auto memrefTy = llvm::dyn_cast<MemRefType>(ty)) {
    if (!memrefTy.getElementType().isIntOrFloat())
      return emitOpError(
          "buffer operand element type must be integer or float");
    return success();
  }
  if (!ty.isIntOrFloat())
    return emitOpError("scalar operand must be integer or float");
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

static LogicalResult verifyElementwiseUnary(Operation *op, Value srcV,
                                            Value outV) {
  auto src = llvm::cast<MemRefType>(srcV.getType());
  auto out = llvm::cast<MemRefType>(outV.getType());
  if (src.getElementType() != out.getElementType())
    return op->emitOpError("operand element types must match");
  if (src.getShape() != out.getShape())
    return op->emitOpError("operand shapes must match");
  return success();
}

LogicalResult AbsOp::verify() {
  return verifyElementwiseUnary(getOperation(), getSrc(), getOut());
}

LogicalResult NotOp::verify() {
  return verifyElementwiseUnary(getOperation(), getSrc(), getOut());
}

LogicalResult FlipOp::verify() {
  auto src = llvm::cast<MemRefType>(getSrc().getType());
  auto dst = llvm::cast<MemRefType>(getDst().getType());
  if (src.getElementType() != dst.getElementType())
    return emitOpError("src and dst element types must match");
  if (src.getShape() != dst.getShape())
    return emitOpError("src and dst shapes must match");
  if (static_cast<int64_t>(getAxis()) >= src.getRank())
    return emitOpError("axis out of range for operand rank");
  return success();
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
LogicalResult ArangeOp::verify() {
  auto dst = llvm::cast<MemRefType>(getDst().getType());
  if (static_cast<int64_t>(getStrides().size()) != dst.getRank())
    return emitOpError("number of strides (")
           << getStrides().size() << ") must match destination rank ("
           << dst.getRank() << ")";
  return success();
}
LogicalResult ReshapeOp::verify() {
  auto src = llvm::cast<MemRefType>(getSrc().getType());
  auto result = llvm::cast<MemRefType>(getResult().getType());
  if (src.getElementType() != result.getElementType())
    return emitOpError("src/result element types must match");
  if (!src.hasStaticShape() || !result.hasStaticShape())
    return emitOpError("reshape requires statically shaped src/result");
  if (src.getNumElements() != result.getNumElements())
    return emitOpError("src/result must have the same number of elements");
  return success();
}
LogicalResult PowOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}

LogicalResult XorOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult OrOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}
LogicalResult AndOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(), getOut());
}

static LogicalResult verifyElementwiseUnary(Operation *op, Value srcV,
                                            Value outV) {
  auto src = llvm::cast<MemRefType>(srcV.getType());
  auto out = llvm::cast<MemRefType>(outV.getType());
  if (src.getElementType() != out.getElementType())
    return op->emitOpError("operand element types must match");
  if (src.getShape() != out.getShape())
    return op->emitOpError("operand shapes must match");
  return success();
}

LogicalResult SqrtOp::verify() {
  return verifyElementwiseUnary(getOperation(), getSrc(), getOut());
}

static LogicalResult verifyElementwiseUnary(Operation *op, Value srcV,
                                            Value outV) {
  auto src = llvm::cast<MemRefType>(srcV.getType());
  auto out = llvm::cast<MemRefType>(outV.getType());
  if (src.getElementType() != out.getElementType())
    return op->emitOpError("operand element types must match");
  if (src.getShape() != out.getShape())
    return op->emitOpError("operand shapes must match");
  return success();
}

LogicalResult RsqrtOp::verify() {
  return verifyElementwiseUnary(getOperation(), getSrc(), getOut());
}
