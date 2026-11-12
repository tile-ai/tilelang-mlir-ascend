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
