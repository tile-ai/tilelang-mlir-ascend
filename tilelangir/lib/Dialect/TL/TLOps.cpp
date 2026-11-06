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
