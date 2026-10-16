// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Dialect/TL/TLDialect.h"
#include "tilelangir/Dialect/TL/TLOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tl;

#include "tilelangir/Dialect/TL/TLEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tilelangir/Dialect/TL/TLAttrs.cpp.inc"

#include "tilelangir/Dialect/TL/TLDialect.cpp.inc"

void TLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tilelangir/Dialect/TL/TLOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tilelangir/Dialect/TL/TLAttrs.cpp.inc"
      >();
}
