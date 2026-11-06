// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TILELANGIR_DIALECT_TL_TLOPS_H
#define TILELANGIR_DIALECT_TL_TLOPS_H

#include "tilelangir/Dialect/TL/TLDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "tilelangir/Dialect/TL/TLOps.h.inc"

#endif // TILELANGIR_DIALECT_TL_TLOPS_H
