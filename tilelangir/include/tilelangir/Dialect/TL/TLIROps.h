// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
/*!
 * \file tilelangir/include/tilelangir/Dialect/TL/TLIROps.h
 * \brief TLIR dialect: types, attrs, ops (TableGen declarations).
 *
 */
#ifndef TILELANGIR_DIALECT_TL_TLIROPS_H
#define TILELANGIR_DIALECT_TL_TLIROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "tilelangir/Dialect/TL/TLIRDialect.h.inc"

#include "tilelangir/Dialect/TL/TLIRAttrs.enum.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tilelangir/Dialect/TL/TLIRAttrs.h.inc"

#define GET_OP_CLASSES
#include "tilelangir/Dialect/TL/TLIROps.h.inc"

#endif // TILELANGIR_DIALECT_TL_TLIROPS_H
