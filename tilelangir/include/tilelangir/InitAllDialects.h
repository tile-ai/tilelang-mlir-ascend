// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/include/tilelangir/InitAllDialects.h
 * \brief TileLangIR dialect registration (registerAllDialects).
 *
 */

#ifndef TILELANGIR_INITALLDIALECTS_H
#define TILELANGIR_INITALLDIALECTS_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "tilelangir/Dialect/TL/TLIROps.h"

namespace tilelangir {

/// Register all TileLangIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::tlir::TLIRDialect>();
}

/// Append all TileLangIR dialects to the registry contained in the given
/// context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  tilelangir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace tilelangir

#endif // TILELANGIR_INITALLDIALECTS_H
