// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
/*!
 * \file target/tir_to_tlir.h
 * \brief TIR -> TLIR import pass (v1: add-kernel slice only).
 *
 * Mirrors the shape of CodeGenTileLangNPUIRAPI (codegen_npuir_api.cc/.h):
 * an ExprFunctor/StmtFunctor visitor over a PrimFunc, driven by AddFunction,
 * finished with Finish(). Registered as a TVM build target the same way
 * (see rt_mod_npuir.cc), so it plugs into TVM's existing dispatch rather
 * than bypassing it.
 *
 * v1 scope (deliberately restricted, matching the FYP proposal's "restricted
 * dialect" framing):
 *   - Emits tl.alloc / tl.copy / tl.add only.
 *   - No ABI padding (FFTS addr, sync-lock/workspace args, grid info) --
 *     those are A5/hacc runtime concerns orthogonal to demonstrating the
 *     TIR -> TLIR translation itself.
 *   - No core-type MIX splitting.
 *   - Only whole-buffer region accesses (offset 0, full shape) -- matches
 *     tl.add's existing no-broadcast verifier constraint.
 */
#ifndef TVM_TL_TARGET_TIR_TO_TLIR_H_
#define TVM_TL_TARGET_TIR_TO_TLIR_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "tilelangir/Dialect/TL/TLIROps.h"

#include <tvm/ir/module.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace codegen {

class CodeGenTIRToTLIR final : public tir::ExprFunctor<void(const PrimExpr &)>,
                                public tir::StmtFunctor<void(const tir::Stmt &)> {
public:
  CodeGenTIRToTLIR();

  void AddFunction(const GlobalVar &gvar, const tir::PrimFunc &f);
  std::string Finish();

  void VisitStmt_(const tir::AllocateNode *op) final;
  void VisitStmt_(const tir::EvaluateNode *op) final;
  void VisitStmt_(const tir::SeqStmtNode *op) final;
  void VisitStmt_(const tir::DeclBufferNode *op) final;
  void VisitStmt_(const tir::AttrStmtNode *op) final;

  // Only Call is meaningful for the v1 add-kernel slice; other expr kinds
  // are not visited standalone (they only ever appear nested inside a Call
  // we already special-case) so no VisitExpr_ overloads are declared for
  // them -- calling VisitExpr on an unsupported node hits the default
  // ExprFunctor error, which is the desired "documented unsupported path"
  // behaviour per the proposal's fallback policy (section 5.3).
  void VisitExpr_(const tir::CallNode *op) final;

private:
  // Resolves a `T.region(BufferLoad(buf), rank, extent0, extent1, ...)`
  // call argument to the tl.alloc'd (or function-arg) mlir::Value for
  // `buf`, and ICHECKs that the region covers the whole buffer (v1's
  // whole-buffer-only restriction).
  mlir::Value ResolveWholeBufferRegionArg(const PrimExpr &arg);

  mlir::MLIRContext context;
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
  std::unordered_map<const Object *, mlir::Value> var_map_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_TIR_TO_TLIR_H_
