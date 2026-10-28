// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file rewrite_fdiv_to_mul.cc
 * \brief Turn ``x / const`` into ``x * (1 / const)`` for float division.
 *
 * The legacy (v0.1.4) ``tir.Simplify`` performed this strength reduction, which
 * is visible in the golden corpus (``T.npuir_mul(acc, 0.000244140625)`` for
 * ``acc / 4096.0``).  The v0.1.15 ``tirx.Simplify`` keeps the division, which
 * then reaches the code generators as ``hivm.hir.vdiv``.
 *
 * The rewrite only fires when the reciprocal is exact (``c * (1/c) == 1``), so
 * it never changes the result of the division.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class FDivToMulRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    FDivToMulRewriter rewriter;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = rewriter(std::move(fptr->body));
    return f;
  }

private:
  PrimExpr VisitExpr_(const DivNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    if (op->dtype.is_float()) {
      if (const auto *fp = rhs.as<FloatImmNode>()) {
        double divisor = fp->value;
        if (divisor != 0.0) {
          double reciprocal = 1.0 / divisor;
          if (divisor * reciprocal == 1.0) {
            return Mul(lhs, FloatImm(op->dtype, reciprocal));
          }
        }
      }
    }
    if (lhs.same_as(op->a) && rhs.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    }
    return Div(lhs, rhs);
  }
};

PrimFunc TLRewriteFDivToMul(PrimFunc f) { return FDivToMulRewriter::Rewrite(std::move(f)); }

tirx::transform::Pass RewriteFDivToMul() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return TLRewriteFDivToMul(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RewriteFDivToMul", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.RewriteFDivToMul", RewriteFDivToMul);
}

} // namespace tl
} // namespace tvm