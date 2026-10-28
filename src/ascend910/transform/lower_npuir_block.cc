// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file lower_npuir_block.cc
 * \brief Strip residual TIR Block/BlockRealize shells for NPUIR codegen.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class LowerNpuirBlockMutator : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    LowerNpuirBlockMutator mutator;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    Stmt body = VisitStmt(op->block);
    PrimExpr predicate = VisitExpr(op->predicate);
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, std::move(body));
    }
    return body;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt body = VisitStmt(op->body);
    if (op->init.defined()) {
      ffi::Array<Stmt> stmts{VisitStmt(op->init.value()), body};
      body = SeqStmt::Flatten(stmts);
    }
    return body;
  }
};

namespace transform {

tvm::transform::Pass LowerNpuirBlock() {
  auto pass_func = [=](PrimFunc f, IRModule m,
                       tvm::transform::PassContext ctx) {
    return LowerNpuirBlockMutator::Substitute(std::move(f));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "tl.LowerNpuirBlock",
                                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.LowerNpuirBlock", LowerNpuirBlock);
}

} // namespace transform
} // namespace tl
} // namespace tvm
