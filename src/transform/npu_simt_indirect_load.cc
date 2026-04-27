/*
 * Copyright (c) Tile-AI Corporation.
 * Licensed under the MIT License.
 */

/*!
 * \file npu_simt_indirect_load.cc
 * \brief Lower the phase-1 A5 SIMT indirect-load pattern.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr const char *kFeature = "A5 SIMT indirect load phase 1";
constexpr int64_t kSupportedBlock = 256;

bool IsSharedScope(const Buffer &buffer) {
  return buffer.scope() == "shared" || buffer.scope() == "shared.dyn";
}

bool SameExpr(const PrimExpr &lhs, const PrimExpr &rhs) {
  return StructuralEqual()(lhs, rhs);
}

PrimExpr MakeRegion(Buffer buffer, PrimExpr offset, int access_mask,
                    PrimExpr extent) {
  Op region_op = Op::Get("tl.region");
  PrimExpr load = BufferLoad(buffer, {offset});
  Array<PrimExpr> args = {load, make_const(DataType::Int(32), access_mask),
                          extent};
  return Call(DataType::Handle(), region_op, args);
}

PrimExpr MatchTailMask(PrimExpr condition, const Var &loop_var) {
  const auto *lt = condition.as<LTNode>();
  if (!lt) {
    return PrimExpr();
  }
  if (!SameExpr(lt->a, loop_var)) {
    return PrimExpr();
  }
  return lt->b;
}

class NpuSimtIndirectLoadRewriter : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kParallel) {
      return StmtMutator::VisitStmt_(op);
    }

    Optional<Stmt> rewritten = TryRewrite(op);
    if (rewritten.defined()) {
      return rewritten.value();
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  Optional<Stmt> TryRewrite(const ForNode *op) {
    const auto *if_node = op->body.as<IfThenElseNode>();
    if (!if_node || if_node->else_case.defined()) {
      return NullOpt;
    }

    PrimExpr valid_extent = MatchTailMask(if_node->condition, op->loop_var);
    if (!valid_extent.defined()) {
      return NullOpt;
    }

    const auto *store = if_node->then_case.as<BufferStoreNode>();
    if (!store) {
      return NullOpt;
    }
    if (store->indices.size() != 1U) {
      return NullOpt;
    }
    if (!SameExpr(store->indices[0], op->loop_var)) {
      return NullOpt;
    }

    const auto *src_load = store->value.as<BufferLoadNode>();
    if (!src_load || src_load->indices.size() != 1U) {
      return NullOpt;
    }
    const auto *idx_load = src_load->indices[0].as<BufferLoadNode>();
    if (!idx_load || idx_load->indices.size() != 1U) {
      return NullOpt;
    }
    if (!SameExpr(idx_load->indices[0], op->loop_var)) {
      return NullOpt;
    }

    ValidateBoundary(op, store, src_load, idx_load, valid_extent);
    return BuildIndirectLoad(op, store, src_load, idx_load, valid_extent);
  }

  void ValidateBoundary(const ForNode *op, const BufferStoreNode *store,
                        const BufferLoadNode *src_load,
                        const BufferLoadNode *idx_load,
                        const PrimExpr &valid_extent) {
    const auto *extent = op->extent.as<IntImmNode>();
    ICHECK(extent) << kFeature
                   << ": T.Parallel extent must be a compile-time constant";
    ICHECK_EQ(extent->value, kSupportedBlock)
        << kFeature << ": only BLOCK=256 is supported in phase 1, got "
        << extent->value;
    ICHECK(is_zero(op->min))
        << kFeature << ": expected T.Parallel loop min to be 0, got "
        << op->min;
    ICHECK(valid_extent.dtype().is_int() || valid_extent.dtype().is_uint())
        << kFeature << ": valid extent must be integer typed, got "
        << valid_extent.dtype();

    ICHECK(src_load->buffer.scope() == "global")
        << kFeature << ": expected src X buffer in global scope, got "
        << src_load->buffer.scope();
    ICHECK(IsSharedScope(idx_load->buffer))
        << kFeature
        << ": expected IDX_UB in shared/shared.dyn scope. Stage indices from "
           "GM with T.copy before the T.Parallel indirect load.";
    ICHECK(IsSharedScope(store->buffer))
        << kFeature
        << ": expected O_UB in shared/shared.dyn scope. Write the indirect "
           "load result to alloc_shared, then use T.copy to store it to GM.";

    ICHECK(src_load->buffer->dtype == DataType::Float(32))
        << kFeature << ": expected src X dtype float32, got "
        << src_load->buffer->dtype;
    ICHECK(idx_load->buffer->dtype == DataType::Int(32))
        << kFeature << ": expected IDX_UB dtype int32, got "
        << idx_load->buffer->dtype;
    ICHECK(store->buffer->dtype == src_load->buffer->dtype)
        << kFeature << ": expected O_UB dtype to match src dtype "
        << src_load->buffer->dtype << ", got " << store->buffer->dtype;
  }

  Stmt BuildIndirectLoad(const ForNode *op, const BufferStoreNode *store,
                         const BufferLoadNode *src_load,
                         const BufferLoadNode *idx_load,
                         const PrimExpr &valid_extent) {
    PrimExpr zero = make_zero(op->loop_var.dtype());
    PrimExpr block = op->extent;

    PrimExpr src_region = MakeRegion(src_load->buffer, zero, 1,
                                     make_const(DataType::Int(32), 1));
    PrimExpr idx_region = MakeRegion(idx_load->buffer, zero, 1, block);
    PrimExpr dst_region = MakeRegion(store->buffer, zero, 2, block);

    Op indirect_load_op = Op::Get("tl.npuir_indirect_load");
    Array<PrimExpr> args = {src_region, idx_region, dst_region, valid_extent};
    PrimExpr call = Call(DataType::Void(), indirect_load_op, args);
    return Evaluate(call);
  }
};

}  // namespace

using namespace tir::transform;

tvm::transform::Pass NpuSimtIndirectLoad() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto *new_pf = f.CopyOnWrite();
    new_pf->body = NpuSimtIndirectLoadRewriter()(std::move(new_pf->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NpuSimtIndirectLoad", {});
}

TVM_REGISTER_GLOBAL("tl.transform.NpuSimtIndirectLoad")
    .set_body_typed(NpuSimtIndirectLoad);

}  // namespace tl
}  // namespace tvm
