// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file legalize_npuir_bf16.cc
 * \brief Legalize BF16 tl.npuir_add into fp32 compute plus casts.
 */

#include <tvm/node/structural_equal.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../op/op.h"
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

class LegalizeNpuirBF16Mutator : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LegalizeNpuirBF16Mutator mutator(&analyzer);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return f;
  }

private:
  using IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  struct PreparedRegion {
    PrimExpr region;
    Array<Stmt> prefix;
    Buffer temp_fp32;
    bool needs_cast_back = false;
  };

  struct CachedSourceCast {
    Buffer src_buffer;
    Array<Range> ranges;
    String scope;
    Buffer temp_f32;
  };

  // Map: real_bf16_buffer -> temp_fp32_buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      bf16_to_fp32_alias_;

  std::vector<std::vector<Buffer>> block_temp_buffers_;
  std::vector<std::vector<CachedSourceCast>> block_source_cast_cache_;

  bool IsBF16(const Buffer &buffer) const {
    return buffer.defined() && buffer->dtype == DataType::BFloat(16);
  }

  bool IsBinaryOp(const CallNode *call) const {
    if (call == nullptr) {
      return false;
    }
    std::vector<std::string> binary_ops = {
        "tl.npuir_add", "tl.npuir_mul", "tl.npuir_max", "tl.npuir_min",
        "tl.npuir_div", "tl.npuir_sub", "tl.npuir_cmp"};
    return std::any_of(binary_ops.begin(), binary_ops.end(),
                       [&](const std::string &op_name) {
                         return call->op.same_as(Op::Get(op_name));
                       });
  }

  bool IsUnaryOp(const CallNode *call) const {
    if (call == nullptr) {
      return false;
    }
    std::vector<std::string> unary_ops = {
        "tl.npuir_exp", "tl.npuir_relu", "tl.npuir_sigmoid",
        "tl.npuir_ln",  "tl.npuir_sqrt", "tl.npuir_rsqrt",
        "tl.npuir_abs", "tl.npuir_rec",  "tl.npuir_reduce"};
    return std::any_of(unary_ops.begin(), unary_ops.end(),
                       [&](const std::string &op_name) {
                         return call->op.same_as(Op::Get(op_name));
                       });
  }

  Buffer CreateTempBuffer(const Array<Range> &region, DataType dtype,
                          const String &scope, const std::string &prefix) {
    Array<PrimExpr> shape;
    for (const Range &r : region) {
      shape.push_back(r->extent);
    }

    std::string name = prefix + "_" + std::to_string(temp_buffer_id_++);
    Buffer buf(Var(name, PointerType(PrimType(dtype), scope)), dtype, shape, {},
               PrimExpr(0), name, 0, 0, kDefault);
    ICHECK(!block_temp_buffers_.empty())
        << "BF16 legalization expects to create temporaries inside a Block.";
    block_temp_buffers_.back().push_back(buf);
    return buf;
  }

  PrimExpr CreateRegion(const Buffer &buffer, const Array<Range> &region,
                        int access_mask) const {
    Array<PrimExpr> indices;
    Array<PrimExpr> args;

    for (const Range &r : region) {
      indices.push_back(make_zero(r->min.dtype()));
    }

    args.push_back(BufferLoad(buffer, indices));
    args.push_back(make_const(DataType::Int(32), access_mask));
    for (const Range &r : region) {
      args.push_back(r->extent);
    }
    return Call(DataType::Handle(), Op::Get("tl.region"), args);
  }

  Stmt CreateCastStmt(const PrimExpr &src_region,
                      const PrimExpr &dst_region) const {
    Array<PrimExpr> args{src_region, dst_region, StringImm("rint")};
    return Evaluate(Call(DataType::Void(), Op::Get("tl.npuir_cast"), args));
  }

  PreparedRegion PrepareSourceRegion(const RegionOp &src, const String &scope) {
    const Buffer &src_buf = src.GetBuffer();
    // If this buffer is currently aliased to a temp FP32 buffer, use it
    // directly
    auto it = bf16_to_fp32_alias_.find(src_buf);
    if (it != bf16_to_fp32_alias_.end()) {
      PreparedRegion prepared{
          CreateRegion(it->second, src.GetRanges(), /*access_mask=*/1), {}};
      return prepared;
    }

    PreparedRegion prepared{
        CreateRegion(src.GetBuffer(), src.GetRanges(), /*access_mask=*/1), {}};
    if (!IsBF16(src.GetBuffer())) {
      return prepared;
    }

    ICHECK(!block_source_cast_cache_.empty())
        << "BF16 legalization source-cache expects to run inside a Block.";

    for (const CachedSourceCast &cached : block_source_cast_cache_.back()) {
      if (cached.src_buffer.same_as(src.GetBuffer()) && cached.scope == scope &&
          StructuralEqual()(cached.ranges, src.GetRanges())) {
        prepared.region =
            CreateRegion(cached.temp_f32, src.GetRanges(), /*access_mask=*/1);
        return prepared;
      }
    }

    Buffer temp = CreateTempBuffer(src.GetRanges(), DataType::Float(32), scope,
                                   "bf16_src_f32");
    PrimExpr temp_dst_region =
        CreateRegion(temp, src.GetRanges(), /*access_mask=*/2);
    prepared.prefix.push_back(CreateCastStmt(prepared.region, temp_dst_region));
    prepared.region = CreateRegion(temp, src.GetRanges(), /*access_mask=*/1);
    block_source_cast_cache_.back().push_back(
        CachedSourceCast{src.GetBuffer(), src.GetRanges(), scope, temp});
    return prepared;
  }

  PreparedRegion PrepareDestinationRegion(const RegionOp &dst) {
    const Buffer &dst_buf = dst.GetBuffer();
    if (!IsBF16(dst.GetBuffer())) {
      return {CreateRegion(dst_buf, dst.GetRanges(), /*access_mask=*/2),
              {},
              Buffer(),
              false};
    }
    // Reuse temp_fp32 if already exists (in-place update)
    auto it = bf16_to_fp32_alias_.find(dst_buf);
    if (it != bf16_to_fp32_alias_.end()) {
      // No need to create a new temp, and no cast-back needed
      return {CreateRegion(it->second, dst.GetRanges(), /*access_mask=*/2),
              {},
              it->second,
              true};
    }
    // Otherwise, allocate a new temp FP32 buffer for the destination, and cast
    // back after the op
    Buffer temp = CreateTempBuffer(dst.GetRanges(), DataType::Float(32),
                                   dst.GetBuffer().scope(), "bf16_dst_f32");
    bf16_to_fp32_alias_[dst_buf] = temp;
    return {
        CreateRegion(temp, dst.GetRanges(), /*access_mask=*/2), {}, temp, true};
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    block_temp_buffers_.push_back({});
    block_source_cast_cache_.push_back({});

    Block new_block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));

    Array<Buffer> new_allocs = new_block->alloc_buffers;
    for (const Buffer &buf : block_temp_buffers_.back()) {
      new_allocs.push_back(buf);
    }
    block_temp_buffers_.pop_back();
    block_source_cast_cache_.pop_back();

    if (new_allocs.same_as(new_block->alloc_buffers)) {
      return new_block;
    }
    bf16_to_fp32_alias_.clear();
    new_block.CopyOnWrite()->alloc_buffers = std::move(new_allocs);
    return new_block;
  }

  Stmt processBinaryOp(const PrimExpr &new_value) {
    const auto *call = new_value.as<CallNode>();
    const auto *src0_call = call->args[0].as<CallNode>();
    const auto *src1_call = call->args[1].as<CallNode>();
    const auto *dst_call = call->args[2].as<CallNode>();
    // Consider the vector-scalar case
    if (src0_call == nullptr || dst_call == nullptr ||
        !src0_call->op.same_as(Op::Get("tl.region")) ||
        !dst_call->op.same_as(Op::Get("tl.region"))) {
      return Evaluate(new_value);
    }

    RegionOp src0_region(src0_call->args, BufferMap{});
    bool src1_not_bf16 =
        (src1_call == nullptr) || !src1_call->op.same_as(Op::Get("tl.region"));
    if (!src1_not_bf16) {
      RegionOp src1_region(src1_call->args, BufferMap{});
      src1_not_bf16 = !IsBF16(src1_region.GetBuffer());
    }
    RegionOp dst_region(dst_call->args, BufferMap{});
    if (!IsBF16(src0_region.GetBuffer()) && src1_not_bf16 &&
        !IsBF16(dst_region.GetBuffer())) {
      return Evaluate(new_value);
    }

    String compute_scope = dst_region.GetBuffer().scope();
    PreparedRegion src0 = PrepareSourceRegion(src0_region, compute_scope);

    PreparedRegion dst = PrepareDestinationRegion(dst_region);

    Array<Stmt> seq;
    for (const Stmt &stmt : src0.prefix) {
      seq.push_back(stmt);
    }

    PrimExpr src1_expr = call->args[1];
    // no need to process src1 if it's scalar or not bf16
    if (!src1_not_bf16) {
      RegionOp src1_region(src1_call->args, BufferMap{});
      PreparedRegion src1 = PrepareSourceRegion(src1_region, compute_scope);
      for (const Stmt &stmt : src1.prefix) {
        seq.push_back(stmt);
      }
      src1_expr = src1.region;
    }

    Array<PrimExpr> op_args{src0.region, src1_expr, dst.region};
    if (call->args.size() > 3) {
      // Pass through additional arguments (e.g., for cmp)
      for (size_t i = 3; i < call->args.size(); ++i) {
        op_args.push_back(call->args[i]);
      }
    }
    seq.push_back(Evaluate(Call(DataType::Void(), call->op, op_args)));
    for (const Stmt &stmt : dst.prefix) {
      seq.push_back(stmt);
    }

    // Insert cast-back for the destination if needed
    if (dst.needs_cast_back) {
      PrimExpr temp_src_region =
          CreateRegion(dst.temp_fp32, dst_region.GetRanges(),
                       /*access_mask=*/1);
      PrimExpr real_dst_region = CreateRegion(
          dst_region.GetBuffer(), dst_region.GetRanges(), /*access_mask=*/2);
      seq.push_back(CreateCastStmt(temp_src_region, real_dst_region));
    }
    return SeqStmt::Flatten(seq);
  }

  Stmt processUnaryOp(const PrimExpr &new_value) {
    const auto *call = new_value.as<CallNode>();
    const auto *src_call = call->args[0].as<CallNode>();
    const auto *dst_call = call->args[1].as<CallNode>();
    if (src_call == nullptr || dst_call == nullptr ||
        !src_call->op.same_as(Op::Get("tl.region")) ||
        !dst_call->op.same_as(Op::Get("tl.region"))) {
      return Evaluate(new_value);
    }

    RegionOp src_region(src_call->args, BufferMap{});
    RegionOp dst_region(dst_call->args, BufferMap{});
    if (!IsBF16(src_region.GetBuffer()) && !IsBF16(dst_region.GetBuffer())) {
      return Evaluate(new_value);
    }

    String compute_scope = dst_region.GetBuffer().scope();
    PreparedRegion src = PrepareSourceRegion(src_region, compute_scope);
    PreparedRegion dst = PrepareDestinationRegion(dst_region);

    Array<Stmt> seq;
    for (const Stmt &stmt : src.prefix) {
      seq.push_back(stmt);
    }

    Array<PrimExpr> op_args{src.region, dst.region};
    if (call->args.size() > 2) {
      // Pass through additional arguments (e.g., for reduce)
      for (size_t i = 2; i < call->args.size(); ++i) {
        op_args.push_back(call->args[i]);
      }
    }
    seq.push_back(Evaluate(Call(DataType::Void(), call->op, op_args)));
    for (const Stmt &stmt : dst.prefix) {
      seq.push_back(stmt);
    }

    // Insert cast-back for the destination if needed
    if (dst.needs_cast_back) {
      PrimExpr temp_src_region =
          CreateRegion(dst.temp_fp32, dst_region.GetRanges(),
                       /*access_mask=*/1);
      PrimExpr real_dst_region = CreateRegion(
          dst_region.GetBuffer(), dst_region.GetRanges(), /*access_mask=*/2);
      seq.push_back(CreateCastStmt(temp_src_region, real_dst_region));
    }
    return SeqStmt::Flatten(seq);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    PrimExpr new_value = this->VisitExpr(op->value);
    const auto *call = new_value.as<CallNode>();
    if ((!IsBinaryOp(call) && !IsUnaryOp(call)) ||
        block_temp_buffers_.empty()) {
      if (new_value.same_as(op->value)) {
        return GetRef<Stmt>(op);
      }
      return Evaluate(new_value);
    }
    if (IsBinaryOp(call)) {
      return processBinaryOp(new_value);
    }

    return processUnaryOp(new_value);
  }

  int temp_buffer_id_{0};
};

namespace transform {

tvm::transform::Pass LegalizeNpuirBF16() {
  auto pass_func = [=](PrimFunc f, IRModule m,
                       tvm::transform::PassContext ctx) {
    return LegalizeNpuirBF16Mutator::Substitute(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.LegalizeNpuirBF16", {});
}

TVM_REGISTER_GLOBAL("tl.transform.LegalizeNpuirBF16")
    .set_body_typed(LegalizeNpuirBF16);

} // namespace transform
} // namespace tl
} // namespace tvm
