// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_TARGET_NPUIR_FIXPIPE_COMPAT_H_
#define TVM_TL_TARGET_NPUIR_FIXPIPE_COMPAT_H_

#include <bishengir/Dialect/HIVM/IR/HIVM.h>
#include <mlir/IR/Builders.h>
#include <type_traits>
#include <utility>

namespace tvm {
namespace codegen {
namespace detail {

// The pinned IR uses enable_nz2nd; CANN 9.0.0 IR uses dma_mode.
// Detect the actual API instead of relying on a release version macro.
template <typename Op, typename = void> struct FixpipeBuilder {
  static void Create(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::TypeRange results, mlir::Value src, mlir::Value dst,
                     mlir::UnitAttr nz2nd,
                     mlir::hivm::FixpipePreQuantModeAttr quant,
                     mlir::hivm::FixpipePreReluModeAttr relu,
                     mlir::BoolAttr channelSplit) {
    builder.create<Op>(loc, results, src, dst, nz2nd, quant, relu,
                       channelSplit);
  }
};

template <typename Op>
struct FixpipeBuilder<
    Op, std::void_t<decltype(std::declval<Op>().getDmaModeAttr())>> {
  static void Create(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::TypeRange results, mlir::Value src, mlir::Value dst,
                     mlir::UnitAttr nz2nd,
                     mlir::hivm::FixpipePreQuantModeAttr quant,
                     mlir::hivm::FixpipePreReluModeAttr relu,
                     mlir::BoolAttr channelSplit) {
    using DmaAttr = decltype(std::declval<Op>().getDmaModeAttr());
    using DmaMode = decltype(std::declval<Op>().getDmaMode());
    using DualDstAttr = decltype(std::declval<Op>().getDualDstModeAttr());
    auto dma = DmaAttr::get(builder.getContext(),
                            nz2nd ? DmaMode::NZ2ND : DmaMode::NZ2NZ);
    builder.create<Op>(loc, results, src, dst, dma, DualDstAttr{}, quant, relu,
                       channelSplit);
  }
};

} // namespace detail

inline void CreateFixpipeCompat(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::TypeRange results, mlir::Value src,
                                mlir::Value dst, mlir::UnitAttr nz2nd,
                                mlir::hivm::FixpipePreQuantModeAttr quant,
                                mlir::hivm::FixpipePreReluModeAttr relu,
                                mlir::BoolAttr channelSplit) {
  detail::FixpipeBuilder<mlir::hivm::FixpipeOp>::Create(
      builder, loc, results, src, dst, nz2nd, quant, relu, channelSplit);
}

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_NPUIR_FIXPIPE_COMPAT_H_
