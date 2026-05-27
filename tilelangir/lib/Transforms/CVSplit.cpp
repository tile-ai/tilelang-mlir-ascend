// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/CVSplit.cpp
 * \brief TileLangIR Cube/Vector 拆分 pass。
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "tilelangir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

namespace mlir::tilelangir {

#define GEN_PASS_DEF_TILELANGIRCVSPLIT
#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-cv-split"
#define LDBG(X)                                                                \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << X << '\n')

struct TileLangIRCVSplit : impl::TileLangIRCVSplitBase<TileLangIRCVSplit> {

  struct StageGroup {
    hivm::TCoreType coreType = hivm::TCoreType::CUBE_OR_VECTOR;
    SmallVector<Operation *> localAllocs;
    SmallVector<Operation *> ops;
  };

  void runOnOperation() override {
    SmallVector<scf::ForOp> pipelineLoops;
    getOperation().walk([&](scf::ForOp forOp) {
      if (isPipelineFor(forOp))
        pipelineLoops.push_back(forOp);
    });

    for (auto forOp : pipelineLoops) {
      LDBG("Processing pipeline loop " << forOp);
      currentForOp = forOp;

      DenseMap<Operation *, std::size_t> cubeGroupOfOp;
      SmallVector<StageGroup> cubeGroups = collectCubeGroups(cubeGroupOfOp);
      SmallVector<StageGroup> orderedGroups =
          collectVectorGroupsByOrder(cubeGroups, cubeGroupOfOp);

      assignLocalAllocsToGroups(orderedGroups);
      packGroups(orderedGroups);
    }
  }

private:
  scf::ForOp currentForOp;

  // 当前 CVSplit 只处理 T.pipeline lowering 后带 num_stages 标记的循环。
  // 普通 scf.for 可能只是 gather、mask 或标量控制流，不能被当成 C/V 分割边界。
  bool isPipelineFor(scf::ForOp forOp) {
    return forOp->hasAttr("tilelangir.num_stages");
  }

  // 临时策略：Cube 阶段只以 GEMM/mmadL1 为锚点。workspace copy 是阶段边界，
  // 不是阶段根节点；继续以 copy 为 seed 会把真实计算拆成大量 copy-only scope。
  bool isCubeAnchor(Operation *op) {
    auto name = op->getName().getStringRef();
    return name == "hivm.hir.mmadL1" || name.contains("mmad");
  }

  // 声明和元数据不是实际的 C/V 执行语句，先不作为计算 op 参与分组。
  // 后续 assignLocalAllocsToGroups 会把只属于单个 scope 的普通 alloc 下沉，
  // 其余跨 scope / multi-buffer alloc 继续留在父 loop 中供多个 scope 捕获。
  // 其中 annotation.mark 只描述 multi_buffer 等属性，若误放入 VECTOR scope，
  // InferMemScope 会把被标记的 local buffer 当作 UB 使用，和真实 CUBE GEMM
  // 的 L1 约束冲突。
  bool shouldKeepOutsideScope(Operation *op) {
    return isa<memref::AllocOp, bishengir::memref_ext::AllocWorkspaceOp,
               annotation::MarkOp>(op);
  }

  bool containsOp(const StageGroup &group, Operation *op) {
    return llvm::any_of(group.ops,
                        [&](Operation *groupOp) { return groupOp == op; });
  }

  void addOpToGroup(StageGroup &group, Operation *op) {
    if (!op || shouldKeepOutsideScope(op) || containsOp(group, op))
      return;
    if (getTopLevelOpInCurrentFor(op) != op)
      return;
    group.ops.push_back(op);
  }

  // copy 的 source/target 可能不是 workspace/GM 本身，而是它们上面的
  // subview/reinterpret_cast/collapse_shape 等 view。
  // 这些 view producer 和 copy 是同一个 C/V 或 GM->Cube 边界的一部分，必须一起进 scope；
  // 否则 copy 被搬入 scope 后会使用仍留在父 region、甚至位于 scope 之后的 view。
  void addViewProducerChainToGroup(StageGroup &group, Value value) {
    llvm::SmallPtrSet<Operation *, 8> visited;
    addViewProducerChainToGroup(group, value, visited);
  }

  void addViewProducerChainToGroup(
      StageGroup &group, Value value,
      llvm::SmallPtrSetImpl<Operation *> &visited) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !visited.insert(definingOp).second)
      return;

    auto viewOp = dyn_cast<ViewLikeOpInterface>(definingOp);
    if (!viewOp)
      return;

    addViewProducerChainToGroup(group, viewOp.getViewSource(), visited);
    addOpToGroup(group, definingOp);
  }

  void addBoundaryCopyToGroup(StageGroup &group, Operation *op) {
    if (!op)
      return;

    auto copyOp = dyn_cast<CopyOpInterface>(op);
    if (copyOp) {
      addViewProducerChainToGroup(group, copyOp.getSource());
      addViewProducerChainToGroup(group, copyOp.getTarget());
    }
    addOpToGroup(group, op);
  }

  void sortGroupOps(StageGroup &group) {
    std::sort(group.ops.begin(), group.ops.end(),
              [](Operation *lhs, Operation *rhs) {
                return lhs->isBeforeInBlock(rhs);
              });
  }

  // 寻找当前 GEMM/mmadL1 前面给某个 local alloc 写数据的 copy，并把它作为
  // Cube scope 的输入搬运一起收进去。
  //
  // 当前规则：
  // 1. workspace -> local：这是 mix 模式里显式的 C/V 边界，必须由 Cube
  //    scope 回灌到 L1/cbuf 后再参与 GEMM。
  // 2. GM -> local：这是普通 FA 这类写法里的 K/V 输入搬运。虽然它不是
  //    workspace 边界，但目的 local alloc 随后作为 GEMM 输入使用，因此也
  //    应归 Cube scope；否则会被剩余 op 分到 Vector scope，InferMemScope
  //    会对同一个 local alloc 同时推导出 UB 和 L1。
  //
  // source/target 往往不是 workspace 或 GM 原值，而是 subview /
  // reinterpret_cast / collapse_shape 等 view。判断时会沿 ViewLikeOpInterface
  // 递归追溯 view source，所以能识别：
  //   memref.subview %workspace[...] -> local
  //   memref.subview %reinterpret_cast_gm[...] -> local
  Operation *findCubeInputCopyBefore(
      Operation *anchor, Operation *localAlloc,
      const llvm::SmallPtrSetImpl<Operation *> &alreadyInCube) {
    Operation *candidate = nullptr;
    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (&op == anchor)
        break;
      if (alreadyInCube.contains(&op))
        continue;

      auto copyOp = dyn_cast<CopyOpInterface>(&op);
      if (!copyOp)
        continue;
      if (isValAtCubeBoundary(copyOp.getSource()) &&
          getLocalAllocRoot(copyOp.getTarget()) == localAlloc)
        candidate = &op;
    }
    return candidate;
  }

  // 寻找当前 GEMM/mmadL1 后面从某个 local alloc 搬出的 copy，并作为 Cube
  // scope 的输出搬运一起收进去。当前只收 local -> workspace/GM：
  // - local -> workspace：写给后续 Vector scope 或下一个 Cube scope 使用；
  // - local -> GM：直接由 Cube 结果写回全局内存的场景。
  // 这里同样会通过 view producer 追溯 subview/reinterpret_cast 的真实来源。
  Operation *findCubeOutputCopyAfter(
      Operation *anchor, Operation *localAlloc,
      const llvm::SmallPtrSetImpl<Operation *> &alreadyInCube) {
    bool seenAnchor = false;
    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (&op == anchor) {
        seenAnchor = true;
        continue;
      }
      if (!seenAnchor)
        continue;
      if (isCubeAnchor(&op))
        return nullptr;
      if (alreadyInCube.contains(&op))
        continue;

      auto copyOp = dyn_cast<CopyOpInterface>(&op);
      if (!copyOp)
        continue;
      if (getLocalAllocRoot(copyOp.getSource()) == localAlloc &&
          isValAtCubeBoundary(copyOp.getTarget()))
        return &op;
    }
    return nullptr;
  }

  void addCubeBoundaryCopies(
      Operation *anchor, StageGroup &group,
      const llvm::SmallPtrSetImpl<Operation *> &alreadyInCube) {
    for (auto operand : anchor->getOperands()) {
      Operation *localAlloc = getLocalAllocRoot(operand);
      if (!localAlloc)
        continue;

      addBoundaryCopyToGroup(
          group, findCubeInputCopyBefore(anchor, localAlloc, alreadyInCube));
      addBoundaryCopyToGroup(
          group, findCubeOutputCopyAfter(anchor, localAlloc, alreadyInCube));
    }
  }

  SmallVector<StageGroup>
  collectCubeGroups(DenseMap<Operation *, std::size_t> &cubeGroupOfOp) {
    SmallVector<StageGroup> cubeGroups;
    llvm::SmallPtrSet<Operation *, 32> alreadyInCube;

    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (!isCubeAnchor(&op) || alreadyInCube.contains(&op))
        continue;

      StageGroup group;
      group.coreType = hivm::TCoreType::CUBE;
      addCubeBoundaryCopies(&op, group, alreadyInCube);
      addOpToGroup(group, &op);
      sortGroupOps(group);

      if (group.ops.empty())
        continue;

      std::size_t groupId = cubeGroups.size();
      for (auto *groupOp : group.ops) {
        cubeGroupOfOp[groupOp] = groupId;
        alreadyInCube.insert(groupOp);
      }
      cubeGroups.push_back(std::move(group));
    }

    return cubeGroups;
  }

  // Cube 组先由 GEMM anchor 固定下来。剩余 op 按 pipeline body 中的源码顺序归入
  // Vector 组，遇到 Cube 组就切断当前 Vector 段并插入 Cube 组。
  SmallVector<StageGroup> collectVectorGroupsByOrder(
      const SmallVector<StageGroup> &cubeGroups,
      const DenseMap<Operation *, std::size_t> &cubeGroupOfOp) {
    SmallVector<StageGroup> orderedGroups;
    SmallVector<char> emittedCube(cubeGroups.size(), 0);
    StageGroup pendingVector;
    pendingVector.coreType = hivm::TCoreType::VECTOR;

    auto flushVector = [&]() {
      if (pendingVector.ops.empty())
        return;
      orderedGroups.push_back(std::move(pendingVector));
      pendingVector = StageGroup{};
      pendingVector.coreType = hivm::TCoreType::VECTOR;
    };

    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      auto cubeIt = cubeGroupOfOp.find(&op);
      if (cubeIt != cubeGroupOfOp.end()) {
        flushVector();
        std::size_t groupId = cubeIt->second;
        if (!emittedCube[groupId]) {
          orderedGroups.push_back(cubeGroups[groupId]);
          emittedCube[groupId] = 1;
        }
        continue;
      }

      if (shouldKeepOutsideScope(&op))
        continue;

      Operation *nextCubeOp = findNextCubeOpAfter(&op, cubeGroupOfOp);
      llvm::SmallPtrSet<Operation *, 8> escapingVisited;
      if (hasResultEscapingVectorSegment(&op, nextCubeOp, cubeGroupOfOp,
                                         escapingVisited)) {
        flushVector();
        continue;
      }

      pendingVector.ops.push_back(&op);
    }

    flushVector();
    return orderedGroups;
  }

  // 普通 local alloc 如果所有有效使用都落在同一个 C/V stage 内，就可以下沉到
  // 该 scope。这样 NPUIR memory planner 能按 scope 生命周期复用 CC/UB。
  // 带 multi_buffer 标记的 alloc 必须留在外层，后续 EnableLocalBuffer 需要
  // 在 EnableMultiBuffer 生成的 stage loop 上给它增加 stage 维度。
  void assignLocalAllocsToGroups(SmallVectorImpl<StageGroup> &groups) {
    DenseMap<Operation *, std::size_t> groupOfOp;
    for (std::size_t groupId = 0; groupId < groups.size(); ++groupId) {
      for (Operation *op : groups[groupId].ops)
        groupOfOp[op] = groupId;
    }

    for (auto &op : currentForOp.getBody()->getOperations()) {
      auto allocOp = dyn_cast<memref::AllocOp>(&op);
      if (!allocOp || hasAnnotationMark(allocOp) || hasMultiBufferMark(allocOp))
        continue;

      std::size_t targetGroup = 0;
      bool hasTarget = false;
      bool canMove = true;

      for (Operation *user : allocOp.getResult().getUsers()) {
        Operation *topLevelUser = getTopLevelOpInCurrentFor(user);
        if (!topLevelUser || topLevelUser == allocOp) {
          canMove = false;
          break;
        }

        auto it = groupOfOp.find(topLevelUser);
        if (it == groupOfOp.end()) {
          canMove = false;
          break;
        }

        if (!hasTarget) {
          targetGroup = it->second;
          hasTarget = true;
        } else if (targetGroup != it->second) {
          canMove = false;
          break;
        }
      }

      if (canMove && hasTarget)
        groups[targetGroup].localAllocs.push_back(allocOp);
    }
  }

  bool hasAnnotationMark(memref::AllocOp allocOp) {
    return llvm::any_of(allocOp.getResult().getUsers(), [](Operation *user) {
      return isa<annotation::MarkOp>(user);
    });
  }

  bool hasMultiBufferMark(memref::AllocOp allocOp) {
    for (Operation *user : allocOp.getResult().getUsers()) {
      auto markOp = dyn_cast<annotation::MarkOp>(user);
      if (markOp && markOp->getAttr("hivm.multi_buffer"))
        return true;
    }
    return false;
  }

  Operation *findNextCubeOpAfter(
      Operation *op, const DenseMap<Operation *, std::size_t> &cubeGroupOfOp) {
    bool seenOp = false;
    for (auto &candidate : currentForOp.getBody()->getOperations()) {
      if (&candidate == op) {
        seenOp = true;
        continue;
      }
      if (!seenOp)
        continue;
      if (cubeGroupOfOp.find(&candidate) != cubeGroupOfOp.end())
        return &candidate;
    }
    return nullptr;
  }

  // arith/subview/reinterpret_cast 这类 helper 可以进入 scope，但前提是结果只在
  // 当前 Vector 段内使用。若结果跨过下一个 Cube 段，或直接被 Cube 段使用，
  // 就先留在父 loop 中，避免生成 EnableMultiBuffer 目前不会消解的 scope result。
  // 这里递归检查 user 的结果是否逃逸，保证跨 stage 标量的 producer 链也留在外面。
  bool hasResultEscapingVectorSegment(
      Operation *op, Operation *nextCubeOp,
      const DenseMap<Operation *, std::size_t> &cubeGroupOfOp,
      llvm::SmallPtrSetImpl<Operation *> &visited) {
    if (!visited.insert(op).second)
      return false;

    for (OpResult result : op->getResults()) {
      for (auto &use : result.getUses()) {
        Operation *user = getTopLevelOpInCurrentFor(use.getOwner());
        if (!user)
          return true;
        if (user == op)
          continue;
        if (cubeGroupOfOp.find(user) != cubeGroupOfOp.end())
          return true;
        if (nextCubeOp && !user->isBeforeInBlock(nextCubeOp))
          return true;
        if (hasResultEscapingVectorSegment(user, nextCubeOp, cubeGroupOfOp,
                                           visited))
          return true;
      }
    }
    return false;
  }

  // 在每组第一个 op 的原位置创建 scope，再按原序搬入组内 op。这样既保持 C/V stage
  // 的执行顺序，也允许 alloc/workspace 留在父循环里被多个 scope 捕获。
  void packGroups(SmallVectorImpl<StageGroup> &groups) {
    for (auto &group : groups) {
      if (group.ops.empty())
        continue;
      sortGroupOps(group);

      OpBuilder builder(group.ops.front());
      auto scope =
          builder.create<scope::ScopeOp>(builder.getUnknownLoc(), TypeRange());

      scope->setAttr(hivm::TCoreTypeAttr::name,
                     builder.getAttr<hivm::TCoreTypeAttr>(group.coreType));

      auto &scopeBody = scope.getRegion().emplaceBlock();
      for (auto *alloc : group.localAllocs)
        alloc->moveBefore(&scopeBody, scopeBody.end());
      for (auto *op : group.ops)
        op->moveBefore(&scopeBody, scopeBody.end());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&scopeBody);
      builder.create<scope::ReturnOp>(builder.getUnknownLoc());

      LDBG("Packed a " << hivm::stringifyTCoreType(group.coreType)
                       << "-core scope:\n"
                       << scope);
    }
  }

  Operation *getLocalAllocRoot(Value val) {
    auto definingOp = val.getDefiningOp();
    if (!definingOp)
      return nullptr;
    if (isa<memref::AllocOp>(definingOp))
      return definingOp;
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(definingOp))
      return getLocalAllocRoot(viewOp.getViewSource());
    return nullptr;
  }

  // 将任意 op 映射为 currentForOp body 的直接子 op。
  // 如果 op 在嵌套 scf.for/scf.if 内部，这里返回外层那个直接子 op；
  // 如果 op 不属于当前 pipeline loop，则返回 nullptr。
  Operation *getTopLevelOpInCurrentFor(Operation *op) {
    auto *body = currentForOp.getBody();
    Operation *cur = op;
    while (cur && cur->getBlock() != body)
      cur = cur->getParentOp();
    return cur;
  }

  // 判断一个 value 是否来自 workspace alloc；subview/reinterpret_cast 这类 view op
  // 会递归追溯到原始 source。
  bool isValFromWorkspace(Value val) {
    auto definingOp = val.getDefiningOp();
    return definingOp &&
           TypeSwitch<Operation *, bool>(definingOp)
               .Case([&](ViewLikeOpInterface op) {
                 return isValFromWorkspace(op.getViewSource());
               })
               .Case([](bishengir::memref_ext::AllocWorkspaceOp op) {
                 return true;
               })
               .Default(false);
  }

  // 判断一个 value 是否是 GM memref，或来自 GM memref 的 view。FA 这类 kernel
  // 的 K/V 输入通常是：
  //   arg(gm) -> reinterpret_cast(gm) -> subview(gm) -> local
  // 因此不能只看 defining op 是否是 workspace alloc，还要检查 memref 类型上的
  // #hivm.address_space<gm>，并沿 view source 继续回溯。
  bool isValFromGM(Value val) {
    if (auto memrefType = dyn_cast<MemRefType>(val.getType())) {
      auto addrSpace = hivm::getOptionalHIVMAddressSpace(memrefType);
      if (addrSpace.has_value() && *addrSpace == hivm::AddressSpace::GM)
        return true;
    }

    auto definingOp = val.getDefiningOp();
    if (auto viewOp = dyn_cast_or_null<ViewLikeOpInterface>(definingOp))
      return isValFromGM(viewOp.getViewSource());
    return false;
  }

  // Cube scope 边界 copy 的外侧来源/去处。workspace 是混编流水的显式跨核
  // 中转点；GM 是普通输入/输出全局内存。二者的 view 都在各自判断里处理。
  bool isValAtCubeBoundary(Value val) {
    return isValFromWorkspace(val) || isValFromGM(val);
  }

};
#undef DEBUG_TYPE

} // namespace mlir::tilelangir
