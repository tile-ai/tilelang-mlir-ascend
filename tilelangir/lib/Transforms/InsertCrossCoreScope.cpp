// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/InsertCrossCoreScope.cpp
 * \brief TileLangIR cross-core scope insertion pass.
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

#define GEN_PASS_DEF_TILELANGIRINSERTCROSSCORESCOPE
#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-insert-cross-core-scope"
#define LDBG(X)                                                                \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << X << '\n')

struct TileLangIRInsertCrossCoreScope
    : impl::TileLangIRInsertCrossCoreScopeBase<
          TileLangIRInsertCrossCoreScope> {

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

  // InsertCrossCoreScope only processes loops marked with num_stages after
  // T.pipeline lowering. A plain scf.for may just be gather, mask, or scalar
  // control flow, so it must not be treated as a C/V partition boundary.
  bool isPipelineFor(scf::ForOp forOp) {
    return forOp->hasAttr("tilelangir.num_stages");
  }

  // Temporary strategy: use only GEMM/mmadL1 as Cube-stage anchors. Workspace
  // copies are stage boundaries, not stage roots; using copies as seeds would
  // split real computation into many copy-only scopes.
  bool isCubeAnchor(Operation *op) {
    auto name = op->getName().getStringRef();
    return name == "hivm.hir.mmadL1" || name.contains("mmad");
  }

  // Declarations and metadata are not real C/V execution operations, so they are
  // excluded from compute grouping first. Later assignLocalAllocsToGroups sinks
  // plain allocs used by only one scope, while cross-scope or multi-buffer allocs
  // stay in the parent loop and are captured by multiple scopes.
  // annotation.mark only describes attributes such as multi_buffer. If it is
  // incorrectly moved into a VECTOR scope, InferMemScope can infer the marked
  // local buffer as UB, conflicting with the L1 requirement of the actual CUBE
  // GEMM.
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

  // A copy source/target may be a view on top of workspace/GM, such as
  // subview/reinterpret_cast/collapse_shape, rather than workspace/GM itself.
  // These view producers and the copy belong to the same C/V or GM->Cube
  // boundary and must move into the scope together. Otherwise a moved copy can
  // use a view that remains in the parent region, possibly after the scope.
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

  // Find the copy before the current GEMM/mmadL1 that writes into a local alloc,
  // and include it as the input movement for the Cube scope.
  //
  // Current rules:
  // 1. workspace -> local: this is the explicit C/V boundary in mix mode. The
  //    Cube scope must load it back into L1/cbuf before GEMM.
  // 2. GM -> local: this is the K/V input movement used by kernels such as
  //    regular FA. It is not a workspace boundary, but the destination local
  //    alloc is later used as a GEMM input, so it also belongs to the Cube scope.
  //    Otherwise the remaining-op pass would place it in a Vector scope, and
  //    InferMemScope could infer both UB and L1 for the same local alloc.
  //
  // The source/target is often a view such as subview/reinterpret_cast/
  // collapse_shape, not the raw workspace or GM value. Boundary checks follow the
  // ViewLikeOpInterface source recursively, so they can recognize:
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

  // Find the copy after the current GEMM/mmadL1 that moves data out from a local
  // alloc, and include it as the output movement for the Cube scope. Currently
  // only local -> workspace/GM is accepted:
  // - local -> workspace: data consumed by a later Vector scope or the next Cube
  //   scope;
  // - local -> GM: Cube results written directly back to global memory.
  // This path also follows view producers to recover the real source behind a
  // subview/reinterpret_cast.
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

  // Cube groups are fixed first by GEMM anchors. Remaining operations are assigned
  // to Vector groups in pipeline-body source order. When a Cube group is reached,
  // the current Vector segment is flushed and the Cube group is inserted.
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

  // A plain local alloc can be sunk into a scope when all of its valid uses are in
  // the same C/V stage. This lets the NPUIR memory planner reuse CC/UB according
  // to scope lifetime. Allocs marked multi_buffer must stay outside because
  // EnableLocalBuffer later adds a stage dimension on the stage loop generated by
  // EnableMultiBuffer.
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

  // Helpers such as arith/subview/reinterpret_cast may enter a scope only when
  // their results are used exclusively in the current Vector segment. If a result
  // crosses the next Cube segment or is used directly by a Cube segment, keep it
  // in the parent loop to avoid scope results that EnableMultiBuffer cannot
  // currently eliminate. Recursively checking user results keeps cross-stage
  // scalar producer chains outside as well.
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

  // Create each scope at the original position of the group's first operation,
  // then move grouped operations into it in source order. This preserves C/V stage
  // execution order while allowing allocs/workspaces to stay in the parent loop
  // and be captured by multiple scopes.
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

  // Map any operation to the direct child operation in currentForOp's body. If an
  // operation is nested inside scf.for/scf.if, return that outer direct child. If
  // it does not belong to the current pipeline loop, return nullptr.
  Operation *getTopLevelOpInCurrentFor(Operation *op) {
    auto *body = currentForOp.getBody();
    Operation *cur = op;
    while (cur && cur->getBlock() != body)
      cur = cur->getParentOp();
    return cur;
  }

  // Check whether a value comes from a workspace alloc. View operations such as
  // subview/reinterpret_cast are followed recursively to their original source.
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

  // Check whether a value is a GM memref, or a view derived from a GM memref. K/V
  // inputs in kernels such as FA usually look like:
  //   arg(gm) -> reinterpret_cast(gm) -> subview(gm) -> local
  // Therefore it is not enough to inspect whether the defining op is a workspace
  // alloc. The memref type's #hivm.address_space<gm> must also be checked, and
  // view sources must be followed recursively.
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

  // External source/destination for copies at Cube-scope boundaries. Workspace is
  // the explicit cross-core staging point for mix pipelines, while GM is regular
  // input/output global memory. Views of both are handled by their own predicates.
  bool isValAtCubeBoundary(Value val) {
    return isValFromWorkspace(val) || isValFromGM(val);
  }

};
#undef DEBUG_TYPE

} // namespace mlir::tilelangir
