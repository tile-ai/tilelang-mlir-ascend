// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "tilelangir/Transforms/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InferHIVMMemScope.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINFERMEMSCOPE
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-infer-mem-scope"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace hivm;

// ===----------------------------------------------------------------------===//
// Replaces AscendNPU-IR's MemScopeInferAndPropagateHelper with a more general
// approach: instead of a whitelist for single-result ops, we propagate the scope
// to *every* memref-typed result of any user op.
// ===----------------------------------------------------------------------===//
class MemScopePropagator {
public:
  LogicalResult run(Value operand, AddressSpaceAttr targetScope) {
    auto memRefType = dyn_cast<BaseMemRefType>(operand.getType());
    if (!memRefType)
      return failure();
    if (memRefType.getMemorySpace())
      return success();

    setBaseMemRefTypeScope(operand, targetScope);
    return propagateToUsers(operand);
  }

private:
  static BlockArgument getTiedWhileBodyIterArg(scf::WhileOp op,
                                               OpOperand *opOperand) {
    auto argsMutable = op.getInitsMutable();
    auto *it = llvm::find(argsMutable, *opOperand);
    if (it == argsMutable.end())
      return {};
    return op.getAfterArguments()[std::distance(argsMutable.begin(), it)];
  }

  LogicalResult propagateToUsers(Value val) {
    auto memrefScope = getHIVMAddressSpaceAttr(val.getType());

    for (OpOperand &use : val.getUses()) {
      Operation *userOp = use.getOwner();
      LogicalResult res =
          TypeSwitch<Operation *, LogicalResult>(userOp)
              .Case<scf::YieldOp>([&](scf::YieldOp op) {
                Operation *parentOp = op->getParentOp();
                Value yieldOperand = op.getOperand(use.getOperandNumber());
                if (!isa<BaseMemRefType>(yieldOperand.getType()))
                  return success();
                Value parentResult =
                    parentOp->getResult(use.getOperandNumber());
                setBaseMemRefTypeScope(parentResult, memrefScope);
                return propagateToUsers(parentResult);
              })
              .Case<scf::ForOp>([&](scf::ForOp op) {
                Value result = op.getTiedLoopResult(&use);
                setBaseMemRefTypeScope(result, memrefScope);
                Value bbArg = op.getTiedLoopRegionIterArg(&use);
                setBaseMemRefTypeScope(bbArg, memrefScope);
                return success(
                    propagateToUsers(bbArg).succeeded() &&
                    propagateToUsers(result).succeeded());
              })
              .Case<scf::WhileOp>([&](scf::WhileOp op) {
                BlockArgument bbArg =
                    cast<BlockArgument>(op.getTiedLoopRegionIterArg(&use));
                auto yield = op.getTiedLoopYieldedValue(bbArg);
                BlockArgument afterArg = getTiedWhileBodyIterArg(op, &use);
                setBaseMemRefTypeScope(bbArg, memrefScope);
                setBaseMemRefTypeScope(yield->get(), memrefScope);
                setBaseMemRefTypeScope(afterArg, memrefScope);
                return success(
                    propagateToUsers(afterArg).succeeded() &&
                    propagateToUsers(bbArg).succeeded() &&
                    propagateToUsers(yield->get()).succeeded());
              })
              .Case<func::CallOp>([&](auto) { return success(); })
              .Default([&](Operation *op) {
                if (op->getNumResults() == 0)
                  return success();
                for (OpResult result : op->getResults()) {
                  if (!isa<BaseMemRefType>(result.getType()))
                    continue;
                  setBaseMemRefTypeScope(result, memrefScope);
                  if (failed(propagateToUsers(result)))
                    return failure();
                }
                return success();
              });
      if (failed(res))
        return failure();
    }
    return success();
  }
};

/// Set address space on a root value and propagate to all users.
static LogicalResult setAllocScope(Value rootVal,
                                   hivm::AddressSpace space) {
  auto memRefType = dyn_cast<BaseMemRefType>(rootVal.getType());
  if (!memRefType)
    return success();
  if (memRefType.getMemorySpace())
    return success();

  auto spaceAttr = AddressSpaceAttr::get(rootVal.getContext(), space);
  MemScopePropagator propagator;
  return propagator.run(rootVal, spaceAttr);
}

struct AllocScopeConstraints {
  // Strong constraints come from role-specific users: VECTOR ops require UB,
  // and MmadL1 operands require L1/L0C according to A/B/C/bias roles.
  bool needUB = false;
  bool needL1 = false;
  bool needL0C = false;
  // A generic CUBE-scope use is only a fallback. For example, a copy after
  // MmadL1 still consumes the CC output buffer inside a CUBE scope.
  bool weakL1 = false;
};

using ConstraintMap = llvm::DenseMap<Operation *, AllocScopeConstraints>;

static bool hasMemScope(Value value) {
  auto memRefType = dyn_cast<BaseMemRefType>(value.getType());
  return memRefType && memRefType.getMemorySpace();
}

static void addConstraint(ConstraintMap &constraints, Value value,
                          hivm::AddressSpace space, bool strong) {
  if (!isa<BaseMemRefType>(value.getType()))
    return;

  auto rootAlloc = utils::tracebackMemRefToAlloc(value);
  if (!rootAlloc.has_value())
    return;

  if (hasMemScope(rootAlloc->getMemref()))
    return;

  auto &constraint = constraints[rootAlloc->getOperation()];
  if (!strong) {
    if (space == hivm::AddressSpace::L1)
      constraint.weakL1 = true;
    return;
  }

  if (space == hivm::AddressSpace::UB) {
    constraint.needUB = true;
  } else if (space == hivm::AddressSpace::L1) {
    constraint.needL1 = true;
  } else if (space == hivm::AddressSpace::L0C) {
    constraint.needL0C = true;
  }
}

static LogicalResult addMmadConstraint(ConstraintMap &constraints,
                                       hivm::MmadL1Op op, Value value,
                                       hivm::AddressSpace space,
                                       llvm::StringRef operandName) {
  auto rootAlloc = utils::tracebackMemRefToAlloc(value);
  if (!rootAlloc.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for "
                           << operandName << " of this op.";
    return failure();
  }

  if (hasMemScope(rootAlloc->getMemref()))
    return success();

  auto &constraint = constraints[rootAlloc->getOperation()];
  if (space == hivm::AddressSpace::L1) {
    constraint.needL1 = true;
  } else if (space == hivm::AddressSpace::L0C) {
    constraint.needL0C = true;
  } else if (space == hivm::AddressSpace::UB) {
    constraint.needUB = true;
  }
  return success();
}

static std::optional<hivm::TCoreType>
getUseSiteCoreType(Operation *op, func::FuncOp funcOp) {
  if (auto coreIface = dyn_cast<hivm::CoreTypeInterface>(op)) {
    auto coreType = coreIface.getCoreType();
    if (coreType.has_value())
      return *coreType;
  }

  Operation *parent = op->getParentOp();
  while (parent && parent != funcOp.getOperation()) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(parent)) {
      if (auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
              hivm::TCoreTypeAttr::name))
        return attr.getTcoretype();
    }
    parent = parent->getParentOp();
  }

  return std::nullopt;
}

static LogicalResult collectMmadConstraints(func::FuncOp funcOp,
                                            ConstraintMap &constraints) {
  auto result = funcOp.walk([&](hivm::MmadL1Op op) -> WalkResult {
    if (!op.hasPureBufferSemantics()) {
      op->emitOpError("Run infer memory scope after bufferization.");
      return WalkResult::interrupt();
    }

    auto *mA = op.getDpsInputOperand(0);
    auto *mB = op.getDpsInputOperand(1);
    auto *mC = op.getDpsInitOperand(0);

    if (failed(addMmadConstraint(constraints, op, mA->get(),
                                 hivm::AddressSpace::L1, "mA")) ||
        failed(addMmadConstraint(constraints, op, mB->get(),
                                 hivm::AddressSpace::L1, "mB")) ||
        failed(addMmadConstraint(constraints, op, mC->get(),
                                 hivm::AddressSpace::L0C, "mC")))
      return WalkResult::interrupt();

    if (auto bias = op.getPerChannelBias()) {
      if (failed(addMmadConstraint(constraints, op, bias,
                                   hivm::AddressSpace::L1, "bias")))
        return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

static void collectUseSiteConstraints(func::FuncOp funcOp,
                                      ConstraintMap &constraints) {
  funcOp.walk([&](Operation *op) {
    // Mmad has per-operand roles: A/B are L1, C is L0C. Treating it as a
    // generic CUBE use would lose that distinction, so it is handled above.
    if (isa<hivm::MmadL1Op>(op))
      return;

    auto coreType = getUseSiteCoreType(op, funcOp);
    if (!coreType.has_value())
      return;

    std::optional<hivm::AddressSpace> space;
    bool strong = true;
    if (*coreType == hivm::TCoreType::VECTOR) {
      space = hivm::AddressSpace::UB;
    } else if (*coreType == hivm::TCoreType::CUBE) {
      // Generic CUBE uses are only a fallback. Mmad outputs, for example, are
      // CC even though the following copy is still inside a CUBE scope.
      space = hivm::AddressSpace::L1;
      strong = false;
    }

    if (!space.has_value())
      return;

    for (OpOperand &operand : op->getOpOperands())
      addConstraint(constraints, operand.get(), *space, strong);
  });
}

/// Final fallback for allocs that have no real use-site constraint.
static std::optional<hivm::AddressSpace>
getFallbackScope(memref::AllocOp allocOp, func::FuncOp funcOp) {
  if (allocOp.getType().getMemorySpace())
    return std::nullopt;

  Operation *parent = allocOp->getParentOp();
  while (parent && parent != funcOp.getOperation()) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(parent)) {
      if (auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
              hivm::TCoreTypeAttr::name)) {
        auto ct = attr.getTcoretype();
        if (ct == hivm::TCoreType::VECTOR)
          return hivm::AddressSpace::UB;
        if (ct == hivm::TCoreType::CUBE)
          return hivm::AddressSpace::L1;
      }
    }
    parent = parent->getParentOp();
  }

  auto funcCoreType = hivm::queryFuncCoreType(funcOp);
  if (funcCoreType.has_value()) {
    if (*funcCoreType == hivm::TFuncCoreType::AIC)
      return hivm::AddressSpace::L1;
    if (*funcCoreType == hivm::TFuncCoreType::AIV)
      return hivm::AddressSpace::UB;
  }

  return std::nullopt;
}

static std::optional<hivm::AddressSpace>
resolveScope(memref::AllocOp allocOp, const AllocScopeConstraints *constraint,
             func::FuncOp funcOp, bool &failed) {
  failed = false;
  if (!constraint)
    return getFallbackScope(allocOp, funcOp);

  if (constraint->needUB &&
      (constraint->needL1 || constraint->needL0C || constraint->weakL1)) {
    allocOp.emitOpError(
        "conflicting memory scope constraints: VECTOR/UB use and CUBE use")
        << " for the same local buffer";
    failed = true;
    return std::nullopt;
  }
  if (constraint->needL1 && constraint->needL0C) {
    allocOp.emitOpError("conflicting memory scope constraints: L1 and L0C")
        << " for the same local buffer";
    failed = true;
    return std::nullopt;
  }

  if (constraint->needUB)
    return hivm::AddressSpace::UB;
  if (constraint->needL0C)
    return hivm::AddressSpace::L0C;
  if (constraint->needL1)
    return hivm::AddressSpace::L1;
  if (constraint->weakL1)
    return hivm::AddressSpace::L1;

  return getFallbackScope(allocOp, funcOp);
}

struct TileLangIRInferMemScope
    : impl::TileLangIRInferMemScopeBase<TileLangIRInferMemScope> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    LLVM_DEBUG(DBGS() << "processing function: " << funcOp.getSymName()
                      << "\n");

    // Phase 1: workspace buffers are global memory by definition.
    funcOp.walk([&](bishengir::memref_ext::AllocWorkspaceOp op) {
      LLVM_DEBUG(DBGS() << "Phase 1 workspace: " << *op << "\n");
      if (failed(setAllocScope(op.getMemref(), hivm::AddressSpace::GM)))
        return signalPassFailure();
    });

    // Phase 2: function arguments are external GM buffers.
    LLVM_DEBUG(DBGS() << "Phase 2 func args -> GM\n");
    if (failed(hivm::inferAndPropagateMemScopeForFunc(funcOp)))
      return signalPassFailure();

    // Phase 3: collect all local-buffer constraints before mutating types.
    ConstraintMap constraints;
    if (failed(collectMmadConstraints(funcOp, constraints)))
      return signalPassFailure();
    collectUseSiteConstraints(funcOp, constraints);

    // Phase 4: resolve each local memref.alloc from its actual use sites.
    funcOp.walk([&](memref::AllocOp op) {
      bool failedResolve = false;
      auto iter = constraints.find(op.getOperation());
      const AllocScopeConstraints *constraint =
          iter == constraints.end() ? nullptr : &iter->second;
      auto scope = resolveScope(op, constraint, funcOp, failedResolve);
      if (failedResolve)
        return signalPassFailure();
      if (!scope.has_value())
        return;
      LLVM_DEBUG(DBGS() << "Phase 4 local alloc -> "
                        << hivm::stringifyAddressSpace(*scope) << ": " << *op
                        << "\n");
      if (failed(setAllocScope(op, *scope)))
        signalPassFailure();
    });
  }
};

#undef DBGS
#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
