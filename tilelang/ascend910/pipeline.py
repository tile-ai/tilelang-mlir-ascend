"""Ascend 910 (A3) lowering pipeline.

The frontend already produces NPUIR calls and a kernel launch frame.  These
passes materialize the launch, expose allocations, vectorize the residual
parallel loops into NPUIR intrinsics and simplify the folded 1-D grid before
the MLIR emitter runs.
"""

from __future__ import annotations

import os

from tvm import IRModule, tirx
from tvm.ir import CallingConv
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline


def _variant() -> str:
    """Pipeline variant selector.

    ``full`` (default) mirrors the legacy pass order, including the ported
    ``NpuLoopVectorize``/``LowerNpuirBlock`` passes that fold residual
    ``T.Parallel`` loops into ``tl.npuir_*`` intrinsics.
    ``TILELANG_ASCEND910_NPU_LOOP_VECTORIZE=0`` (or ``A910_PIPE=legacy``)
    selects the older pipeline without loop vectorization, kept for A/B
    debugging.
    """

    if os.environ.get("A910_PIPE"):
        return os.environ["A910_PIPE"]
    if os.environ.get("TILELANG_ASCEND910_NPU_LOOP_VECTORIZE") == "0":
        return "legacy"
    return "full"


def Ascend910PassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    # The frontend builds one PrimFunc per kernel.  Mark it as a device kernel
    # so the host/device filter in engine.lower keeps it for codegen.
    for gv in list(mod.functions):
        func = mod[gv]
        if not isinstance(func, tirx.PrimFunc):
            continue
        # The eager frontend keeps dynamic shape symbols as free vars in buffer
        # shapes. A2 codegen expects them as explicit scalar kernel arguments.
        known = set(func.params)
        extra = []
        for buf in func.buffer_map.values():
            for dim in buf.shape:
                for var in tirx.analysis.undefined_vars(dim, []):
                    if var not in known and var not in extra and str(var.dtype).startswith("int"):
                        extra.append(var)
        if extra:
            func = tirx.PrimFunc(
                list(func.params) + extra,
                func.body,
                func.ret_type,
                func.buffer_map,
                func.attrs,
            )
        mod[gv] = func.with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH)

    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)

    if os.environ.get("A910_DUMP_TIR"):
        print("====== A910 TIR (after MaterializeKernelLaunch) ======")
        print(mod.script())

    # v0.1.15's eager frontend leaves Python temporaries as Let-bound scalar
    # variables (``bx_1 = bx // 4 * 128``) that the legacy frontend had already
    # inlined.  Inline them so the TIR handed to the code generator matches.

    variant = _variant()

    def _dump(tag, module):
        if os.environ.get("A910_DUMP_TIR"):
            print(f"====== A910 TIR ({tag}) ======")
            print(module.script())

    # v0.1.15's eager frontend keeps Python temporaries as Let-bound scalars
    # (``bx_1 = bx // 4 * 128``) where the legacy frontend had already inlined
    # them; inline them so the code generators see the legacy shape.
    if os.environ.get("A910_LETINLINE", "1") != "0":
        mod = tilelang.transform.LetInline()(mod)

    # Strength-reduce exact float divisions (legacy Simplify behaviour).
    mod = tilelang.transform.RewriteFDivToMul()(mod)

    if variant in ("nlev_only", "nlev_simplify", "full"):
        # Vectorize residual T.Parallel loops into tl.npuir_* intrinsics, then
        # place the allocations. Mirrors the legacy NPUIR pipeline
        # (LowerAndLegalize + OptimizeForTarget).
        if variant == "full":
            mod = tirx.transform.Simplify()(mod)
            mod = tirx.transform.RemoveNoOp()(mod)
        _dump("before LegalizeNpuirBF16", mod)
        mod = tilelang.transform.LegalizeNpuirBF16()(mod)
        _dump("after LegalizeNpuirBF16", mod)
        _dump("before NpuLoopVectorize", mod)
        mod = tilelang.transform.NpuLoopVectorize()(mod)
        _dump("after NpuLoopVectorize", mod)
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        _dump("after PlanAndUpdate", mod)
        mod = tilelang.transform.LowerOpaqueBlock()(mod)
        _dump("after LowerOpaqueBlock", mod)
        if variant == "full":
            mod = tilelang.transform.LowerNpuirBlock()(mod)
            _dump("after LowerNpuirBlock", mod)
            mod = tirx.transform.RemoveNoOp()(mod)
        else:
            mod = tirx.transform.Simplify()(mod)
        _dump("final", mod)
        return mod

    # Legacy variant: no loop vectorization.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tirx.transform.Simplify()(mod)
    _dump("final (legacy)", mod)
    return mod


ascend910_pipeline = PassPipeline("ascend910", Ascend910PassPipelineBody)
