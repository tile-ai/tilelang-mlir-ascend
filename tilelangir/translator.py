"""TIRX to TileLangIR MLIR translation."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import json
from typing import Any

from tvm import IRModule, tirx
from tvm.tirx import PrimFunc, PyStmtExprVisitor

from .dialect import configure_context
from ._tilelang_ops_gen import (
    CopyOp,
    GemmOp,
    ScopeOp,
    VcvtOp,
    VexpdifOp,
    VmaxOp,
    VmulsOp,
)


@tirx.functor.visitor
class TileLangIRTranslator(PyStmtExprVisitor):
    """Translate TIRX nodes into their TileLangIR MLIR representation.

    This initial storage-focused implementation intentionally flattens TIRX
    control and scope nodes into the ``func.func`` entry block.  It preserves
    the values needed by ``tl.tileop.vadd`` while Kernel, For, SBlock, and copy
    conversion are introduced separately.
    """

    def __init__(self) -> None:
        super().__init__()
        self.value_map: dict[int, Any] = {} # value_map stores mappings from TIRX object identities to MLIR Values, ensuring that the same TIRX object maps to the same MLIR Value
        self._arith: Any = None
        self._ir: Any = None
        self._func: Any = None
        self._linalg: Any = None
        self._memref: Any = None
        self._scf: Any = None
        self._execution_scope_modes: list[str] = []

    def translate(self, source_module: IRModule) -> str:
        self._jit_functions = []
        # 1. Check that the input is a tvm.IRModule
        if not isinstance(source_module, IRModule):
            raise TypeError(
                "TileLangIR codegen expects tvm.IRModule, "
                f"but received {type(source_module).__name__}."
            )
        # 2. Load and validate the MLIR Python bindings
        try:
            from mlir import ir
            from mlir.dialects import arith, func, linalg, memref, scf
        except ImportError as exc:
            raise ImportError(
                "TileLangIR codegen requires LLVM's official MLIR Python bindings. "
                "Install them with `python -m pip install mlir-python-bindings "
                "-f https://llvm.github.io/eudsl`."
            ) from exc

        self._arith = arith
        self._ir = ir
        self._func = func
        self._linalg = linalg
        self._memref = memref
        self._scf = scf

        with ir.Context() as context:
            try:
                configure_context(context)
                with ir.Location.unknown(context):
                    # 3. Create an empty MLIR module
                    module = ir.Module.create()
                    with ir.InsertionPoint(module.body):
                        # 4. Iterate over the PrimFuncs
                        for global_var, base_func in source_module.functions.items():
                            if not isinstance(base_func, PrimFunc):
                                raise TypeError(
                                    "TileLangIR codegen currently accepts only TIRX PrimFunc "
                                    f"entries, but @{global_var.name_hint} is "
                                    f"{type(base_func).__name__}."
                                )
                            self._emit_prim_func(global_var.name_hint, base_func)
                    module.operation.verify()
                    # A Python-side handoff only. OpenTileAS sees an ordinary
                    # comment and does not need to preserve custom attributes.
                    source = "// tilelang.launch.v1 " + json.dumps(
                        {"functions": self._jit_functions}, separators=(",", ":")
                    ) + "\n" + str(module)
            finally:
                self.value_map.clear()
                self._arith = None
                self._ir = None
                self._func = None
                self._linalg = None
                self._memref = None
                self._scf = None
                self._execution_scope_modes.clear()

        return source

    def _emit_prim_func(self, name: str, prim_func: PrimFunc) -> None:
        """Emit one function and materialize its TIRX Buffer parameters."""

        self.value_map.clear()
        block_idx_bindings = self._collect_block_idx_bindings(prim_func.body)
        self._jit_functions.append({
            "name": name,
            "kinds": ["handle" if param in prim_func.buffer_map else
                      ("int32" if str(param.dtype) == "bool" else str(param.dtype))
                      for param in prim_func.params],
            "axes": [{"tag": tag, "extent": self._static_int(extent, tag + " extent")}
                     for _var, extent, tag in block_idx_bindings],
        })

        # 1. Parse the function parameters and create the MLIR function
        input_types = [self._parameter_type(param, prim_func) for param in prim_func.params]
        arg_base = len(input_types)
        i32_type = self._ir.IntegerType.get_signless(32)
        input_types.extend(i32_type for _ in block_idx_bindings)
        function_type = self._ir.FunctionType.get(input_types, [])
        function_op = self._func.FuncOp(name, function_type)
        self._set_block_idx_attr(function_op, block_idx_bindings, arg_base)
        entry_block = function_op.add_entry_block()

        # 2. Set the MLIR insertion point inside func.func
        with self._ir.InsertionPoint(entry_block):
            # 3. Process the function parameters
            for param, block_argument in zip(prim_func.params, entry_block.arguments):
                self._insert_value(param, block_argument)
                if param in prim_func.buffer_map:
                    buffer = prim_func.buffer_map[param]
                    self._insert_value(buffer.data, block_argument)

            for param in prim_func.params:
                if param in prim_func.buffer_map:
                    self._emit_parameter_buffer(prim_func.buffer_map[param])

            for offset, (var, _extent, _tag) in enumerate(block_idx_bindings):
                self._insert_value(var, entry_block.arguments[arg_base + offset])

            # 4. Traverse the TIRX tree
            self.visit_stmt(prim_func.body)
            self._func.ReturnOp([])
    #############
    # Function parameter handling
    #############
    def _parameter_type(self, param: Any, prim_func: PrimFunc) -> Any:
        if param not in prim_func.buffer_map:
            return self._dtype_type(param.dtype)

        buffer = prim_func.buffer_map[param]
        dynamic = self._ir.ShapedType.get_dynamic_size()
        return self._ir.MemRefType.get(
            [dynamic],
            self._dtype_type(buffer.dtype),
            memory_space=self._memory_scope(buffer),
        )

    @staticmethod
    def _collect_block_idx_bindings(body: Any) -> list[tuple[Any, Any, str]]:
        bindings: list[tuple[Any, Any, str]] = []
        seen: set[int] = set()

        @tirx.functor.visitor
        class Collector(PyStmtExprVisitor):
            def visit_attr_stmt_(self, op: tirx.AttrStmt) -> None:
                if str(op.attr_key) == "thread_extent":
                    node = op.node
                    tag = str(getattr(node, "thread_tag", "") or "")
                    var = getattr(node, "var", None)
                    if tag.startswith("blockIdx.") and var is not None:
                        key = id(var)
                        if key not in seen:
                            seen.add(key)
                            bindings.append((var, op.value, tag))
                self.visit_stmt(op.body)

        Collector().visit_stmt(body)
        order = {"blockIdx.x": 0, "blockIdx.y": 1, "blockIdx.z": 2}
        bindings.sort(key=lambda item: order.get(item[2], 99))
        return bindings

    def _set_block_idx_attr(
        self,
        function_op: Any,
        block_idx_bindings: list[tuple[Any, Any, str]],
        arg_base: int,
    ) -> None:
        """Record which `func.func` arguments carry `blockIdx.x` / `blockIdx.y`.

        Each attribute is a parameter position; downstream replaces the
        argument with npu.get_block_idx (cid) / npu.get_sub_block_idx (vid).
        """
        for offset, (_var, extent, tag) in enumerate(block_idx_bindings):
            i64 = self._ir.IntegerType.get_signless(64)
            if tag == "blockIdx.x":
                function_op.attributes["BlockIdx"] = self._ir.IntegerAttr.get(
                    i64, arg_base + offset
                )
            elif tag == "blockIdx.y":
                # Second kernel axis is the 1:2 vec-core split (vid).
                extent_value = self._static_int(extent, "blockIdx.y extent")
                if extent_value != 2:
                    raise ValueError(
                        "blockIdx.y is the 1:2 vec-core split (vid), "
                        f"so its extent must be 2, but got {extent_value}"
                    )
                function_op.attributes["SubBlockIdx"] = self._ir.IntegerAttr.get(
                    i64, arg_base + offset
                )
            elif tag == "blockIdx.z":
                raise ValueError(
                    "The tile target supports at most two kernel axes: "
                    "blockIdx.x (cid) and blockIdx.y (vid, extent 2)."
                )

    def _emit_parameter_buffer(self, buffer: Any) -> Any:
        backing = self._get_value(buffer.data)
        shape = self._static_shape(buffer)
        strides = self._buffer_strides(buffer, shape)
        offset = self._static_int(buffer.elem_offset, f"{buffer.name}.elem_offset")
        result_type = self._buffer_memref_type(buffer, shape, strides, offset)
        view = self._memref.ReinterpretCastOp(
            result_type,
            backing,
            [],
            [],
            [],
            [offset],
            shape,
            strides,
        ).result
        self._insert_value(buffer, view)
        return view

    def _buffer_memref_type(
        self,
        buffer: Any,
        shape: list[int],
        strides: list[int],
        offset: int,
    ) -> Any:
        layout = self._ir.StridedLayoutAttr.get(offset, strides)
        return self._ir.MemRefType.get(
            shape,
            self._dtype_type(buffer.dtype),
            layout=layout,
            memory_space=self._memory_scope(buffer),
        )
    #############
    # TIRX tree traversal helper functions
    #############
    def _emit_alloc_buffer(self, buffer: Any) -> Any:
        scope = str(buffer.scope())
        if scope in ("shared", "shared.dyn"):
            return self._emit_shared_buffer(buffer)

        # Register fragments are logical typed allocations.  Unlike shared
        # memory, they do not use a reusable byte-level backing allocation.
        shape = self._static_shape(buffer)
        result_type = self._ir.MemRefType.get(
            shape,
            self._dtype_type(buffer.dtype),
            memory_space=self._memory_scope(buffer),
        )
        allocated = self._memref.AllocOp(result_type, [], []).result
        self._insert_value(buffer.data, allocated)
        self._insert_value(buffer, allocated)
        return allocated

    def _emit_shared_buffer(self, buffer: Any) -> Any:
        """Emit shared storage as a byte allocation plus a typed view."""

        shape = self._static_shape(buffer)
        byte_size = self._num_elements(shape) * self._dtype_nbytes(buffer.dtype)
        memory_space = self._memory_scope(buffer)
        raw_type = self._ir.MemRefType.get(
            [byte_size],
            self._ir.IntegerType.get_signless(8),
            memory_space=memory_space,
        )
        raw = self._memref.AllocOp(raw_type, [], []).result
        self._insert_value(buffer.data, raw)

        typed_type = self._ir.MemRefType.get(
            shape,
            self._dtype_type(buffer.dtype),
            memory_space=memory_space,
        )
        index_type = self._ir.IndexType.get()
        byte_shift = self._arith.ConstantOp(
            index_type, self._ir.IntegerAttr.get(index_type, 0)
        ).result
        typed = self._memref.ViewOp(typed_type, raw, byte_shift, []).result
        self._insert_value(buffer, typed)
        return typed

    def _emit_linear_offset(
        self,
        buffer: Any,
        indices: list[tirx.PrimExpr],
        buffer_strides: list[int],
    ) -> Any:
        if isinstance(buffer.elem_offset, tirx.IntImm) and all(
            isinstance(index, tirx.IntImm) for index in indices
        ):
            offset = int(buffer.elem_offset.value)
            offset += sum(
                int(index.value) * stride for index, stride in zip(indices, buffer_strides)
            )
            return offset

        index_type = self._ir.IndexType.get()
        offset = self._cast_to_index(self._get_or_create_expr_value(buffer.elem_offset))
        for index, stride in zip(indices, buffer_strides):
            index_value = self._cast_to_index(self._get_or_create_expr_value(index))
            stride_value = self._arith.ConstantOp(
                index_type, self._ir.IntegerAttr.get(index_type, stride)
            ).result
            scaled_index = self._arith.MulIOp(index_value, stride_value).result
            offset = self._arith.AddIOp(offset, scaled_index).result
        return offset

    def _is_identity_region_view(self, backing: Any, result_type: Any) -> bool:
        """Return whether a region type describes the existing memref exactly."""

        try:
            backing_type = self._ir.MemRefType(backing.type)
            region_type = self._ir.MemRefType(result_type)
            backing_strides, backing_offset = backing_type.get_strides_and_offset()
            region_strides, region_offset = region_type.get_strides_and_offset()
        except (TypeError, ValueError):
            return False

        return (
            tuple(backing_type.shape) == tuple(region_type.shape)
            and backing_type.element_type == region_type.element_type
            and backing_type.memory_space == region_type.memory_space
            and tuple(backing_strides) == tuple(region_strides)
            and backing_offset == region_offset
        )

    def _emit_region(self, call: tirx.Call) -> Any:
        if len(call.args) < 3:
            raise ValueError("tl.tileop.region expects BufferLoad, access type, and extents")

        buffer_load = call.args[0]
        if not isinstance(buffer_load, tirx.BufferLoad):
            raise TypeError(
                "tl.tileop.region expects a BufferLoad as arg0, "
                f"but received {type(buffer_load).__name__}."
            )

        buffer = buffer_load.buffer
        backing = self._get_value(buffer)
        buffer_shape = self._static_shape(buffer)
        buffer_strides = self._buffer_strides(buffer, buffer_shape)
        indices = list(buffer_load.indices)
        if len(indices) != len(buffer_strides):
            raise ValueError(
                f"Region rank mismatch for {buffer.name}: "
                f"{len(indices)} indices for rank-{len(buffer_strides)} Buffer."
            )

        raw_extents = list(call.args[2:])
        if len(raw_extents) > len(buffer_strides):
            raise ValueError(
                f"Region rank mismatch for {buffer.name}: "
                f"{len(raw_extents)} extents for rank-{len(buffer_strides)} Buffer."
            )
        region_strides = buffer_strides[-len(raw_extents) :]
        if str(buffer.scope()) == "local.fragment":
            # Fragment regions must keep their unit dims. Squeezing a (1,)
            # fragment region down to rank-0 makes its reaching-def vector rank
            # inconsistent with the fragment's other producers (vmax/vreduce),
            # which breaks the vexpdif RHS type check.
            sizes = [
                self._static_int(extent, f"{buffer.name} extent")
                for extent in raw_extents
            ]
            strides = region_strides
        else:
            sizes, strides = self._squeeze_unit_dims(raw_extents, region_strides)

        offset = self._emit_linear_offset(buffer, indices, buffer_strides)
        dynamic = self._ir.ShapedType.get_dynamic_size()
        if isinstance(offset, int):
            dynamic_offsets = []
            static_offsets = [offset]
            layout_offset = offset
        else:
            dynamic_offsets = [offset]
            static_offsets = [dynamic]
            layout_offset = dynamic

        static_sizes: list[int] = []
        size_operands: list[Any] = []
        for size in sizes:
            if isinstance(size, int):
                static_sizes.append(size)
            else:
                static_sizes.append(dynamic)
                size_operands.append(
                    self._cast_to_index(self._get_or_create_expr_value(size))
                )

        layout = self._ir.StridedLayoutAttr.get(layout_offset, strides)
        result_type = self._ir.MemRefType.get(
            static_sizes,
            self._dtype_type(buffer.dtype),
            layout=layout,
            memory_space=self._memory_scope(buffer),
        )
        if (
            not dynamic_offsets
            and not size_operands
            and self._is_identity_region_view(backing, result_type)
        ):
            self._insert_value(call, backing)
            return backing

        view_op = self._memref.ReinterpretCastOp(
            result_type,
            backing,
            dynamic_offsets,
            size_operands,
            [],
            static_offsets,
            static_sizes,
            strides,
        )
        self._insert_value(call, view_op.result)
        return view_op.result

    @staticmethod
    def _squeeze_unit_dims(
        extents: list[Any],
        strides: list[int],
    ) -> tuple[list[Any], list[int]]:
        sizes: list[Any] = []
        kept_strides: list[int] = []
        for extent, stride in zip(extents, strides):
            static = TileLangIRTranslator._try_static_int(extent)
            if static == 1:
                continue
            sizes.append(static if static is not None else extent)
            kept_strides.append(stride)
        return sizes, kept_strides

    def _emit_vadd(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(f"tl.tileop.vadd expects 3 arguments, but received {len(call.args)}")
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])
        self._linalg.add(src0, src1, outs=[dst])

    def _emit_vmul(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(f"tl.tileop.vmul expects 3 arguments, but received {len(call.args)}")
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])
        self._linalg.mul(src0, src1, outs=[dst])

    def _emit_broadcast(self, call: tirx.Call) -> None:
        if len(call.args) != 2:
            raise ValueError(
                f"tl.tileop.broadcast expects 2 arguments, but received {len(call.args)}"
            )
        src = self._get_or_create_expr_value(call.args[0])
        dst = self._get_or_create_expr_value(call.args[1])
        src_type = self._ir.MemRefType(src.type)
        dst_type = self._ir.MemRefType(dst.type)
        if dst_type.rank < src_type.rank:
            raise ValueError(
                f"tl.tileop.broadcast expects dst rank >= src rank, but got "
                f"{src_type.rank} -> {dst_type.rank}"
            )
        src_shape = list(src_type.shape)
        dst_shape = list(dst_type.shape)
        broadcast_dims = self._infer_broadcast_dims(src_shape, dst_shape)
        self._linalg.broadcast(src, outs=[dst], dimensions=broadcast_dims)
    def _emit_vdiv(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(f"tl.tileop.vdiv expects 3 arguments, but received {len(call.args)}")
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])
        src1_type = self._ir.MemRefType(src1.type)
        dst_type = self._ir.MemRefType(dst.type)

        # 判断是否需要广播：src1 rank < dst rank（仿 _emit_vsub）
        if src1_type.rank < dst_type.rank:
            broadcasted_src1 = self._memref.AllocOp(dst_type, [], []).result
            rank_diff = dst_type.rank - src1_type.rank
            broadcast_dims = list(range(dst_type.rank - rank_diff, dst_type.rank))
            self._linalg.broadcast(
                src1,
                outs=[broadcasted_src1],
                dimensions=broadcast_dims,
            )
            self._linalg.div(src0, broadcasted_src1, outs=[dst])
        else:
            self._linalg.div(src0, src1, outs=[dst])

    def _emit_vsub(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(f"tl.tileop.vsub expects 3 arguments, but received {len(call.args)}")
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])
        src1_type = self._ir.MemRefType(src1.type)
        dst_type = self._ir.MemRefType(dst.type)

        # 判断是否需要广播：src1 rank < dst rank（同 _emit_vdiv）
        if src1_type.rank < dst_type.rank:
            broadcasted_src1 = self._memref.AllocOp(dst_type, [], []).result
            rank_diff = dst_type.rank - src1_type.rank
            broadcast_dims = list(range(dst_type.rank - rank_diff, dst_type.rank))
            self._linalg.broadcast(
                src1,
                outs=[broadcasted_src1],
                dimensions=broadcast_dims,
            )
            self._linalg.sub(src0, broadcasted_src1, outs=[dst])
        else:
            self._linalg.sub(src0, src1, outs=[dst])

    def _emit_vreduce_max(self, call: tirx.Call) -> None:
        if len(call.args) < 2:
            raise ValueError(
                f"tl.tileop.vreduce_max expects at least 2 arguments, but received {len(call.args)}"
            )
        src = self._get_or_create_expr_value(call.args[0])
        dst = self._get_or_create_expr_value(call.args[1])
        dtype_name = self._call_arg_dtype_name(call.args[0])
        explicit_dims = self._explicit_reduce_dims(call.args[2:])
        self._emit_linalg_reduce(src, dst, "maxf", dtype_name, explicit_dims)

    def _emit_vreduce_sum(self, call: tirx.Call) -> None:
        if len(call.args) < 2:
            raise ValueError(
                f"tl.tileop.vreduce_sum expects at least 2 arguments, but received {len(call.args)}"
            )
        src = self._get_or_create_expr_value(call.args[0])
        dst = self._get_or_create_expr_value(call.args[1])
        dtype_name = self._call_arg_dtype_name(call.args[0])
        explicit_dims = self._explicit_reduce_dims(call.args[2:])
        self._emit_linalg_reduce(src, dst, "addf", dtype_name, explicit_dims)

    @staticmethod
    def _explicit_reduce_dims(args: list) -> list[int] | None:
        """Parse trailing IntImm reduce-dim arguments; None when no dim was supplied."""
        if len(args) == 0:
            return None
        dims: list[int] = []
        for arg in args:
            if not isinstance(arg, tirx.IntImm):
                raise ValueError(
                    f"Reduce dim arguments must be IntImm, but received {type(arg).__name__}"
                )
            dims.append(int(arg.value))
        return dims

    def _emit_linalg_reduce(
        self,
        src: Any,
        dst: Any,
        kind: str,
        dtype_name: str,
        explicit_dims: list[int] | None = None,
    ) -> None:
        src_type = self._ir.MemRefType(src.type)
        dst_type = self._ir.MemRefType(dst.type)
        element_type = src_type.element_type
        is_float = isinstance(element_type, self._ir.FloatType)

        src_shape = list(src_type.shape)
        dst_shape = list(dst_type.shape)

        # Infer reduction dimensions by comparing src and dst shapes,
        # instead of hardcoding the last dimension.
        reduce_dims = self._resolve_reduce_dims(src_shape, dst_shape, explicit_dims)

        # Compute the expected init shape (src shape with reduce_dims removed).
        init_shape = [src_shape[i] for i in range(len(src_shape)) if i not in reduce_dims]

        # Create the identity constant value.
        if is_float:
            if kind == "maxf":
                identity_val = float("-inf")
            else:
                identity_val = 0.0
            identity_attr = self._ir.FloatAttr.get(element_type, identity_val)
        else:
            if kind == "maxf":
                if dtype_name.startswith("uint"):
                    identity_val = 0
                else:
                    bit_width = element_type.width
                    identity_val = -(1 << (bit_width - 1))
            else:
                identity_val = 0
            identity_attr = self._ir.IntegerAttr.get(element_type, identity_val)
        identity_const = self._arith.ConstantOp(element_type, identity_attr).result

        # Prepare the init buffer: use dst directly if shapes match,
        # otherwise create a rank-reduced view of dst.
        if init_shape == dst_shape:
            init = dst
        else:
            init = self._create_rank_reduced_view(dst, dst_type, init_shape, element_type)

        # Fill the init buffer with the identity value using linalg.fill.
        self._linalg.fill(identity_const, outs=[init])

        # Create linalg.reduce with memref semantics (no tensor conversions).
        reduce_op = self._ir.Operation.create(
            "linalg.reduce",
            results=[],
            operands=[src, init],
            attributes={"dimensions": self._ir.DenseI64ArrayAttr.get(reduce_dims)},
            regions=1,
        )

        # Build the combiner body.
        body = reduce_op.regions[0].blocks.append(element_type, element_type)
        with self._ir.InsertionPoint(body):
            if is_float:
                if kind == "maxf":
                    val = self._arith.MaximumFOp(body.arguments[0], body.arguments[1]).result
                else:
                    val = self._arith.AddFOp(body.arguments[0], body.arguments[1]).result
            else:
                if kind == "maxf":
                    if dtype_name.startswith("uint"):
                        val = self._arith.MaxUIOp(body.arguments[0], body.arguments[1]).result
                    else:
                        val = self._arith.MaxSIOp(body.arguments[0], body.arguments[1]).result
                else:
                    val = self._arith.AddIOp(body.arguments[0], body.arguments[1]).result
            self._ir.Operation.create("linalg.yield", results=[], operands=[val])

    @staticmethod
    def _resolve_reduce_dims(
        src_shape: list[int], dst_shape: list[int], explicit_dims: list[int] | None
    ) -> list[int]:
        """Resolve reduction dims: explicit dims (negative normalized) or shape inference."""
        if explicit_dims is None:
            return TileLangIRTranslator._infer_reduce_dims(src_shape, dst_shape)

        rank = len(src_shape)
        dims = [d + rank if d < 0 else d for d in explicit_dims]
        for d in dims:
            if d < 0 or d >= rank:
                raise ValueError(f"Reduce dim {d} is out of range for source rank {rank}")
        if len(set(dims)) != len(dims):
            raise ValueError(f"Duplicate reduce dims: {dims}")
        dims.sort()

        expected_shape = [src_shape[i] for i in range(rank) if i not in dims]
        if expected_shape != dst_shape:
            raise ValueError(
                f"Reduce dims {dims} on source shape {src_shape} produce {expected_shape}, "
                f"but destination shape is {dst_shape}"
            )
        return dims

    @staticmethod
    def _infer_reduce_dims(
        src_shape: list[int], dst_shape: list[int]
    ) -> list[int]:
        """Infer which dimensions of src are reduced by comparing with dst shape."""
        if len(dst_shape) == 0:
            # All dimensions are reduced.
            return list(range(len(src_shape)))
        if len(dst_shape) == len(src_shape):
            # Same rank: dims where sizes differ are the reduction dims.
            return [i for i in range(len(src_shape)) if src_shape[i] != dst_shape[i]]
        # Different rank: match from left to right to find kept dims.  The
        # leading convention matches the broadcast convention (a row-reduced
        # (BR, BC) region yields (BR,) and broadcasts back over columns).
        reduce_dims: list[int] = []
        dst_idx = 0
        for src_idx in range(len(src_shape)):
            if dst_idx < len(dst_shape) and src_shape[src_idx] == dst_shape[dst_idx]:
                dst_idx += 1
            else:
                reduce_dims.append(src_idx)
        return reduce_dims

    @staticmethod
    def _infer_broadcast_dims(
        src_shape: list[int], dst_shape: list[int]
    ) -> list[int]:
        """Infer linalg.broadcast dimensions by matching src dims to dst dims.

        src dims are matched in order to dst dims with equal extents; dst dims
        left unmatched are broadcast, and extent-1 src dims are treated as
        broadcastable. This supports both (M,) -> (M, N) and (N,) -> (M, N)
        broadcasts as well as same-rank extent-1 broadcasts like
        (M, 1) -> (M, N) or (1, N) -> (M, N).
        """
        broadcast_dims: list[int] = []
        src_idx = 0
        for dst_idx, dst_extent in enumerate(dst_shape):
            if src_idx >= len(src_shape):
                broadcast_dims.append(dst_idx)
            elif src_shape[src_idx] == dst_extent:
                src_idx += 1  # matched dim, kept as-is
            elif src_shape[src_idx] == 1:
                broadcast_dims.append(dst_idx)  # extent-1 src dim is broadcast
                src_idx += 1
            else:
                broadcast_dims.append(dst_idx)  # no src dim for this dst dim
        if src_idx < len(src_shape):
            raise ValueError(
                f"tl.tileop.broadcast cannot broadcast shape {src_shape} to {dst_shape}"
            )
        return broadcast_dims

    def _create_rank_reduced_view(
        self, dst: Any, dst_type: Any, init_shape: list[int], element_type: Any
    ) -> Any:
        """Create a rank-reduced memref view of dst matching init_shape."""
        memory_space = dst_type.memory_space
        if len(init_shape) == 0:
            # Scalar result: create a rank-0 view of dst's first element.
            init_type = self._ir.MemRefType.get(
                [],
                element_type,
                layout=self._ir.StridedLayoutAttr.get(0, []),
                memory_space=memory_space,
            )
            return self._memref.ReinterpretCastOp(
                init_type, dst, [], [], [], [0], [], []
            ).result
        # Compute contiguous strides for the init shape.
        strides: list[int] = []
        stride = 1
        for size in reversed(init_shape):
            strides.insert(0, stride)
            stride *= size
        init_type = self._ir.MemRefType.get(
            init_shape,
            element_type,
            layout=self._ir.StridedLayoutAttr.get(0, strides),
            memory_space=memory_space,
        )
        return self._memref.ReinterpretCastOp(
            init_type, dst, [], [], [], [0], init_shape, strides
        ).result

    @staticmethod
    def _call_arg_dtype_name(arg: Any) -> str:
        """Extract the dtype name string from a TIRX call argument."""
        if hasattr(arg, "buffer") and hasattr(arg.buffer, "dtype"):
            return str(arg.buffer.dtype)
        if hasattr(arg, "dtype"):
            return str(arg.dtype)
        return "float32"

    def _emit_vexpdif(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(
                f"tl.tileop.vexpdif expects 3 arguments, but received {len(call.args)}"
            )
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])
        VexpdifOp(src0, src1, dst)

    def _emit_copy(self, call: tirx.Call) -> None:
        if len(call.args) < 2:
            raise ValueError(
                f"tl.tileop.copy expects at least 2 arguments, but received {len(call.args)}"
            )
        src = self._get_or_create_expr_value(call.args[0])
        dst = self._get_or_create_expr_value(call.args[1])
        ann = self._call_annotations(call)
        split_dim = self._optional_int(ann.get("split_dim"))
        if split_dim is not None:
            # The NPU CopyOp verifier only accepts 0 (ROW) or 1 (COLUMN);
            # reject every other value instead of guessing an auto split dim.
            if split_dim not in (0, 1):
                raise ValueError(
                    f"split_dim must be 0 (ROW) or 1 (COLUMN), but got {split_dim}"
                )
            # The NPU CopyOp verifier requires static rank-2 memrefs for split
            # copies.
            self._require_rank2_memrefs(src, dst)
        CopyOp(
            src,
            dst,
            split_dim=split_dim,
            transpose=True if self._as_bool(ann.get("transpose", False)) else None,
        )

    def _require_rank2_memrefs(self, src: Any, dst: Any) -> None:
        """Require rank-2 memrefs for split_dim copies."""
        # TODO: only rank-2 memrefs are supported for now.
        for value, name in ((src, "source"), (dst, "destination")):
            rank = self._ir.MemRefType(value.type).rank
            if rank != 2:
                raise ValueError(
                    f"split_dim copy requires rank-2 memrefs, but the {name} "
                    f"operand has rank {rank}"
                )

    def _emit_gemm(self, call: tirx.Call) -> None:
        # T.gemm Call arg layout (see tilelang/language/gemm_op.py):
        #   0: A region, 1: B region, 2: C region,
        #   3: transpose_A (bool), 4: transpose_B (bool),
        #   5: M, 6: N, 7: K, 8: policy,
        #   9: clear_accum (bool), 10-18: strides/offsets/etc (unused by GemmOp)
        if len(call.args) < 10:
            raise ValueError(
                f"tl.tileop.gemm expects at least 10 arguments, but received {len(call.args)}"
            )
        a = self._get_or_create_expr_value(call.args[0])
        b = self._get_or_create_expr_value(call.args[1])
        c = self._get_or_create_expr_value(call.args[2])
        transpose_a = self._as_bool(call.args[3])
        transpose_b = self._as_bool(call.args[4])
        clear_accum = self._as_bool(call.args[9])
        GemmOp(
            a,
            b,
            c,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            clear_accum=clear_accum,
        )
    def _emit_vcvt(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(f"tl.tileop.vcvt expects 3 arguments, but received {len(call.args)}")
        dtype_arg = call.args[2]
        src = self._get_or_create_expr_value(call.args[0])
        dst = self._get_or_create_expr_value(call.args[1])
        src_dtype = self._region_dtype_name(call.args[0])
        if src_dtype is None:
            src_dtype = str(src.type.element_type)
        VcvtOp(src, dst, src_dtype=src_dtype, dst_dtype=dtype_arg.value)

    @staticmethod
    def _region_dtype_name(region_call: Any) -> str | None:
        try:
            if not isinstance(region_call, tirx.Call):
                return None
            buffer_load = region_call.args[0]
            if not isinstance(buffer_load, tirx.BufferLoad):
                return None
            return str(buffer_load.buffer.dtype)
        except (AttributeError, IndexError):
            return None

    def _emit_vmuls(self, call: tirx.Call) -> None:
        if len(call.args) != 3:
            raise ValueError(
                f"tl.tileop.vmuls expects 3 arguments, but received {len(call.args)}"
            )
        src = self._get_or_create_expr_value(call.args[0])
        scalar = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])

        src_type = src.type.element_type
        scalar_type = getattr(scalar.type, "element_type", scalar.type)
        dst_type = dst.type.element_type
        if not (src_type == scalar_type == dst_type):
            raise TypeError(
                f"tl.tileop.vmuls expects src, scalar, and dst to have the same dtype, "
                f"but received {src_type}, {scalar_type}, and {dst_type}"
            )
        VmulsOp(src, scalar, dst)

    def _emit_if_then_else(self, call: tirx.Call) -> Any:
        """Lower the ``tirx.if_then_else`` expression into ``arith.select``.

        TIRX enforces a Bool condition and identical branch dtypes, which is
        exactly the contract of ``arith.select`` (see tvm/tirx/op/op.cc).
        """
        if len(call.args) != 3:
            raise ValueError(
                f"tirx.if_then_else expects 3 arguments, but received {len(call.args)}"
            )
        cond = self._get_or_create_expr_value(call.args[0])
        true_value = self._get_or_create_expr_value(call.args[1])
        false_value = self._get_or_create_expr_value(call.args[2])
        result = self._arith.select(cond, true_value, false_value)
        self._insert_value(call, result)
        return result
    def _emit_vexp(self, call: tirx.Call) -> None:
        if len(call.args) != 2:
            raise ValueError(
                f"tl.tileop.vexp expects 2 arguments, but received {len(call.args)}"
            )
        src = self._get_or_create_expr_value(call.args[0])
        dst = self._get_or_create_expr_value(call.args[1])

        src_type = self._ir.MemRefType(src.type)
        dst_type = self._ir.MemRefType(dst.type)
        if tuple(src_type.shape) != tuple(dst_type.shape):
            raise ValueError(
                "tl.tileop.vexp expects src and dst to have the same shape, "
                f"but received {src_type.shape} and {dst_type.shape}"
            )
        self._linalg.exp(src, outs=[dst])

    def _emit_vmax(self, call: tirx.Call) -> None:
        if len(call.args) < 3:
            raise ValueError(
                f"tl.tileop.vmax expects at least 3 arguments, but received {len(call.args)}"
            )
        src0 = self._get_or_create_expr_value(call.args[0])
        src1 = self._get_or_create_expr_value(call.args[1])
        dst = self._get_or_create_expr_value(call.args[2])

        # 无符号信息仅在 TIRX 层保留（_dtype_type 把 int/uint 都压成 signless）。
        # 从 tl.tileop.region Call 解包出 BufferLoad，读取原始 TIRX dtype。
        tirx_dtype = self._region_dtype_name(call.args[0])
        if tirx_dtype is None:
            tirx_dtype = str(dst.type.element_type)
        VmaxOp(src0, src1, dst, dtype=tirx_dtype)

    def _cast_to_index(self, value: Any) -> Any:
        index_type = self._ir.IndexType.get()
        if value.type == index_type:
            return value
        if isinstance(value.type, self._ir.IntegerType):
            return self._arith.IndexCastOp(index_type, value).result
        raise TypeError(f"Cannot cast MLIR value of type {value.type} to index.")

    def _unify_integer_operands(self, lhs: Any, rhs: Any) -> tuple[Any, Any]:
        if lhs.type == rhs.type:
            return lhs, rhs
        index_type = self._ir.IndexType.get()
        if lhs.type == index_type or rhs.type == index_type:
            return self._cast_to_index(lhs), self._cast_to_index(rhs)
        raise TypeError(
            f"Cannot unify MLIR integer operand types {lhs.type} and {rhs.type}."
        )

    def _memory_scope(self, buffer: Any) -> Any:
        scope = str(buffer.scope())
        if scope in ("", "global"):
            address_space = 0
        elif scope in ("shared", "shared.dyn"):
            address_space = 1
        elif scope == "local.fragment":
            address_space = 2
        else:
            raise NotImplementedError(
                f"TileLangIR has no memory-space mapping for TIRX scope {scope!r}."
            )
        return self._ir.IntegerAttr.get(self._ir.IntegerType.get_signless(64), address_space)

    @staticmethod
    def _num_elements(shape: list[int]) -> int:
        count = 1
        for extent in shape:
            count *= extent
        return count

    @staticmethod
    def _dtype_nbytes(dtype: Any) -> int:
        bits = int(dtype.bits) * int(dtype.lanes)
        return max(1, (bits + 7) // 8)

    def _dtype_type(self, dtype: Any) -> Any:
        dtype_name = str(dtype)
        if dtype_name == "bool":
            return self._ir.IntegerType.get_signless(1)
        if dtype_name == "bfloat16":
            return self._ir.BF16Type.get()
        if dtype_name == "float16":
            return self._ir.F16Type.get()
        if dtype_name == "float32":
            return self._ir.F32Type.get()
        if dtype_name == "float64":
            return self._ir.F64Type.get()
        if dtype_name.startswith(("int", "uint")):
            prefix_length = 3 if dtype_name.startswith("int") else 4
            return self._ir.IntegerType.get_signless(int(dtype_name[prefix_length:]))
        raise TypeError(f"Unsupported TIRX dtype in TileLangIR codegen: {dtype_name}.")

    def _static_shape(self, buffer: Any) -> list[int]:
        return [self._static_int(extent, f"{buffer.name} shape") for extent in buffer.shape]

    def _buffer_strides(self, buffer: Any, shape: list[int]) -> list[int]:
        if buffer.strides:
            return [self._static_int(stride, f"{buffer.name} stride") for stride in buffer.strides]

        strides = [1] * len(shape)
        for index in range(len(shape) - 2, -1, -1):
            strides[index] = strides[index + 1] * shape[index + 1]
        return strides

    @staticmethod
    def _static_int(expr: Any, description: str) -> int:
        value = TileLangIRTranslator._try_static_int(expr)
        if value is not None:
            return value
        raise NotImplementedError(
            f"Current TileLangIR storage lowering requires static {description}, but got {expr}."
        )

    @staticmethod
    def _try_static_int(expr: Any) -> int | None:
        if isinstance(expr, tirx.IntImm):
            return int(expr.value)
        if isinstance(expr, int):
            return expr
        return None

    @staticmethod
    def _call_op_name(call: tirx.Call) -> str:
        return call.op.name

    #############
    # value_map helper methods
    #############
    # Insert a mapping from a TIRX object into value_map
    def _insert_value(self, tirx_object: object, mlir_value: Any) -> None:
        key = tirx_object.id_
        if key in self.value_map:
            if self.value_map[key] == mlir_value:
                return
            raise ValueError(f"TIRX value is already present in value_map: {tirx_object}")
        self.value_map[key] = mlir_value

    # Get the MLIR Value mapped to a TIRX object
    def _get_value(self, tirx_object: object) -> Any:
        key = tirx_object.id_
        try:
            return self.value_map[key]
        except KeyError as exc:
            raise KeyError(f"TIRX value has not been lowered yet: {tirx_object}") from exc

    def _get_or_create_expr_value(self, expr: tirx.PrimExpr) -> Any:
        """Get or emit the MLIR Value represented by a TIRX PrimExpr."""

        key = expr.id_
        if key in self.value_map:
            return self._get_value(expr)

        self.visit_expr(expr)

        try:
            return self._get_value(expr)
        except KeyError as exc:
            raise NotImplementedError(
                "TIRX expression lowering did not produce an MLIR Value for "
                f"{type(expr).__name__}: {expr}"
            ) from exc

    @contextmanager
    def _scoped_value_map(self) -> Generator[None, None, None]:
        """Lexically scope the value_map to the enclosing ``with`` block.

        Values created inside a child region (e.g. a ``tilelang.scope`` or
        ``scf.for`` body) must not be referenced by the enclosing block,
        otherwise an SSA value defined in the child region could be used where
        it does not dominate the use site. Save the current key set on entry
        and drop every newly-added key on exit, even if an exception
        propagates.
        """
        saved_keys = set(self.value_map)
        try:
            yield
        finally:
            for key in list(self.value_map):
                if key not in saved_keys:
                    del self.value_map[key]

    @contextmanager
    def _execution_scope(self, mode: str) -> Generator[None, None, None]:
        """Record the execution mode while translating a scope body."""

        self._execution_scope_modes.append(mode)
        try:
            yield
        finally:
            self._execution_scope_modes.pop()

    #############
    # Overridden visitor methods
    #############
    # Handle different TIRX node types; the override method names are fixed
    # The following methods are dispatched by visit_stmt.
    def visit_sblock_realize_(self, op: tirx.SBlockRealize) -> None:
        self.visit_stmt(op.block)

    def visit_sblock_(self, op: tirx.SBlock) -> None:
        name = str(getattr(op, "name_hint", "") or "")
        annotations = getattr(op, "annotations", {})
        mode_value = annotations.get("tl.execution_scope")
        mode = str(mode_value) if mode_value is not None else None

        # Keep accepting older SimdVF blocks while callers migrate to the
        # explicit execution-scope annotation contract.
        if mode is None and name == "SimdVF":
            mode = "simd"

        if mode is not None:
            if mode not in ("simd", "simt"):
                raise ValueError(
                    "Unsupported TileLang execution scope mode "
                    f"{mode!r} in block {name!r}."
                )

            threads = None
            if mode == "simt":
                raw_threads = annotations.get("tl.simt_threads")
                if raw_threads is None:
                    raise ValueError("A SIMT execution scope requires tl.simt_threads")
                threads = self._static_int(raw_threads, "SIMT threads")
                if not 1 <= threads <= (1 << 32) - 1:
                    raise ValueError(
                        "SIMT threads must be in the uint32 range [1, 4294967295]"
                    )

            scope_op = ScopeOp(mode=mode, threads=threads)
            body_block = self._ir.Block.create_at_start(scope_op.body)
            with (
                self._ir.InsertionPoint(body_block),
                self._scoped_value_map(),
                self._execution_scope(mode),
            ):
                for buffer in op.alloc_buffers:
                    self._emit_alloc_buffer(buffer)
                self.visit_stmt(op.body)
            return

        for buffer in op.alloc_buffers:
            self._emit_alloc_buffer(buffer)
        self.visit_stmt(op.body)

    def visit_for_(self, op: tirx.For) -> None:
        parallel_rank = self._simt_parallel_rank(op)
        if parallel_rank is not None:
            self._emit_simt_forall(op, parallel_rank)
            return

        index_type = self._ir.IndexType.get()
        lower = self._cast_to_index(self._get_or_create_expr_value(op.min))
        extent = self._cast_to_index(self._get_or_create_expr_value(op.extent))
        upper = self._arith.AddIOp(lower, extent).result
        if op.step is not None:
            step = self._cast_to_index(self._get_or_create_expr_value(op.step))
        else:
            step = self._arith.ConstantOp(
                index_type, self._ir.IntegerAttr.get(index_type, 1)
            ).result

        for_op = self._scf.ForOp(lower, upper, step)
        loop_kind = self._loop_kind_attr(op)
        if loop_kind is not None:
            for_op.attributes["tilelang.loop_kind"] = loop_kind
        self._copy_loop_annotations(op, for_op)

        # Scope the value_map to this region. TIRX may alias the same node (e.g.
        # an interned IntImm loop bound) at several tree positions. Values created
        # inside the scf.for body must not be reused outside it, otherwise an
        # arith.index_cast emitted in a parent region could reference an operand
        # defined in this (child) region, violating SSA dominance.
        body_keys = set(self.value_map)
        self._insert_value(op.loop_var, for_op.induction_variable)
        with self._ir.InsertionPoint(for_op.body):
            self.visit_stmt(op.body)
            self._scf.YieldOp([])
        for key in list(self.value_map):
            if key not in body_keys:
                del self.value_map[key]

    def _simt_parallel_rank(self, op: tirx.For) -> int | None:
        """Infer the rank of a perfect SIMT parallel loop nest."""

        if (
            not self._execution_scope_modes
            or self._execution_scope_modes[-1] != "simt"
        ):
            return None
        if op.kind != tirx.ForKind.PARALLEL:
            return None

        # TIRX represents T.Parallel(X, Y, Z) as a perfect nest of parallel
        # For nodes. During rapid bring-up, a directly nested sequence of
        # one-dimensional T.Parallel calls has the same logical-thread meaning,
        # so infer the domain structurally and avoid a frontend-only rank marker.
        rank = 1
        current = op
        while True:
            nested = self._unwrap_single_stmt(current.body)
            if (
                not isinstance(nested, tirx.For)
                or nested.kind != tirx.ForKind.PARALLEL
            ):
                break
            rank += 1
            if rank > 3:
                raise ValueError(
                    "T.Parallel inside T.SimtVF currently supports one to three "
                    f"logical thread dimensions, but received rank {rank}."
                )
            current = nested
        return rank

    @staticmethod
    def _unwrap_single_stmt(stmt: Any) -> Any:
        """Remove a one-element SeqStmt introduced by TIRX normalization."""

        if isinstance(stmt, tirx.SeqStmt) and len(stmt.seq) == 1:
            return stmt.seq[0]
        return stmt

    def _emit_simt_forall(self, outer: tirx.For, rank: int) -> None:
        """Emit one target-neutral logical thread domain for ``T.Parallel``."""

        loops: list[tirx.For] = []
        current: Any = outer
        for dimension in range(rank):
            if (
                not isinstance(current, tirx.For)
                or current.kind != tirx.ForKind.PARALLEL
            ):
                raise ValueError(
                    "Malformed SIMT parallel loop nest at logical dimension "
                    f"{dimension}."
                )
            loops.append(current)
            if dimension + 1 < rank:
                current = self._unwrap_single_stmt(current.body)

        logical_body = loops[-1].body
        lower_bounds = []
        upper_bounds = []
        steps = []
        index_type = self._ir.IndexType.get()
        for loop in loops:
            lower = self._cast_to_index(self._get_or_create_expr_value(loop.min))
            extent = self._cast_to_index(
                self._get_or_create_expr_value(loop.extent)
            )
            lower_bounds.append(lower)
            upper_bounds.append(self._arith.AddIOp(lower, extent).result)
            if loop.step is not None:
                steps.append(
                    self._cast_to_index(self._get_or_create_expr_value(loop.step))
                )
            else:
                steps.append(
                    self._arith.ConstantOp(
                        index_type, self._ir.IntegerAttr.get(index_type, 1)
                    ).result
                )

        forall_op = self._scf.ForallOp(lower_bounds, upper_bounds, steps)
        forall_op.attributes["tilelang.loop_kind"] = self._ir.StringAttr.get(
            "parallel"
        )
        forall_op.attributes["tilelang.logical_thread_axes"] = self._ir.ArrayAttr.get(
            [self._ir.StringAttr.get(axis) for axis in ("x", "y", "z")[:rank]]
        )

        body_keys = set(self.value_map)
        for loop, induction_variable in zip(loops, forall_op.induction_variables):
            self._insert_value(loop.loop_var, induction_variable)
        with self._ir.InsertionPoint(forall_op.body):
            self.visit_stmt(logical_body)
            self._scf.InParallelOp()
        for key in list(self.value_map):
            if key not in body_keys:
                del self.value_map[key]

    def _loop_kind_attr(self, op: tirx.For) -> Any:
        """Classify a TIRX For into a ``tilelang.loop_kind`` string attribute.

        Pipelined loops are represented in TIRX as Serial loops carrying
        ``num_stages``/``tl_pipeline_*`` annotations (see README.md).
        """
        kind = op.kind
        if kind == tirx.ForKind.SERIAL:
            if op.annotations and "num_stages" in op.annotations:
                return self._ir.StringAttr.get("pipelined")
            return self._ir.StringAttr.get("serial")
        if kind == tirx.ForKind.PARALLEL:
            return self._ir.StringAttr.get("parallel")
        if kind == tirx.ForKind.VECTORIZED:
            return self._ir.StringAttr.get("vectorized")
        if kind == tirx.ForKind.UNROLLED:
            return self._ir.StringAttr.get("unrolled")
        # THREAD_BINDING is handled by tilelang.launch_thread, not scf.for.
        return None

    _PIPELINE_ANNOTATION_KEYS = (
        "num_stages",
        "tl_pipeline_order",
        "tl_pipeline_stage",
        "tl_pipeline_group",
    )

    def _copy_loop_annotations(self, op: tirx.For, for_op: Any) -> None:
        """Mirror pipeline and loop annotations onto the scf.for operation."""
        if not op.annotations:
            return
        for key in self._PIPELINE_ANNOTATION_KEYS:
            if key not in op.annotations:
                continue
            try:
                for_op.attributes[key] = self._annotation_value_to_attr(op.annotations[key])
            except NotImplementedError:
                continue

    def _annotation_value_to_attr(self, value: Any) -> Any:
        if isinstance(value, tirx.IntImm):
            return self._ir.IntegerAttr.get(
                self._ir.IntegerType.get_signless(64), int(value.value)
            )
        if isinstance(value, tirx.FloatImm):
            return self._ir.FloatAttr.get(self._ir.F64Type.get(), float(value.value))
        # tirx.Array and Python list/tuple both iterate element-wise.
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            elements = [self._annotation_value_to_attr(v) for v in value]
            return self._ir.ArrayAttr.get(elements)
        raise NotImplementedError(
            f"TileLangIR loop annotation lowering does not support {type(value).__name__}: {value}"
        )

    def visit_attr_stmt_(self, op: tirx.AttrStmt) -> None:
        self.visit_stmt(op.body)

    # The following methods are dispatched by visit_expr.
    def visit_var_(self, op: tirx.Var) -> None:
        self._get_value(op)

    def visit_int_imm_(self, op: tirx.IntImm) -> None:
        if op.id_ in self.value_map:
            return
        result_type = self._dtype_type(op.dtype)
        value = self._arith.ConstantOp(
            result_type,
            self._ir.IntegerAttr.get(result_type, int(op.value)),
        ).result
        self._insert_value(op, value)

    def visit_float_imm_(self, op: tirx.FloatImm) -> None:
        if op.id_ in self.value_map:
            return
        result_type = self._dtype_type(op.dtype)
        value = self._arith.ConstantOp(
            result_type,
            self._ir.FloatAttr.get(result_type, float(op.value)),
        ).result
        self._insert_value(op, value)

    def visit_buffer_load_(self, op: tirx.BufferLoad) -> None:
        """
        Lower TIRX BufferLoad:

            buffer[i, j]

        into:

            memref.load %buffer[%i, %j]
        """

        buffer = op.buffer

        memref_value = self._get_value(buffer)

        indices = [
            self._cast_to_index(
                self._get_or_create_expr_value(index)
            )
            for index in op.indices
        ]

        loaded = self._memref.LoadOp(
            memref_value,
            indices,
        ).result

        self._insert_value(op, loaded)


    def visit_buffer_store_(self, op: tirx.BufferStore) -> None:
        """
        Lower TIRX BufferStore:

            buffer[i, j] = value

        into:

            memref.store value, buffer[%i, %j]
        """

        buffer = op.buffer

        memref_value = self._get_value(buffer)

        value = self._get_or_create_expr_value(op.value)

        indices = [
            self._cast_to_index(
                self._get_or_create_expr_value(index)
            )
            for index in op.indices
        ]

        self._memref.StoreOp(
            value,
            memref_value,
            indices,
        )

    def _arith_binary_op(
        self, dtype: Any, lhs: Any, rhs: Any, integer_op: str, float_op: str
    ) -> Any:
        """Emit an arith binary op, selecting the integer or float variant by the TIRX dtype."""
        is_float = isinstance(self._dtype_type(dtype), self._ir.FloatType)
        if not is_float:
            lhs, rhs = self._unify_integer_operands(lhs, rhs)
        name = float_op if is_float else integer_op
        return getattr(self._arith, name)(lhs, rhs).result

    def visit_add_(self, op: tirx.Add) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "AddIOp", "AddFOp"))

    def visit_sub_(self, op: tirx.Sub) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "SubIOp", "SubFOp"))

    def visit_mul_(self, op: tirx.Mul) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "MulIOp", "MulFOp"))

    def visit_div_(self, op: tirx.Div) -> None:
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        self._insert_value(op, self._arith_binary_op(op.dtype, lhs, rhs, "DivSIOp", "DivFOp"))

    # NOTE: The dispatch method names of tirx PyStmtExprVisitor are explicitly
    # registered by functor.py via _tvm_metadata["methods"]; FloorMod maps to
    # visit_floor_mod_ (with a trailing underscore).
    def visit_floor_mod_(self, op: tirx.FloorMod) -> None:
        if isinstance(self._dtype_type(op.dtype), self._ir.FloatType):
            raise NotImplementedError(f"FloorMod on float operands is not supported: {op}")
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        lhs, rhs = self._unify_integer_operands(lhs, rhs)
        # TIR floormod follows floor semantics (the sign of the result follows
        # the divisor), whereas arith.remsi uses truncation semantics (the sign
        # of the result follows the dividend). When the remainder is nonzero and
        # its sign differs from that of the divisor, rem + rhs yields the floor
        # result.
        rem = self._arith.RemSIOp(lhs, rhs).result
        zero = self._arith.ConstantOp(
            rem.type, self._ir.IntegerAttr.get(rem.type, 0)
        ).result
        rem_nonzero = self._arith.cmpi(self._arith.CmpIPredicate.ne, rem, zero)
        rem_negative = self._arith.cmpi(self._arith.CmpIPredicate.slt, rem, zero)
        rhs_negative = self._arith.cmpi(self._arith.CmpIPredicate.slt, rhs, zero)
        sign_mismatch = self._arith.XOrIOp(rem_negative, rhs_negative).result
        adjust = self._arith.AndIOp(rem_nonzero, sign_mismatch).result
        adjusted = self._arith.AddIOp(rem, rhs).result
        self._insert_value(op, self._arith.select(adjust, adjusted, rem))

    def visit_floor_div_(self, op: tirx.FloorDiv) -> None:
        if isinstance(self._dtype_type(op.dtype), self._ir.FloatType):
            raise NotImplementedError(f"FloorDiv on float operands is not supported: {op}")
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        lhs, rhs = self._unify_integer_operands(lhs, rhs)
        # TIR floordiv follows floor semantics (rounding toward negative
        # infinity), whereas arith.divsi uses truncation semantics (rounding
        # toward zero). When the remainder is nonzero and its sign differs from
        # that of the divisor, the quotient must be decremented by one.
        quotient = self._arith.DivSIOp(lhs, rhs).result
        rem = self._arith.RemSIOp(lhs, rhs).result
        zero = self._arith.ConstantOp(
            rem.type, self._ir.IntegerAttr.get(rem.type, 0)
        ).result
        one = self._arith.ConstantOp(
            quotient.type, self._ir.IntegerAttr.get(quotient.type, 1)
        ).result
        rem_nonzero = self._arith.cmpi(self._arith.CmpIPredicate.ne, rem, zero)
        rem_negative = self._arith.cmpi(self._arith.CmpIPredicate.slt, rem, zero)
        rhs_negative = self._arith.cmpi(self._arith.CmpIPredicate.slt, rhs, zero)
        sign_mismatch = self._arith.XOrIOp(rem_negative, rhs_negative).result
        adjust = self._arith.AndIOp(rem_nonzero, sign_mismatch).result
        adjusted = self._arith.SubIOp(quotient, one).result
        self._insert_value(op, self._arith.select(adjust, adjusted, quotient))

    def _emit_comparison(
        self, op: tirx.PrimExpr, int_pred: str, float_pred: str
    ) -> None:
        """Lower a TIRX comparison (EQ/NE/LT/...) into arith.cmpi/cmpf.

        The predicate kind follows the operand dtype: ``op.dtype`` is the
        boolean result type, not the operand type, so floatness is decided
        from ``op.a.dtype``.
        """
        lhs = self._get_or_create_expr_value(op.a)
        rhs = self._get_or_create_expr_value(op.b)
        operand_dtype = op.a.dtype
        is_float = isinstance(self._dtype_type(operand_dtype), self._ir.FloatType)
        if is_float:
            predicate = getattr(self._arith.CmpFPredicate, float_pred)
            result = self._arith.cmpf(predicate, lhs, rhs)
        else:
            lhs, rhs = self._unify_integer_operands(lhs, rhs)
            predicate = getattr(self._arith.CmpIPredicate, int_pred)
            result = self._arith.cmpi(predicate, lhs, rhs)
        self._insert_value(op, result)

    def visit_eq_(self, op: tirx.EQ) -> None:
        self._emit_comparison(op, "eq", "OEQ")

    def visit_ne_(self, op: tirx.NE) -> None:
        self._emit_comparison(op, "ne", "UNE")

    def visit_lt_(self, op: tirx.LT) -> None:
        self._emit_comparison(op, "slt", "OLT")

    def visit_le_(self, op: tirx.LE) -> None:
        self._emit_comparison(op, "sle", "OLE")

    def visit_gt_(self, op: tirx.GT) -> None:
        self._emit_comparison(op, "sgt", "OGT")

    def visit_ge_(self, op: tirx.GE) -> None:
        self._emit_comparison(op, "sge", "OGE")

    def visit_if_then_else_(self, op: tirx.IfThenElse) -> None:
        """Lower a statement-level TIRX IfThenElse into ``scf.if``.

        The condition must already lower to an i1 value (comparisons produce
        i1 via arith.cmpi/cmpf). The then/else bodies are translated inside
        fresh scoped value maps so region-local SSA values cannot escape
        their non-dominating region.
        """
        cond = self._get_or_create_expr_value(op.condition)
        if str(cond.type) != "i1":
            raise NotImplementedError(
                "Statement-level if condition must lower to an i1 value, got "
                f"{cond.type}: {op.condition}"
            )
        has_else = op.else_case is not None
        if_op = self._scf.IfOp(cond, has_else=has_else)
        with self._ir.InsertionPoint(if_op.then_block), self._scoped_value_map():
            self.visit_stmt(op.then_case)
            self._scf.YieldOp([])
        if has_else:
            with self._ir.InsertionPoint(if_op.else_block), self._scoped_value_map():
                self.visit_stmt(op.else_case)
                self._scf.YieldOp([])

    # Add all handlers for native TIRX expressions here.

    # Operations registered under src/op are represented as Call nodes.
    def visit_call_(self, op: tirx.Call) -> None:
        op_name = self._call_op_name(op)
        if op_name == "tl.tileop.region":
            self._emit_region(op)
        elif op_name == "tl.tileop.vadd":
            self._emit_vadd(op)
        elif op_name == "tl.tileop.vmul":
            self._emit_vmul(op)
        elif op_name == "tl.tileop.broadcast":
            self._emit_broadcast(op)
        elif op_name == "tl.tileop.vdiv":
            self._emit_vdiv(op)
        elif op_name == "tl.tileop.vsub":
            self._emit_vsub(op)
        elif op_name == "tl.tileop.vreduce_max":
            self._emit_vreduce_max(op)
        elif op_name == "tl.tileop.vreduce_sum":
            self._emit_vreduce_sum(op)
        elif op_name == "tl.tileop.vexpdif":
            self._emit_vexpdif(op)
        elif op_name == "tl.tileop.copy":
            self._emit_copy(op)
        elif op_name == "tl.tileop.gemm":
            self._emit_gemm(op)
        elif op_name == "tirx.if_then_else":
            self._emit_if_then_else(op)
        elif op_name == "tl.tileop.vexp":
            self._emit_vexp(op)
        elif op_name == "tl.tileop.vmax":
            self._emit_vmax(op)
        elif op_name == "tl.tileop.vmuls":
            self._emit_vmuls(op)
        elif op_name == "tl.tileop.fill":
            self._emit_fill(op)
        elif op_name == "tl.infinity":
            self._emit_infinity(op)
        elif op_name == "tl.tileop.vcvt":
            self._emit_vcvt(op)
    def _emit_fill(self, op: tirx.Call) -> None:
        if len(op.args) != 2:
            raise ValueError(f"tl.tileop.fill expects 2 arguments, but received {len(op.args)}")

        # arg0 is a tl.tileop.region Call node that has not yet been lowered; you must use _get_or_create_expr_value.
        # First trigger _emit_region to generate a memref view, then obtain its Value (consistent with _emit_vadd).
        # arg1 is the fill value.
        buffer = self._get_or_create_expr_value(op.args[0])
        value = self._get_or_create_expr_value(op.args[1])
        self._linalg.fill(value, outs=[buffer])

    def _emit_infinity(self, op: tirx.Call) -> None:
        if len(op.args) != 1:
            raise ValueError(f"tl.infinity expects 1 argument, but received {len(op.args)}")

        # op.args[0] is a StringImm (e.g., "float16"), and its .dtype is "handle",
        # which cannot be used as the element type. The actual element dtype comes
        # from the call's return dtype (op.dtype). This is consistent with the
        # approach used in visit_float_imm_, which also uses op.dtype.
        result_type = self._dtype_type(op.dtype)
        inf_attr = self._ir.FloatAttr.get(result_type, float("inf"))
        value = self._arith.ConstantOp(result_type, inf_attr).result
        self._insert_value(op, value)

    @staticmethod
    def _call_annotations(call: tirx.Call) -> dict[str, Any]:
        ann = getattr(call, "annotations", None)
        if ann is None:
            attrs = getattr(call, "attrs", None)
            if attrs is None:
                return {}
            try:
                return dict(attrs)
            except TypeError:
                return {}
        try:
            return dict(ann)
        except TypeError:
            return {}

    @staticmethod
    def _as_bool(expr: Any, default: bool = False) -> bool:
        if isinstance(expr, tirx.IntImm):
            return bool(expr.value)
        if isinstance(expr, bool):
            return expr
        return default

    @staticmethod
    def _optional_int(expr: Any) -> int | None:
        if expr is None:
            return None
        if isinstance(expr, tirx.IntImm):
            return int(expr.value)
        if isinstance(expr, int):
            return expr
        return None
