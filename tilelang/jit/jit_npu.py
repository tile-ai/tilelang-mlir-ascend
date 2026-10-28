"""Compatibility layer for the legacy ``tilelang.jit.jit_npu`` module.

Before the v0.1.15 rebase the Ascend backend lived in this module.  The rebased
tree implements it in :mod:`tilelang.ascend910`; the dynamic-shape helpers and
the argument packing rules are kept here under their historical names so the
existing test-suite keeps importing them.
"""

from __future__ import annotations

from tvm import tirx as tir

from tilelang.ascend910.jit import (  # noqa: F401
    _collect_tir_vars,
    _detect_affine,
    _eval_tir_expr,
)

__all__ = [
    "_collect_tir_vars",
    "_detect_affine",
    "_eval_tir_expr",
    "_process_dynamic_symbolic",
    "_symbolic_var_promoter_pass",
    "_transform_stmt",
]


def _transform_stmt(stmt, symbolic_var_names):
    """Rebuild *stmt* without the ``Bind``s that define symbolic shape vars."""

    if stmt is None:
        return None

    if isinstance(stmt, tir.SBlock):
        new_body = _transform_stmt(stmt.body, symbolic_var_names)
        if new_body is None:
            return None
        return tir.SBlock(
            stmt.iter_vars,
            stmt.reads,
            stmt.writes,
            stmt.name_hint,
            new_body,
            stmt.init,
            stmt.alloc_buffers,
            getattr(stmt, "match_buffers", None),
            getattr(stmt, "annotations", None),
            getattr(stmt, "span", None),
        )

    if isinstance(stmt, tir.SeqStmt):
        kept = [
            transformed for transformed in (_transform_stmt(inner, symbolic_var_names) for inner in stmt.seq) if transformed is not None
        ]
        return tir.SeqStmt(kept) if kept else None

    bind_cls = getattr(tir, "Bind", None)
    if bind_cls is not None and isinstance(stmt, bind_cls):
        name = getattr(getattr(stmt, "var", None), "name", None)
        return None if name in symbolic_var_names else stmt

    return stmt


def _process_dynamic_symbolic(func):
    """Map every shape ``Var`` to the buffer dim that can recover it at runtime."""

    params = list(func.params)
    buffer_map = func.buffer_map
    dynamic_symbolic_map = {}

    all_vars = []
    seen_var_names = set()
    for param in params:
        if param not in buffer_map:
            continue
        buffer = buffer_map[param]
        for shape in buffer.shape:
            for var in _collect_tir_vars(shape):
                if var.name not in seen_var_names:
                    seen_var_names.add(var.name)
                    all_vars.append(var)

    for var in all_vars:
        if var in params:
            continue

        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var) and shape.same_as(var):
                    dynamic_symbolic_map[var] = (i, j)
                    break
            if var in dynamic_symbolic_map:
                break

        if var in dynamic_symbolic_map:
            continue

        for i, param in enumerate(params):
            if param not in buffer_map:
                continue
            buffer = buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var):
                    continue
                affine = _detect_affine(shape, var)
                if affine is not None and affine[0] != 0:
                    a, b = affine
                    dynamic_symbolic_map[var] = (i, j, a, b)
                    break
            if var in dynamic_symbolic_map:
                break

    return dynamic_symbolic_map


def _symbolic_var_promoter_pass(func):
    """Append symbolic shape vars as explicit scalar params (legacy contract)."""

    dynamic_symbolic_map = _process_dynamic_symbolic(func)
    symbolic_vars = list(dynamic_symbolic_map.keys())
    if not symbolic_vars:
        return func, {}

    new_params = list(func.params) + symbolic_vars
    new_body = _transform_stmt(func.body, {str(v) for v in symbolic_vars})

    new_primfunc = tir.PrimFunc(
        params=new_params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
        span=func.span,
    )
    return new_primfunc, dynamic_symbolic_map
