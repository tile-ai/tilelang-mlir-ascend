"""Unit test for composite dynamic shape support in _process_dynamic_symbolic."""

import tilelang.language as T
from tvm import tir
from tilelang.jit.jit_npu import (
    _collect_tir_vars,
    _detect_affine,
    _eval_tir_expr,
    _process_dynamic_symbolic,
    _symbolic_var_promoter_pass,
)


def test_helpers():
    batch_size = tir.Var("batch_size", "int32")

    # _collect_tir_vars
    vars_found = _collect_tir_vars(batch_size + 1)
    assert len(vars_found) == 1 and vars_found[0].name == "batch_size"

    vars_found = _collect_tir_vars(batch_size * 2 + 3)
    assert len(vars_found) == 1 and vars_found[0].name == "batch_size"

    # _detect_affine
    assert _detect_affine(batch_size + 1, batch_size) == (1, 1)
    assert _detect_affine(batch_size * 2 + 3, batch_size) == (2, 3)
    assert _detect_affine(batch_size, batch_size) == (1, 0)
    assert _detect_affine(tir.IntImm("int32", 42), batch_size) == (0, 42)
    assert _detect_affine(batch_size - 5, batch_size) == (1, -5)

    # _eval_tir_expr
    dynamic_val = {"batch_size": 7}
    assert _eval_tir_expr(batch_size + 1, dynamic_val) == 8
    assert _eval_tir_expr(batch_size * 2, dynamic_val) == 14
    assert _eval_tir_expr(batch_size * 2 + 3, dynamic_val) == 17

    print("[PASS] test_helpers")


def test_process_dynamic_symbolic_composite():
    """Test that _process_dynamic_symbolic detects vars inside composite expressions."""
    raw_batch_size = T.dynamic("raw_batch_size")
    cp_batch_size = T.dynamic("cp_batch_size")

    @T.prim_func
    def kernel(
        ht_buffer: T.Tensor([cp_batch_size, 2, 128, 128], "float16"),
        seq_map_r2c: T.Tensor([raw_batch_size + 1], "int32"),
        cp_h0: T.Tensor([cp_batch_size, 2, 128, 128], "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            seq_map_r2c[0] = seq_map_r2c[0]

    dyn_map = _process_dynamic_symbolic(kernel)

    # raw_batch_size should be registered with affine binding
    raw_entries = [
        (str(k), v) for k, v in dyn_map.items() if str(k) == "raw_batch_size"
    ]
    assert len(raw_entries) == 1, (
        f"Expected 1 entry for raw_batch_size, got {len(raw_entries)}"
    )
    _, raw_pos = raw_entries[0]
    assert len(raw_pos) == 4, (
        f"Expected affine binding (4-tuple), got {len(raw_pos)}-tuple"
    )
    assert raw_pos[2] == 1 and raw_pos[3] == 1, (
        f"Expected a=1, b=1, got a={raw_pos[2]}, b={raw_pos[3]}"
    )

    # cp_batch_size should be registered with direct binding
    cp_entries = [(str(k), v) for k, v in dyn_map.items() if str(k) == "cp_batch_size"]
    assert len(cp_entries) == 1
    _, cp_pos = cp_entries[0]
    assert len(cp_pos) == 2, (
        f"Expected direct binding (2-tuple), got {len(cp_pos)}-tuple"
    )

    print("[PASS] test_process_dynamic_symbolic_composite")


def test_process_dynamic_symbolic_prefer_direct():
    """When a var has both a direct and a composite binding, prefer direct."""
    batch_size = T.dynamic("batch_size")

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_size], "float16"),
        B: T.Tensor([batch_size + 1], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            A[0] = A[0]

    dyn_map = _process_dynamic_symbolic(kernel)

    bs_entries = [(str(k), v) for k, v in dyn_map.items() if str(k) == "batch_size"]
    assert len(bs_entries) == 1
    _, pos = bs_entries[0]
    assert len(pos) == 2, f"Expected direct binding preferred, got {len(pos)}-tuple"

    print("[PASS] test_process_dynamic_symbolic_prefer_direct")


def test_symbolic_var_promoter_with_composite():
    """Test that _symbolic_var_promoter_pass correctly promotes vars from composite shapes."""
    raw_batch_size = T.dynamic("raw_batch_size")

    @T.prim_func
    def kernel(
        seq_map_r2c: T.Tensor([raw_batch_size + 1], "int32"),
        out: T.Tensor([raw_batch_size + 1], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            seq_map_r2c[0] = seq_map_r2c[0]

    new_func, new_map = _symbolic_var_promoter_pass(kernel)
    assert len(new_func.params) == len(kernel.params) + 1, (
        "Should promote 1 symbolic var"
    )
    assert any(str(k) == "raw_batch_size" for k in new_map.keys())

    print("[PASS] test_symbolic_var_promoter_with_composite")


def test_runtime_resolution():
    """Simulate runtime resolution: given actual_dim, compute var value via affine inverse."""
    raw_batch_size = T.dynamic("raw_batch_size")

    @T.prim_func
    def kernel(
        seq_map_r2c: T.Tensor([raw_batch_size + 1], "int32"),
        out: T.Tensor([raw_batch_size + 1], "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            seq_map_r2c[0] = seq_map_r2c[0]

    dyn_map = _process_dynamic_symbolic(kernel)

    # Simulate _calcu_grid resolution
    actual_dim = 9  # the actual seq_map_r2c.shape[0]
    for key, pos in dyn_map.items():
        if str(key) == "raw_batch_size":
            assert len(pos) == 4
            a, b = pos[2], pos[3]
            resolved_value = (actual_dim - b) // a
            assert resolved_value == 8, (
                f"Expected raw_batch_size=8, got {resolved_value}"
            )

    # Simulate output shape evaluation
    dynamic_val = {"raw_batch_size": 8}
    output_shape_expr = raw_batch_size + 1
    output_dim = _eval_tir_expr(output_shape_expr, dynamic_val)
    assert output_dim == 9, f"Expected output dim=9, got {output_dim}"

    print("[PASS] test_runtime_resolution")


if __name__ == "__main__":
    test_helpers()
    test_process_dynamic_symbolic_composite()
    test_process_dynamic_symbolic_prefer_direct()
    test_symbolic_var_promoter_with_composite()
    test_runtime_resolution()
    print("\nAll tests passed!")
