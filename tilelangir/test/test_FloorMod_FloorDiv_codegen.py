"""TileLangIR FloorDiv / FloorMod codegen tests.

Verifies that ``T.floordiv`` and ``T.floormod`` on integer operands lower
to the correct ``arith.divsi`` / ``arith.remsi`` with sign-mismatch
adjustment logic, matching the floor-toward-negative-infinity semantics.
These are pure IR codegen tests and require no Ascend hardware.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


def _floor_div_mod_kernel(
    M: int = 256,
    N: int = 256,
    block_M: int = 32,
):
    """Build a minimal kernel exercising both FloorDiv and FloorMod on integers."""
    num_blocks = M // block_M
    dtype = "int32"

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C_div: T.Tensor((M, N), dtype),
        C_mod: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            a_shared = T.alloc_shared((block_M, N), dtype)
            b_shared = T.alloc_shared((block_M, N), dtype)
            c_div_shared = T.alloc_shared((block_M, N), dtype)
            c_mod_shared = T.alloc_shared((block_M, N), dtype)

            T.copy(A[bx * block_M : (bx + 1) * block_M, 0:N], a_shared)
            T.copy(B[bx * block_M : (bx + 1) * block_M, 0:N], b_shared)

            for r in range(0, block_M):
                for c in range(0, N):
                    # Integer FloorDiv and FloorMod — must trigger sign-adjust logic
                    c_div_shared[r, c] = T.floordiv(a_shared[r, c], b_shared[r, c])
                    c_mod_shared[r, c] = T.floormod(a_shared[r, c], b_shared[r, c])

            T.copy(c_div_shared, C_div[bx * block_M : (bx + 1) * block_M, 0:N])
            T.copy(c_mod_shared, C_mod[bx * block_M : (bx + 1) * block_M, 0:N])

    return main


def _lower_to_tile_source(kernel):
    """Lower ``kernel`` to tile target and return the generated MLIR source."""
    from tilelang import lower as _lower

    artifact = _lower(kernel, target="tile")
    return artifact.kernel_source


def test_floordiv_lowers_to_arith_divsi_with_adjust():
    """FloorDiv on int32 must emit arith.divsi plus sign-mismatch correction."""
    source = _lower_to_tile_source(_floor_div_mod_kernel())
    # Core truncating division
    assert "arith.divsi" in source, (
        f"Expected arith.divsi in lowered source, got:\n{source}"
    )
    # Sign-mismatch detection requires xor + and + select
    assert "arith.xori" in source, (
        f"Expected arith.xori for sign-mismatch check, got:\n{source}"
    )
    assert "arith.select" in source, (
        f"Expected arith.select for conditional adjustment, got:\n{source}"
    )
    # Adjustment subtracts 1 from quotient when needed
    assert "arith.subi" in source, (
        f"Expected arith.subi for quotient adjustment, got:\n{source}"
    )


def test_floormod_lowers_to_arith_remsi_with_adjust():
    """FloorMod on int32 must emit arith.remsi plus sign-mismatch correction."""
    source = _lower_to_tile_source(_floor_div_mod_kernel())
    # Core truncating remainder
    assert "arith.remsi" in source, (
        f"Expected arith.remsi in lowered source, got:\n{source}"
    )
    # When sign mismatches, floor mod = rem + rhs
    assert "arith.addi" in source, (
        f"Expected arith.addi for remainder adjustment, got:\n{source}"
    )
    assert "arith.select" in source, (
        f"Expected arith.select for conditional adjustment, got:\n{source}"
    )


def test_floor_ops_use_integer_types():
    """Both ops should operate on integer types, not float."""
    source = _lower_to_tile_source(_floor_div_mod_kernel())
    # Ensure we're using signed integer comparisons, not float
    assert "arith.cmpi" in source, (
        f"Expected arith.cmpi for integer comparison, got:\n{source}"
    )
    # Must NOT contain float division/remainder for integer floor ops
    assert "arith.divf" not in source, (
        f"Unexpected arith.divf for integer FloorDiv:\n{source}"
    )
    assert "arith.remf" not in source, (
        f"Unexpected arith.remf for integer FloorMod:\n{source}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
