"""TileLangIR ``T.Pipelined`` scheduling annotation codegen tests.

The frontend consumes the DSL scheduling annotations while it lowers a
pipelined loop into ``tilelang.pipeline_stage`` units:

* ``num_stages`` stays on the ``scf.for`` for the automatic pipeline path;
* manual ``stage`` / ``order`` are mirrored one entry per scheduling unit;
* ``group`` merges its members into one ``tilelang.pipeline_stage`` unit at
  the first member's position (members keep program order, labels join with
  ``+``) and ``tl_pipeline_group`` is dropped so a downstream pass cannot
  schedule the same unit twice.

Pure IR codegen tests: they lower with ``tilelang.lower(..., target="tile")``
and assert on the generated MLIR source string. No Ascend hardware required.
"""

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing


def _source(kernel) -> str:
    artifact = tilelang.lower(kernel, target="tile")
    return artifact.kernel_source


def _stage_ops(source: str) -> list[str]:
    """Split the source into one text segment per ``tilelang.pipeline_stage``."""
    return source.split('"tilelang.pipeline_stage"')[1:]


def _pair_kernel(num_stages=0, stage=None, order=None):
    """Two plain copies: the minimal schedulable pipelined loop body."""

    options = {}
    if num_stages:
        options["num_stages"] = num_stages
    if stage is not None:
        options["stage"] = stage
    if order is not None:
        options["order"] = order

    @T.prim_func
    def main(A: T.Tensor((128, 16), "float32"), B: T.Tensor((128, 16), "float32"), n: T.int32):
        with T.Kernel(1):
            x = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(n, **options):
                T.copy(A[i, :], x)
                T.copy(x, B[i, :])

    return main


def _group_kernel(group, stage, order):
    """Three named sblocks so labels expose the merged member sequence."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 16), "float32"),
        A2: T.Tensor((128, 16), "float32"),
        B: T.Tensor((128, 16), "float32"),
        n: T.int32,
    ):
        with T.Kernel(1):
            x = T.alloc_shared((16,), "float32")
            y = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(n, stage=stage, order=order, group=group):
                with T.sblock("load_x"):
                    T.copy(A[i, :], x)
                with T.sblock("load_y"):
                    T.copy(A2[i, :], y)
                with T.sblock("consume"):
                    T.copy(x, B[i, :])

    return main


def test_num_stages_is_left_to_the_backend():
    source = _source(_pair_kernel(num_stages=3))
    assert source.count('"tilelang.pipeline_stage"') == 2
    assert 'tilelang.loop_kind = "pipelined"' in source
    assert "num_stages = 3 : i64" in source
    assert "tl_pipeline_stage" not in source
    assert "tl_pipeline_order" not in source


def test_manual_stage_and_order_mirror_one_entry_per_unit():
    source = _source(_pair_kernel(stage=[0, 3], order=[1, 0]))
    assert source.count('"tilelang.pipeline_stage"') == 2
    assert "tl_pipeline_stage = [0, 3]" in source
    assert "tl_pipeline_order = [1, 0]" in source
    assert "num_stages" not in source


def test_group_merges_members_into_the_first_member_slot():
    source = _source(_group_kernel([[0, 1]], stage=[0, 3], order=[0, 1]))
    ops = _stage_ops(source)
    assert len(ops) == 2
    # The merged unit sits where its first member was and keeps both copies.
    assert 'label = "load_x+load_y"' in ops[0]
    assert ops[0].count('"tilelang.copy"') == 2
    assert 'label = "consume"' in ops[1]
    assert ops[1].count('"tilelang.copy"') == 1
    # Manual arrays number the merged units and the raw grouping is consumed.
    assert "tl_pipeline_stage = [0, 3]" in source
    assert "tl_pipeline_order = [0, 1]" in source
    assert "tl_pipeline_group" not in source


def test_group_after_the_first_unit_keeps_program_order():
    source = _source(_group_kernel([[1, 2]], stage=[0, 3], order=[0, 1]))
    ops = _stage_ops(source)
    assert len(ops) == 2
    assert 'label = "load_x"' in ops[0]
    assert ops[0].count('"tilelang.copy"') == 1
    assert 'label = "load_y+consume"' in ops[1]
    assert ops[1].count('"tilelang.copy"') == 2
    assert "tl_pipeline_group" not in source


def test_empty_group_merges_nothing():
    source = _source(_group_kernel([], stage=[0, 3, 3], order=[0, 1, 2]))
    assert source.count('"tilelang.pipeline_stage"') == 3
    assert "tl_pipeline_stage = [0, 3, 3]" in source
    assert "tl_pipeline_order = [0, 1, 2]" in source
    assert "tl_pipeline_group" not in source


def test_merged_label_names_only_named_members():
    @T.prim_func
    def main(
        A: T.Tensor((128, 16), "float32"),
        A2: T.Tensor((128, 16), "float32"),
        B: T.Tensor((128, 16), "float32"),
        n: T.int32,
    ):
        with T.Kernel(1):
            x = T.alloc_shared((16,), "float32")
            y = T.alloc_shared((16,), "float32")
            for i in T.Pipelined(n, stage=[0, 3], order=[0, 1], group=[[0, 1]]):
                with T.sblock("load_x"):
                    T.copy(A[i, :], x)
                T.copy(A2[i, :], y)
                with T.sblock("consume"):
                    T.copy(x, B[i, :])

    ops = _stage_ops(_source(main))
    assert len(ops) == 2
    assert 'label = "load_x"' in ops[0]
    assert ops[0].count('"tilelang.copy"') == 2


def test_annotation_size_counts_merged_units():
    with pytest.raises(ValueError, match="expected 2, got 3"):
        _source(_group_kernel([[0, 1]], stage=[0, 3, 3], order=[0, 1, 2]))


def test_annotation_size_checks_each_unit_without_group():
    with pytest.raises(ValueError, match="expected 3, got 2"):
        _source(_group_kernel(None, stage=[0, 3], order=[0, 1]))


@pytest.mark.parametrize(
    "group, message",
    [
        ([[0, 1, 1]], "appears in more than one group"),
        ([[0, 3]], "out of range"),
        ([[0, 1], [1, 2]], "appears in more than one group"),
        ([[]], "must not be empty"),
        ([[0.5]], "must be integers"),
    ],
)
def test_malformed_group_is_rejected(group, message):
    with pytest.raises(ValueError, match=message):
        _source(_group_kernel(group, stage=[0, 3], order=[0, 1]))


if __name__ == "__main__":
    tilelang.testing.main()
