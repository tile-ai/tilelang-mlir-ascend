"""Grouped GEMM contract and NPU correctness tests, independent of TileOPs.

TILELANG_ASCEND_MODE=Expert python -m pytest example/grouped_gemm/test_grouped.py -q
The seven PR manifest dtype cases are retained, with additional ragged, empty,
tail, rerouting and input-preservation checks. Only absent NPU hardware skips
device tests; compilation and execution failures are errors.
"""

import pytest
import torch

from .grouped import GroupedGemmFwdOp, _validate


def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


npu_only = pytest.mark.skipif(not _npu_available(), reason="Ascend NPU is unavailable")
LAYOUTS = [(False, True), (False, False), (True, False)]
MANIFEST_CASES = [
    ("moe-fwd-bfloat16", 4096, 16, 4096, 4096, False, True, torch.bfloat16),
    ("moe-fwd-float16", 4096, 16, 4096, 4096, False, True, torch.float16),
    ("moe-dgrad-float16", 4096, 16, 4096, 4096, False, False, torch.float16),
    ("moe-wgrad-float16", 4096, 16, 4096, 4096, True, False, torch.float16),
    ("qwen3-30b-ep8-empty-bfloat16", 8, 16, 1536, 2048, False, True, torch.bfloat16),
    (
        "qwen3-30b-ep8-routed1k-bfloat16",
        1024,
        16,
        1536,
        2048,
        False,
        True,
        torch.bfloat16,
    ),
    ("qwen3-30b-ep8-ragged-bfloat16", 257, 16, 1536, 2048, False, True, torch.bfloat16),
]


def make_inputs(sizes, n, k, transpose_a, transpose_b, dtype, device="cpu"):
    sizes = torch.tensor(sizes, dtype=torch.int32, device=device)
    offsets = sizes.cumsum(0, dtype=torch.int32) - sizes
    numel = int(sizes.sum())  # Test fixture only, never in the production path.
    experts = sizes.numel()
    a_shape = (numel, n if transpose_a else k)
    b_shape = (
        (numel, k)
        if transpose_a
        else ((experts, n, k) if transpose_b else (experts, k, n))
    )
    a = (torch.randn(a_shape, device=device) * 0.25).to(dtype)
    b = (torch.randn(b_shape, device=device) * 0.25).to(dtype)
    return a, b, sizes, offsets


def reference(a, b, sizes, offsets, transpose_a, transpose_b):
    """Independent CPU per-group matmul; FP32 accumulation then one output cast."""
    a, b = a.cpu(), b.cpu()
    sizes, offsets = sizes.cpu().tolist(), offsets.cpu().tolist()
    if transpose_a:
        shape = (len(sizes), a.shape[1], b.shape[1])
    else:
        shape = (a.shape[0], b.shape[1] if transpose_b else b.shape[2])
    out = torch.zeros(shape, dtype=a.dtype)
    for group, (rows, start) in enumerate(zip(sizes, offsets, strict=True)):
        if transpose_a:
            out[group] = (
                a[start : start + rows].float().T @ b[start : start + rows].float()
            )
        elif rows:
            weight = b[group].float().T if transpose_b else b[group].float()
            out[start : start + rows] = a[start : start + rows].float() @ weight
    return out


def check(a, b, sizes, offsets, transpose_a, transpose_b, op=None):
    inputs = a, b, sizes, offsets
    before = tuple(x.clone() for x in inputs)
    op = op or GroupedGemmFwdOp(transpose_a=transpose_a, transpose_b=transpose_b)
    actual = op(*inputs)
    expected = reference(*inputs, transpose_a, transpose_b)
    assert actual.device == a.device and actual.dtype == a.dtype
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
    for value, previous in zip(inputs, before, strict=True):
        assert torch.equal(value, previous)
    if transpose_a:
        assert (actual[sizes == 0] == 0).all()
    return actual


@npu_only
@pytest.mark.parametrize("case", MANIFEST_CASES, ids=lambda c: c[0])
def test_manifest(case):
    _, m, groups, n, k, ta, tb, dtype = case
    sizes = [m // groups + (g < m % groups) for g in range(groups)]
    inputs = make_inputs(sizes, n, k, ta, tb, dtype, "npu")
    check(*inputs, ta, tb)


@npu_only
@pytest.mark.parametrize("ta,tb", LAYOUTS, ids=["NT", "NN", "TN"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,k", [(17, 19), (129, 79)])
def test_ragged_tails_empty_groups_and_rerouting(ta, tb, dtype, n, k):
    # Cross 16/32/64-row boundaries, leave internal and trailing empty groups.
    sizes = [1, 16, 17, 32, 64, 65, 0, 3] + [0] * 8
    a, b, group_sizes, offsets = make_inputs(sizes, n, k, ta, tb, dtype, "npu")
    op = GroupedGemmFwdOp(transpose_a=ta, transpose_b=tb)
    check(a, b, group_sizes, offsets, ta, tb, op)
    # The same compiled shape must consume new device metadata on every call.
    group_sizes = group_sizes.flip(0).contiguous()
    offsets = group_sizes.cumsum(0, dtype=torch.int32) - group_sizes
    check(a, b, group_sizes, offsets, ta, tb, op)


@npu_only
@pytest.mark.parametrize("ta,tb", LAYOUTS, ids=["NT", "NN", "TN"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_empty_and_single_token(ta, tb, dtype):
    for sizes in ([0] * 16, [0, 1] + [0] * 14):
        check(*make_inputs(sizes, 7, 5, ta, tb, dtype, "npu"), ta, tb)


@npu_only
@pytest.mark.parametrize("ta,tb", LAYOUTS, ids=["NT", "NN", "TN"])
def test_contiguous_views_with_nonzero_storage_offsets(ta, tb):
    inputs = list(make_inputs([1, 0, 17], 33, 65, ta, tb, torch.float16, "npu"))
    for index, value in enumerate(inputs):
        storage = torch.empty(value.numel() + 1, dtype=value.dtype, device=value.device)
        inputs[index] = storage[1:].view(value.shape).copy_(value)
    check(*inputs, ta, tb)


@pytest.mark.parametrize("ta,tb", LAYOUTS, ids=["NT", "NN", "TN"])
@pytest.mark.parametrize(
    "fault", ["dtype", "metadata_dtype", "metadata_shape", "dimensions", "contiguity"]
)
def test_invalid_contract_is_rejected_before_compilation(ta, tb, fault):
    a, b, sizes, offsets = make_inputs([1, 2], 17, 19, ta, tb, torch.float16)
    if fault == "dtype":
        a = a.float()
    elif fault == "metadata_dtype":
        sizes = sizes.long()
    elif fault == "metadata_shape":
        offsets = offsets[:1]
    elif fault == "dimensions":
        b = b[:-1].contiguous()
    else:
        a = a.T.contiguous().T
    with pytest.raises(ValueError):
        GroupedGemmFwdOp(transpose_a=ta, transpose_b=tb)(a, b, sizes, offsets)


@pytest.mark.parametrize("ta,tb", LAYOUTS)
def test_valid_cpu_input_has_no_silent_fallback(ta, tb):
    args = make_inputs([1, 0, 2], 17, 19, ta, tb, torch.bfloat16)
    assert _validate(*args, ta, tb) == (3, 3, 17, 19)
    with pytest.raises(RuntimeError, match="NPU"):
        GroupedGemmFwdOp(transpose_a=ta, transpose_b=tb)(*args)


@pytest.mark.parametrize("flags", [(True, True), (1, False), (False, "NT")])
def test_unsupported_layout(flags):
    with pytest.raises(ValueError):
        GroupedGemmFwdOp(transpose_a=flags[0], transpose_b=flags[1])
