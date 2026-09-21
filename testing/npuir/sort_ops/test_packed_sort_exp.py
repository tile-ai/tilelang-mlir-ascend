# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Native packed sort: exact index-bit checks, no numeric index conversion."""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

pytestmark = [pytest.mark.op("sort"), pytest.mark.mode("Expert")]


def packed_sort_kernel():
    @T.prim_func
    def kernel(
        src: T.Tensor((512,), "float32"),
        idx: T.Tensor((512,), "int32"),
        dst: T.Tensor((1024,), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            values = T.alloc_ub((512,), "float32")
            indices = T.alloc_ub((512,), "int32")
            ping = T.alloc_ub((1024,), "float32")
            pong = T.alloc_ub((1024,), "float32")
            T.copy(src, values)
            T.copy(idx, indices)
            T.vsort32(values, indices, ping, repeat_times=16)
            T.vmrgsort(
                ping[0:1024],
                ping[64:1024],
                ping[128:1024],
                ping[192:1024],
                pong,
                (32, 32, 32, 32),
                repeat_times=4,
            )
            T.vmrgsort(
                pong[0:256],
                pong[256:512],
                pong[512:768],
                pong[768:1024],
                ping,
                (128, 128, 128, 128),
            )
            T.copy(ping, dst)

    return kernel


def merge_kernel(lengths):
    a, b, c, d = [max(8, n * 2) for n in lengths]
    size = sum(lengths) * 2
    valid = (1 << sum(n > 0 for n in lengths)) - 1

    @T.prim_func
    def kernel(
        x0: T.Tensor((a,), "float32"),
        x1: T.Tensor((b,), "float32"),
        x2: T.Tensor((c,), "float32"),
        x3: T.Tensor((d,), "float32"),
        out: T.Tensor((size,), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            u0 = T.alloc_ub((a,), "float32")
            u1 = T.alloc_ub((b,), "float32")
            u2 = T.alloc_ub((c,), "float32")
            u3 = T.alloc_ub((d,), "float32")
            dst = T.alloc_ub((size,), "float32")
            T.copy(x0, u0)
            T.copy(x1, u1)
            T.copy(x2, u2)
            T.copy(x3, u3)
            T.vmrgsort(u0, u1, u2, u3, dst, lengths, valid_bit=valid)
            T.copy(dst, out)

    return kernel


def test_sort512_packed():
    torch.manual_seed(1207)
    src = torch.randperm(512).float().sub(256).npu()
    idx = (torch.arange(512, dtype=torch.int64) + 0x80000000).to(torch.int32).npu()
    dst = torch.empty(1024, dtype=torch.float32).npu()
    fn = tilelang.compile(packed_sort_kernel(), target="npuir")
    fn(src, idx, dst)
    expected, order = torch.sort(src.cpu(), descending=True)
    got = dst.cpu()
    assert torch.equal(got[0::2], expected)
    assert torch.equal(got.view(torch.int32)[1::2], idx.cpu()[order])


def test_extract_preserves_index_bits():
    @T.prim_func
    def kernel(
        src: T.Tensor((4096,), "float32"),
        values: T.Tensor((2048,), "float32"),
        indices: T.Tensor((2048,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            packed = T.alloc_ub((4096,), "float32")
            val = T.alloc_ub((2048,), "float32")
            idx = T.alloc_ub((2048,), "int32")
            T.copy(src, packed)
            T.vextract_pairs(packed, val, idx)
            T.copy(val, values)
            T.copy(idx, indices)

    source = torch.zeros(4096, dtype=torch.float32)
    source[::2] = torch.arange(2048).float().sub(1024)
    expected_indices = (torch.arange(2048, dtype=torch.int64) + 0x80000000).to(
        torch.int32
    )
    source.view(torch.int32)[1::2] = expected_indices
    src = source.npu()
    val = torch.empty(2048, dtype=torch.float32).npu()
    idx = torch.empty(2048, dtype=torch.int32).npu()
    fn = tilelang.compile(kernel, target="npuir")
    fn(src, val, idx)
    assert torch.equal(val.cpu(), source[::2])
    assert torch.equal(idx.cpu(), expected_indices)


def test_packed_sort_in_mixcv():
    """Check bitcode linkage and both AIV lanes in a Cube+Vector kernel."""

    @T.prim_func
    def kernel(
        a: T.Tensor((32, 32), "float16"),
        b: T.Tensor((64, 32), "float16"),
        workspace: T.Tensor((32, 64), "float32"),
        output: T.Tensor((2, 128), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("Cube"):
                a_l1 = T.alloc_L1((32, 32), "float16")
                b_l1 = T.alloc_L1((64, 32), "float16")
                c_l0 = T.alloc_L0C((32, 64), "float32")
                T.copy(a, a_l1)
                T.copy(b, b_l1)
                T.gemm(a_l1, b_l1, c_l0, initC=True, b_transpose=True)
                with T.rs("PIPE_FIX"):
                    T.copy(c_l0, workspace)
                    T.sync_block_set(0)
            with T.Scope("Vector"):
                scores = T.alloc_ub((64,), "float32")
                indices = T.alloc_ub((64,), "int32")
                groups = T.alloc_ub((128,), "float32")
                result = T.alloc_ub((128,), "float32")
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(0)
                    T.copy(workspace[vid, :], scores)
                T.arange(indices, [1])
                T.vsort32(scores, indices, groups, repeat_times=2)
                T.vmrgsort(
                    groups[:64],
                    groups[64:128],
                    groups[:64],
                    groups[:64],
                    result,
                    (32, 32, 0, 0),
                    valid_bit=3,
                )
                T.copy(result, output[vid, :])

    torch.manual_seed(9007)
    a = torch.randint(-2, 3, (32, 32)).half().npu()
    b = torch.randint(-2, 3, (64, 32)).half().npu()
    workspace = torch.empty((32, 64), dtype=torch.float32).npu()
    output = torch.empty((2, 128), dtype=torch.float32).npu()
    fn = tilelang.compile(kernel, target="npuir")
    fn(a, b, workspace, output)
    reference = a.cpu().float() @ b.cpu().float().T
    got = output.cpu()
    for row in range(2):
        ids = got.view(torch.int32)[row, 1::2].long()
        assert torch.equal(ids.sort().values, torch.arange(64))
        assert torch.equal(got[row, ::2], reference[row].sort(descending=True).values)
        assert torch.equal(got[row, ::2], reference[row][ids])


@pytest.mark.parametrize(
    "lengths", [(2048, 512, 0, 0), (2048, 512, 512, 0), (2048, 512, 512, 512)]
)
def test_merge_packed(lengths):
    torch.manual_seed(1207)
    scores = torch.randperm(sum(lengths)).float().sub(sum(lengths) // 2)
    inputs, expected_indices = [], []
    start = 0
    for n in lengths:
        values, order = scores[start : start + n].sort(descending=True)
        indices = (order.to(torch.int64) + start + 0x80000000).to(torch.int32)
        packed = torch.zeros(max(8, n * 2), dtype=torch.float32)
        packed[: n * 2 : 2] = values
        packed.view(torch.int32)[1 : n * 2 : 2] = indices
        inputs.append(packed.npu())
        expected_indices.append(indices)
        start += n
    out = torch.empty(sum(lengths) * 2, dtype=torch.float32).npu()
    fn = tilelang.compile(merge_kernel(lengths), target="npuir")
    fn(*inputs, out)
    expected, order = scores.sort(descending=True)
    got = out.cpu()
    assert torch.equal(got[::2], expected)
    assert torch.equal(
        got.view(torch.int32)[1::2],
        (order.to(torch.int64) + 0x80000000).to(torch.int32),
    )
