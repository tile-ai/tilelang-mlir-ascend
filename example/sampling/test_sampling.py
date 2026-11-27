"""Self-contained NPUIR port of the TileOPs PR #50 sampling tests.

Source: https://github.com/yyttt6/TileOPs/pull/50
Revision: d3aa853509f3ca68d98f699a225e639aff83be47
Run from the repository root with TILELANG_ASCEND_MODE=Developer:
    python -m pytest example/sampling/test_sampling.py -q

PyTorch references and fixture helpers live here so no TileOPs package or
external workload framework is required. Missing NPU hardware skips tests;
compiler/runtime failures remain errors.
"""

import numpy as np
import pytest
import torch

from .ops import (
    ChainSpeculativeSamplingFwdOp,
    MinPMaskFwdOp,
    SamplingFromProbsFwdOp,
    TopKMaskFwdOp,
    TopKTopPMaskFwdOp,
    TopPMaskFwdOp,
)


import hashlib
from dataclasses import dataclass


def npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


def run_device():
    return torch.device("npu" if npu_available() else "cpu")


def manifest_call(op, dtype_case=None, **shape):
    return op, dtype_case or {"T": "float32"}, shape


class CallWorkload:
    def __init__(self, call):
        self.op, self.dtype_case, self.shape = call

    def rng(self, name, device):
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        return torch.Generator(device=device).manual_seed(seed)

    def gen_inputs(self):
        device = run_device()
        shape = self.shape
        k_list = shape.get("k_list", [1, 5])
        batch = shape.get("B", len(k_list))
        vocab = shape["V"]
        dtype = getattr(torch, self.dtype_case["T"])
        rng = self.rng("inputs", device)
        x = torch.randn(batch, vocab, dtype=dtype, device=device, generator=rng)
        k = torch.tensor(k_list, dtype=torch.int32, device=device)
        p = torch.full((batch,), 0.9, device=device)
        seed = torch.tensor([1234], dtype=torch.int64, device=device)
        offset = torch.tensor([7], dtype=torch.int64, device=device)
        if self.op == "TopKMaskFwdOp":
            return x, k
        if self.op in ("MinPMaskFwdOp", "TopPMaskFwdOp"):
            return x, p
        if self.op == "TopKTopPMaskFwdOp":
            return x, k, p
        if self.op == "SamplingFromProbsFwdOp":
            return x, seed, offset
        drafts = shape["N"]
        draft = torch.randn(batch, drafts, vocab, device=device, generator=rng)
        target = torch.randn(batch, drafts + 1, vocab, device=device, generator=rng)
        ids = torch.randint(
            vocab, (batch, drafts), dtype=torch.int32, device=device, generator=rng
        )
        return draft, ids, target, seed, offset


@dataclass
class Custom:
    validate: object
    description: str
    probe: object = None


def compare_outputs(got, expected, verification):
    verification.validate(got, expected)


def logit_mask_validator(logits, near):
    def validate(got, expected):
        assert got.shape == expected.shape == logits.shape
        assert got.dtype == expected.dtype == logits.dtype
        removed = got == -float("inf")
        expected_removed = expected == -float("inf")
        assert not ((removed != expected_removed) & ~near).any()
        storage = torch.int32 if logits.dtype == torch.float32 else torch.int16
        kept = ~removed
        assert torch.equal(got.view(storage)[kept], logits.view(storage)[kept])

    return validate


class TestBase:
    @staticmethod
    def check(workload, op, *args, runs):
        verification = workload.verification(*args)
        compare_outputs(runs(*args), workload.ref_program(*args), verification)
        if verification.probe is not None:
            verification.probe(runs, args)


_INF = float("inf")
_MASK = 0xFFFFFFFF
# Philox4x32 round multipliers and Weyl key increments (Salmon et al., SC'11).
_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85


def _mulhilo(a: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The high and low 32-bit words of ``a * m`` for 32-bit ``a`` and ``m``, without int64 overflow."""
    low = (a & 0xFFFF) * m
    high = (a >> 16) * m
    mid = low + ((high & 0xFFFF) << 16)
    return ((high >> 16) + (mid >> 32)) & _MASK, mid & _MASK


def philox_uniform(
    seed: torch.Tensor, offset: torch.Tensor, rows: int, draws: int
) -> torch.Tensor:
    """``[rows, draws]`` float32 uniforms in ``[0, 1)``, a function of ``seed`` and ``offset`` alone.

    Draw ``j`` of row ``r`` is the first output word of Philox4x32-10 with key
    ``(seed_lo, seed_hi)`` and counter ``(j, r, offset_lo, offset_hi)``, its top 24 bits
    scaled by ``2**-24``.
    """
    device = seed.device
    s, o = seed.reshape(()), offset.reshape(()).to(device)
    shape = (rows, draws)
    c0 = torch.arange(draws, device=device).expand(shape)
    c1 = torch.arange(rows, device=device)[:, None].expand(shape)
    c2 = (o & _MASK).expand(shape)
    c3 = ((o >> 32) & _MASK).expand(shape)
    k0, k1 = s & _MASK, (s >> 32) & _MASK
    for step in range(10):
        if step:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0 >> 8).to(torch.float32) * 2.0**-24


def _draw(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Per row, the index an inverse-CDF draw at ``u * total`` lands on; never a zero weight.

    ``weights`` is ``[B, V]``, non-negative with a positive row total; ``u`` is ``[B]`` in ``[0, 1)``.
    """
    batch, vocab = weights.shape
    cdf = weights.double().cumsum(-1)
    point = u.double() * cdf[:, -1]
    index = torch.searchsorted(cdf, point[:, None], right=True).clamp(max=vocab - 1)
    # A draw on a zero weight, which cumsum rounding allows, moves to the next positive one.
    positive = weights > 0
    columns = torch.arange(vocab, device=weights.device).expand(batch, vocab)
    next_positive = (
        torch.where(positive, columns, vocab).flip(-1).cummin(-1).values.flip(-1)
    )
    chosen = next_positive.gather(1, index)[:, 0]
    last_positive = vocab - 1 - positive.flip(-1).int().argmax(-1)
    return torch.where(chosen == vocab, last_positive, chosen).to(torch.int32)


def top_k_mask(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Keep every logit at least the ``k[b]``-th largest of row ``b``; ``k[b] >= V`` keeps the row."""
    batch, vocab = logits.shape
    k = k.view(batch).to(logits.device, torch.long)
    if vocab == 0:
        return logits.clone()
    values = logits.float()
    kth = values.sort(-1, descending=True).values.gather(
        1, (k.clamp(max=vocab) - 1)[:, None]
    )
    kept = (values >= kth) | (k >= vocab)[:, None]
    return logits.masked_fill(~kept, -_INF)


def min_p_mask(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Mask every logit below ``max_logit + log(min_p[b])``."""
    batch, _vocab = logits.shape
    values = logits.float()
    threshold = values.amax(-1) + min_p.view(batch).to(logits.device).log()
    return logits.masked_fill(values < threshold[:, None], -_INF)


def probability_above(probs: torch.Tensor) -> torch.Tensor:
    """Per token, the total probability of the tokens strictly more probable than it.

    Tokens of equal probability get the same value, whatever order a sort leaves them in.
    """
    ordered, order = probs.sort(-1, descending=True)
    exclusive = ordered.cumsum(-1) - ordered
    # The first position of each run of equal values, where the exclusive sum excludes the run.
    # A row of NaN probabilities, which a row whose largest logit is not finite gives, has no
    # position a search settles on; it is clamped into the row, and its total stays NaN.
    first = torch.searchsorted(-ordered, -ordered, side="left").clamp(
        max=probs.shape[-1] - 1
    )
    return torch.empty_like(probs).scatter_(-1, order, exclusive.gather(-1, first))


def top_p_mask(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Mask every token whose more probable tokens already hold at least ``p[b]``.

    Tokens tied at the boundary are kept or masked together.
    """
    batch, _vocab = logits.shape
    probs = logits.float().softmax(-1)
    removed = probability_above(probs) >= p.view(batch, 1).to(logits.device)
    return logits.masked_fill(removed, -_INF)


def sampling_from_probs(
    probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """One index per row, drawn with probability proportional to ``probs[b]``."""
    batch, _vocab = probs.shape
    return _draw(probs, philox_uniform(seed, offset, batch, 1)[:, 0])


def chain_speculative_sampling(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify each row's draft chain; draws ``0..N-1`` test acceptance, draw ``N`` the token.

    Draft ``i`` is accepted while ``u_i * draft < target`` at its id. The token after the
    accepted prefix is drawn from ``max(0, target - draft)`` at that position, or from target
    row ``N`` when every draft is accepted; later positions are ``-1``.
    """
    batch, n, vocab = draft_probs.shape
    ids = draft_token_ids.view(batch, n).to(draft_probs.device, torch.long)
    target = target_probs.view(batch, n + 1, vocab)
    u = philox_uniform(seed, offset, batch, n + 1)
    d = draft_probs.gather(2, ids[..., None])[..., 0]
    q = target[:, :n].gather(2, ids[..., None])[..., 0]
    num = (u[:, :n] * d < q).int().cumprod(1).sum(1)
    padded_draft = torch.cat([draft_probs, torch.zeros_like(target[:, :1])], 1)
    at = num.long()[:, None, None].expand(batch, 1, vocab)
    residual = (target.gather(1, at) - padded_draft.gather(1, at))[:, 0].clamp_min(0)
    token = _draw(residual, u[:, n]).long()
    position = torch.arange(n + 1, device=ids.device)[None]
    drafts = torch.cat(
        [ids, torch.full((batch, 1), -1, dtype=ids.dtype, device=ids.device)], 1
    )
    stop = num.long()[:, None]
    tokens = torch.where(
        position < stop, drafts, torch.where(position == stop, token[:, None], -1)
    )
    return tokens.to(torch.int32), num.to(torch.int32)


class TopKMaskWorkload(CallWorkload):
    """Logits and per-row ``k`` of one ``TopKMaskFwdOp`` call."""

    def ref_program(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return top_k_mask(logits, k)

    def verification(self, *inputs):
        return top_k_mask_verification(inputs[0])


class MinPMaskWorkload(CallWorkload):
    """Logits of one ``MinPMaskFwdOp`` call, with ``min_p`` uniform in ``[0.05, 0.25)``.

    The first two rows take the contract's endpoints instead: 1, which keeps only the logits
    equal to the row max, and 0, which masks nothing. 1 comes first so that a one-row call
    still exercises a threshold.
    """

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits, min_p = CallWorkload.gen_inputs(self)
        min_p = torch.rand_like(min_p) * 0.2 + 0.05
        min_p[:2] = torch.tensor([1.0, 0.0], device=min_p.device)[: min_p.numel()]
        return logits, min_p

    def ref_program(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        return min_p_mask(logits, min_p)

    def verification(self, *inputs):
        return min_p_mask_verification(*inputs)


class TopPMaskWorkload(CallWorkload):
    """Logits of one ``TopPMaskFwdOp`` call, with ``p`` uniform in ``[0.5, 0.95)``.

    The first row takes 1 instead, the contract's upper endpoint, which keeps every token
    the float32 softmax gives a probability to. The lower endpoint 0 masks a row whole, a
    row no comparator agrees on, so ``tests/ops/test_sampling.py`` covers it rather than a
    workload every benchmark reads.
    """

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        logits, p = CallWorkload.gen_inputs(self)
        p = torch.rand_like(p) * 0.45 + 0.5
        p[0] = 1.0
        return logits, p

    def ref_program(self, logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return top_p_mask(logits, p)

    def verification(self, *inputs):
        return top_p_mask_verification(*inputs)


class TopKTopPMaskWorkload(CallWorkload):
    """Logits and per-row ``k`` of one ``TopKTopPMaskFwdOp`` call, with ``p`` uniform in ``[0.5, 0.95)``."""

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, k, p = CallWorkload.gen_inputs(self)
        return logits, k, torch.rand_like(p) * 0.45 + 0.5

    def ref_program(
        self, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        return top_p_mask(top_k_mask(logits, k), p)

    def verification(self, *inputs):
        logits, k, p = inputs
        return top_p_mask_verification(logits, p, k=k)


class SamplingFromProbsWorkload(CallWorkload):
    """One ``SamplingFromProbsFwdOp`` call: each row the softmax of standard normal logits."""

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, seed, offset = CallWorkload.gen_inputs(self)
        return probs.softmax(-1), seed, offset

    def ref_program(
        self, probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        return sampling_from_probs(probs, seed, offset)

    def verification(self, *inputs):

        probs, seed, offset = inputs

        def validate(tokens, expected):
            assert tokens.dtype == torch.int32
            assert ((tokens >= 0) & (tokens < probs.shape[1])).all()
            assert (probs.gather(1, tokens.long()[:, None]) > 0).all()

        def distribution(draw, _args):
            # An additional fixed statistical experiment distinguishes a real categorical
            # sampler from a callable that always returns any positive-weight token.
            draws, vocab = 65536, 64
            weights = torch.rand(
                vocab,
                device=probs.device,
                generator=self.rng("distribution", device=probs.device),
            )
            weights[::4] = 0
            share = (weights / weights.sum()).double()
            trial = (weights / weights.sum()).expand(draws, vocab).contiguous()
            tokens = draw(trial, seed, offset)
            assert tokens.shape == (draws,) and tokens.dtype == torch.int32
            assert ((tokens >= 0) & (tokens < vocab)).all()
            _assert_follows(tokens, share)

        return Custom(
            validate,
            "categorical support and six-sigma distribution",
            probe=distribution,
        )


class ChainSpeculativeSamplingWorkload(CallWorkload):
    """One ``ChainSpeculativeSamplingFwdOp`` call: draft and target rows the softmax of
    standard normal logits, the draft ids the row's generated ones."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        draft_probs, draft_token_ids, target_probs, seed, offset = (
            CallWorkload.gen_inputs(self)
        )
        return (
            draft_probs.softmax(-1),
            draft_token_ids,
            target_probs.softmax(-1),
            seed,
            offset,
        )

    def ref_program(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return chain_speculative_sampling(
            draft_probs, draft_token_ids, target_probs, seed, offset
        )

    def verification(self, *inputs):

        draft, draft_ids, target, seed, offset = inputs
        batch, num_draft, vocab = draft.shape

        def validate(got, expected):
            tokens, accepted = got
            assert ((tokens >= -1) & (tokens < vocab)).all()
            assert ((accepted >= 0) & (accepted <= num_draft)).all()
            positions = torch.arange(num_draft + 1, device=tokens.device)[None, :]
            assert torch.equal(tokens == -1, positions > accepted[:, None])
            prefix = positions[:, :num_draft] < accepted[:, None]
            assert torch.equal(tokens[:, :num_draft][prefix], draft_ids[prefix])
            support = target[:, :num_draft].gather(-1, draft_ids.long()[..., None])[
                ..., 0
            ]
            assert (support[prefix] > 0).all(), (
                "accepted a draft the target cannot draw"
            )
            # Each implementation draws its own stream, so two of one rule agree on how the
            # accepted lengths are distributed, never on the batch they drew: each length's
            # count stays within 5 sigma of two independent batches, plus 5.
            got_counts = torch.bincount(
                accepted.long(), minlength=num_draft + 1
            ).double()
            want = torch.bincount(expected[1].long(), minlength=num_draft + 1).double()
            share = want / batch
            bound = 5 * (2 * batch * share * (1 - share)).sqrt() + 5
            assert ((got_counts - want).abs() <= bound).all(), (got_counts, want, bound)
            rows = torch.arange(batch, device=tokens.device)
            padded = torch.cat([draft, torch.zeros_like(draft[:, :1])], 1)
            weights = (
                target[rows, accepted.long()] - padded[rows, accepted.long()]
            ).clamp_min(0)
            assert (weights[rows, tokens[rows, accepted.long()].long()] > 0).all()

        def distribution(draw, _args):
            n, drafts, vocabulary = 65536, 2, 16
            generator = self.rng("chain-distribution", device=draft.device)
            trial_draft = (
                torch.rand(drafts, vocabulary, device=draft.device, generator=generator)
                ** 3
            )
            trial_draft /= trial_draft.sum(-1, keepdim=True)
            trial_target = (
                torch.rand(
                    drafts + 1, vocabulary, device=draft.device, generator=generator
                )
                ** 3
            )
            trial_target /= trial_target.sum(-1, keepdim=True)
            ids = (
                torch.multinomial(trial_draft, n, replacement=True, generator=generator)
                .T.to(torch.int32)
                .contiguous()
            )
            args = (
                trial_draft.expand(n, drafts, vocabulary).contiguous(),
                ids,
                trial_target.expand(n, drafts + 1, vocabulary).contiguous(),
                seed,
                offset,
            )
            got = draw(*args)
            _assert_verifies_chains(
                *got,
                ids,
                trial_draft,
                trial_target,
                torch.minimum(trial_draft, trial_target[:drafts]).sum(-1),
            )

        return Custom(
            validate,
            "accepted-length and residual-token distribution",
            probe=distribution,
        )


def top_k_mask_verification(logits):

    return Custom(
        logit_mask_validator(logits, torch.zeros_like(logits, dtype=torch.bool)),
        "exact top-k mask and retained logit bits",
    )


def min_p_mask_verification(logits, min_p):

    values = logits.float()
    threshold = values.amax(-1, keepdim=True) + min_p[:, None].log()
    return Custom(
        logit_mask_validator(logits, (values - threshold).abs() <= 1e-4),
        "min-p boundary within 1e-4 logit distance",
    )


def top_p_mask_verification(logits, p, *, k=None):

    selected = logits if k is None else top_k_mask(logits, k)
    near = (probability_above(selected.float().softmax(-1)) - p[:, None]).abs() <= 1e-4
    return Custom(
        logit_mask_validator(logits, near),
        "nucleus boundary within 1e-4 probability mass",
    )


def _assert_follows(samples, probs):
    """Each index's count within ``6 sigma + 5`` of its expectation; a zero-probability index
    never drawn. The ``+ 5`` covers indices expected fewer than a few times."""
    n = samples.numel()
    count = torch.bincount(samples.long(), minlength=probs.numel()).double()
    p = probs.double()
    bound = 6 * (n * p * (1 - p)).sqrt() + 5 * (p > 0)
    assert ((count - n * p).abs() <= bound).all(), (count, n * p)


def _assert_verifies_chains(tokens, num, draft_ids, draft, target, accepted):
    """The accepted prefix is the drafts and ``-1`` follows the drawn token; the drawn token
    follows the residual of its position, or target row ``N`` after a whole chain; the first
    token follows target row 0; an accepted draft follows ``min(draft, target)`` of its
    position; ``num`` follows the acceptance probabilities ``accepted``."""
    assert tokens.dtype == num.dtype == torch.int32
    num_draft = draft.shape[0]
    position = torch.arange(num_draft + 1, device=tokens.device)[None]
    prefix = position[:, :-1] < num[:, None]
    assert torch.equal(tokens[:, :-1][prefix], draft_ids[prefix])
    assert (tokens[position > num[:, None]] == -1).all()
    for stop in range(num_draft + 1):
        weights = (
            target[stop]
            if stop == num_draft
            else (target[stop] - draft[stop]).clamp_min(0)
        )
        _assert_follows(tokens[num == stop, stop], weights / weights.sum())
    _assert_follows(tokens[:, 0], target[0])
    for step in range(num_draft):
        kept = torch.minimum(draft[step], target[step])
        _assert_follows(draft_ids[num > step, step], kept / kept.sum())
    a0, a1 = accepted
    _assert_follows(num, torch.stack([1 - a0, a0 * (1 - a1), a0 * a1]))


pytestmark = pytest.mark.skipif(not npu_available(), reason="Ascend NPU is unavailable")

_DTYPES = ["float16", "bfloat16", "float32"]
_INF = float("inf")


def _run(op, *inputs):
    return op(*inputs)


def _assert_top_set(masked, logits):
    """Per row: the largest logit is kept, and no masked logit exceeds a kept one."""
    kept = masked != -_INF
    values = logits.float()
    assert kept.gather(1, values.argmax(-1, keepdim=True)).all()
    lowest_kept = values.masked_fill(~kept, _INF).amin(-1)
    highest_masked = values.masked_fill(kept, -_INF).amax(-1)
    assert (highest_masked <= lowest_kept).all()


def _assert_nucleus(masked, logits, p, *, margin):
    """The kept tokens hold at least ``p``, less than ``p`` without their least probable value,
    and every token of that value."""
    _assert_top_set(masked, logits)
    kept = masked != -_INF
    probs = logits.float().softmax(-1)
    lowest = probs.masked_fill(~kept, _INF).amin(-1, keepdim=True)
    assert ((probs * kept).sum(-1) >= p - margin).all()
    assert ((probs * (kept & (probs > lowest))).sum(-1) < p + margin).all()
    # Tokens tied with the least probable kept one are kept with it.
    assert not (~kept & (probs == lowest)).any()


def _tie_at_the_boundary(logits, p):
    """Row 1 becomes three equal top tokens and ``p[1] = 0.5``: all three survive top-p."""
    logits[1] = -100.0
    logits[1, :3] = 0.0
    p[1] = 0.5


@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_k_mask(dtype):
    vocab = 32000
    # k = 1, a k that bf16 ties overrun, k == vocab and k > vocab.
    workload = TopKMaskWorkload(
        manifest_call(
            "TopKMaskFwdOp", {"T": dtype}, V=vocab, k_list=[1, 50, vocab, vocab + 7]
        )
    )
    logits, k = workload.gen_inputs()
    ref = workload.ref_program(logits, k)
    _assert_top_set(ref, logits)
    kept = (ref != -_INF).sum(-1)
    assert (kept >= k.clamp(max=vocab)).all()
    assert torch.equal(kept[k >= vocab], torch.full_like(kept[k >= vocab], vocab))
    out = _run(TopKMaskFwdOp(), logits, k)
    compare_outputs(out, ref, workload.verification(logits, k))


def _special_rows(vocab: int, dtype: torch.dtype) -> torch.Tensor:
    """Six rows: noise, both zeros, both infinities, NaN, one repeated value, and two."""
    device = run_device()
    rows = torch.randn(6, vocab, device=device).to(dtype)
    rows[1, ::2], rows[1, 1::2] = 0.0, -0.0
    rows[2, :7], rows[2, 7:14] = _INF, -_INF
    rows[3, :5] = float("nan")
    rows[4] = 1.5
    rows[5] = torch.where(torch.arange(vocab, device=device) % 3 == 0, 2.0, -2.0).to(
        dtype
    )
    return rows


@pytest.mark.parametrize(
    "dtype, vocab",
    [
        # 16-bit keys with the row on the vector, 16-bit and 32-bit keys with it off.
        (torch.bfloat16, 4096),
        (torch.float16, 999),
        (torch.float32, 999),
        (torch.bfloat16, 4099),
        (torch.float16, 4099),
        (torch.float32, 4099),
    ],
)
def test_top_k_mask_matches_the_reference_bit_for_bit(dtype: torch.dtype, vocab: int):
    """Each row is cut where its own special values sit: at a zero, at an infinity, at a
    NaN, among equal values, and at a k the row leaves whole."""
    logits = _special_rows(vocab, dtype)
    k = torch.tensor(
        [1, vocab // 2, 3, 2, vocab // 4, vocab],
        dtype=torch.int32,
        device=logits.device,
    )
    ref = top_k_mask(logits, k)
    out = _run(TopKMaskFwdOp(), logits, k)
    # Bit patterns, so that -0.0 kept where the reference keeps 0.0 is a failure.
    compare_outputs(out, ref, top_k_mask_verification(logits))
    k[3] = vocab
    out = _run(TopKMaskFwdOp(), logits, k)
    compare_outputs(out, top_k_mask(logits, k), top_k_mask_verification(logits))


def test_top_k_mask_cuts_a_row_its_samples_misplace():
    """Every value a sample can land on is the row's smallest, so the bracket the samples
    give holds no rank the k-th value can take."""
    vocab = 8192
    device = run_device()
    columns = torch.arange(vocab, device=device)
    logits = torch.where(columns % 4 == 0, 0.0, 1.0).to(torch.float32)[None]
    k = torch.tensor([vocab // 2], dtype=torch.int32, device=device)
    out = _run(TopKMaskFwdOp(), logits, k)
    compare_outputs(out, top_k_mask(logits, k), top_k_mask_verification(logits))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_min_p_mask(dtype):
    # Allow threshold-rounding differences only for tokens this close to the cutoff.
    margin = 1e-4
    vocab = 32000
    workload = MinPMaskWorkload(
        manifest_call("MinPMaskFwdOp", {"T": dtype}, B=4, V=vocab)
    )
    logits, min_p = workload.gen_inputs()
    ref = workload.ref_program(logits, min_p)
    _assert_top_set(ref, logits)
    probs = logits.float().softmax(-1)
    relative = probs / probs.amax(-1, keepdim=True) - min_p[:, None]
    assert (relative[ref == -_INF] < margin).all() and (
        relative[ref != -_INF] > -margin
    ).all()
    out = _run(MinPMaskFwdOp(), logits, min_p)
    compare_outputs(out, ref, workload.verification(logits, min_p))


def test_min_p_mask_rows_without_a_finite_threshold():
    """A row whose maximum is not a number passes through, at each shape the kernel plans for.

    ``V`` picks the plan: a row split across blocks, which reduces across a grid barrier; a
    row one block holds, which takes no barrier; and a row whose bytes are not a whole number
    of 16-byte vectors, which is read and written element by element.
    """
    device = run_device()
    for vocab in (32000, 512, 4099):
        logits = torch.randn(5, vocab, device=device)
        logits[1] = -_INF
        logits[2, vocab // 2] = float("nan")
        logits[3, 0] = _INF
        min_p = torch.tensor([0.0, 0.5, 0.5, 0.5, 1.0], device=device)
        out = _run(MinPMaskFwdOp(), logits, min_p)
        ref = min_p_mask(logits, min_p)
        compare_outputs(out, ref, min_p_mask_verification(logits, min_p))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_p_mask(dtype):
    # Allow threshold-rounding differences only for tokens this close to the cutoff.
    margin = 1e-4
    vocab = 32000
    workload = TopPMaskWorkload(
        manifest_call("TopPMaskFwdOp", {"T": dtype}, B=4, V=vocab)
    )
    logits, p = workload.gen_inputs()
    _tie_at_the_boundary(logits, p)
    ref = workload.ref_program(logits, p)
    _assert_nucleus(ref, logits, p, margin=margin)
    out = _run(TopPMaskFwdOp(), logits, p)
    compare_outputs(out, ref, workload.verification(logits, p))


def test_top_p_mask_rows_at_the_contract_endpoints():
    """``p = 0`` masks a row whole, ``p = 1`` keeps every token the softmax gives mass to,
    and a row with no finite maximum passes through, at each shape the kernel plans for.

    ``V`` picks the plan: a row split across blocks, which reduces across a grid barrier; a
    row one block holds, which takes no barrier; and a row whose bytes are not a whole number
    of 16-byte vectors, which is read and written element by element. The endpoints are held
    to the contract rather than to the reference, which rounds its own cumulative sum past 1
    at ``p = 1``; the rows with no finite maximum are held to the reference, which leaves
    them whole because a NaN never reaches ``p``.
    """
    device = run_device()
    p = torch.tensor([0.0, 1.0, 0.5], device=device)
    for vocab in (32000, 512, 4099):
        logits = torch.randn(3, vocab, device=device)
        out = _run(TopPMaskFwdOp(), logits, p)
        assert (out[0] == -_INF).all(), vocab
        assert torch.equal(out[1] != -_INF, logits.float().softmax(-1)[1] > 0), vocab
        # p=1 retains positive probabilities, excluding logits whose softmax underflows.
        logits[1, :5] = torch.tensor(
            [0.0, -90.0, -100.0, -200.0, -1000.0], device=device
        )
        out = _run(TopPMaskFwdOp(), logits, p)
        assert torch.equal(out[1] != -_INF, logits.float().softmax(-1)[1] > 0), vocab
        logits[0, vocab // 2] = float("nan")
        logits[1, 0] = _INF
        logits[2] = -_INF
        out = _run(TopPMaskFwdOp(), logits, p)
        ref = top_p_mask(logits, p)
        compare_outputs(out, ref, top_p_mask_verification(logits, p))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_top_k_top_p_mask(dtype):
    # Allow threshold-rounding differences only for tokens this close to the cutoff.
    margin = 1e-4
    vocab = 32000
    call = manifest_call(
        "TopKTopPMaskFwdOp", {"T": dtype}, V=vocab, k_list=[1, 50, vocab, vocab + 7]
    )
    workload = TopKTopPMaskWorkload(call)
    logits, k, p = workload.gen_inputs()
    _tie_at_the_boundary(logits, p)
    ref = workload.ref_program(logits, k, p)
    top_k = top_k_mask(logits, k)
    assert ((ref != -_INF) <= (top_k != -_INF)).all()
    _assert_nucleus(ref, top_k, p, margin=margin)
    out = _run(TopKTopPMaskFwdOp(), logits, k, p)
    compare_outputs(out, ref, workload.verification(logits, k, p))


@pytest.mark.parametrize(
    "dtype, vocab",
    [
        # 16-bit keys with the row on the vector, 16-bit and 32-bit keys with it off.
        (torch.bfloat16, 4096),
        (torch.float16, 999),
        (torch.float32, 999),
    ],
)
@pytest.mark.parametrize("repetitions", [1, 3])
def test_top_k_top_p_mask_cuts_special_rows_where_the_reference_does(
    dtype, vocab, repetitions
):
    """Row 1's top-k bound lands on a zero and the other zero is kept with it; rows 2 and 3
    have no finite largest logit once ``k`` leaves them whole, so the reference's
    probabilities are NaN, top-p removes nothing and the kept sets have to agree exactly.
    The NaN row is cut both ways: at a ``k`` that drops its NaNs and at one that keeps them."""
    # Allow threshold-rounding differences only for tokens this close to the cutoff.
    logits = _special_rows(vocab, dtype).repeat(repetitions, 1)
    device = logits.device
    logits[3::6, 5:] = -1000.0
    logits[3::6, 5:10] = torch.tensor(
        [5.0, 4.0, 3.0, 2.0, 1.0], dtype=dtype, device=device
    )
    p = torch.tensor(
        [0.9, 0.95, 0.8, 0.5, 0.6, 0.5], dtype=torch.float32, device=device
    ).repeat(repetitions)
    for nan_k in (2, 10, vocab):
        k = torch.tensor(
            [1, vocab // 2, vocab, nan_k, vocab // 4, vocab // 3],
            dtype=torch.int32,
            device=device,
        ).repeat(repetitions)
        ref = top_p_mask(top_k_mask(logits, k), p)
        out = _run(TopKTopPMaskFwdOp(), logits, k, p)
        compare_outputs(out, ref, top_p_mask_verification(logits, p, k=k))
        if nan_k == 10:
            assert ((out[3::6] != -_INF).sum(-1) == 1).all()


def test_top_k_masks_hold_a_row_in_one_cta():
    """Thread block clusters start at SM90, so on SM89 one CTA holds qwen3's bfloat16 row."""
    vocab = 151936
    k_list = [1, 50, vocab]
    top_k = TopKMaskWorkload(
        manifest_call("TopKMaskFwdOp", {"T": "bfloat16"}, V=vocab, k_list=k_list)
    )
    logits, k = top_k.gen_inputs()
    out = TopKMaskFwdOp()(logits, k)
    compare_outputs(out, top_k.ref_program(logits, k), top_k.verification(logits, k))

    nucleus = TopKTopPMaskWorkload(
        manifest_call("TopKTopPMaskFwdOp", {"T": "bfloat16"}, V=vocab, k_list=k_list)
    )
    logits, k, p = nucleus.gen_inputs()
    out = TopKTopPMaskFwdOp()(logits, k, p)
    compare_outputs(
        out, nucleus.ref_program(logits, k, p), nucleus.verification(logits, k, p)
    )


def _wide_row(vocab: int):
    device = run_device()
    logits = torch.randn(1, vocab, device=device, dtype=torch.bfloat16)
    return logits, torch.ones(1, dtype=torch.int32, device=device)


@pytest.mark.parametrize("combined", [False, True])
def test_split_row_masks_handle_rows_larger_than_ub(combined):
    """Replace the source's CUDA capacity refusals with NPUIR split-row coverage."""
    logits, k = _wide_row(1 << 19)
    if combined:
        p = torch.full((1,), 0.9, device=run_device())
        out = TopKTopPMaskFwdOp()(logits, k, p)
        compare_outputs(
            out,
            top_p_mask(top_k_mask(logits, k), p),
            top_p_mask_verification(logits, p, k=k),
        )
    else:
        out = TopKMaskFwdOp()(logits, k)
        compare_outputs(out, top_k_mask(logits, k), top_k_mask_verification(logits))


def test_sampling_from_probs():
    """Unnormalized rows with every fourth weight zero, drawn in 65536 identical rows."""
    n, vocab = 65536, 64
    device = run_device()
    weights = torch.rand(vocab, device=device)
    weights[::4] = 0
    probs = weights.expand(n, vocab).contiguous()
    seed = torch.tensor([1234], dtype=torch.int64, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    workload = SamplingFromProbsWorkload(
        manifest_call("SamplingFromProbsFwdOp", B=n, V=vocab)
    )
    ref = workload.ref_program(probs, seed, offset)
    assert torch.equal(ref, workload.ref_program(probs, seed, offset))
    op = SamplingFromProbsFwdOp()
    out = _run(op, probs, seed, offset)
    assert out.dtype == torch.int32
    TestBase.check(
        workload, op, probs, seed, offset, runs=lambda *args: _run(op, *args)
    )
    assert torch.equal(out, op(probs, seed, offset))


def test_sampling_from_probs_draws_the_same_token_at_every_launch_shape():
    """A row drawn inside batches that take different launches gives one token, and weight.

    The batch settles how many blocks share a row, and with it how the prefix sums the
    search descends are grouped. ``B = 1`` splits the row across the device and takes the
    grid barrier; a batch past the block count leaves the row to one block and takes none.
    Every seventh weight is zero, so the same case says the split row draws no zero weight.
    """
    device = run_device()
    torch.manual_seed(3)
    weights = torch.rand(151936, device=device)
    weights[::7] = 0
    row = weights / weights.sum()
    seed = torch.tensor([1234], dtype=torch.int64, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    drawn = {
        batch: _run(
            SamplingFromProbsFwdOp(), row.expand(batch, -1).contiguous(), seed, offset
        )[0]
        for batch in (1, 17, 300)
    }
    assert len(set(int(token) for token in drawn.values())) == 1, drawn
    assert (row[torch.stack(list(drawn.values())).long()] > 0).all()
    assert int(drawn[1]) == int(sampling_from_probs(row[None], seed, offset)[0])


def test_chain_speculative_sampling():
    """Two drafts per row over 16 tokens, each draft drawn from its draft row, in 65536 rows."""
    n, num_draft, vocab = 65536, 2, 16
    device = run_device()
    draft = torch.rand(num_draft, vocab, device=device) ** 3
    draft /= draft.sum(-1, keepdim=True)
    target = torch.rand(num_draft + 1, vocab, device=device) ** 3
    target /= target.sum(-1, keepdim=True)
    draft_ids = (
        torch.multinomial(draft, n, replacement=True).T.to(torch.int32).contiguous()
    )
    # A draft drawn from its row is accepted with probability sum(min(draft, target)).
    inputs = (
        draft.expand(n, num_draft, vocab).contiguous(),
        draft_ids,
        target.expand(n, num_draft + 1, vocab).contiguous(),
        torch.tensor([1234], dtype=torch.int64, device=device),
        torch.tensor([7], dtype=torch.int64, device=device),
    )
    call = manifest_call("ChainSpeculativeSamplingFwdOp", B=n, N=num_draft, V=vocab)
    workload = ChainSpeculativeSamplingWorkload(call)
    ref = workload.ref_program(*inputs)
    assert all(map(torch.equal, ref, workload.ref_program(*inputs)))
    op = ChainSpeculativeSamplingFwdOp()
    out = _run(op, *inputs)
    TestBase.check(workload, op, *inputs, runs=lambda *args: _run(op, *args))
    assert all(map(torch.equal, out, op(*inputs)))


@pytest.mark.parametrize(
    "batch, vocab, num_draft",
    [
        # A row split across blocks, a row one block holds, a row whose bytes are not a whole
        # number of 16-byte vectors, which is folded weight by weight, and a chain longer than
        # the block, whose uniforms one thread each takes over several rounds.
        (2, 151936, 3),
        (300, 32000, 3),
        (8, 4099, 3),
        (4, 64, 600),
    ],
)
def test_chain_speculative_sampling_accepts_the_reference_prefix(
    batch, vocab, num_draft
):
    """``num_accepted`` and the accepted prefix are the reference's, at every launch shape.

    The uniforms and the float32 ``u * draft < target`` test are the reference's, and neither
    names a launch fact, so the verification agrees exactly however the row is split. The
    token after the prefix follows the same distribution but is not required to be the
    reference's index.
    """
    device = run_device()
    torch.manual_seed(11)
    draft = torch.randn(batch, num_draft, vocab, device=device).softmax(-1)
    target = torch.randn(batch, num_draft + 1, vocab, device=device).softmax(-1)
    ids = torch.multinomial(draft.reshape(-1, vocab), 1).view(batch, num_draft)
    inputs = (
        draft,
        ids.to(torch.int32).contiguous(),
        target,
        torch.tensor([1234], dtype=torch.int64, device=device),
        torch.tensor([7], dtype=torch.int64, device=device),
    )
    workload = ChainSpeculativeSamplingWorkload(
        manifest_call("ChainSpeculativeSamplingFwdOp", B=batch, N=num_draft, V=vocab)
    )
    op = ChainSpeculativeSamplingFwdOp()
    # The dedicated chain test exercises the shared distribution probe once.
    # These cases cover launch boundaries, exact acceptance and residual-token support.
    out, ref = _run(op, *inputs), workload.ref_program(*inputs)
    compare_outputs(out, ref, workload.verification(*inputs))
    assert torch.equal(out[1], ref[1])
    position = torch.arange(num_draft + 1, device=device)[None]
    accepted = position < ref[1][:, None]
    assert torch.equal(out[0][accepted], ref[0][accepted])


_SMALL_CALLS = {
    TopKMaskFwdOp: (TopKMaskWorkload, {"T": "float32"}, {"V": 64, "k_list": [1, 5]}),
    MinPMaskFwdOp: (MinPMaskWorkload, {"T": "float32"}, {"B": 2, "V": 64}),
    TopPMaskFwdOp: (TopPMaskWorkload, {"T": "float32"}, {"B": 2, "V": 64}),
    TopKTopPMaskFwdOp: (
        TopKTopPMaskWorkload,
        {"T": "float32"},
        {"V": 64, "k_list": [1, 5]},
    ),
    SamplingFromProbsFwdOp: (SamplingFromProbsWorkload, None, {"B": 2, "V": 64}),
    ChainSpeculativeSamplingFwdOp: (
        ChainSpeculativeSamplingWorkload,
        None,
        {"B": 2, "N": 2, "V": 64},
    ),
}


@pytest.mark.parametrize("cls", list(_SMALL_CALLS), ids=lambda cls: cls.__name__)
def test_generated_checks_reject_an_integer_first_input(cls):
    workload_cls, dtype_case, row = _SMALL_CALLS[cls]
    inputs = list(
        workload_cls(manifest_call(cls.__name__, dtype_case, **row)).gen_inputs()
    )
    inputs[0] = inputs[0].to(torch.int32)
    with pytest.raises(ValueError):
        cls()(*inputs)


@pytest.mark.parametrize("cls", list(_SMALL_CALLS), ids=lambda cls: cls.__name__)
def test_probabilities_off_a_vector_boundary_give_the_aligned_result(cls):
    """Logits or probabilities starting one element past a 16-byte boundary sample as their
    aligned copy does; the draws are seeded."""
    workload_cls, dtype_case, row = _SMALL_CALLS[cls]
    inputs = list(
        workload_cls(manifest_call(cls.__name__, dtype_case, **row)).gen_inputs()
    )
    want = _run(cls(), *inputs)
    # The float inputs a row is read from: chain speculative sampling's draft and target.
    for i in (0, 2) if cls is ChainSpeculativeSamplingFwdOp else (0,):
        moved = torch.empty(
            inputs[i].numel() + 1, dtype=inputs[i].dtype, device=run_device()
        )
        inputs[i] = moved[1:].copy_(inputs[i].reshape(-1)).view(inputs[i].shape)
    got = cls()(*inputs)
    if isinstance(want, tuple):
        assert all(map(torch.equal, got, want))
    else:
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    "dtype, storage, nan_bits, one_bits, negative_zero",
    [
        (torch.float16, torch.int16, 0x7C01, 0x3C00, -32768),
        (torch.bfloat16, torch.int16, 0x7F81, 0x3F80, -32768),
        (torch.float32, torch.int32, 0x7FA12345, 0x3F800000, -2147483648),
    ],
)
def test_min_p_mask_preserves_retained_storage_bits(
    dtype, storage, nan_bits, one_bits, negative_zero
):
    """Promoting and narrowing retained logits must not quiet NaNs or change their payloads."""
    device = run_device()
    bits = torch.tensor(
        [
            [nan_bits, one_bits, negative_zero, 0],
            [one_bits, negative_zero, 0, one_bits],
        ],
        dtype=storage,
        device=device,
    )
    logits = bits.view(dtype)
    min_p = torch.tensor([1.0, 0.0], device=device)
    out = _run(MinPMaskFwdOp(), logits, min_p)
    # The first row's NaN threshold and the second row's disabled mask retain every bit.
    assert torch.equal(out.view(storage), bits)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cls", [TopPMaskFwdOp, TopKTopPMaskFwdOp])
def test_top_p_keeps_probabilities_tied_by_float32_rounding(dtype, cls):
    """Distinct logits whose softmax probabilities tie must cross the boundary together."""
    device = run_device()
    logits = torch.tensor([[0.0, 1e-8, -200.0]], dtype=dtype, device=device)
    p = torch.tensor([0.25], device=device)
    if cls is TopPMaskFwdOp:
        out = _run(cls(), logits, p)
    else:
        k = torch.tensor([2], dtype=torch.int32, device=device)
        out = _run(cls(), logits, k, p)
    probs = logits.float().softmax(-1)
    assert probs[0, 0] == probs[0, 1]
    assert torch.equal(out != -_INF, torch.tensor([[True, True, False]], device=device))
    assert torch.equal(out[:, :2], logits[:, :2])


def test_sampling_from_probs_never_draws_a_trailing_zero_probability():
    """A uniform near one must still select positive support after accumulated roundoff."""
    weights = np.random.default_rng(77).random((48, 256), dtype=np.float32)[-1]
    weights /= weights.sum()
    probs = torch.zeros((1, 258 * 8192), dtype=torch.float32)
    probs[0, : 256 * 8192 : 8192] = torch.from_numpy(weights)
    device = run_device()
    probs = probs.to(device)
    seed = torch.tensor([2160754], dtype=torch.int64, device=device)
    offset = torch.tensor([7], dtype=torch.int64, device=device)
    token = _run(SamplingFromProbsFwdOp(), probs, seed, offset)
    assert ((token >= 0) & (token < probs.shape[1])).all()
    assert (probs.gather(1, token.long()[:, None]) > 0).all()
