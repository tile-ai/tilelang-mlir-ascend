"""Standalone sampling entry points. All numerical work runs in NPUIR kernels.

Inputs must share an NPU device. Logits support fp16/bf16/fp32; sampling
probabilities are float32. Probability rows must be finite, nonnegative and
have positive mass (normalized for speculative sampling); draft IDs must be
in range and have positive draft probability. These value preconditions are
not copied to the host for validation. k >= 1 and p in [0, 1] are required.
"""

import torch

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _tensor(x, name, shape=None, dtypes=_FLOATS, device=None):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if x.dtype not in dtypes:
        raise ValueError(f"{name} has unsupported dtype {x.dtype}")
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(x.shape)}")
    if device is not None and x.device != device:
        raise ValueError(f"{name} must be on {device}")


def _matrix(x, name, dtypes=_FLOATS):
    _tensor(x, name, dtypes=dtypes)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError(f"{name} must be [batch, positive vocabulary]")
    return x.shape


def _npu(x):
    if x.device.type != "npu":
        raise RuntimeError(
            "sampling requires an Ascend NPU; no CPU fallback is installed"
        )


def _aligned(x):
    x = x.contiguous()
    # A contiguous view can still start inside an unaligned storage allocation.
    return x.clone() if x.storage_offset() * x.element_size() % 32 else x


def _raw(x):
    return x.view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def _dtype(x):
    return str(x.dtype).removeprefix("torch.")


def _empty(x, shape, dtype):
    return torch.empty(shape, dtype=dtype, device=x.device)


def _maximum(x):
    from .min_p import build_maximum

    maximum = _empty(x, (x.shape[0],), torch.float32)
    build_maximum(*x.shape, _dtype(x))(x, maximum)
    return maximum


class TopKMaskFwdOp:
    def __call__(self, logits, k):
        batch, vocab = _matrix(logits, "logits")
        _tensor(k, "k", (batch,), (torch.int32,), logits.device)
        _npu(logits)
        if batch == 0:
            return torch.empty_like(logits)
        from .radix_reduce import build_cutoff
        from .top_k_split import build_keys, build_top_k_write

        x, k = _aligned(logits), _aligned(k)
        keys = _empty(x, x.shape, torch.int64)
        cutoff = _empty(x, (batch,), torch.int64)
        unused_weights = _empty(x, (batch, 1), torch.float32)
        out = torch.empty_like(x)
        build_keys(batch, vocab, _dtype(x))(_raw(x), keys)
        build_cutoff(batch, vocab, "int64", x.element_size() * 8)(
            keys, unused_weights, k, cutoff
        )
        build_top_k_write(batch, vocab, _dtype(x))(_raw(x), keys, k, cutoff, _raw(out))
        return out


class MinPMaskFwdOp:
    def __call__(self, logits, min_p):
        batch, vocab = _matrix(logits, "logits")
        _tensor(min_p, "min_p", (batch,), (torch.float32,), logits.device)
        _npu(logits)
        if batch == 0:
            return torch.empty_like(logits)
        from .min_p import build_min_p_write

        x, p = _aligned(logits), _aligned(min_p)
        out = torch.empty_like(x)
        build_min_p_write(batch, vocab, _dtype(x))(
            x, _raw(x), p, _maximum(x), _raw(out)
        )
        return out


class TopPMaskFwdOp:
    def __call__(self, logits, p):
        batch, vocab = _matrix(logits, "logits")
        _tensor(p, "p", (batch,), (torch.float32,), logits.device)
        _npu(logits)
        if batch == 0:
            return torch.empty_like(logits)
        from .masks import build_probabilities, build_top_p_write
        from .radix_reduce import build_cutoff

        x, p = _aligned(logits), _aligned(p)
        maximum = _maximum(x)
        probabilities = _empty(x, x.shape, torch.float32)
        cutoff = _empty(x, (batch,), torch.int64)
        out = torch.empty_like(x)
        build_probabilities(batch, vocab, _dtype(x))(x, maximum, probabilities)
        bits = probabilities.view(torch.int32)
        build_cutoff(batch, vocab, "int32", 31, True)(bits, probabilities, p, cutoff)
        build_top_p_write(batch, vocab, _dtype(x))(
            _raw(x), bits, p, maximum, cutoff, _raw(out)
        )
        return out


class TopKTopPMaskFwdOp:
    def __call__(self, logits, k, p):
        batch, _ = _matrix(logits, "logits")
        _tensor(k, "k", (batch,), (torch.int32,), logits.device)
        _tensor(p, "p", (batch,), (torch.float32,), logits.device)
        return TopPMaskFwdOp()(TopKMaskFwdOp()(logits, k), p)


def _seed_offset(seed, offset, x):
    _tensor(seed, "seed", (1,), (torch.int64,), x.device)
    _tensor(offset, "offset", (1,), (torch.int64,), x.device)


def _uniforms(x, seed, offset, draws):
    from .chain_prepare import build_uniforms

    uniform = _empty(x, (x.shape[0] * draws,), torch.float32)
    build_uniforms(x.shape[0], draws)(_aligned(seed), _aligned(offset), uniform)
    return uniform.view(x.shape[0], draws)


def _draw(draft, ids, target, uniform, accepted, chain):
    from .draw_reduce import build_parts
    from .draws import build_draw
    from .radix_reduce import TILE

    batch, draws, vocab = target.shape
    weights = _empty(target, (batch, vocab), torch.float32)
    parts = _empty(target, (batch, (vocab + TILE - 1) // TILE), torch.float32)
    tokens = _empty(target, (batch, draws), torch.int32)
    build_parts(batch, draws - 1, vocab, chain)(draft, target, accepted, weights, parts)
    build_draw(batch, draws - 1, vocab)(weights, parts, uniform, ids, accepted, tokens)
    return tokens


class SamplingFromProbsFwdOp:
    def __call__(self, probs, seed, offset):
        batch, vocab = _matrix(probs, "probs", (torch.float32,))
        _seed_offset(seed, offset, probs)
        _npu(probs)
        if batch == 0:
            return _empty(probs, (0,), torch.int32)
        x = _aligned(probs)
        uniform = _uniforms(x, seed, offset, 1)
        accepted = torch.zeros(batch, dtype=torch.int32, device=x.device)
        ids = _empty(x, (batch, 1), torch.int32)
        target = x.view(batch, 1, vocab)
        return _draw(target, ids, target, uniform, accepted, False).view(batch)


class ChainSpeculativeSamplingFwdOp:
    def __call__(self, draft_probs, draft_token_ids, target_probs, seed, offset):
        _tensor(draft_probs, "draft_probs", dtypes=(torch.float32,))
        if draft_probs.ndim != 3 or draft_probs.shape[2] == 0:
            raise ValueError("draft_probs must be [batch, drafts, positive vocabulary]")
        batch, drafts, vocab = draft_probs.shape
        _tensor(
            draft_token_ids,
            "draft_token_ids",
            (batch, drafts),
            (torch.int32,),
            draft_probs.device,
        )
        _tensor(
            target_probs,
            "target_probs",
            (batch, drafts + 1, vocab),
            (torch.float32,),
            draft_probs.device,
        )
        _seed_offset(seed, offset, draft_probs)
        _npu(draft_probs)
        if batch == 0:
            return _empty(draft_probs, (0, drafts + 1), torch.int32), _empty(
                draft_probs, (0,), torch.int32
            )
        if drafts == 0:
            tokens = SamplingFromProbsFwdOp()(target_probs[:, 0], seed, offset)
            return tokens[:, None], torch.zeros(
                batch, dtype=torch.int32, device=draft_probs.device
            )
        from .chain_prepare import build_acceptance

        draft, ids, target = map(_aligned, (draft_probs, draft_token_ids, target_probs))
        uniform = _uniforms(draft, seed, offset, drafts + 1)
        accepted = _empty(draft, (batch,), torch.int32)
        build_acceptance(batch, drafts, vocab)(draft, ids, target, uniform, accepted)
        return _draw(draft, ids, target, uniform, accepted, True), accepted
