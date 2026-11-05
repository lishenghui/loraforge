# SPDX-License-Identifier: Apache-2.0
"""Experimental NF4/BF16 projection kernels; imports perform no GPU work.

Weights keep canonical bitsandbytes row-major NF4 bytes. Expanded FP32 block
scales may be prepared once; complete BF16 weight matrices are never cached.
The dequantized value is rounded to BF16 before MFMA, as in the native fallback.
These are inference-only TP1 benchmark candidates, not installed dispatch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _code_value(code):
    # Canonical NF4 codebook, expressed as register selections instead of
    # per-element indirect global loads. Exact FP32 constants, not an approximation.
    a = tl.where((code & 1) != 0, -0.6961928009986877, -1.0)
    b = tl.where((code & 1) != 0, -0.39491748809814453, -0.5250730514526367)
    c = tl.where((code & 1) != 0, -0.18477343022823334, -0.28444138169288635)
    d = tl.where((code & 1) != 0, 0.0, -0.09105003625154495)
    e = tl.where((code & 1) != 0, 0.16093020117282867, 0.07958029955625534)
    f = tl.where((code & 1) != 0, 0.33791524171829224, 0.24611230194568634)
    g = tl.where((code & 1) != 0, 0.5626170039176941, 0.44070982933044434)
    h = tl.where((code & 1) != 0, 1.0, 0.7229568362236023)
    ab = tl.where((code & 2) != 0, b, a)
    cd = tl.where((code & 2) != 0, d, c)
    ef = tl.where((code & 2) != 0, f, e)
    gh = tl.where((code & 2) != 0, h, g)
    low = tl.where((code & 4) != 0, cd, ab)
    high = tl.where((code & 4) != 0, gh, ef)
    return tl.where((code & 8) != 0, high, low)


@triton.jit
def _values(Q, S, C, index, valid):
    byte = tl.load(Q + index // 2, mask=valid, other=0).to(tl.int32)
    code = tl.where(index % 2 == 0, byte >> 4, byte & 15)
    value = _code_value(code)
    scale = tl.load(S + index // 64, mask=valid, other=0)
    return (value * scale).to(tl.bfloat16)


@triton.jit
def _dequant(Q, S, C, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, _values(Q, S, C, i, i < SIZE), mask=i < SIZE)


@triton.jit
def _matmul(
    X,
    Q,
    S,
    C,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(K, BK)):
        ki = start * BK + k
        a = tl.load(
            X + m[:, None] * SX + ki[None, :],
            mask=(m[:, None] < M) & (ki[None, :] < K),
            other=0,
        )
        index = n[:, None] * K + ki[None, :]
        b = _values(Q, S, C, index, (n[:, None] < N) & (ki[None, :] < K))
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(
        Y + m[:, None] * N + n[None, :],
        acc.to(tl.bfloat16),
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def prepare_scales(state):
    """Own a lossless FP32 scale vector; run before capture, for immutable weights."""
    from bitsandbytes.functional import dequantize_blockwise

    if (
        state.quant_type != "nf4"
        or state.blocksize != 64
        or state.dtype != torch.bfloat16
    ):
        raise ValueError("requires NF4 block64 BF16 source weights")
    expected = torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020117282867,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=torch.float32,
        device=state.code.device,
    )
    if not torch.equal(expected, state.code):
        raise ValueError("requires the canonical NF4 codebook")
    if state.nested:
        return (
            dequantize_blockwise(state.absmax, state.state2) + state.offset
        ).contiguous()
    return state.absmax.contiguous()


def dequantize(qweight, scales, code, shape, *, out=None):
    if out is None:
        out = torch.empty(shape, device=qweight.device, dtype=torch.bfloat16)
    _dequant[(triton.cdiv(out.numel(), 1024),)](
        qweight,
        scales,
        code,
        out,
        out.numel(),
        1024,
        enable_fp_fusion=False,
    )
    return out


def matmul(x, qweight, scales, code, shape, *, config=(16, 64, 64, 4), out=None):
    """Return BF16 X @ dequant(Q).T with FP32 accumulation on current stream."""
    n, k = shape
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != k or x.stride(1) != 1:
        raise ValueError("requires row-contiguous BF16 [M,K] activations")
    if out is None:
        out = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    bm, bn, bk, warps = config
    _matmul[(triton.cdiv(x.shape[0], bm), triton.cdiv(n, bn))](
        x,
        qweight,
        scales,
        code,
        out,
        x.shape[0],
        n,
        k,
        x.stride(0),
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=2,
        matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    return out
