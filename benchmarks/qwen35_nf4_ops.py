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

from loraforge.ops._nf4_kernels import _code_value, _dequant_pairs, _values  # noqa: F401
from loraforge.ops.nf4 import (
    _dequantize,
    prepare_scales,  # noqa: F401
)


def dequantize(qweight, scales, code, shape, *, out=None):
    return _dequantize(qweight, scales, code, shape, byte_pairs=False, out=out)


@triton.jit
def _split_matmul(
    X,
    Q,
    S,
    C,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    start = tl.program_id(2) * tl.cdiv(K, SPLIT * BK) * BK
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(K, SPLIT * BK)):
        ki = start + step * BK + k
        a = tl.load(
            X + m[:, None] * SX + ki[None, :],
            mask=(m[:, None] < M) & (ki[None, :] < K),
            other=0,
        )
        idx = n[:, None] * K + ki[None, :]
        b = _values(Q, S, C, idx, (n[:, None] < N) & (ki[None, :] < K))
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(
        P + (tl.program_id(2) * M + m[:, None]) * N + n[None, :],
        acc,
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _reduce_split(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for s in tl.static_range(SPLIT):
        total += tl.load(P + s * SIZE + i, mask=i < SIZE, other=0)
    tl.store(Y + i, total.to(tl.bfloat16), mask=i < SIZE)


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


def dequantize_pairs(
    qweight, scales, code, shape, *, block=1024, warps=4, lookup=False, out=None
):
    """Decode each packed byte once, preserving exact BF16 reconstruction."""
    if out is None:
        out = torch.empty(shape, device=qweight.device, dtype=torch.bfloat16)
    _dequant_pairs[(triton.cdiv(out.numel(), block * 2),)](
        qweight,
        scales,
        code,
        out,
        out.numel(),
        block,
        lookup,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out


def split_matmul(x, qweight, scales, code, shape, *, config=(16, 32, 64, 8, 4)):
    """Experimental split-K, FP32 partials; reduction order differs from native."""
    n, k = shape
    m = x.shape[0]
    bm, bn, bk, split, warps = config
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _split_matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        x,
        qweight,
        scales,
        code,
        partial,
        m,
        n,
        k,
        x.stride(0),
        bm,
        bn,
        bk,
        split,
        num_warps=warps,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    _reduce_split[(triton.cdiv(y.numel(), 256),)](
        partial,
        y,
        y.numel(),
        split,
        256,
    )
    return y


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
