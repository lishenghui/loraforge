# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Adapted from probe GEMV kernels; see docs/provenance.md.
"""Split-K decode kernels for SGLang-compatible packed fused slices."""

import triton
import triton.language as tl


@triton.jit
def shrink_kernel(
    X,
    A,
    P,
    MAP,
    RANKS,
    M,
    K: tl.constexpr,
    RMAX: tl.constexpr,
    S: tl.constexpr,
    SK: tl.constexpr,
    KPS: tl.constexpr,
    SX,
    BR: tl.constexpr,
    BK: tl.constexpr,
    RLO: tl.constexpr = -1,
    PW: tl.constexpr = 0,
):
    t = tl.program_id(0)
    ks = tl.program_id(1)
    s = tl.program_id(2)
    g = tl.load(MAP + t)
    rank = tl.minimum(tl.load(RANKS + g), RMAX)
    if RLO >= 0:  # rank-bucketed launch: this launch serves RLO < rank <= BR only
        if (rank <= RLO) | (rank > BR):
            return
    rr = tl.arange(0, BR)
    kk0 = tl.arange(0, BK)
    acc = tl.zeros((BR,), tl.float32)
    for k in range(ks * KPS, (ks + 1) * KPS, BK):
        kk = k + kk0
        xv = tl.load(X + t * SX + kk, kk < K, 0).to(tl.float32)
        av = tl.load(
            A + g * S * RMAX * K + (s * rank + rr[:, None]) * K + kk[None, :],
            (rr[:, None] < rank) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        acc += tl.sum(av * xv[None, :], 1)
    if PW > 0:
        tl.store(P + ((s * M + t) * SK + ks) * PW + rr, acc, rr < PW)
    else:
        tl.store(P + ((s * M + t) * SK + ks) * RMAX + rr, acc, rr < RMAX)


@triton.jit
def reduce_kernel(
    P,
    OUT,
    MAP,
    RANKS,
    M,
    RMAX: tl.constexpr,
    S: tl.constexpr,
    SK: tl.constexpr,
    BSK: tl.constexpr,
    BR: tl.constexpr,
    RLO: tl.constexpr = -1,
    PW: tl.constexpr = 0,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    g = tl.load(MAP + t)
    rank = tl.minimum(tl.load(RANKS + g), RMAX)
    if RLO >= 0:
        if (rank <= RLO) | (rank > BR):
            return
    rr = tl.arange(0, BR)
    kk = tl.arange(0, BSK)
    W: tl.constexpr = PW if PW > 0 else RMAX
    vals = tl.load(
        P + ((s * M + t) * SK + kk[:, None]) * W + rr[None, :],
        (kk[:, None] < SK) & (rr[None, :] < W),
        0,
    )
    acc = tl.sum(vals, 0)
    tl.store(OUT + t * S * RMAX + s * rank + rr, acc, rr < rank)


@triton.jit
def expand_kernel(
    V,
    B,
    Y,
    MAP,
    RANKS,
    SCALES,
    OFFSETS,
    N: tl.constexpr,
    RMAX: tl.constexpr,
    S: tl.constexpr,
    SY0,
    SY1,
    BR: tl.constexpr,
    BN: tl.constexpr,
    RLO: tl.constexpr = -1,
):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    s = tl.program_id(2)
    g = tl.load(MAP + t)
    rank = tl.minimum(tl.load(RANKS + g), RMAX)
    if rank == 0:
        return
    if RLO >= 0:
        if (rank <= RLO) | (rank > BR):
            return
    start = tl.load(OFFSETS + s)
    end = tl.load(OFFSETS + s + 1)
    nn = start + nb * BN + tl.arange(0, BN)
    rr = tl.arange(0, BR)
    vv = tl.load(V + t * S * RMAX + s * rank + rr, rr < rank, 0).to(tl.float32)
    bb = tl.load(
        B + g * N * RMAX + nn[:, None] * RMAX + rr[None, :],
        (nn[:, None] < end) & (rr[None, :] < rank),
        0,
    ).to(tl.float32)
    delta = tl.sum(bb * vv[None, :], 1) * tl.load(SCALES + g)
    # Native CSGMV rounds the LoRA delta to the intermediate dtype before add.
    delta = delta.to(V.dtype.element_ty).to(tl.float32)
    yp = Y + t * SY0 + nn * SY1
    base = tl.load(yp, nn < end, 0).to(tl.float32)
    tl.store(yp, delta + base, nn < end)


@triton.jit
def shrink_dot_kernel(
    X, A, Y, MAP, RANKS, K: tl.constexpr, RMAX: tl.constexpr, S: tl.constexpr, SX
):
    """One routed token per tile, with sequential FP32 matrix-dot blocks."""
    t, block = tl.program_id(0), tl.program_id(1)
    g = tl.load(MAP + t)
    rank = tl.minimum(tl.load(RANKS + g), RMAX)
    rows = tl.arange(0, 16)
    cols = block * 16 + tl.arange(0, 16)
    kk = tl.arange(0, 256)
    acc = tl.zeros((16, 16), tl.float32)
    for start in range(tl.cdiv(K, 256)):
        k = start * 256 + kk
        x = tl.load(X + t * SX + k[None, :], (rows[:, None] == 0) & (k[None, :] < K), 0)
        a = tl.load(
            A + g * S * RMAX * K + cols[None, :] * K + k[:, None],
            (k[:, None] < K) & (cols[None, :] < S * rank),
            0,
        )
        acc += tl.dot(x, a)
    tl.store(
        Y + t * S * RMAX + cols[None, :],
        acc,
        (rows[:, None] == 0) & (cols[None, :] < S * rank),
    )


@triton.jit
def expand_dot_kernel(
    V,
    B,
    Y,
    MAP,
    RANKS,
    SCALES,
    OFFSETS,
    N: tl.constexpr,
    RMAX: tl.constexpr,
    S: tl.constexpr,
    SY0,
    SY1,
):
    t, block, sl = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    g = tl.load(MAP + t)
    rank = tl.minimum(tl.load(RANKS + g), RMAX)
    if rank == 0:
        return
    begin, end = tl.load(OFFSETS + sl), tl.load(OFFSETS + sl + 1)
    rows = tl.arange(0, 16)
    cols = begin + block * 64 + tl.arange(0, 64)
    kk = tl.arange(0, 16)
    acc = tl.zeros((16, 64), tl.float32)
    for start in range(tl.cdiv(rank, 16)):
        k = start * 16 + kk
        v = tl.load(
            V + t * S * RMAX + sl * rank + k[None, :],
            (rows[:, None] == 0) & (k[None, :] < rank),
            0,
        )
        b = tl.load(
            B + g * N * RMAX + cols[None, :] * RMAX + k[:, None],
            (k[:, None] < rank) & (cols[None, :] < end),
            0,
        )
        acc += tl.dot(v, b)
    delta = (acc * tl.load(SCALES + g)).to(V.dtype.element_ty)
    ptr = Y + t * SY0 + cols[None, :] * SY1
    mask = (rows[:, None] == 0) & (cols[None, :] < end)
    base = tl.load(ptr, mask, 0)
    tl.store(ptr, delta + base, mask)
