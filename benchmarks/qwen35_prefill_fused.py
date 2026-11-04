# SPDX-License-Identifier: Apache-2.0
"""Rank4 research operators: grouped B-at-SiLU and segmented two-dot A+B.

Native BF16 pools, effective-rank-packed A/U, int32 segment/rank/permutation
metadata and float32 scaling are caller-owned resident tensors. Inference only.
No serving installer and no device work at module import.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def expand_act_kernel(
    Y,
    U,
    B,
    IND,
    GROUP,
    PERM,
    RANK,
    SCALE,
    O,  # noqa: E741 - preserve the measured Triton argument name.
    YS: tl.constexpr,
    US: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ROUND_ACT: tl.constexpr,
):
    seg = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    begin = tl.load(IND + seg)
    end = tl.load(IND + seg + 1)
    rows = begin + tl.arange(0, BM)
    live = rows < end
    physical = tl.load(PERM + rows, mask=live, other=0)
    g = tl.load(GROUP + seg)
    safe = tl.maximum(g, 0)
    r = tl.load(RANK + safe)
    scale = tl.load(SCALE + safe)
    cg = tl.full((BM, BN), 0, tl.float32)
    cu = tl.full((BM, BN), 0, tl.float32)
    for k in tl.static_range(4):
        ug = tl.load(U + physical * US + k, mask=live & (g >= 0) & (k < r), other=0).to(
            tl.float32
        )
        uu = tl.load(
            U + physical * US + r + k, mask=live & (g >= 0) & (k < r), other=0
        ).to(tl.float32)
        bg = tl.load(
            B + (safe * 2 * N + col) * 4 + k,
            mask=(col < N) & (g >= 0) & (k < r),
            other=0,
        ).to(tl.float32)
        bu = tl.load(
            B + (safe * 2 * N + N + col) * 4 + k,
            mask=(col < N) & (g >= 0) & (k < r),
            other=0,
        ).to(tl.float32)
        cg = cg + ug[:, None] * bg[None, :]
        cu = cu + uu[:, None] * bu[None, :]
    gate = tl.load(
        Y + physical[:, None] * YS + col[None, :],
        mask=live[:, None] & (col[None, :] < N),
        other=0,
    ).to(tl.float32)
    up = tl.load(
        Y + physical[:, None] * YS + N + col[None, :],
        mask=live[:, None] & (col[None, :] < N),
        other=0,
    ).to(tl.float32)
    gate = (
        (gate + (cg * scale).to(tl.bfloat16).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    up = (
        (up + (cu * scale).to(tl.bfloat16).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    act = gate / (1 + tl.exp(-gate))
    if ROUND_ACT:
        act = act.to(tl.bfloat16).to(tl.float32)
    value = (act * up).to(tl.bfloat16)
    tl.store(
        O + physical[:, None] * N + col[None, :],
        value,
        mask=live[:, None] & (col[None, :] < N),
    )


@triton.jit
def segmented_ab_kernel(
    X,
    A,
    B,
    Y,
    IND,
    GROUP,
    PERM,
    RANK,
    SCALE,
    XS: tl.constexpr,
    YS: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    SPLIT0: tl.constexpr,
    SPLIT1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    seg = tl.program_id(0)
    start = tl.program_id(1) * BN
    begin = tl.load(IND + seg)
    end = tl.load(IND + seg + 1)
    g = tl.load(GROUP + seg)
    safe = tl.maximum(g, 0)
    r = tl.load(RANK + safe)
    if (g >= 0) & (r > 0) & (begin < end):
        rows = begin + tl.arange(0, BM)
        live = rows < end
        physical = tl.load(PERM + rows, mask=live, other=0)
        rr = tl.arange(0, 16)
        kk = tl.arange(0, BK)
        cols = start + tl.arange(0, BN)
        sl = 0
        if S == 2:
            sl = start >= SPLIT0
        elif S == 3:
            sl = (start >= SPLIT0).to(tl.int32) + (start >= SPLIT1).to(tl.int32)
        elif S == 4:
            sl = start // SPLIT0
        acc = tl.full((BM, 16), 0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            ki = block * BK + kk
            xx = tl.load(
                X + physical[:, None] * XS + ki[None, :],
                mask=live[:, None] & (ki[None, :] < K),
                other=0,
            )
            aa = tl.load(
                A + (safe * S * 4 + sl * r + rr[None, :]) * K + ki[:, None],
                mask=(ki[:, None] < K) & (rr[None, :] < r),
                other=0,
            )
            acc += tl.dot(xx, aa)
        u = acc.to(tl.bfloat16)
        bb = tl.load(
            B + (safe * N + cols[None, :]) * 4 + rr[:, None],
            mask=(cols[None, :] < N) & (rr[:, None] < r),
            other=0,
        )
        delta = tl.dot(u, bb)
        scale = tl.load(SCALE + safe)
        base = tl.load(
            Y + physical[:, None] * YS + cols[None, :],
            mask=live[:, None] & (cols[None, :] < N),
            other=0,
        ).to(tl.float32)
        out = (base + (delta * scale).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(
            Y + physical[:, None] * YS + cols[None, :],
            out,
            mask=live[:, None] & (cols[None, :] < N),
        )


def expand_act(y, u, b, info, ranks, *, tile=128, round_act=True, out=None):
    if out is None:
        out = torch.empty((y.shape[0], 3584), dtype=y.dtype, device=y.device)
    segments = (
        info.weight_indices.shape[0] if info.use_cuda_graph else info.num_segments
    )
    expand_act_kernel[(segments, triton.cdiv(3584, tile))](
        y,
        u,
        b,
        info.seg_indptr,
        info.weight_indices,
        info.permutation,
        ranks,
        info.scalings,
        out,
        y.stride(0),
        u.shape[1],
        3584,
        16,
        tile,
        round_act,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def segmented_ab(x, a, b, y, info, ranks, slices, offsets, *, tile=256, block_k=64):
    assert block_k in (64, 256)
    segments = (
        info.weight_indices.shape[0] if info.use_cuda_graph else info.num_segments
    )
    n = y.shape[1]
    assert all(o % tile == 0 for o in offsets)
    segmented_ab_kernel[(segments, triton.cdiv(n, tile))](
        x,
        a,
        b,
        y,
        info.seg_indptr,
        info.weight_indices,
        info.permutation,
        ranks,
        info.scalings,
        x.stride(0),
        y.stride(0),
        x.shape[1],
        n,
        slices,
        offsets[1] if slices > 1 else n,
        offsets[2] if slices == 3 else 0,
        16,
        tile,
        block_k,
        num_warps=4,
    )
    return y
