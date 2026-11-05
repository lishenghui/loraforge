# SPDX-License-Identifier: Apache-2.0
"""Per-row tiny B-at-SiLU candidate; no padded16-row grouping."""

import torch
import triton
import triton.language as tl


@triton.jit
def row_act_kernel(
    Y,
    U,
    B,
    ROUTE,
    RANK,
    SCALE,
    OUT,
    YS: tl.constexpr,
    US: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    g = tl.load(ROUTE + row)
    safe = tl.maximum(g, 0)
    r = tl.load(RANK + safe)
    scale = tl.load(SCALE + safe)
    cg = tl.full((BN,), 0, tl.float32)
    cu = tl.full((BN,), 0, tl.float32)
    for k in tl.static_range(4):
        ug = tl.load(U + row * US + k, mask=(g >= 0) & (k < r), other=0).to(tl.float32)
        uu = tl.load(U + row * US + r + k, mask=(g >= 0) & (k < r), other=0).to(
            tl.float32
        )
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
        cg = cg + ug * bg
        cu = cu + uu * bu
    gate = tl.load(Y + row * YS + col, mask=col < N, other=0).to(tl.float32)
    up = tl.load(Y + row * YS + N + col, mask=col < N, other=0).to(tl.float32)
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
    act = (gate / (1 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    value = (act * up).to(tl.bfloat16)
    tl.store(OUT + row * N + col, value, mask=col < N)


def row_act(y, u, b, routes, ranks, scales, *, tile=256, out=None):
    if out is None:
        out = torch.empty((y.shape[0], 3584), dtype=y.dtype, device=y.device)
    row_act_kernel[(y.shape[0], triton.cdiv(3584, tile))](
        y,
        u,
        b,
        routes,
        ranks,
        scales,
        out,
        y.stride(0),
        u.shape[1],
        3584,
        tile,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
