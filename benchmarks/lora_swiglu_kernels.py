# SPDX-License-Identifier: Apache-2.0
"""New first-order SwiGLU training kernels; lazy-loaded by the CPU-safe probe."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(G, U, H, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < SIZE, 0).to(tl.float32)
    u = tl.load(U + i, i < SIZE, 0).to(tl.float32)
    act = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(H + i, act * u, i < SIZE)


@triton.jit
def _backward(
    DH, G, U, DG, DU, H, SIZE: tl.constexpr, WRITE_H: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < SIZE, 0).to(tl.float32)
    u = tl.load(U + i, i < SIZE, 0).to(tl.float32)
    dy = tl.load(DH + i, i < SIZE, 0).to(tl.float32)
    sig = tl.sigmoid(g)
    act = (g * sig).to(tl.bfloat16).to(tl.float32)
    da = (dy * u).to(tl.bfloat16).to(tl.float32)
    dg = da * sig * (1.0 + g * (1.0 - sig))
    tl.store(DG + i, dg, i < SIZE)
    tl.store(DU + i, dy * act, i < SIZE)
    if WRITE_H:
        tl.store(H + i, act * u, i < SIZE)


def forward(g, u):
    g, u = g.contiguous(), u.contiguous()
    h = torch.empty_like(g)
    _forward[(triton.cdiv(g.numel(), 1024),)](
        g, u, h, g.numel(), 1024, enable_fp_fusion=False
    )
    return h


def backward(dh, g, u, *, rematerialize=False):
    dh, g, u = dh.contiguous(), g.contiguous(), u.contiguous()
    dg, du = torch.empty_like(g), torch.empty_like(u)
    h = torch.empty_like(g) if rematerialize else dg
    _backward[(triton.cdiv(g.numel(), 1024),)](
        dh, g, u, dg, du, h, g.numel(), rematerialize, 1024, enable_fp_fusion=False
    )
    return dg, du, h if rematerialize else None


@triton.jit
def _gate_up(
    GU0,
    UG,
    UU,
    BG,
    BU,
    G,
    U,
    H,
    WIDTH: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    mask = i < SIZE
    bg, bu = tl.full((BLOCK,), 0, tl.float32), tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(RANK):
        ug = tl.load(UG + row * RANK + k, mask, 0).to(tl.float32)
        uu = tl.load(UU + row * RANK + k, mask, 0).to(tl.float32)
        bg += ug * tl.load(BG + col * RANK + k, mask, 0).to(tl.float32)
        bu += uu * tl.load(BU + col * RANK + k, mask, 0).to(tl.float32)
    # Preserve the native BF16 GEMM -> scale -> residual -> SiLU boundaries.
    bg = (bg.to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    bu = (bu.to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    g = (
        (tl.load(GU0 + row * 2 * WIDTH + col, mask, 0).to(tl.float32) + bg)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    u = (
        (tl.load(GU0 + row * 2 * WIDTH + WIDTH + col, mask, 0).to(tl.float32) + bu)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    act = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(G + i, g, mask)
    tl.store(U + i, u, mask)
    tl.store(H + i, act * u, mask)


@triton.jit
def _output(
    Y0,
    UD,
    BD,
    Y,
    WIDTH: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    mask = i < SIZE
    v = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(RANK):
        a = tl.load(UD + row * RANK + k, mask, 0).to(tl.float32)
        b = tl.load(BD + col * RANK + k, mask, 0).to(tl.float32)
        v += a * b
    v = (v.to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    y = tl.load(Y0 + i, mask, 0).to(tl.float32) + v
    tl.store(Y + i, y, mask)


@triton.jit
def _mlp_backward(
    DH0,
    DUD,
    AD,
    G,
    U,
    DGU,
    DSGU,
    H,
    WIDTH: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    SCALE: tl.constexpr,
    COMPACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    mask = i < SIZE
    v = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(RANK):
        a = tl.load(DUD + row * RANK + k, mask, 0).to(tl.float32)
        b = tl.load(AD + k * WIDTH + col, mask, 0).to(tl.float32)
        v += a * b
    dy = (
        (tl.load(DH0 + i, mask, 0).to(tl.float32) + v.to(tl.bfloat16).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    g = tl.load(G + i, mask, 0).to(tl.float32)
    u = tl.load(U + i, mask, 0).to(tl.float32)
    sig = tl.sigmoid(g)
    act = (g * sig).to(tl.bfloat16).to(tl.float32)
    da = (dy * u).to(tl.bfloat16).to(tl.float32)
    dg = (da * sig * (1.0 + g * (1.0 - sig))).to(tl.bfloat16)
    du = (dy * act).to(tl.bfloat16)
    off = row * 2 * WIDTH + col
    tl.store(DGU + off, dg, mask)
    tl.store(DGU + off + WIDTH, du, mask)
    if not COMPACT:
        tl.store(DSGU + off, dg.to(tl.float32) * SCALE, mask)
        tl.store(DSGU + off + WIDTH, du.to(tl.float32) * SCALE, mask)
    tl.store(H + i, act * u, mask)


@triton.jit
def _input_gradient(
    DX0,
    DUG,
    DUU,
    AG,
    AU,
    DX,
    WIDTH: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    mask = i < SIZE
    vg, vu = tl.full((BLOCK,), 0, tl.float32), tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(RANK):
        vg += tl.load(DUG + row * RANK + k, mask, 0).to(tl.float32) * tl.load(
            AG + k * WIDTH + col, mask, 0
        ).to(tl.float32)
        vu += tl.load(DUU + row * RANK + k, mask, 0).to(tl.float32) * tl.load(
            AU + k * WIDTH + col, mask, 0
        ).to(tl.float32)
    v = (
        (vg.to(tl.bfloat16).to(tl.float32) + vu.to(tl.bfloat16).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    tl.store(DX + i, tl.load(DX0 + i, mask, 0).to(tl.float32) + v, mask)


def gate_up(gu0, ug, uu, bg, bu, scale):
    rows, rank = ug.shape
    width = gu0.shape[-1] // 2
    g = torch.empty((rows, width), device=gu0.device, dtype=gu0.dtype)
    u, h = torch.empty_like(g), torch.empty_like(g)
    _gate_up[(triton.cdiv(g.numel(), 256),)](
        gu0,
        ug,
        uu,
        bg,
        bu,
        g,
        u,
        h,
        width,
        g.numel(),
        rank,
        scale,
        256,
        enable_fp_fusion=False,
    )
    return g, u, h


def output(y0, ud, bd, scale):
    y = torch.empty_like(y0)
    _output[(triton.cdiv(y.numel(), 256),)](
        y0,
        ud,
        bd,
        y,
        y.shape[-1],
        y.numel(),
        ud.shape[-1],
        scale,
        256,
        enable_fp_fusion=False,
    )
    return y


def mlp_backward(dh0, dud, ad, g, u, scale, *, compact=False):
    rows, width = g.shape
    dgu = torch.empty((rows, 2 * width), device=g.device, dtype=g.dtype)
    dsgu, h = None if compact else torch.empty_like(dgu), torch.empty_like(g)
    _mlp_backward[(triton.cdiv(g.numel(), 256),)](
        dh0,
        dud,
        ad,
        g,
        u,
        dgu,
        dgu if compact else dsgu,
        h,
        width,
        g.numel(),
        dud.shape[-1],
        scale,
        compact,
        256,
        enable_fp_fusion=False,
    )
    return dgu, dsgu, h


def input_gradient(dx0, dug, duu, ag, au):
    dx = torch.empty_like(dx0)
    _input_gradient[(triton.cdiv(dx.numel(), 256),)](
        dx0,
        dug,
        duu,
        ag,
        au,
        dx,
        dx.shape[-1],
        dx.numel(),
        dug.shape[-1],
        256,
        enable_fp_fusion=False,
    )
    return dx


@triton.jit
def _gate_up_separate(
    G0,
    U0,
    UG,
    UU,
    BG,
    BU,
    G,
    U,
    H,
    WIDTH: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // WIDTH, i % WIDTH
    mask = i < SIZE
    bg, bu = tl.full((BLOCK,), 0, tl.float32), tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(RANK):
        bg += tl.load(UG + row * RANK + k, mask, 0).to(tl.float32) * tl.load(
            BG + col * RANK + k, mask, 0
        ).to(tl.float32)
        bu += tl.load(UU + row * RANK + k, mask, 0).to(tl.float32) * tl.load(
            BU + col * RANK + k, mask, 0
        ).to(tl.float32)
    bg = (bg.to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    bu = (bu.to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    g = (tl.load(G0 + i, mask, 0).to(tl.float32) + bg).to(tl.bfloat16).to(tl.float32)
    u = (tl.load(U0 + i, mask, 0).to(tl.float32) + bu).to(tl.bfloat16).to(tl.float32)
    act = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(G + i, g, mask)
    tl.store(U + i, u, mask)
    tl.store(H + i, act * u, mask)


def gate_up_separate(g0, u0, ug, uu, bg, bu, scale):
    shape = g0.shape
    g0, u0, ug, uu = [v.contiguous() for v in (g0, u0, ug, uu)]
    g, u, h = torch.empty_like(g0), torch.empty_like(u0), torch.empty_like(g0)
    _gate_up_separate[(triton.cdiv(g0.numel(), 256),)](
        g0,
        u0,
        ug,
        uu,
        bg,
        bu,
        g,
        u,
        h,
        shape[-1],
        g0.numel(),
        ug.shape[-1],
        scale,
        256,
        enable_fp_fusion=False,
    )
    return g, u, h
