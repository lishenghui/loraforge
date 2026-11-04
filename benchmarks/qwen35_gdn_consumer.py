# SPDX-License-Identifier: Apache-2.0
"""GPU research prototype: BF16 Qwen3.5-0.8B expand at contiguous unpack.

This benchmark module has no serving installer. The caller owns validated,
resident routing/rank/scale metadata, same-device contiguous weights and U/V,
unit column strides for base projections, and inference-only execution.
Q/K/V/Z widths are 2048 each; b/a widths are 16 each. Inactive routes have
rank zero (negative route IDs are also masked). This is not a general GDN op.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    Q,
    BA,
    U,
    V,
    BQ,
    BB,
    MAP,
    RQ,
    RB,
    SCALE,
    O,  # noqa: E741 - output pointer, kept identical to the measured kernel
    Z,
    B,
    A,
    QS: tl.constexpr,
    BS: tl.constexpr,
    R: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.program_id(1) * TILE + tl.arange(0, TILE)
    g = tl.load(MAP + row)
    safe = tl.maximum(g, 0)
    rq = tl.load(RQ + safe)
    rb = tl.load(RB + safe)
    scale = tl.load(SCALE + safe)
    isq = j < 8192
    jb = j - 8192
    qacc = tl.full((TILE,), 0, tl.float32)
    bacc = tl.full((TILE,), 0, tl.float32)
    for k in tl.static_range(R):
        u = tl.load(
            U + row * 4 * R + (j // 2048) * rq + k,
            mask=isq & (g >= 0) & (k < rq),
            other=0,
        ).to(tl.float32)
        w = tl.load(
            BQ + (safe * 8192 + j) * R + k, mask=isq & (g >= 0) & (k < rq), other=0
        ).to(tl.float32)
        qacc = qacc + u * w
        v = tl.load(
            V + row * 2 * R + (jb // 16) * rb + k,
            mask=(~isq) & (j < 8224) & (g >= 0) & (k < rb),
            other=0,
        ).to(tl.float32)
        w = tl.load(
            BB + (safe * 32 + jb) * R + k,
            mask=(~isq) & (j < 8224) & (g >= 0) & (k < rb),
            other=0,
        ).to(tl.float32)
        bacc = bacc + v * w
    qbase = tl.load(Q + row * QS + j, mask=isq, other=0).to(tl.float32)
    bbase = tl.load(BA + row * BS + jb, mask=(~isq) & (j < 8224), other=0).to(
        tl.float32
    )
    qdelta = (qacc * scale).to(tl.bfloat16).to(tl.float32)
    bdelta = (bacc * scale).to(tl.bfloat16).to(tl.float32)
    qout = (qbase + qdelta).to(tl.bfloat16)
    bout = (bbase + bdelta).to(tl.bfloat16)
    tl.store(O + row * 6144 + j, qout, mask=j < 6144)
    tl.store(Z + row * 2048 + j - 6144, qout, mask=(j >= 6144) & isq)
    tl.store(B + row * 16 + jb, bout, mask=(jb >= 0) & (jb < 16))
    tl.store(A + row * 16 + jb - 16, bout, mask=(jb >= 16) & (jb < 32))


def consume(q, ba, u, v, bq, bb, routes, rq, rb, scales):
    t = q.shape[0]
    r = bq.shape[-1]
    assert r in (1, 2, 4, 8) and q.shape[1] == 8192 and ba.shape[1] == 32
    o = torch.empty((t, 6144), device=q.device, dtype=q.dtype)
    z = torch.empty((t, 16, 128), device=q.device, dtype=q.dtype)
    b = torch.empty((t, 16), device=q.device, dtype=q.dtype)
    a = torch.empty_like(b)
    kernel[(t, triton.cdiv(8224, 256))](
        q,
        ba,
        u,
        v,
        bq,
        bb,
        routes,
        rq,
        rb,
        scales,
        o,
        z,
        b,
        a,
        q.stride(0),
        ba.stride(0),
        r,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return o, z, b, a
