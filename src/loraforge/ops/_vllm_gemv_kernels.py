# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Adapted from probe 5fd6ae060d106066b306959f9a34b726a281ef4c.
# See docs/provenance.md.
"""GEMV kernels for vLLM padded slot layouts; imported only on use."""

import triton
import triton.language as tl


@triton.jit
def _shrink_kernel(
    x_ptr,
    a_ptr,
    out_ptr,
    map_ptr,
    K,
    R,
    K_PER_SPLIT,
    sxm,
    sag,
    sar,
    som,
    scale,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    t = tl.program_id(0)
    ks = tl.program_id(1)
    g = tl.load(map_ptr + t)
    if g < 0:
        return
    offs_r = tl.arange(0, BLOCK_R)
    offs_k = tl.arange(0, BLOCK_K)
    mr = offs_r < R
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    k0 = ks * K_PER_SPLIT
    for k in range(k0, k0 + K_PER_SPLIT, BLOCK_K):
        kk = k + offs_k
        mk = kk < K
        xv = tl.load(x_ptr + t * sxm + kk, mask=mk, other=0.0).to(tl.float32)
        a = tl.load(
            a_ptr + g * sag + offs_r[:, None] * sar + kk[None, :],
            mask=mr[:, None] & mk[None, :],
            other=0.0,
        )
        acc += tl.sum(a.to(tl.float32) * xv[None, :], axis=1)
    acc = acc * scale
    if SPLIT_K == 1:
        tl.store(out_ptr + t * som + offs_r, acc, mask=mr)
    else:
        tl.atomic_add(out_ptr + t * som + offs_r, acc, mask=mr)


@triton.jit
def _expand_kernel(
    v_ptr,
    b_ptr,
    y_ptr,
    map_ptr,
    N,
    R,
    y_off,
    svm,
    sbg,
    sbn,
    sym,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ADD: tl.constexpr,
):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    g = tl.load(map_ptr + t)
    if g < 0:
        return
    offs_r = tl.arange(0, BLOCK_R)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mr = offs_r < R
    mn = offs_n < N
    v = tl.load(v_ptr + t * svm + offs_r, mask=mr, other=0.0)
    b = tl.load(
        b_ptr + g * sbg + offs_n[:, None] * sbn + offs_r[None, :],
        mask=mn[:, None] & mr[None, :],
        other=0.0,
    )
    o = tl.sum(b.to(tl.float32) * v[None, :], axis=1)
    yp = y_ptr + t * sym + y_off + offs_n
    if ADD:
        o = o + tl.load(yp, mask=mn, other=0.0).to(tl.float32)
    tl.store(yp, o.to(y_ptr.dtype.element_ty), mask=mn)
