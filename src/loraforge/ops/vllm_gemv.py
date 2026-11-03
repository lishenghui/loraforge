# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Adapted from probe 5fd6ae060d106066b306959f9a34b726a281ef4c.
# See docs/provenance.md.
"""Lazy GEMV for padded vLLM slot layouts, independent of vLLM imports.

Routing values must be -1 or valid slot indices. Callers own device metadata
validation, input ordering, and fixed-address lifetimes during graph replay.
"""


def _check(x, weights, routes):
    import torch

    if x.ndim != 2 or x.stride(-1) != 1 or x.device.type != "cuda":
        raise ValueError("expected GPU x[tokens, width] with contiguous columns")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("expected floating-point inputs")
    if (
        routes.device != x.device
        or routes.dtype != torch.int32
        or routes.shape != (x.shape[0],)
        or not routes.is_contiguous()
    ):
        raise ValueError("expected same-device contiguous int32 routes[tokens]")
    if not weights:
        raise ValueError("at least one weight slice is required")
    slots = weights[0].shape[0]
    for weight in weights:
        if (
            weight.ndim not in (3, 4)
            or (weight.ndim == 4 and weight.shape[1] != 1)
            or min(weight.shape) < 1
            or weight.shape[0] != slots
            or weight.stride(-1) != 1
            or weight.device != x.device
            or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        ):
            raise ValueError("invalid same-device padded weight slice")
    if any(t.requires_grad for t in (x, *weights)):
        raise ValueError("GEMV is inference-only")


def shrink(x, weights, out, routes, *, scale=1.0):
    """Write FP32 out[slices, tokens, rank] = scale * A[route] @ x.

    A is [slots, rank, width] or [slots, 1, rank, width]. Base routes (-1)
    produce zero. Split-K uses FP32 atomic accumulation, as in the source.
    """
    import torch
    import triton

    from ._vllm_gemv_kernels import _shrink_kernel

    _check(x, weights, routes)
    m, k = x.shape
    rank = weights[0].shape[-2]
    if k < 1 or not 1 <= rank <= 256:
        raise ValueError("width must be positive and rank must be in 1..256")
    if any(a.shape[-2:] != (rank, k) for a in weights):
        raise ValueError("A slice shape mismatch")
    if (
        out.device != x.device
        or out.dtype != torch.float32
        or out.shape != (len(weights), m, rank)
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError("expected contiguous same-device FP32 output buffer")
    with torch.cuda.device(x.device), torch.no_grad():
        out.zero_()
        if m == 0:
            return
        block_k = 256
        splits = max(1, min(triton.cdiv(k, block_k), triton.cdiv(440, m)))
        per_split = triton.cdiv(triton.cdiv(k, splits), block_k) * block_k
        splits = triton.cdiv(k, per_split)
        for s, a in enumerate(weights):
            _shrink_kernel[(m, splits)](
                x,
                a,
                out[s],
                routes,
                k,
                rank,
                per_split,
                x.stride(0),
                a.stride(0),
                a.stride(-2),
                out.stride(1),
                scale,
                BLOCK_R=triton.next_power_of_2(rank),
                BLOCK_K=block_k,
                SPLIT_K=splits,
                num_warps=4,
            )


def expand(v, weights, y, routes, *, offset=0, add=True):
    """Apply B[route] @ v to output slices in place, with FP32 accumulation.

    Unlike native vLLM matrix-dot, the source GEMV retains FP32 intermediates
    before multiplying by B. Numerical parity is tolerance-based, not bitwise.
    Base-route rows and columns outside the slices remain unchanged, even
    for add=False, matching native vLLM's skipped-token behavior.
    """
    import torch
    import triton

    from ._vllm_gemv_kernels import _expand_kernel

    _check(y, weights, routes)
    m = y.shape[0]
    rank = weights[0].shape[-1]
    if not 1 <= rank <= 256 or any(b.shape[-1] != rank for b in weights):
        raise ValueError("B slices must share rank in 1..256")
    if (
        v.device != y.device
        or v.dtype != torch.float32
        or v.shape != (len(weights), m, rank)
        or not v.is_contiguous()
        or v.requires_grad
    ):
        raise ValueError("expected same-device contiguous FP32 intermediate")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if offset + sum(b.shape[-2] for b in weights) > y.shape[1]:
        raise ValueError("output slices exceed output width")
    with torch.cuda.device(y.device), torch.no_grad():
        if m == 0:
            return
        for s, b in enumerate(weights):
            n = b.shape[-2]
            _expand_kernel[(m, triton.cdiv(n, 128))](
                v[s],
                b,
                y,
                routes,
                n,
                rank,
                offset,
                v.stride(1),
                b.stride(0),
                b.stride(-2),
                y.stride(0),
                BLOCK_R=triton.next_power_of_2(rank),
                BLOCK_N=128,
                ADD=add,
                num_warps=4,
            )
            offset += n
