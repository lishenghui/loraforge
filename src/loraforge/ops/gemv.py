# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Adapted from probe decode GEMV; see docs/provenance.md.
"""Lazy graph-compatible decode operators; metadata stays on device."""


def _inputs(x, weights, ranks, token_map):
    import torch

    if not all(isinstance(t, torch.Tensor) for t in (x, weights, ranks, token_map)):
        raise TypeError("inputs must be tensors")
    if x.ndim != 2 or weights.ndim != 3 or min(weights.shape) <= 0:
        raise ValueError(
            "expected x[tokens, width] and nonempty weights[slots, rows, cols]"
        )
    if x.device.type != "cuda" or any(
        t.device != x.device for t in (weights, ranks, token_map)
    ):
        raise ValueError("all inputs must share a CUDA/ROCm device")
    if (
        x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or weights.dtype != x.dtype
    ):
        raise ValueError(
            "matching float32/float16/bfloat16 factors and activations required"
        )
    if x.requires_grad or weights.requires_grad:
        raise ValueError("decode operators are inference-only")
    if not weights.is_contiguous() or x.stride(1) != 1:
        raise ValueError(
            "weights must be contiguous and x must have unit column stride"
        )
    if (
        ranks.ndim != 1
        or ranks.numel() != weights.shape[0]
        or token_map.ndim != 1
        or token_map.numel() < x.shape[0]
        or not ranks.is_contiguous()
        or not token_map.is_contiguous()
        or ranks.dtype not in (torch.int32, torch.int64)
        or token_map.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("expected contiguous integer slot ranks and token mapping")


def lora_shrink(x, weights, ranks, token_map, num_slices: int = 1):
    """Apply packed LoRA A to decode tokens, preserving the activation dtype.

    A is [slots, slices*max_rank, input]; actual slices are packed with stride
    rank[slot]. Caller guarantees device metadata values are valid: mappings in
    [0, slots), ranks in [0, max_rank]. Rank zero represents base-model tokens.
    BF16 uses sequential matrix-dot blocks to avoid split-K rounding drift.
    No device values are read on the host, so warmed calls support graph capture.
    """
    import torch

    _inputs(x, weights, ranks, token_map)
    if type(num_slices) is not int or num_slices <= 0 or weights.shape[1] % num_slices:
        raise ValueError("num_slices must divide the packed A row capacity")
    rank = weights.shape[1] // num_slices
    if not 1 <= rank <= 256 or x.shape[1] != weights.shape[2]:
        raise ValueError("rank capacity must be 1..256 and input widths must match")
    import triton

    from ._gemv_kernels import reduce_kernel, shrink_kernel

    tokens, width = x.shape
    out = torch.zeros((tokens, num_slices * rank), device=x.device, dtype=x.dtype)
    if not tokens:
        return out
    if x.dtype == torch.bfloat16:
        from ._gemv_kernels import shrink_dot_kernel

        with torch.cuda.device(x.device):
            shrink_dot_kernel[(tokens, triton.cdiv(num_slices * rank, 16))](
                x,
                weights,
                out,
                token_map,
                ranks,
                width,
                rank,
                num_slices,
                x.stride(0),
                tokens,
                num_warps=4,
            )
        return out
    block = 256
    splits = max(1, min(triton.cdiv(width, block), triton.cdiv(440, tokens)))
    per_split = triton.cdiv(triton.cdiv(width, splits), block) * block
    splits = triton.cdiv(width, per_split)
    partial = torch.empty(
        (num_slices, tokens, splits, rank), device=x.device, dtype=torch.float32
    )
    with torch.cuda.device(x.device):
        shrink_kernel[(tokens, splits, num_slices)](
            x,
            weights,
            partial,
            token_map,
            ranks,
            tokens,
            width,
            rank,
            num_slices,
            splits,
            per_split,
            x.stride(0),
            BR=triton.next_power_of_2(rank),
            BK=block,
            num_warps=4,
        )
        reduce_kernel[(tokens, num_slices)](
            partial,
            out,
            token_map,
            ranks,
            tokens,
            rank,
            num_slices,
            splits,
            BSK=triton.next_power_of_2(splits),
            BR=triton.next_power_of_2(rank),
            num_warps=4,
        )
    return out


def lora_expand(
    x, weights, ranks, scales, token_map, offsets, max_slice_size: int, base=None
):
    """Apply padded LoRA B and add a scaled delta to optional base in place.

    B is [slots, total_output, max_rank]. x has the packed shrink layout. Slice
    offsets are increasing, start at zero and end at total_output; each slice
    length is <= max_slice_size. Caller guarantees these device values, metadata
    ranges and non-overlapping output storage. The scaled delta is rounded to
    the activation dtype before base addition. BF16 uses sequential matrix-dot
    blocks matching the pinned SGLang default reduction geometry.
    """
    import torch

    _inputs(x, weights, ranks, token_map)
    slots, width, rank = weights.shape
    if not x.is_contiguous() or not 1 <= rank <= 256:
        raise ValueError(
            "intermediate must be contiguous; rank capacity must be 1..256"
        )
    if (
        not isinstance(offsets, torch.Tensor)
        or offsets.device != x.device
        or offsets.ndim != 1
        or offsets.numel() < 2
        or not offsets.is_contiguous()
        or offsets.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("expected contiguous integer device slice offsets")
    slices = offsets.numel() - 1
    if x.shape[1] != slices * rank:
        raise ValueError("intermediate width does not match slices and rank capacity")
    if (
        not isinstance(scales, torch.Tensor)
        or scales.device != x.device
        or scales.ndim != 1
        or scales.numel() != slots
        or not scales.is_contiguous()
        or scales.requires_grad
        or scales.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        raise ValueError("expected contiguous floating device scales per slot")
    if type(max_slice_size) is not int or not 1 <= max_slice_size <= width:
        raise ValueError("max_slice_size must be within the output width")
    if base is not None and (
        not isinstance(base, torch.Tensor)
        or base.device != x.device
        or base.dtype != x.dtype
        or base.ndim != 2
        or base.shape[0] != x.shape[0]
        or base.shape[1] < width
        or base.requires_grad
        or base.stride(0) <= 0
        or base.stride(1) <= 0
    ):
        raise ValueError(
            "base must match tokens, device and dtype with sufficient output width"
        )
    import triton

    from ._gemv_kernels import expand_kernel

    output = (
        torch.zeros((x.shape[0], width), device=x.device, dtype=x.dtype)
        if base is None
        else base
    )
    if x.shape[0] and x.dtype == torch.bfloat16:
        from ._gemv_kernels import expand_dot_kernel

        with torch.cuda.device(x.device):
            expand_dot_kernel[(x.shape[0], triton.cdiv(max_slice_size, 64), slices)](
                x,
                weights,
                output,
                token_map,
                ranks,
                scales,
                offsets,
                width,
                rank,
                slices,
                output.stride(0),
                output.stride(1),
                x.shape[0],
                num_warps=4,
            )
        return output
    if x.shape[0]:
        block = 128 if rank <= 32 else 64 if rank <= 64 else 32
        with torch.cuda.device(x.device):
            expand_kernel[(x.shape[0], triton.cdiv(max_slice_size, block), slices)](
                x,
                weights,
                output,
                token_map,
                ranks,
                scales,
                offsets,
                width,
                rank,
                slices,
                output.stride(0),
                output.stride(1),
                BR=triton.next_power_of_2(rank),
                BN=block,
                num_warps=4,
            )
    return output
