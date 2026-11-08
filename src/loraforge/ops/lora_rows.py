# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Kernels extracted from probe at 5fd6ae060d106066b306959f9a34b726a281ef4c.
"""BF16 retained-rank shrink and compact-B RowExpand without framework imports.

Device metadata values are trusted inputs: see docs/row-operators.md. Structural
validation does not copy metadata to the CPU or synchronize the current stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loraforge.annotations import ExperimentalAPI

if TYPE_CHECKING:
    from torch import Tensor


def _tensor(tensor, name, ndim, dtype, device=None):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim or tensor.dtype != dtype:
        raise ValueError(f"{name} must be {ndim}D with dtype {dtype}")
    if any(size >= 2**31 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must fit signed int32 indexing")
    if tensor.device.type != "cuda" or (device is not None and tensor.device != device):
        raise ValueError(f"{name} must be on the same CUDA/ROCm GPU")
    if tensor.requires_grad:
        raise ValueError("row operators are inference-only")


def _vector(tensor, name, dtype, device, size=None):
    _tensor(tensor, name, 1, dtype, device)
    if not tensor.is_contiguous() or (size is not None and tensor.numel() != size):
        raise ValueError(f"{name} must be contiguous with the required length")


def _distinct_output(output, inputs):
    # Conservatively reject shared storage, including disjoint views. This does
    # not read device data and catches output aliasing during graph capture too.
    ptr = output.untyped_storage().data_ptr()
    if output.numel() and any(
        t.numel() and t.untyped_storage().data_ptr() == ptr for t in inputs
    ):
        raise ValueError("output must not share storage with an input")


@ExperimentalAPI(
    reason="BF16 retained-rank inference operator; L40S CUDA validated, caller-owned device metadata."
)
def row_shrink(
    x: Tensor,
    a: Tensor,
    output: Tensor,
    slots: Tensor,
    ranks: Tensor,
    *,
    num_slices: int = 1,
    max_rank: int = 64,
) -> None:
    """Write live packed A @ x prefixes in place, rounding FP32 sums to BF16.

    x is [tokens, width], A [slots, slices * capacity, width], output
    [tokens, slices * capacity]. For slot rank r, slice s starts at s*r
    in both A and output. Slots -1 and rank-zero rows are left unchanged;
    unused output columns are also unchanged. All live ranks must be <=max_rank.
    Input width and row stride must be multiples of eight BF16 elements.
    """
    import torch

    _tensor(x, "x", 2, torch.bfloat16)
    _tensor(a, "a", 3, torch.bfloat16, x.device)
    _tensor(output, "output", 2, torch.bfloat16, x.device)
    if type(num_slices) is not int or not 1 <= num_slices <= 3:
        raise ValueError("num_slices must be in 1..3")
    if type(max_rank) is not int or not 1 <= max_rank <= 64:
        raise ValueError("max_rank must be in 1..64")
    if (
        a.shape[0] < 1
        or a.shape[1] % num_slices
        or a.shape[1] // num_slices < max_rank
        or a.shape[2] != x.shape[1]
        or output.shape != (x.shape[0], a.shape[1])
    ):
        raise ValueError("A/output shapes must agree with slices and rank capacity")
    if (
        x.shape[1] < 1
        or x.shape[1] % 8
        or x.stride(1) != 1
        or x.stride(0) < x.shape[1]
        or x.stride(0) % 8
        or x.data_ptr() % 16
        or a.data_ptr() % 16
        or not a.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("expected aligned A/x and contiguous A/output")
    _vector(slots, "slots", torch.int32, x.device, x.shape[0])
    _vector(ranks, "ranks", torch.int32, x.device, a.shape[0])
    _distinct_output(output, (x, a, slots, ranks))
    if x.shape[0] == 0:
        return
    from ._row_extension import extension

    with torch.cuda.device(x.device), torch.no_grad():
        extension().shrink(x, a, output, slots, ranks, num_slices, max_rank)


@ExperimentalAPI(
    reason="Compact-B RowExpand; L40S CUDA validated, fixed-address metadata/lifetime contract."
)
def row_expand(
    z: Tensor,
    b: Tensor,
    output: Tensor,
    *,
    b_offsets: Tensor,
    b_strides: Tensor,
    ranks: Tensor,
    scales: Tensor,
    segment_indptr: Tensor,
    segment_slots: Tensor,
    token_indices: Tensor,
    slice_offsets: Tensor,
    segment_count: Tensor,
    blocks: int | None = None,
    threads: int = 256,
) -> None:
    """Add compact B @ z to BF16 output, reusing B across <=16 token segments.

    z is [tokens, slices * capacity], with each live slice packed at s*r.
    B is flat; b_offsets are element offsets and b_strides are power-of-two
    row widths in 1..64, at least the corresponding rank. B padding is zero.
    Each segment has one slot and at most 16 distinct token indices. Tokens
    occur at most once across active segments. slice_offsets partition output
    columns; segment_count is a one-element int32 device vector.

    FP32 dot products are scaled, rounded to BF16, added to the BF16 base,
    and rounded again. Rank-zero and slot -1 segments leave output unchanged.
    Warm the operator before capturing a graph. Callers own all buffer lifetimes
    and stream ordering, including writes to metadata before replay.
    """
    import torch

    _tensor(z, "z", 2, torch.bfloat16)
    _tensor(b, "b", 1, torch.bfloat16, z.device)
    _tensor(output, "output", 2, torch.bfloat16, z.device)
    _vector(ranks, "ranks", torch.int32, z.device)
    nslots = ranks.numel()
    for tensor, name, dtype in (
        (b_offsets, "b_offsets", torch.int64),
        (b_strides, "b_strides", torch.int32),
        (scales, "scales", torch.float32),
    ):
        _vector(tensor, name, dtype, z.device, nslots)
    _vector(segment_slots, "segment_slots", torch.int32, z.device)
    _vector(
        segment_indptr,
        "segment_indptr",
        torch.int32,
        z.device,
        segment_slots.numel() + 1,
    )
    _vector(token_indices, "token_indices", torch.int32, z.device)
    _vector(slice_offsets, "slice_offsets", torch.int32, z.device)
    _vector(segment_count, "segment_count", torch.int32, z.device, 1)
    slices = slice_offsets.numel() - 1
    if (
        nslots < 1
        or not 1 <= slices <= 3
        or z.shape[1] < slices
        or z.shape[1] % slices
        or output.shape[0] != z.shape[0]
        or output.shape[1] < 1
    ):
        raise ValueError("invalid slots, slice count or activation/output shape")
    if (
        not z.is_contiguous()
        or not b.is_contiguous()
        or b.data_ptr() % 16
        or output.stride(1) != 1
        or output.stride(0) < output.shape[1]
    ):
        raise ValueError(
            "expected contiguous z/aligned B and nonoverlapping output rows"
        )
    if blocks is not None and (type(blocks) is not int or not 1 <= blocks <= 65535):
        raise ValueError("blocks must be in 1..65535")
    if type(threads) is not int or threads not in (128, 256):
        raise ValueError("threads must be 128 or 256")
    column_blocks = (output.shape[1] + threads - 1) // threads
    if segment_slots.numel() * column_blocks >= 2**31:
        raise ValueError("segment work count must fit signed int32 indexing")
    _distinct_output(
        output,
        (
            z,
            b,
            b_offsets,
            b_strides,
            ranks,
            scales,
            segment_indptr,
            segment_slots,
            token_indices,
            slice_offsets,
            segment_count,
        ),
    )
    if z.shape[0] == 0:
        return
    from ._row_extension import extension

    with torch.cuda.device(z.device), torch.no_grad():
        if blocks is None:
            blocks = (
                2 * torch.cuda.get_device_properties(z.device).multi_processor_count
            )
        extension().expand(
            z,
            b,
            b_offsets,
            b_strides,
            output,
            segment_indptr,
            segment_slots,
            token_indices,
            ranks,
            scales,
            slice_offsets,
            segment_count,
            output.shape[1],
            z.shape[1] // slices,
            blocks,
            threads,
        )
