# SPDX-License-Identifier: Apache-2.0
# Derived from SGLang 0.5.22 via lishenghui/probe, commit 5fd6ae060d106066b306959f9a34b726a281ef4c.
# Probe modification: Shenghui Li, skip column blocks beyond actual adapter rank.
# See docs/provenance.md for upstream attribution and extraction scope.
"""Pinned SGLang segmented shrink with probe's actual-rank early exit."""

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.lora.dense.lora_tuning_config import get_lora_shrink_config


@triton.jit(do_not_specialize=["num_segs"])
def _rank_aware_shrink_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths,ranks and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # Meta parameters
    N: tl.constexpr,  # num_slices * r
    K: tl.constexpr,  # input_dim
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA shrink operations.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * num_slices]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA A weights for all available adapters,
            with shape `(num_lora, N, K)` where N = num_slices * r.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K

    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    pid_n = tl.program_id(0)

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    if seg_start == seg_end:
        return

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    # Adjust N dim according to the specific LoRA adapter
    cur_n = tl.minimum(N, rank * NUM_SLICES)
    if (
        pid_n * BLOCK_N >= cur_n
    ):  # PROBE: column block beyond this adapter's actual rank -> no work (outputs there are masked anyway)
        return

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end, other=0
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def shrink(x, weights, info, slices):
    from sglang.srt.environ import envs

    # Preserve the caller's explicit native split-K configuration.
    if envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.get():
        from sglang.kernels.ops.lora.dense.chunked_sgmv_shrink import (
            chunked_sgmv_lora_shrink_forward,
        )

        return chunked_sgmv_lora_shrink_forward(x, weights, info, slices)
    config = get_lora_shrink_config(
        K=weights.shape[2],
        R=weights.shape[1] // slices,
        num_slices=slices,
        chunk_size=info.max_len,
    )
    segments = (
        info.weight_indices.shape[0] if info.use_cuda_graph else info.num_segments
    )
    output = torch.empty((x.shape[0], weights.shape[1]), device=x.device, dtype=x.dtype)
    if not x.shape[0] or not segments:
        return output
    options = {k: config[k] for k in ("num_warps", "num_stages") if k in config}
    _rank_aware_shrink_kernel[
        (triton.cdiv(weights.shape[1], config["BLOCK_N"]), segments)
    ](
        x,
        weights,
        output,
        info.seg_indptr,
        info.weight_indices,
        info.lora_ranks,
        info.permutation,
        segments,
        N=weights.shape[1],
        K=weights.shape[2],
        NUM_SLICES=slices,
        BLOCK_M=info.max_len,
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        **options,
    )
    return output
