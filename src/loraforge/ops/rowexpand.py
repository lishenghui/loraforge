# SPDX-License-Identifier: Apache-2.0
"""Probe Fleet direct shrink and staged RowExpand, lazily compiled.

These low-level inference operations consume trusted device routing metadata.
The owner must validate route/segment values on CPU before uploading them:
segments contain at most 16 rows; live routes and permutation rows are in range;
rank is zero or <=64, bucket is the next power of two, and compact regions fit.
Metadata remains device data during graph replay. Compile before capture.
"""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@lru_cache(maxsize=1)
def rowexpand_extension():
    import torch
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "rowexpand.cu"
    digest = sha256(
        source.read_bytes() + repr((torch.__version__, torch.version.hip)).encode()
    ).hexdigest()[:16]
    return load(
        name=f"loraforge_rowexpand_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def direct_shrink(x, weights, output, routes, ranks, slices, max_rank):
    """Write live retained A rows; inactive output columns remain unspecified."""
    if any(t.requires_grad for t in (x, weights, output)):
        raise ValueError("RowExpand serving operators do not support autograd")
    rowexpand_extension().shrink(x, weights, output, routes, ranks, slices, max_rank)
    return output


def row_expand(
    z,
    store,
    offsets,
    buckets,
    output,
    segments,
    slots,
    permutation,
    ranks,
    scales,
    slice_offsets,
    segment_count,
    capacity,
    *,
    blocks=880,
    threads=256,
):
    """Accumulate BF16-rounded scaled delta into output on its current stream.

    Z uses effective-rank packed slices inside a capacity-strided row. B uses
    fixed compact regions owned by RetainedLayout. Output may have padded rows.
    Routing, metadata and operands must live on the same device. No host rank
    reads, synchronization or allocations occur after the extension is loaded.
    """
    if any(t.requires_grad for t in (z, store, output, scales)):
        raise ValueError("RowExpand serving operators do not support autograd")
    rowexpand_extension().expand(
        z,
        store,
        offsets,
        buckets,
        output,
        segments,
        slots,
        permutation,
        ranks,
        scales,
        slice_offsets,
        segment_count,
        output.shape[1],
        capacity,
        blocks,
        threads,
    )
    return output
