# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only rank-capacity-four narrow specialization; same trusted contract
as qwen35_fused_lora. K=2048/3584, N=1024, BF16, unit column strides, live
int32 routes/ranks (0..4), float32 scaling, and inference-only execution.
Prebuild on CPU for gfx90a before graph capture. No import-time device work.
"""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load

    if not torch.version.hip:
        raise RuntimeError("this benchmark requires ROCm gfx90a")
    source = Path(__file__).with_suffix(".cu")
    digest = sha256(
        source.read_bytes() + repr((torch.__version__, torch.version.hip)).encode()
    ).hexdigest()[:16]
    return load(
        name=f"lf_qwen35_rank4_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
