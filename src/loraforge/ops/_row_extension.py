# SPDX-License-Identifier: Apache-2.0
"""Lazy compilation of the retained-rank row kernels, separate from FlashTSQR."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load

    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm row operators before graph capture")
    source = Path(__file__).with_name("csrc") / "lora_rows.cu"
    digest = sha256(
        source.read_bytes()
        + repr((torch.__version__, torch.version.cuda, torch.version.hip)).encode()
    ).hexdigest()[:16]
    return load(
        name=f"loraforge_rows_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
