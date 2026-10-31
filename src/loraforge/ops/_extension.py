# SPDX-License-Identifier: Apache-2.0
"""Lazy, content-addressed compilation of packaged FlashTSQR sources."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

SOURCE_DIR = Path(__file__).with_name("csrc")
SOURCE_NAMES = ("bindings.cpp", "tsqr_full.cu", "tsqr_wide.cu", "jacobi_eigh.cu")


def source_hashes() -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, sha256((SOURCE_DIR / name).read_bytes()).hexdigest())
        for name in SOURCE_NAMES
    )


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load

    digest = sha256(
        repr((source_hashes(), torch.__version__, torch.version.hip)).encode()
    ).hexdigest()[:16]
    return load(
        name=f"loraforge_flashtsqr_{digest}",
        sources=[str(SOURCE_DIR / name) for name in SOURCE_NAMES],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
