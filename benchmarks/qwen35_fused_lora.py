# SPDX-License-Identifier: Apache-2.0
"""Lazy loaders for the MI250 Qwen3.5 tiny-LoRA research kernels.

These benchmark functions are inference-only, trusted-input operators, not an
installed serving backend. Compile before graph capture. The caller owns BF16
resident factors, request routes, retained ranks and adapter scaling, all on
one gfx90a GPU. All tensors except base projections must be contiguous; base
projections may have padded rows but need unit column stride. A rows are packed
by *effective* rank inside capacity-strided rows, matching RetainedLayout.

Live routes index allocated slots, negative routes are inactive, and ranks are
zero or at most min(capacity, 8). Metadata uses int32 routes/ranks and float32
scales. Buffers must not alias factors, metadata or inputs. No operands require
gradients. These values/ownership conditions are caller duties; the native
entrypoints are not general public validation boundaries.

Narrow: A [G,R,K], B [G,1024,R], K=2048/3584, tile128/256/512/1024.
Wide: A [G,S*R,1024], B [G,N,R], gate/up S=2,N=7168 or QKV S=3,N=5120.
GDN: Q/K/V/Z widths2048 each; b/a widths16 each; input width1024.
Shared-A: Q/K/V factors must be exactly equal, checked by the caller at every
slot load; Z remains independent. This reduces arithmetic but often not latency.

The loaders also work for CPU precompilation in an explicitly configured ROCm
container (for example PYTORCH_ROCM_ARCH=gfx90a). Importing this module performs
no Torch import, device initialization or compilation. These files are research
sources alongside the benchmark, not wheel-installed LoRAForge operators.
"""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@lru_cache(maxsize=2)
def load_variant(variant="narrow"):
    """Build/load an explicit HIP variant; caller supplies the target build env."""
    if variant not in ("narrow", "wide"):
        raise ValueError("variant must be narrow or wide")
    import torch
    from torch.utils.cpp_extension import load

    if not torch.version.hip:
        raise RuntimeError("these benchmark kernels require ROCm and 64-lane gfx90a")
    source = Path(__file__).with_name(
        "qwen35_tiny_fused.cu" if variant == "narrow" else "qwen35_wide_fused.cu"
    )
    digest = sha256(
        source.read_bytes() + repr((torch.__version__, torch.version.hip)).encode()
    ).hexdigest()[:16]
    return load(
        name=f"lf_qwen35_{variant}_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def narrow_update(x, a, b, base, routes, ranks, scales, *, tile=256):
    """Accumulate one tiny A+B update into an existing narrow base projection."""
    load_variant("narrow").fused_ab(x, a, b, base, routes, ranks, scales, tile)
    return base


def wide_update(x, a, b, base, routes, ranks, scales, *, slices):
    """Accumulate gate/up or QKV updates; only tile-aligned slice boundaries."""
    if slices == 2:
        split0, split1, width = 3584, 0, 7168
    elif slices == 3:
        split0, split1, width = 4096, 4608, 5120
    else:
        raise ValueError("this wrapper covers gate/up or QKV only")
    if base.shape[1] != width:
        raise ValueError("unexpected projection width")
    load_variant("wide").wide_ab(
        x, a, b, base, routes, ranks, scales, slices, split0, split1
    )
    return base


def gdn_update(q, ba, x, aq, ab, bq, bb, routes, rq, rb, scales):
    """Produce corrected pre-convolution QKV/Z/b/a, including inactive rows."""
    import torch

    t = x.shape[0]
    out = torch.empty((t, 6144), device=x.device, dtype=x.dtype)
    z = torch.empty((t, 16, 128), device=x.device, dtype=x.dtype)
    b = torch.empty((t, 16), device=x.device, dtype=x.dtype)
    a = torch.empty_like(b)
    load_variant("wide").gdn_ab_unpack(
        x, aq, ab, bq, bb, q, ba, out, z, b, a, routes, rq, rb, scales
    )
    return out, z, b, a
