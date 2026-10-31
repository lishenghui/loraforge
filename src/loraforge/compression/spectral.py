# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Adapted from probe/FlashTSQR/flashtsqr.py; see docs/provenance.md.
"""One-sided spectral LoRA refactoring through FlashTSQR and a small FP64 core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loraforge.ops import ExecutionInfo, qr_factor
from loraforge.ops._extension import extension

if TYPE_CHECKING:
    from torch import Tensor

EighBackend = Literal["auto", "torch", "jacobi", "cpu"]


@dataclass(frozen=True)
class CompressionInfo:
    qr: ExecutionInfo
    eigh_backend: str
    original_rank: int
    retained_rank: int
    padded: bool = False


@dataclass(frozen=True)
class LoRAResult:
    """FP32 factors, descending singular values, and per-call execution metadata.

    ``a`` and ``b`` preserve the input batching convention. ``singular_values``
    contains the complete original-rank spectrum, even when factors are truncated.
    This tensor API represents B @ A; external adapter scaling is unchanged and
    must still be applied by the caller. Results are detached from autograd.
    """

    a: Tensor
    b: Tensor
    singular_values: Tensor
    info: CompressionInfo


def _validate(a: Tensor, b: Tensor) -> bool:
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be torch.Tensor instances")
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("expected a[r, in], b[out, r] or matching batched 3D tensors")
    if any(size == 0 for size in (*a.shape, *b.shape)):
        raise ValueError("input factors must be nonempty")
    if a.device.type != "cuda" or a.device != b.device:
        raise ValueError("a and b must be on the same CUDA/ROCm GPU")
    if a.dtype != b.dtype or a.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(
            "a and b must have the same float16, bfloat16 or float32 dtype"
        )
    if a.shape[-2] != b.shape[-1] or a.shape[:-2] != b.shape[:-2]:
        raise ValueError("factor batch and rank dimensions must match")
    if a.shape[-2] > min(a.shape[-1], b.shape[-2]):
        raise ValueError("rank must not exceed either ambient dimension")
    return a.ndim == 2


def _eigh(g: Tensor, backend: EighBackend) -> tuple[Tensor, Tensor, str]:
    import torch

    n = g.shape[-1]
    padded_n = n + n % 2
    fits = (2 * padded_n * padded_n + padded_n + 256) * 8 <= 64 * 1024
    if backend == "auto":
        backend = ("jacobi" if fits else "cpu") if torch.version.hip else "torch"
    if backend == "cpu":
        values, vectors = torch.linalg.eigh(g.cpu())
        return values.flip(-1).to(g.device), vectors.flip(-1).to(g.device), backend
    if backend == "torch":
        values, vectors = torch.linalg.eigh(g)
        return values.flip(-1), vectors.flip(-1), backend
    if not fits:
        raise ValueError("Jacobi core does not fit its 64 KiB shared-memory policy")
    source = torch.nn.functional.pad(g, (0, n % 2, 0, n % 2))
    values, vectors, _ = extension().jacobi_eigh(source.contiguous(), 60, 1e-14)
    values, vectors = values[:, :n], vectors[:, :n, :n]
    order = values.argsort(dim=-1, descending=True)
    values = values.gather(1, order)
    vectors = vectors.gather(2, order.unsqueeze(1).expand(-1, n, -1))
    residual = (g @ vectors - vectors * values.unsqueeze(1)).norm(dim=(-2, -1))
    if not (residual <= 1e-11 * g.norm(dim=(-2, -1)).clamp_min(1e-300)).all().item():
        raise RuntimeError("FlashTSQR Jacobi eigensolver did not converge")
    return values, vectors, backend


def refactor_lora(
    a: Tensor, b: Tensor, *, eigh_backend: EighBackend = "auto"
) -> LoRAResult:
    """Refactor B @ A into singular-value order, using our GPU TSQR.

    Inputs are unmodified 2D factors or same-shape 3D batches. Outputs remain on
    the input GPU in float32. QR uses the smaller ambient side; h and h @ h.T
    are formed in FP64 before eigendecomposition. Singular values are split
    evenly between factors. Numerical null directions below rank * eps64 times
    the largest Gram eigenvalue are set to zero. No generic QR is substituted.

    EVD auto uses torch on CUDA, Jacobi on ROCm for small cores, and CPU FP64
    EVD for larger ROCm cores, as in probe. The selected EVD is always reported.
    Set eigh_backend explicitly to disallow that auto selection. This operation
    may synchronize with the host and does not support autograd or graph capture.
    """
    import torch

    single = _validate(a, b)
    if eigh_backend not in ("auto", "torch", "jacobi", "cpu"):
        raise ValueError(f"unknown EVD backend: {eigh_backend}")
    with (
        torch.no_grad(),
        torch.cuda.device(a.device),
        torch.autocast("cuda", enabled=False),
    ):
        if single:
            a, b = a.unsqueeze(0), b.unsqueeze(0)
        left = b.shape[1] <= a.shape[2]
        x = (b if left else a.transpose(1, 2)).float().contiguous()
        other = (a if left else b.transpose(1, 2)).double()
        factor = qr_factor(x)
        h = factor.r.double() @ other
        values, vectors, backend = _eigh(h @ h.transpose(1, 2), eigh_backend)
        cutoff = values[:, :1].clamp_min(0) * (
            a.shape[1] * torch.finfo(torch.float64).eps
        )
        values = torch.where(values > cutoff, values, torch.zeros_like(values))
        singular = values.sqrt()
        root = singular.sqrt()
        inverse = torch.where(root > 0, root.reciprocal(), torch.zeros_like(root))
        side_q = factor.apply_q((vectors * root.unsqueeze(1)).float().contiguous())
        side_h = (inverse.unsqueeze(2) * (vectors.transpose(1, 2) @ h)).float()
        if left:
            a_out, b_out = side_h.contiguous(), side_q.contiguous()
        else:
            a_out = side_q.transpose(1, 2).contiguous()
            b_out = side_h.transpose(1, 2).contiguous()
        singular = singular.float()
        if single:
            a_out, b_out, singular = a_out[0], b_out[0], singular[0]
        return LoRAResult(
            a_out,
            b_out,
            singular,
            CompressionInfo(factor.info, backend, a.shape[1], a.shape[1]),
        )


def truncate_lora(
    a: Tensor,
    b: Tensor,
    keep: int,
    *,
    pad_to_original_rank: bool = False,
    eigh_backend: EighBackend = "auto",
) -> LoRAResult:
    """Return rank-keep factors, compact by default, with the full spectrum.

    keep=0 produces empty rank dimensions. Explicit padding preserves nominal
    shapes for legacy consumers but does not reduce stored tensor bytes.
    Inputs follow refactor_lora's shape, dtype, scaling and device contracts.
    """
    from dataclasses import replace

    import torch

    _validate(a, b)
    rank = a.shape[-2]
    if isinstance(keep, bool) or not isinstance(keep, int):
        raise TypeError("keep must be an integer")
    if not 0 <= keep <= rank:
        raise ValueError("keep must be between zero and the original rank")
    if not isinstance(pad_to_original_rank, bool):
        raise TypeError("pad_to_original_rank must be a boolean")
    result = refactor_lora(a, b, eigh_backend=eigh_backend)
    # A prefix can already be contiguous while retaining the original allocation.
    # Clone both factors so compact output owns only its retained tensor storage.
    compact_a = result.a[..., :keep, :].clone(memory_format=torch.contiguous_format)
    compact_b = result.b[..., :keep].clone(memory_format=torch.contiguous_format)
    if pad_to_original_rank:
        padded_a, padded_b = (
            result.a.new_zeros(result.a.shape),
            result.b.new_zeros(result.b.shape),
        )
        padded_a[..., :keep, :] = compact_a
        padded_b[..., :keep] = compact_b
        compact_a, compact_b = padded_a, padded_b
    return LoRAResult(
        compact_a,
        compact_b,
        result.singular_values,
        replace(result.info, retained_rank=keep, padded=pad_to_original_rank),
    )
