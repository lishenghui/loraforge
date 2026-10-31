# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Adapted from probe/FlashTSQR/flashtsqr.py; see docs/provenance.md.
"""Batched Householder TSQR with owned, reusable factorization state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._extension import extension, source_hashes

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class ExecutionInfo:
    """Implementation selected for this factorization, without global state."""

    path: str
    device: str
    apply_mode: str
    source_sha256: tuple[tuple[str, str], ...]
    backend: str = "FlashTSQR"


def _check_tensor(x: Tensor, name: str, ndim: int = 3) -> None:
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != ndim or any(size == 0 for size in x.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}D tensor")
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be on a CUDA or ROCm GPU; no CPU QR fallback")
    if x.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype float32")


def _check_capture() -> None:
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FlashTSQR graph capture is not supported in this version")


class QRFactorization:
    """Owned QR state for ``x[batch, rows, columns]``.

    ``r`` is [batch, columns, columns]. ``apply_q(z)`` evaluates the implicit
    thin Q times z without materializing Q. Keep r and the private reflector
    state unmodified. Later factorizations do not invalidate this object.
    Applications on other streams wait for factorization completion. Native
    state is immutable and its allocator use is recorded on consumer streams.
    Callers remain responsible for ordering writes to their input tensors.
    Autograd and CUDA graph capture are unsupported.
    """

    def __init__(self, r, shape, apply, tensors, ready, info):
        self.r: Tensor = r
        self.info: ExecutionInfo = info
        self._shape = shape
        self._apply: Callable[[Tensor], Tensor] = apply
        self._tensors = tuple(tensors)
        self._ready = ready

    def apply_q(self, z: Tensor) -> Tensor:
        """Apply Q to GPU float32 z[batch, columns, k], for any positive k."""
        import torch

        _check_tensor(z, "z")
        if z.device != self.r.device:
            raise ValueError("z must be on the factorization device")
        if z.shape[:2] != (self._shape[0], self._shape[2]):
            raise ValueError(
                "z batch and column dimensions must match the factorization"
            )
        with (
            torch.no_grad(),
            torch.cuda.device(z.device),
            torch.autocast("cuda", enabled=False),
        ):
            _check_capture()
            stream = torch.cuda.current_stream(z.device)
            stream.wait_event(self._ready)
            for tensor in (*self._tensors, z):
                tensor.record_stream(stream)
            return self._apply(z.contiguous())


def _leaf_rows(x: Tensor, limit: int) -> int | None:
    import torch

    _, m, n = x.shape
    rows = max(n, min(128 if torch.version.hip else 256, m))
    leaf_bytes = (rows * n + rows + n + 32) * 4
    merge_bytes = (n * n + 3 * n + 32) * 4
    return rows if max(leaf_bytes, merge_bytes) <= limit else None


class _Wide:
    def __init__(self, x: Tensor, ext):
        import torch

        batch, m, n = x.shape
        rows = max(2 * n, 512)
        count = (m + rows - 1) // rows
        self.shape = batch, m, n, count, rows
        self.ext = ext
        padded = x.new_zeros(batch, count * rows, n)
        padded[:, :m] = x
        r, v, tau = ext.wide_factor(padded.reshape(batch * count, rows, n))
        r = r.reshape(batch, count, n, n)
        self.leaf = v, tau
        self.levels = []
        self.tensors = [v, tau]
        while count > 1:
            pairs = (count + 1) // 2
            if count % 2:
                r = torch.cat((r, torch.zeros_like(r[:, :1])), dim=1)
            r, v, tau = ext.wide_factor(r.reshape(batch * pairs, 2 * n, n))
            self.levels.append((v, tau, count, pairs))
            self.tensors.extend((v, tau))
            r = r.reshape(batch, pairs, n, n)
            count = pairs
        self.r = r[:, 0].contiguous()

    def apply(self, z: Tensor) -> Tensor:
        batch, m, n, count, rows = self.shape
        k = z.shape[-1]
        w = z.reshape(batch, 1, n, k)
        for v, tau, inputs, pairs in reversed(self.levels):
            c = z.new_zeros(batch, pairs, 2 * n, k)
            c[:, :, :n] = w
            w = self.ext.wide_apply(v, tau, c.reshape(batch * pairs, 2 * n, k))
            w = w.reshape(batch, 2 * pairs, n, k)[:, :inputs]
        c = z.new_zeros(batch, count, rows, k)
        c[:, :, :n] = w
        return (
            self.ext.wide_apply(*self.leaf, c.reshape(batch * count, rows, k))
            .reshape(batch, count * rows, k)[:, :m]
            .contiguous()
        )


def qr_factor(
    x: Tensor,
    *,
    path: Literal["auto", "full", "wide"] = "auto",
    apply_mode: Literal["serial", "cooperative"] = "serial",
) -> QRFactorization:
    """Factor GPU float32 x[batch, rows >= columns, columns].

    Input is not modified. The default selects the shared-memory FlashTSQR
    kernel, or its global-workspace wide kernel when shared memory cannot fit.
    A compiler and a matching PyTorch CUDA/ROCm toolkit are required on first
    use. Compilation uses PyTorch's extension cache and environment overrides;
    LoRAForge does not change compiler or architecture environment variables.
    ``cooperative`` is an explicit experimental apply mode for the full path.
    """
    import torch

    _check_tensor(x, "x")
    if x.shape[1] < x.shape[2]:
        raise ValueError("x must have rows >= columns")
    if path not in ("auto", "full", "wide"):
        raise ValueError(f"unknown TSQR path: {path}")
    if apply_mode not in ("serial", "cooperative"):
        raise ValueError(f"unknown apply mode: {apply_mode}")
    with (
        torch.no_grad(),
        torch.cuda.device(x.device),
        torch.autocast("cuda", enabled=False),
    ):
        _check_capture()
        ext = extension()
        rows = _leaf_rows(x, ext.shared_memory_limit(x))
        use_wide = path == "wide" or (path == "auto" and rows is None)
        if not use_wide and rows is None:
            raise ValueError(
                "full TSQR does not fit device shared memory; use auto/wide"
            )
        if use_wide and apply_mode != "serial":
            raise ValueError("cooperative application requires the full path")
        if use_wide:
            wide = _Wide(x.contiguous(), ext)
            r, apply, tensors = wide.r, wide.apply, wide.tensors
        else:
            state = ext.factor(x.contiguous(), rows, 256)
            r, tensors = state[0], state
            original_rows = x.shape[1]

            def apply(z):
                return ext.apply_q(
                    z, state, original_rows, 256, apply_mode == "cooperative"
                )

        # Native launches and ATen operations all use this device's current stream.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(x.device))
        info = ExecutionInfo(
            path="tsqr_wide" if use_wide else "tsqr_full",
            device=str(x.device),
            apply_mode=apply_mode,
            source_sha256=source_hashes(),
        )
        return QRFactorization(r, tuple(x.shape), apply, tensors, ready, info)
