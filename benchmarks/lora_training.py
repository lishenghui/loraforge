# SPDX-License-Identifier: Apache-2.0
"""First-order training probes for frozen-base LoRA; no serving-framework hooks.

NF4 weights/scales are immutable. Only the explicit dense reference supports
CPU/float64 checks. LoRA factors may be FP32 master parameters with BF16 compute.
Dropout, bias, DoRA, distributed training and double backward are out of scope.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


@dataclass
class FrozenBase:
    weight: torch.Tensor
    state: object = None
    scales: torch.Tensor | None = None

    @classmethod
    def nf4(cls, weight, state):
        from qwen35_nf4_ops import prepare_scales

        if weight.requires_grad or weight.dtype != torch.uint8:
            raise ValueError("NF4 base must be frozen uint8")
        return cls(weight, state, prepare_scales(state))

    @classmethod
    def dense(cls, weight):
        if weight.requires_grad:
            raise ValueError("base weight must be frozen")
        return cls(weight)

    @property
    def shape(self):
        return tuple(self.weight.shape if self.state is None else self.state.shape)

    def tensors(self):
        if self.state is None:
            return (self.weight,)
        return self.weight, self.scales, self.state.code


def _weight(base, backend):
    if base.state is None:
        return base.weight
    if backend == "bnb":
        from bitsandbytes.functional import dequantize_4bit

        return dequantize_4bit(base.weight, quant_state=base.state)
    from qwen35_nf4_ops import dequantize

    return dequantize(base.weight, base.scales, base.state.code, base.shape)


def _forward_base(x, base, backend):
    if backend == "bnb" and base.state is not None:
        import bitsandbytes as bnb

        return bnb.matmul_4bit(x, base.weight.t(), base.state)
    return F.linear(x, _weight(base, backend))


class _FrozenNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base):
        ctx.base = base
        # Track frozen-storage versions without retaining the full BF16 weight.
        ctx.save_for_backward(*base.tensors())
        return F.linear(x, _weight(base, "cached"))

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        _ = ctx.saved_tensors
        dx = dy @ _weight(ctx.base, "cached") if ctx.needs_input_grad[0] else None
        return dx, None


def _base_linear(x, base, backend):
    if backend == "cached":
        return _FrozenNF4.apply(x, base)
    if backend == "naive":
        # Intentionally bad training transfer: autograd retains BF16 weight.
        return F.linear(x, _weight(base, "cached"))
    return _forward_base(x, base, backend)


class _RecomputeLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, base, scale):
        ac, bc = a.to(x.dtype), b.to(x.dtype)
        y = _forward_base(x, base, "cached")
        y = y + F.linear(F.linear(x, ac), bc) * scale
        ctx.save_for_backward(x, a, b, *base.tensors())
        ctx.base, ctx.scale = base, scale
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, a, b, *_ = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        grad = dy.reshape(-1, dy.shape[-1])
        ac, bc = a.to(x.dtype), b.to(x.dtype)
        # Match native autograd's BF16 rounding of the scaled output gradient.
        ds = grad * ctx.scale
        dh = ds @ bc
        da = (dh.t() @ flat).to(a.dtype) if ctx.needs_input_grad[1] else None
        db = (
            (ds.t() @ F.linear(flat, ac)).to(b.dtype)
            if ctx.needs_input_grad[2]
            else None
        )
        dx = None
        if ctx.needs_input_grad[0]:
            dx = grad @ _weight(ctx.base, "cached") + dh @ ac
            dx = dx.reshape(x.shape)
        return dx, da, db, None, None


def linear(x, a, b, base, *, backend="cached", scale=1.0):
    """Train one plain LoRA on a frozen base; no cached trainable A/B copies."""
    if backend not in ("bnb", "cached", "naive", "recompute"):
        raise ValueError("unknown training backend")
    n, k = base.shape
    if (
        x.ndim not in (2, 3)
        or a.ndim != 2
        or b.ndim != 2
        or x.shape[-1] != k
        or a.shape[1] != k
        or b.shape != (n, a.shape[0])
        or base.weight.requires_grad
    ):
        raise ValueError("requires X[...,K], A[R,K], B[N,R] and a frozen base")
    if base.state is not None and x.dtype != torch.bfloat16:
        raise ValueError("NF4 probe requires BF16 compute")
    if backend == "recompute":
        return _RecomputeLoRA.apply(x, a, b, base, scale)
    y = _base_linear(x, base, backend)
    return y + F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) * scale
