# SPDX-License-Identifier: Apache-2.0
"""Plain LoRA MLP training experiments with a frozen fused BF16 base.

First-order only. Save small U tensors; rematerialize wide H with activation
backward. No in-place saved-activation writes or cached trainable weight copies.
Dense CPU/float64 paths exist solely for derivative and checkpoint checks.
"""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def _act_forward(g, u):
    if g.is_cuda:
        from lora_swiglu_kernels import forward

        if g.dtype != torch.bfloat16:
            raise ValueError("GPU probe requires BF16")
        return forward(g, u)
    return F.silu(g) * u


def _act_backward(dh, g, u, rematerialize=False):
    if g.is_cuda:
        from lora_swiglu_kernels import backward

        return backward(dh, g, u, rematerialize=rematerialize)
    sig = torch.sigmoid(g)
    act = F.silu(g)
    dg = (dh * u) * sig * (1 + g * (1 - sig))
    return dg, dh * act, act * u if rematerialize else None


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        return _act_forward(g, u)

    @staticmethod
    @once_differentiable
    def backward(ctx, dh):
        g, u = ctx.saved_tensors
        dg, du, _ = _act_backward(dh, g, u)
        return dg, du


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ag, bg, au, bu, ad, bd, wgu, wd, scale, fused):
        flat = x.reshape(-1, x.shape[-1])
        gu0 = F.linear(flat, wgu)
        ug, uu = F.linear(flat, ag.to(x.dtype)), F.linear(flat, au.to(x.dtype))
        if fused and x.is_cuda:
            from lora_swiglu_kernels import gate_up

            g, u, h = gate_up(gu0, ug, uu, bg.to(x.dtype), bu.to(x.dtype), scale)
        else:
            g0, u0 = gu0.chunk(2, -1)
            g = g0 + F.linear(ug, bg.to(x.dtype)) * scale
            u = u0 + F.linear(uu, bu.to(x.dtype)) * scale
            h = _act_forward(g, u)
        del gu0
        ud = F.linear(h, ad.to(x.dtype))
        y = F.linear(h, wd)
        if fused and x.is_cuda:
            from lora_swiglu_kernels import output

            y = output(y, ud, bd.to(x.dtype), scale)
        else:
            y = y + F.linear(ud, bd.to(x.dtype)) * scale
        ctx.save_for_backward(x, g, u, ug, uu, ud, ag, bg, au, bu, ad, bd, wgu, wd)
        ctx.scale = scale
        ctx.fused = fused and x.is_cuda
        ctx.compact = fused == 2 and x.is_cuda
        return y.reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, g, u, ug, uu, ud, ag, bg, au, bu, ad, bd, wgu, wd = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        dy = dy.reshape(-1, dy.shape[-1])
        acg, bcg, acu, bcu, acd, bcd = [v.to(x.dtype) for v in (ag, bg, au, bu, ad, bd)]
        ds = dy * ctx.scale
        dud = ds @ bcd
        dbd = (ds.t() @ ud).to(bd.dtype)
        del ds
        dx = None
        if ctx.fused:
            from lora_swiglu_kernels import input_gradient, mlp_backward

            dgu, dsgu, h = mlp_backward(
                dy @ wd, dud, acd, g, u, ctx.scale, compact=ctx.compact
            )
            if not ctx.compact:
                dg, du = dgu.chunk(2, -1)
                dsg, dsu = dsgu.chunk(2, -1)
        else:
            dh = dy @ wd + dud @ acd
            dg, du, h = _act_backward(dh, g, u, rematerialize=True)
            del dh
            dsg, dsu = dg * ctx.scale, du * ctx.scale
        dad = (dud.t() @ h).to(ad.dtype)
        del h, dud
        if ctx.compact:
            if ctx.needs_input_grad[0]:
                dx = dgu @ wgu
            # This is a freshly produced gradient buffer, never a saved input.
            # The base GEMM consumes it first on the same stream. First-order only.
            dgu.mul_(ctx.scale)
            dsg, dsu = dgu.chunk(2, -1)
        dug, duu = dsg @ bcg, dsu @ bcu
        dag = (dug.t() @ flat).to(ag.dtype)
        dbg = (dsg.t() @ ug).to(bg.dtype)
        dau = (duu.t() @ flat).to(au.dtype)
        dbu = (dsu.t() @ uu).to(bu.dtype)
        del dsg, dsu
        if ctx.fused:
            del dsgu
        if ctx.compact:
            del dgu
        if ctx.needs_input_grad[0]:
            # Match the native fused-base projection dimensions. The residual
            # additions can still differ in BF16 association; record that gate.
            if not ctx.compact:
                dx = (dgu if ctx.fused else torch.cat((dg, du), -1)) @ wgu
            if ctx.fused:
                dx = input_gradient(dx, dug, duu, acg, acu)
            else:
                dx = dx + (dug @ acg + duu @ acu)
            dx = dx.reshape(x.shape)
        return dx, dag, dbg, dau, dbu, dad, dbd, None, None, None, None


def mlp(x, ag, bg, au, bu, ad, bd, wgu, wd, *, scale=0.7, backend="native"):
    if backend not in ("native", "activation", "block", "fused", "compact"):
        raise ValueError("unknown MLP backend")
    if wgu.requires_grad or wd.requires_grad:
        raise ValueError("base must remain frozen")
    if backend in ("block", "fused", "compact"):
        fusion = {"block": 0, "fused": 1, "compact": 2}[backend]
        return _MLP.apply(x, ag, bg, au, bu, ad, bd, wgu, wd, scale, fusion)
    g0, u0 = F.linear(x, wgu).chunk(2, -1)
    g = g0 + F.linear(F.linear(x, ag.to(x.dtype)), bg.to(x.dtype)) * scale
    u = u0 + F.linear(F.linear(x, au.to(x.dtype)), bu.to(x.dtype)) * scale
    h = _Activation.apply(g, u) if backend == "activation" else F.silu(g) * u
    return (
        F.linear(h, wd) + F.linear(F.linear(h, ad.to(x.dtype)), bd.to(x.dtype)) * scale
    )
