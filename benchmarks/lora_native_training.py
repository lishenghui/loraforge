# SPDX-License-Identifier: Apache-2.0
"""First-order LoRA consumers retaining separate native base projections."""

import torch
import torch.nn.functional as F
from lora_swiglu_training import _act_backward, _Activation
from torch import nn
from torch.autograd.function import once_differentiable


class _ExpandAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, shrink, b, scale):
        ctx.save_for_backward(shrink, b)
        ctx.scale = scale
        if base.is_cuda:
            if base.dtype != torch.bfloat16 or shrink.dtype != torch.bfloat16:
                raise ValueError("GPU probe requires BF16 compute")
            from lora_swiglu_kernels import output

            return output(base, shrink, b.to(base.dtype), scale)
        return base + F.linear(shrink, b.to(base.dtype)) * scale

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        shrink, b = ctx.saved_tensors
        ds = (dy * ctx.scale).reshape(-1, dy.shape[-1])
        du = (ds @ b.to(dy.dtype)).reshape_as(shrink)
        db = (ds.t() @ shrink.reshape(-1, shrink.shape[-1])).to(b.dtype)
        return dy, du, db, None


class _GateUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g0, u0, ug, uu, bg, bu, scale):
        shape = g0.shape
        bgc, buc = bg.to(g0.dtype), bu.to(g0.dtype)
        if g0.is_cuda:
            if any(v.dtype != torch.bfloat16 for v in (g0, u0, ug, uu)):
                raise ValueError("GPU probe requires BF16 compute")
            from lora_swiglu_kernels import gate_up_separate

            g, u, h = gate_up_separate(g0, u0, ug, uu, bgc, buc, scale)
        else:
            g = g0 + F.linear(ug, bgc) * scale
            u = u0 + F.linear(uu, buc) * scale
            h = F.silu(g) * u
        ctx.save_for_backward(g, u, ug, uu, bg, bu)
        ctx.scale, ctx.shape = scale, shape
        return h.reshape(shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dh):
        g, u, ug, uu, bg, bu = ctx.saved_tensors
        dg, du, _ = _act_backward(dh.reshape_as(g), g, u)
        dsg, dsu = dg * ctx.scale, du * ctx.scale
        rank = ug.shape[-1]
        dsg, dsu = dsg.reshape(-1, g.shape[-1]), dsu.reshape(-1, u.shape[-1])
        ugf, uuf = ug.reshape(-1, rank), uu.reshape(-1, rank)
        dug = (dsg @ bg.to(g.dtype)).reshape_as(ug)
        duu = (dsu @ bu.to(u.dtype)).reshape_as(uu)
        dbg = (dsg.t() @ ugf).to(bg.dtype)
        dbu = (dsu.t() @ uuf).to(bu.dtype)
        return dg.reshape(ctx.shape), du.reshape(ctx.shape), dug, duu, dbg, dbu, None


class NativeTrainingMLP(nn.Module):
    """Preserve PEFT modules/masters, independent base GEMMs, and their autograd."""

    def __init__(self, original: nn.Module, backend: str):
        super().__init__()
        if original.config.hidden_act != "silu":
            raise ValueError("SiLU required")
        if backend not in ("native-activation", "native-consumer", "native-expand"):
            raise ValueError("unsupported native-base consumer")
        self.gate_proj, self.up_proj, self.down_proj = (
            original.gate_proj,
            original.up_proj,
            original.down_proj,
        )
        self.backend = backend
        scales = []
        for layer in (self.gate_proj, self.up_proj, self.down_proj):
            if (
                list(layer.active_adapters) != ["default"]
                or layer.merged
                or layer.disable_adapters
            ):
                raise ValueError("one unmerged active adapter required")
            if getattr(layer, "lora_variant", {}) or layer.use_dora.get(
                "default", False
            ):
                raise ValueError("ordinary LoRA required")
            if getattr(layer.lora_dropout["default"], "p", 0) != 0:
                raise ValueError("dropout must be zero")
            if layer.r["default"] != 4 or layer.get_base_layer().weight.requires_grad:
                raise ValueError("frozen base and rank four required")
            scales.append(layer.scaling["default"])
        if len(set(scales)) != 1:
            raise ValueError("scales must match")
        self.scale = scales[0]

    def forward(self, x):
        if self.backend == "native-expand":

            def projection(layer):
                base = layer.get_base_layer()(x)
                a = layer.lora_A["default"]
                shrink = a(layer._cast_input_dtype(x, a.weight.dtype))
                return _ExpandAdd.apply(
                    base, shrink, layer.lora_B["default"].weight, self.scale
                )

            act = F.silu(projection(self.gate_proj))
            return self.down_proj(act * projection(self.up_proj))
        if self.backend == "native-activation":
            g, u = self.gate_proj(x), self.up_proj(x)
            h = _Activation.apply(g, u)
        else:
            gate, up = self.gate_proj, self.up_proj
            # Keep PEFT's FP32-input casts before AMP linear, and separate
            # base projection nodes. No concatenated base backward is created.
            g0 = gate.get_base_layer()(x)
            ug = gate.lora_A["default"](
                gate._cast_input_dtype(x, gate.lora_A["default"].weight.dtype)
            )
            u0 = up.get_base_layer()(x)
            uu = up.lora_A["default"](
                up._cast_input_dtype(x, up.lora_A["default"].weight.dtype)
            )
            # The low-rank factors are refreshed from the actual masters.
            h = _GateUp.apply(
                g0,
                u0,
                ug,
                uu,
                gate.lora_B["default"].weight,
                up.lora_B["default"].weight,
                self.scale,
            )
        return self.down_proj(h)
