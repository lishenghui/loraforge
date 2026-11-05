# SPDX-License-Identifier: Apache-2.0
"""Explicit, process-local PEFT MLP replacement for the BF16 training probe."""

import torch
from lora_swiglu_training import mlp
from torch import nn


class TrainingMLP(nn.Module):
    """Keep actual PEFT master parameters; own one immutable concatenated base."""

    def __init__(self, original: nn.Module, backend: str = "native") -> None:
        super().__init__()
        if backend not in ("native", "fused", "compact"):
            raise ValueError("unsupported training MLP path")
        if original.config.hidden_act != "silu":
            raise ValueError("training probe requires SiLU")
        layers = [original.gate_proj, original.up_proj, original.down_proj]
        scale = []
        for prefix, layer in zip(("g", "u", "d"), layers, strict=True):
            if (
                list(layer.active_adapters) != ["default"]
                or layer.merged
                or layer.disable_adapters
            ):
                raise ValueError("one unmerged active adapter is required")
            if getattr(layer, "lora_variant", {}) or layer.use_dora.get(
                "default", False
            ):
                raise ValueError("only ordinary LoRA is measured")
            if getattr(layer.lora_dropout["default"], "p", 0) != 0:
                raise ValueError("dropout must be zero")
            a, b = layer.lora_A["default"], layer.lora_B["default"]
            if a.bias is not None or b.bias is not None or a.weight.shape[0] != 4:
                raise ValueError("rank-4 bias-free adapter required")
            base = layer.get_base_layer()
            if base.bias is not None or base.weight.requires_grad:
                raise ValueError("base must be frozen and bias-free")
            setattr(self, "a" + prefix, a.weight)
            setattr(self, "b" + prefix, b.weight)
            scale.append(layer.scaling["default"])
        if len(set(scale)) != 1:
            raise ValueError("projection scales must match")
        self.scale, self.backend = scale[0], backend
        self.register_buffer(
            "wgu",
            torch.cat(
                [layer.get_base_layer().weight.detach() for layer in layers[:2]], 0
            ),
        )
        self.register_buffer("wd", layers[2].get_base_layer().weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parameters stay FP32; only compute copies use the activation dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return mlp(
                x,
                self.ag,
                self.bg,
                self.au,
                self.bu,
                self.ad,
                self.bd,
                self.wgu,
                self.wd,
                scale=self.scale,
                backend=self.backend,
            )


def replace_mlps(model: nn.Module, backend: str) -> int:
    """Replace decoder MLPs without retaining inactive base/module copies."""
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".mlp") and hasattr(module, "gate_proj")
    ]
    for name, original in candidates:
        parent_name, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        if backend in ("native-activation", "native-consumer", "native-expand"):
            from lora_native_training import NativeTrainingMLP

            replacement = NativeTrainingMLP(original, backend)
        else:
            replacement = TrainingMLP(original, backend)
        setattr(parent, attr, replacement)
    return len(candidates)


def adapter_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Stable PEFT names across ordinary and replaced MLPs for receipt checks."""
    result = {}
    replacements = {
        "ag": ("gate_proj", "A"),
        "bg": ("gate_proj", "B"),
        "au": ("up_proj", "A"),
        "bu": ("up_proj", "B"),
        "ad": ("down_proj", "A"),
        "bd": ("down_proj", "B"),
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        stem, suffix = name.rsplit(".", 1)
        if suffix in replacements and stem.endswith(".mlp"):
            proj, factor = replacements[suffix]
            name = f"{stem}.{proj}.lora_{factor}.default.weight"
        if name in result:
            raise ValueError("duplicate adapter parameter")
        result[name] = p
    return dict(sorted(result.items()))
