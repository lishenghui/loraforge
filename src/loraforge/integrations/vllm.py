# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shenghui Li
# Adapted from probe/tools/llm_fleet/vllm_gemv_patch.py at
# 5fd6ae060d106066b306959f9a34b726a281ef4c; see docs/provenance.md.
"""Explicit per-worker vLLM 0.26.0 linear decode integration."""

import os
from functools import lru_cache
from importlib.metadata import version
from threading import RLock

_lock = RLock()
_installed = None


@lru_cache(maxsize=1)
def _operator():
    import torch

    from loraforge.ops.vllm_gemv import expand, shrink

    @torch.library.custom_op("loraforge::vllm_linear", mutates_args=("y",))
    def linear(
        y: torch.Tensor,
        x: torch.Tensor,
        a: list[torch.Tensor],
        b: list[torch.Tensor],
        scale: float,
        routes: torch.Tensor,
        sorted_tokens: torch.Tensor,
        counts: torch.Tensor,
        starts: torch.Tensor,
        ids: torch.Tensor,
        no_lora: torch.Tensor,
        active: torch.Tensor,
        add: bool,
        threshold: int,
    ) -> None:
        # These are native CPU metadata tensors; specialization fixes the
        # dispatch decision for each captured graph. Routing stays on device.
        if no_lora.item():
            return
        with torch.cuda.device(x.device):
            buf = torch.empty(
                (len(a), x.shape[0], b[0].shape[-1]),
                device=x.device,
                dtype=torch.float32,
            )
            if x.shape[0] <= threshold * max(1, int(active.item())):
                shrink(x, a, buf, routes, scale=scale)
                expand(buf, b, y, routes, add=add)
            else:
                from vllm.lora.ops.triton_ops.lora_expand_op import _lora_expand
                from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink

                meta = (routes, sorted_tokens, counts, starts, ids, no_lora, active)
                _lora_shrink(x, a, buf, *meta, scale)
                _lora_expand(buf, b, y, *meta, offset_start=0, add_inputs=add)

    @linear.register_fake
    def fake(*args, **kwargs):
        return None

    return linear


def install_vllm_backend(*, tokens_per_lora: int = 4) -> None:
    """Install in each worker before model construction; repeatable/conflict-aware.

    Only specialized dense-linear decode is replaced. Prefill, unspecialized
    execution, extra keyword contracts and non-linear methods stay native.
    """
    global _installed
    if type(tokens_per_lora) is not int or tokens_per_lora < 1:
        raise ValueError("tokens_per_lora must be a positive integer")
    if version("vllm") != "0.26.0":
        raise RuntimeError("LoRAForge requires exactly vLLM 0.26.0")
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    with _lock:
        current = PunicaWrapperGPU.add_lora_linear
        if _installed is not None:
            patched, threshold = _installed
            if current is not patched or tokens_per_lora != threshold:
                raise RuntimeError("vLLM registration conflicts with existing backend")
            return
        if current.__module__ != "vllm.lora.punica_wrapper.punica_gpu":
            raise RuntimeError("refusing to replace a foreign vLLM backend")
        op = _operator()

        def add_lora_linear(
            self,
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            scale,
            output_slices,
            *,
            buffer=None,
            **kwargs,
        ):
            if (
                self.is_prefill
                or not self.lora_config.specialize_active_lora
                or buffer is not None
                or set(kwargs) - {"add_inputs"}
                or not lora_a_stacked
                or not 1 <= lora_a_stacked[0].shape[-2] <= 256
            ):
                return current(
                    self,
                    y,
                    x,
                    lora_a_stacked,
                    lora_b_stacked,
                    scale,
                    output_slices,
                    buffer=buffer,
                    **kwargs,
                )
            if not (len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)):
                raise ValueError("slice counts must match")
            if tuple(b.shape[-2] for b in lora_b_stacked) != tuple(output_slices):
                raise ValueError("output slice widths must match B")
            x2, y2 = x.view(-1, x.shape[-1]), y.view(-1, y.shape[-1])
            op(
                y2,
                x2,
                list(lora_a_stacked),
                list(lora_b_stacked),
                float(scale),
                *self.token_mapping_meta.meta_args(x2.shape[0], True),
                kwargs.get("add_inputs", True),
                tokens_per_lora,
            )

        PunicaWrapperGPU.add_lora_linear = add_lora_linear
        _installed = add_lora_linear, tokens_per_lora


def register() -> None:
    """vllm.general_plugins entrypoint; inert unless explicitly enabled."""
    if os.environ.get("LORAFORGE_VLLM") == "1":
        install_vllm_backend()
