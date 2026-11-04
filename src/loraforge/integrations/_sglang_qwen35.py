# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen3.5 dense-text adapter packing for SGLang 0.5.22.

The stock loader understands four independent q/k/v/z factors, but HF PEFT
uses a shared qkv A and a separate z A. Preserve both products by repeating
qkv A three times, without merging ranks or changing scaling. The b/a gates
are two independent slices. SGLang still owns batching and recurrent state.
"""

from functools import wraps


def _text_config(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def _is_qwen35(config):
    return getattr(_text_config(config), "model_type", None) == "qwen3_5_text"


def normalize_gdn_weights(weights: dict, config, rank: int) -> None:
    """Pack CPU A/B pairs in place, preserving BF16 factors and global scaling.

    Supports fused qkvz/ba, HF qkv+z/b+a, or independent q/k/v/z; missing
    projections become zero factors. Rejects overlapping formats and bad
    shapes before publishing any changes. No device operations are performed.
    """
    import torch

    config = _text_config(config)
    if not _is_qwen35(config) or type(rank) is not int or rank < 1:
        raise ValueError("Qwen3.5 dense text config and positive rank required")
    hidden = config.hidden_size
    key = config.linear_key_head_dim * config.linear_num_key_heads
    value = config.linear_value_head_dim * config.linear_num_value_heads
    heads = config.linear_num_value_heads
    prefixes = {
        name.split(".linear_attn.")[0] + ".linear_attn."
        for name in weights
        if ".linear_attn.in_proj_" in name
    }
    additions, removals = {}, set()
    for prefix in prefixes:

        def pair(name, width):
            ka, kb = (prefix + name + f".lora_{ab}.weight" for ab in ("A", "B"))
            if ka not in weights and kb not in weights:
                return None
            if ka not in weights or kb not in weights:
                raise ValueError(f"incomplete GDN A/B pair: {prefix}{name}")
            a, b = weights[ka], weights[kb]
            if (
                tuple(a.shape) != (rank, hidden)
                or tuple(b.shape) != (width, rank)
                or a.device.type != "cpu"
                or b.device.type != "cpu"
                or a.dtype != b.dtype
                or not a.is_floating_point()
            ):
                raise ValueError(f"invalid GDN factors: {prefix}{name}")
            removals.update((ka, kb))
            return a, b

        for fused, names, widths in (
            (
                "in_proj_qkvz",
                ("in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"),
                (key, key, value, value),
            ),
            ("in_proj_ba", ("in_proj_b", "in_proj_a"), (heads, heads)),
        ):
            merged = pair(fused, sum(widths))
            parts = [pair(n, w) for n, w in zip(names, widths, strict=True)]
            shared = pair("in_proj_qkv", key * 2 + value) if len(names) == 4 else None
            if merged is not None and (
                shared is not None or any(p is not None for p in parts)
            ):
                raise ValueError("overlapping fused and split GDN factors")
            if shared is not None:
                if any(p is not None for p in parts[:3]):
                    raise ValueError("overlapping qkv and q/k/v GDN factors")
                aa, bb = shared
                parts[:3] = [(aa, b) for b in bb.split(widths[:3], dim=0)]
            if merged is not None:
                aa, bb = merged
                parts = [(aa, b) for b in bb.split(widths, dim=0)]
            present = next((p for p in parts if p is not None), None)
            if present is None:
                continue
            aa, bb = present
            parts = [
                p if p is not None else (torch.zeros_like(aa), bb.new_zeros((w, rank)))
                for p, w in zip(parts, widths, strict=True)
            ]
            additions[prefix + fused + ".lora_A.weight"] = torch.cat(
                [p[0] for p in parts]
            )
            additions[prefix + fused + ".lora_B.weight"] = torch.cat(
                [p[1] for p in parts]
            )
    for name in removals:
        del weights[name]
    weights.update(additions)


def install_qwen35_lora() -> None:
    """Install process-local packing fixes before creating a Qwen3.5 server.

    Use identically for a native csgmv baseline and Fleet. This does not enable
    Fleet or alter attention/DeltaNet kernels. Scope: BF16 dense text, TP=1,
    synchronous adapter loading. Call in every worker (or enable the plugin's
    LORAFORGE_SGLANG_QWEN35=1 switch). Shared SGLang files are never modified.
    """
    from .sglang import _check_version

    _check_version()
    import torch
    from sglang.srt.lora import lora_manager, mem_pool, utils
    from sglang.srt.lora.lora import LoRAAdapter
    from sglang.srt.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )
    from sglang.srt.utils import common

    if getattr(LoRAAdapter, "_loraforge_qwen35", False):
        return
    if "in_proj_ba" in Qwen3_5ForCausalLM.supported_lora_modules:
        raise RuntimeError(
            "SGLang GDN support changed; revalidate this compatibility shim"
        )
    original_normalize = LoRAAdapter._normalize_in_proj_qkvz
    original_targets = utils.get_normalized_target_modules
    original_stacked = utils.get_stacked_multiply
    original_dim = Qwen3_5ForCausalLM.get_hidden_dim
    original_init = mem_pool.LoRAMemoryPool.init_buffers

    @wraps(original_normalize)
    def normalize(self, weights):
        if _is_qwen35(self.base_hf_config):
            normalize_gdn_weights(weights, self.base_hf_config, self.config.r)
        else:
            original_normalize(self, weights)

    @wraps(original_targets)
    def targets(names):
        result = original_targets(names)
        mapping = {
            n: "in_proj_qkvz"
            for n in ("in_proj_qkv", "in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z")
        }
        mapping.update(in_proj_b="in_proj_ba", in_proj_a="in_proj_ba")
        return {mapping.get(n, n) for n in result}

    @wraps(original_stacked)
    def stacked(name, base_model=None):
        if (
            name == "in_proj_ba"
            and base_model is not None
            and _is_qwen35(base_model.config)
        ):
            return 2
        return original_stacked(name, base_model)

    @wraps(original_dim)
    def dimensions(self, name, layer_idx):
        if name == "in_proj_ba":
            return self.config.hidden_size, 2 * self.config.linear_num_value_heads
        return original_dim(self, name, layer_idx)

    @wraps(original_init)
    def init(self, base_model):
        if not _is_qwen35(base_model.config):
            raise ValueError("Qwen3.5 shim requires a dense Qwen3.5 text model")
        if (
            self.tp_size != 1
            or self.enable_lora_overlap_loading
            or self.dtype != torch.bfloat16
        ):
            raise ValueError(
                "Qwen3.5 shim requires BF16, TP=1, non-overlapping loading"
            )
        return original_init(self, base_model)

    LoRAAdapter._normalize_in_proj_qkvz = normalize
    utils.get_normalized_target_modules = targets
    lora_manager.get_normalized_target_modules = targets
    mem_pool.get_normalized_target_modules = targets
    utils.get_stacked_multiply = stacked
    mem_pool.get_stacked_multiply = stacked
    Qwen3_5ForCausalLM.get_hidden_dim = dimensions
    modules = [*Qwen3_5ForCausalLM.supported_lora_modules, "in_proj_ba"]
    Qwen3_5ForCausalLM.supported_lora_modules = modules
    Qwen3_5ForConditionalGeneration.supported_lora_modules = modules
    utils._KNOWN_LORA_TARGET_MODULES |= {"in_proj_ba"}
    # TP is guarded above; membership also prevents base-layer shape inference
    # from mistaking the two gate slices for a single merged A block.
    utils.ATTN_TP_LORA_MODULE_NAMES |= {"in_proj_ba"}
    mem_pool.ATTN_TP_LORA_MODULE_NAMES |= {"in_proj_ba"}
    for name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
        if name not in common.SUPPORTED_LORA_TARGET_MODULES:
            common.SUPPORTED_LORA_TARGET_MODULES.append(name)
    mem_pool.LoRAMemoryPool.init_buffers = init
    LoRAAdapter._loraforge_qwen35 = True
