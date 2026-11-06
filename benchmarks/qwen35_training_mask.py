# SPDX-License-Identifier: Apache-2.0
"""Fixed unpadded full-sequence SDPA policy for the isolated Qwen3.5 probe.

Keep implicit causal SDPA + GQA while capturing/tracing. The original SDPA
function is registered unchanged under a new worker-local interface name.
This is not a mask policy for padding, packing, cached or sliding attention.
"""

import hashlib
import importlib.metadata
import inspect

NAME = "loraforge_fixed_causal_sdpa"


def fixed_positions(ids):
    import torch

    if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 2:
        raise ValueError("positive batch of fixed full sequences required")
    # Native Qwen3.5 constructs four identical axes, then uses [1:] for RoPE
    # and axis zero for automatic packing detection. We provide exactly those
    # three RoPE axes for this already-known unpacked sequence. No positions
    # are removed from the rotary calculation; no custom user positions enter.
    return (
        torch.arange(ids.shape[1], device=ids.device)
        .view(1, 1, -1)
        .expand(3, ids.shape[0], -1)
    )


def validate_fixed_causal(
    *,
    batch_size,
    q_length,
    kv_length,
    q_offset=0,
    kv_offset=0,
    attention_mask=None,
    allow_is_causal_skip=True,
    **kwargs,
):
    if (
        batch_size < 1
        or q_length < 2
        or q_length != kv_length
        or q_offset != 0
        or kv_offset != 0
        or attention_mask is not None
        or not allow_is_causal_skip
    ):
        raise ValueError(
            "fixed unpadded, unpacked, uncached full causal sequence required"
        )


def fixed_causal_mask(*, mask_function, **kwargs):
    from transformers.masking_utils import causal_mask_function

    if mask_function is not causal_mask_function:
        raise ValueError("custom/packed attention patterns are unsupported")
    validate_fixed_causal(**kwargs)
    return None


def install(model):
    from transformers import AttentionInterface
    from transformers import masking_utils as masks
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward,
        use_gqa_in_sdpa,
    )

    version = importlib.metadata.version("transformers")
    if version != "5.10.1":
        raise RuntimeError("fixed-causal probe is scoped to Transformers 5.10.1")
    if model.config._attn_implementation != "sdpa" or model.config.use_cache:
        raise ValueError("native SDPA and disabled KV cache required")
    attention = [m for m in model.modules() if type(m).__name__ == "Qwen3_5Attention"]
    expected = model.config.layer_types.count("full_attention")
    if (
        expected < 1
        or len(attention) != expected
        or not all(m.is_causal for m in attention)
    ):
        raise ValueError("native causal Qwen3.5 attention layers must match the config")
    AttentionInterface.register(NAME, sdpa_attention_forward)
    masks.AttentionMaskInterface.register(NAME, fixed_causal_mask)
    model.set_attn_implementation(NAME)
    assert all(m.config._attn_implementation == NAME for m in attention)
    return dict(
        name=NAME,
        transformers=version,
        attention_layers=len(attention),
        scope="positive rectangular unpadded/unpacked batch, no KV cache, q_length=kv_length, zero offsets, causal SDPA retained",
        native_source_sha256={
            fn.__name__: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            for fn in (
                sdpa_attention_forward,
                use_gqa_in_sdpa,
                masks.is_tracing,
                masks._ignore_causal_mask_sdpa,
            )
        },
    )
