# SPDX-License-Identifier: Apache-2.0
"""Process-local prepared Qwen3.5 NF4 experiment, with cached scales only.

Immutable quantized base weights retain uint8 storage. FP32 per-block scales
are reconstructed once per parameter/state version. BF16 weights are temporary
per invocation, with the usual captured-graph lifetime; there is no resident
BF16 weight cache. Small/unsupported cases retain the original BNB method.
"""

import atexit
import json
import os
from pathlib import Path

SHAPES = {
    (32, 1024),
    (8192, 1024),
    (7168, 1024),
    (5120, 1024),
    (1024, 2048),
    (1024, 3584),
}


def select_state(weight, x):
    states = getattr(weight, "bnb_quant_state", {})
    if str(x.dtype) != "torch.bfloat16" or x.ndim not in (2, 3) or x.stride(-1) != 1:
        return None
    if len(states) != 1 or 0 not in states:
        return None
    state = states[0]
    if (
        tuple(state.shape) not in SHAPES
        or state.quant_type != "nf4"
        or state.blocksize != 64
        or state.dtype != x.dtype
        or x.shape[-1] != state.shape[1]
        or x.numel() // x.shape[-1] <= 4
    ):
        return None
    return state


def scale_key(weight, state):
    # Contents must not be modified outside PyTorch's versioned update contract.
    tensors = [weight, state.absmax, state.code]
    if state.nested:
        tensors += [state.state2.absmax, state.state2.code, state.offset]

    def version(tensor):
        try:
            return tensor._version
        except RuntimeError:
            # Loading under inference_mode creates immutable unversioned tensors.
            return None

    return (id(state), tuple((t.data_ptr(), version(t)) for t in tensors))


def install():
    import qwen35_nf4_ops as ops
    import torch
    import torch.nn.functional as functional
    from sglang import __version__
    from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesLinearMethod

    assert __version__ == "0.5.22"
    assert os.environ.get("LORAFORGE_BENCH_QWEN35_NF4") == "1"
    if getattr(BitsAndBytesLinearMethod, "_lf_nf4_cached_scales", False):
        return
    original = BitsAndBytesLinearMethod._apply_4bit_weight
    byte_pairs = os.environ.get("LORAFORGE_BENCH_QWEN35_NF4_PAIRS") == "1"
    stats = dict(
        optimized_calls=0,
        fallback_calls=0,
        scale_prepares=0,
        pair_exact_checks=0,
        byte_pairs=byte_pairs,
        layers={},
        executions={},
    )

    def apply(self, layer, x, bias=None):
        weight = layer.weight
        state = select_state(weight, x)
        if state is None or x.requires_grad:
            stats["fallback_calls"] += 1
            return original(self, layer, x, bias)
        key = scale_key(weight, state)
        cached = getattr(weight, "_lf_nf4_scales", None)
        if cached is None or cached[0] != key:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "prepare NF4 scales in eager warmup before graph capture"
                )
            scales = ops.prepare_scales(state)
            weight._lf_nf4_scales = (key, scales)
            stats["scale_prepares"] += 1
            stats["layers"][str(weight.data_ptr())] = dict(
                shape=list(state.shape),
                scale_bytes=scales.numel() * scales.element_size(),
                weight_dtype=str(weight.dtype),
            )
            if byte_pairs and state.shape[0] > 32:
                if not torch.equal(
                    ops.dequantize(weight, scales, state.code, state.shape),
                    ops.dequantize_pairs(
                        weight, scales, state.code, state.shape, lookup=True
                    ),
                ):
                    raise RuntimeError("NF4 byte-pair reconstruction is not exact")
                stats["pair_exact_checks"] += 1
        scales = weight._lf_nf4_scales[1]
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        dense = (
            ops.dequantize_pairs(weight, scales, state.code, state.shape, lookup=True)
            if byte_pairs and state.shape[0] > 32
            else ops.dequantize(weight, scales, state.code, state.shape)
        )
        y = functional.linear(flat, dense)
        y = y.reshape(*shape[:-1], state.shape[0])
        if bias is not None:
            y += bias
        stats["optimized_calls"] += 1
        event = (
            ("graph_capture" if torch.cuda.is_current_stream_capturing() else "eager")
            + "/"
            + str(flat.shape[0])
        )
        stats["executions"][event] = stats["executions"].get(event, 0) + 1
        return y

    BitsAndBytesLinearMethod._apply_4bit_weight = apply
    BitsAndBytesLinearMethod._lf_nf4_cached_scales = True

    def save():
        folder = os.environ.get("HYBRID_AUDIT_DIR")
        if folder:
            Path(folder, f"nf4-linear-audit-{os.getpid()}.json").write_text(
                json.dumps(stats, indent=2) + "\n"
            )

    from sglang.srt.model_executor.model_runner import ModelRunner

    original_forward = ModelRunner._forward_raw
    saved_captures = set()

    def forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        if not torch.cuda.is_current_stream_capturing():
            captures = {
                k for k in stats["executions"] if k.startswith("graph_capture/")
            }
            if captures - saved_captures:
                save()
                saved_captures.update(captures)
        return result

    ModelRunner._forward_raw = forward
    atexit.register(save)
