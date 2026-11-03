# SPDX-License-Identifier: Apache-2.0
"""Isolated HIP-A/native-B experiment; external compiled library is explicit.

This is not an installed serving policy. The library must implement the pinned
public probe shrink ABI. No compilation or global-environment changes occur.
"""

import importlib.util
import json
import os


def observe_graph_execution(output):
    """Write each observed phase once, before warmed timed requests."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    previous = ModelRunner._forward_raw
    seen = set()

    def forward(self, batch, *args, **kwargs):
        result = previous(self, batch, *args, **kwargs)
        params = getattr(batch.sampling_info, "custom_params", None) or []
        trace = any(p and "trace_case" in p for p in params)
        phase = batch.forward_mode.name
        graph = bool(result.can_run_graph)
        key = (phase, graph, trace)
        if key not in seen:
            seen.add(key)
            output.mkdir(parents=True, exist_ok=True)
            path = (
                output
                / f"execution-{os.getpid()}-{phase}-{int(graph)}-{int(trace)}.json"
            )
            path.write_text(json.dumps(dict(phase=phase, graph=graph, trace=trace)))
        return result

    ModelRunner._forward_raw = forward


def install(library, output):
    import torch

    from loraforge.integrations.sglang import get_backend

    spec = importlib.util.spec_from_file_location(library.stem, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = get_backend()
    previous = backend._shrink
    observed = set()

    def shrink(self, x, weights, slices=1):
        info = self.batch_info
        capacity = weights.shape[1] // slices
        supported = (
            torch.version.hip is not None
            and x.dtype == torch.bfloat16
            and weights.dtype == x.dtype
            and 1 <= slices <= 3
            and 1 <= capacity <= 64
            and x.shape[1] % 8 == 0
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
            and weights.data_ptr() % 16 == 0
            and weights.is_contiguous()
            and info.req_weight_indices.dtype == torch.int32
            and info.lora_ranks.dtype == torch.int32
        )
        capture = torch.cuda.is_current_stream_capturing()
        key = (supported, capture)
        if key not in observed:
            observed.add(key)
            output.mkdir(parents=True, exist_ok=True)
            (
                output / f"hip-{os.getpid()}-{int(supported)}-{int(capture)}.json"
            ).write_text(
                json.dumps(
                    dict(supported=supported, capture=capture, library=str(library))
                )
            )
        if not supported:
            return previous(self, x, weights, slices)
        result = torch.empty(
            (x.shape[0], weights.shape[1]), device=x.device, dtype=x.dtype
        )
        module.shrink(
            x,
            weights,
            result,
            info.req_weight_indices,
            info.lora_ranks,
            slices,
            capacity,
        )
        return result

    backend._shrink = shrink
