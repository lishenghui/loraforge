# SPDX-License-Identifier: Apache-2.0
"""Native-autograd graph experiment; fixed shape, live original parameters.

Only the standalone GPU benchmark opts in. Capture/setup are outside timing;
each invocation still participates in autograd and a real external optimizer
update. This calls Torch's implementation and copies no graph implementation.
"""

from contextlib import contextmanager

import torch
from torch import nn


def sample_logits(logits):
    tokens = logits.shape[1]
    # Slices + stack avoid constructing a host index Tensor during capture.
    samples = torch.stack(
        [logits[:, p, ::257] for p in (0, tokens // 2, tokens - 2)], dim=1
    )
    return samples[0] if logits.shape[0] == 1 else samples


class LossAndSamples(nn.Module):
    """Expose only loss and detached audit logits to the graph boundary."""

    def __init__(self, model, implicit_causal=False):
        super().__init__()
        self.model = model
        self.implicit_causal = implicit_causal

    def forward(self, ids):
        kwargs = {}
        if self.implicit_causal:
            from qwen35_training_mask import fixed_positions

            kwargs["position_ids"] = fixed_positions(ids)
        output = self.model(input_ids=ids, labels=ids, use_cache=False, **kwargs)
        return (
            output.loss,
            sample_logits(output.logits.detach()),
        )


@contextmanager
def graph_counters():
    """Instrument created graphs; their replay counts survive factory restore."""
    original = torch.cuda.CUDAGraph
    counts = dict(captures=0, replays=0)

    class CountedGraph(original):
        def capture_end(self):
            result = super().capture_end()
            counts["captures"] += 1
            return result

        def replay(self):
            result = super().replay()
            counts["replays"] += 1
            return result

    torch.cuda.CUDAGraph = CountedGraph
    try:
        yield counts
    finally:
        torch.cuda.CUDAGraph = original


def capture(model, ids, *, implicit_causal=False):
    if not ids.is_cuda or not model.training:
        raise ValueError("capture requires CUDA/ROCm tensors and training mode")
    wrapper = LossAndSamples(model, implicit_causal=implicit_causal)
    sample_ids = ids.detach().clone()
    # AMP's weight cache must be disabled: graph execution must recast the live
    # FP32 LoRA masters on every forward, including after AdamW updates/reset.
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        return torch.cuda.make_graphed_callables(
            wrapper, (sample_ids,), num_warmup_iters=3, allow_unused_input=True
        )
