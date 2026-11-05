# SPDX-License-Identifier: Apache-2.0
"""Graph wrapper preserves parameter ownership and instrumentation lifecycle."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "qwen35_training_graph",
    Path(__file__).resolve().parents[1] / "benchmarks/qwen35_training_graph.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_loss_wrapper_keeps_native_parameters_and_detaches_audit():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 5, dtype=torch.float64))

        def forward(self, input_ids, labels, use_cache):
            assert labels is input_ids and use_cache is False
            logits = self.weight[input_ids]
            return SimpleNamespace(loss=logits.square().mean(), logits=logits)

    model = Model()
    wrapper = probe.LossAndSamples(model)
    assert next(wrapper.parameters()) is model.weight
    ids = torch.arange(6).reshape(1, 6)
    loss, audit = wrapper(ids)
    ref = model(ids, ids, False)
    assert torch.equal(audit, ref.logits.detach()[0, [0, 3, 4], ::257])
    assert not audit.requires_grad
    expected = torch.autograd.grad(ref.loss, model.weight)[0]
    loss.backward()
    assert torch.equal(model.weight.grad, expected)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_graph_factory_restored_but_live_graph_still_counts(monkeypatch, raise_inside):
    class Graph:
        def capture_end(self):
            return "captured"

        def replay(self):
            return "replayed"

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    try:
        with probe.graph_counters() as counts:
            graph = torch.cuda.CUDAGraph()
            assert graph.capture_end() == "captured"
            if raise_inside:
                raise RuntimeError("test restoration")
    except RuntimeError:
        assert raise_inside
    assert torch.cuda.CUDAGraph is Graph
    assert graph.replay() == "replayed"
    assert counts == dict(captures=1, replays=1)


def test_cpu_capture_is_rejected():
    with pytest.raises(ValueError, match="CUDA/ROCm"):
        probe.capture(torch.nn.Linear(2, 3), torch.ones(1, 2, dtype=torch.long))
