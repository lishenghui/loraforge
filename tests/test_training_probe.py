# SPDX-License-Identifier: Apache-2.0
"""Derivative and optimizer-state checks for the isolated training experiment."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "lora_training", Path(__file__).resolve().parents[1] / "benchmarks/lora_training.py"
)
probe = importlib.util.module_from_spec(spec)
# dataclasses resolves annotations through sys.modules.

sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


@pytest.mark.parametrize("backend", ["cached", "recompute"])
def test_training_probe_first_derivatives(backend):
    torch.manual_seed(671)
    base = probe.FrozenBase.dense(torch.randn(7, 5, dtype=torch.float64))
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    a = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, a, b: probe.linear(x, a, b, base, backend=backend, scale=0.7),
        (x, a, b),
    )


@pytest.mark.parametrize("backend", ["cached", "recompute"])
@pytest.mark.parametrize("input_grad", [True, False])
def test_training_probe_optimizer_steps_and_checkpoint(backend, input_grad):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(672)
    base = probe.FrozenBase.dense(torch.randn(7, 5, dtype=torch.float64))
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=input_grad)
    source = [
        torch.randn(2, 5, dtype=torch.float64),
        torch.zeros(7, 2, dtype=torch.float64),
    ]
    params = [[torch.nn.Parameter(p.clone()) for p in source] for _ in range(2)]
    optim = [torch.optim.AdamW(p, lr=0.01) for p in params]
    target = torch.randn(3, 7, dtype=torch.float64)
    for step in range(3):
        values = []
        for i, kind in enumerate(("bnb", backend)):
            optim[i].zero_grad(set_to_none=True)

            def fn(z, i=i, kind=kind):
                return probe.linear(z, *params[i], base, backend=kind, scale=0.7)

            out = checkpoint(fn, x, use_reentrant=False) if i else fn(x)
            loss = (out - target).square().mean()
            loss.backward()
            optim[i].step()
            values.append(loss.detach())
        torch.testing.assert_close(*values)
        for a, b in zip(*params, strict=True):
            torch.testing.assert_close(a, b)
