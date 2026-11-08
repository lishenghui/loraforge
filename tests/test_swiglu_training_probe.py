# SPDX-License-Identifier: Apache-2.0
"""Whole LoRA MLP derivatives and checkpointed optimizer updates."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "lora_swiglu_training",
    Path(__file__).resolve().parents[1] / "benchmarks/lora_swiglu_training.py",
)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


def fixture():
    torch.manual_seed(771)
    x = torch.randn(2, 2, 3, dtype=torch.float64) * 0.1
    params = [
        torch.randn(*s, dtype=torch.float64) * 0.1
        for s in [(2, 3), (5, 2), (2, 3), (5, 2), (2, 5), (3, 2)]
    ]
    base = [
        torch.randn(10, 3, dtype=torch.float64) * 0.1,
        torch.randn(3, 5, dtype=torch.float64) * 0.1,
    ]
    return x, params, base


@pytest.mark.parametrize("backend", ["activation", "block", "fused", "compact"])
def test_swiglu_mlp_gradcheck(backend):
    x, params, base = fixture()
    values = [v.requires_grad_(True) for v in (x, *params)]
    assert torch.autograd.gradcheck(
        lambda *v: probe.mlp(*v, *base, backend=backend), tuple(values)
    )


@pytest.mark.parametrize("backend", ["activation", "block", "fused", "compact"])
@pytest.mark.parametrize("input_grad", [False, True])
def test_swiglu_mlp_checkpoint_and_optimizer(backend, input_grad):
    from torch.utils.checkpoint import checkpoint

    x, source, base = fixture()
    for i in (1, 3, 5):
        source[i].zero_()
    params = [[torch.nn.Parameter(v.clone()) for v in source] for _ in range(2)]
    optim = [torch.optim.AdamW(p, lr=0.001) for p in params]
    target = torch.randn_like(x)
    for step in range(3):
        losses = []
        for i, kind in enumerate(("native", backend)):
            optim[i].zero_grad(set_to_none=True)
            z = x.detach().clone().requires_grad_(input_grad)

            def fn(z, i=i, kind=kind):
                return probe.mlp(z, *params[i], *base, backend=kind)

            y = checkpoint(fn, z, use_reentrant=False) if i else fn(z)
            loss = (y - target).square().mean()
            loss.backward()
            optim[i].step()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses)
        for a, b in zip(*params, strict=True):
            torch.testing.assert_close(a, b)
