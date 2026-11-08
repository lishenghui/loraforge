# SPDX-License-Identifier: Apache-2.0
"""Actual PEFT ownership and checkpointed updates through the model MLP wrapper."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
for name in ("lora_swiglu_training", "lora_native_training", "qwen35_training_mlp"):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
probe = sys.modules["qwen35_training_mlp"]


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        self.gate_proj = torch.nn.Linear(4, 7, bias=False)
        self.up_proj = torch.nn.Linear(4, 7, bias=False)
        self.down_proj = torch.nn.Linear(7, 4, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, x):
        return self.mlp(x)


def model():
    torch.manual_seed(771)
    return peft.get_peft_model(
        Model(),
        peft.LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        ),
    ).double()


@pytest.mark.parametrize(
    "backend",
    [
        "native",
        "fused",
        "compact",
        "native-activation",
        "native-consumer",
        "native-expand",
    ],
)
def test_wrapper_preserves_parameters_and_checkpoint_updates(backend):
    from torch.utils.checkpoint import checkpoint

    reference, candidate = model(), model()
    before = probe.adapter_parameters(candidate)
    ids = {k: id(p) for k, p in before.items()}
    assert probe.replace_mlps(candidate, backend) == 1
    after = probe.adapter_parameters(candidate)
    assert {k: id(p) for k, p in after.items()} == ids
    assert set(after) == set(probe.adapter_parameters(reference))
    assert len(after) == 6
    optim = [
        torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.001)
        for m in (reference, candidate)
    ]
    torch.manual_seed(772)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    target = torch.randn_like(x)
    for _ in range(3):
        losses = []
        for index, m in enumerate((reference, candidate)):
            optim[index].zero_grad(set_to_none=True)
            y = checkpoint(m, x, use_reentrant=False) if index else m(x)
            loss = (y - target).square().mean()
            loss.backward()
            losses.append(loss.detach())
            optim[index].step()
        torch.testing.assert_close(*losses)
        for name, p in probe.adapter_parameters(reference).items():
            torch.testing.assert_close(p, after[name])


def test_wrapper_rejects_dropout():
    m = model()
    m.base_model.model.mlp.gate_proj.lora_dropout["default"] = torch.nn.Dropout(0.1)
    with pytest.raises(ValueError, match="dropout"):
        probe.replace_mlps(m, "fused")


def test_separate_consumer_gradcheck():
    torch.manual_seed(774)
    values = [
        torch.randn(*s, dtype=torch.float64, requires_grad=True)
        for s in [(2, 3, 7), (2, 3, 7), (2, 3, 4), (2, 3, 4), (7, 4), (7, 4)]
    ]
    fn = sys.modules["lora_native_training"]._GateUp.apply
    assert torch.autograd.gradcheck(lambda *v: fn(*v, 0.7), tuple(values))


def test_expand_add_gradcheck():
    values = [
        torch.randn(*s, dtype=torch.float64, requires_grad=True)
        for s in [(2, 3, 7), (2, 3, 4), (7, 4)]
    ]
    fn = sys.modules["lora_native_training"]._ExpandAdd.apply
    assert torch.autograd.gradcheck(lambda *v: fn(*v, 0.7), tuple(values))
