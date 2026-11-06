# SPDX-License-Identifier: Apache-2.0
"""Frozen-head chunks preserve gradients, caller storage and loss ownership."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "qwen35_training_head",
    Path(__file__).resolve().parents[1] / "benchmarks/qwen35_training_head.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("strategy", ["chunked", "full-reuse"])
@pytest.mark.parametrize("chunk_rows", [1, 4, 32])
def test_loss_derivatives_and_retained_backward(strategy, chunk_rows):
    torch.manual_seed(788)
    hidden = torch.randn(3, 7, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 5, dtype=torch.float64)
    labels = torch.randint(11, (3, 7))
    labels[1, 2] = -100
    original = [t.detach().clone() for t in (hidden, weight, labels)]
    logits = torch.nn.functional.linear(hidden, weight)
    reference = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1)
    )
    expected = torch.autograd.grad(reference, hidden)[0]
    loss, audit, full = probe.FrozenHeadLoss.apply(
        hidden, weight, labels, chunk_rows, strategy, True
    )
    assert hidden.grad is None and weight.grad is None
    torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
    assert not audit.requires_grad and not full.requires_grad
    torch.testing.assert_close(
        full, logits.detach()[:, [0, 3, 5]], rtol=1e-12, atol=1e-12
    )
    assert torch.equal(audit, full[..., ::257])
    for scale in (1.0, 0.3, 2.0):
        gradient = torch.autograd.grad(loss * scale, hidden, retain_graph=True)[0]
        torch.testing.assert_close(gradient, expected * scale, rtol=1e-12, atol=1e-12)
    assert torch.equal(gradient[:, -1], torch.zeros_like(gradient[:, -1]))
    assert all(
        torch.equal(t, old)
        for t, old in zip((hidden, weight, labels), original, strict=True)
    )


def test_trainable_head_is_rejected():
    with pytest.raises(ValueError, match="frozen head"):
        probe.FrozenHeadLoss.apply(
            torch.randn(1, 3, 4, requires_grad=True),
            torch.randn(7, 4, requires_grad=True),
            torch.zeros(1, 3, dtype=torch.long),
            2,
            "chunked",
            False,
        )


def test_model_wrapper_owns_original_parameters_and_has_no_early_gradients():
    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Parameter(torch.randn(13, 5, dtype=torch.float64))

        def forward(self, input_ids, use_cache):
            assert use_cache is False
            return SimpleNamespace(last_hidden_state=self.embedding[input_ids])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = torch.nn.Linear(5, 13, bias=False, dtype=torch.float64)
            self.lm_head.requires_grad_(False)

        def get_base_model(self):
            return self

    torch.manual_seed(789)
    model = Model()
    wrapped = probe.HeadLossAndSamples(model, chunk_rows=4, strategy="chunked")
    assert {id(p) for p in wrapped.parameters()} == {id(p) for p in model.parameters()}
    ids = torch.arange(12).reshape(2, 6)
    loss, audit = wrapped(ids)
    assert model.model.embedding.grad is None and not audit.requires_grad
    logits = model.lm_head(model.model.embedding[ids])
    reference = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 13), ids[:, 1:].reshape(-1)
    )
    expected = torch.autograd.grad(reference, model.model.embedding)[0]
    loss.backward()
    torch.testing.assert_close(
        model.model.embedding.grad, expected, rtol=1e-12, atol=1e-12
    )
