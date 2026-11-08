# SPDX-License-Identifier: Apache-2.0
"""Causal CE must shift within each sequence and preserve caller tensors."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "qwen35_training_loss",
    Path(__file__).resolve().parents[1] / "benchmarks/qwen35_training_loss.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("denominator", [None, 7])
def test_causal_rows_gradients_and_count(denominator):
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    torch.manual_seed(782)
    logits = torch.randn(3, 5, 11, requires_grad=True)
    labels = torch.randint(11, (3, 5))
    labels[1, 2] = -100
    originals = logits.detach().clone(), labels.clone()

    def fast_ce(values, targets, n_items):
        count = (targets != -100).sum() if n_items is None else n_items
        return (
            functional.cross_entropy(
                values.flatten(0, 1), targets.flatten(), reduction="sum"
            )
            / count
        )

    loss = probe.make_causal_loss(fast_ce)(
        logits,
        labels,
        11,
        num_items_in_batch=denominator,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    )
    expected = functional.cross_entropy(
        logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), reduction="sum"
    ) / ((labels[:, 1:] != -100).sum() if denominator is None else denominator)
    assert torch.equal(loss, expected)
    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    assert torch.equal(actual_grad, torch.autograd.grad(expected, logits)[0])
    assert torch.equal(actual_grad[:, -1], torch.zeros_like(actual_grad[:, -1]))
    assert torch.equal(logits, originals[0]) and torch.equal(labels, originals[1])


def test_external_kernel_source_is_pinned_before_execution(tmp_path):
    (tmp_path / "cross_entropy_loss.py").write_text("raise RuntimeError('bad')\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        probe.checked_sources(tmp_path)
