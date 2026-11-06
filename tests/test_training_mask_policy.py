# SPDX-License-Identifier: Apache-2.0
"""The implicit causal policy must reject inputs outside its fixed contract."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "qwen35_training_mask",
    Path(__file__).resolve().parents[1] / "benchmarks/qwen35_training_mask.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"batch_size": 0},
        {"q_length": 1},
        {"kv_length": 8},
        {"q_offset": 1},
        {"kv_offset": 1},
        {"attention_mask": object()},
        {"allow_is_causal_skip": False},
    ],
)
def test_fixed_causal_contract(change):
    value = dict(batch_size=1, q_length=4, kv_length=4)
    value.update(change)
    if change:
        with pytest.raises(ValueError, match="full causal sequence"):
            probe.validate_fixed_causal(**value)
    else:
        probe.validate_fixed_causal(**value)


def test_implicit_causal_sdpa_does_not_read_future():
    torch = pytest.importorskip("torch")
    torch.manual_seed(779)
    q, k, v = [torch.randn(1, 2, 4, 8, dtype=torch.float64) for _ in range(3)]
    sdpa = torch.nn.functional.scaled_dot_product_attention
    implicit = sdpa(q, k, v, is_causal=True)
    explicit = sdpa(q, k, v, attn_mask=torch.ones(4, 4, dtype=torch.bool).tril())
    torch.testing.assert_close(implicit, explicit, rtol=1e-12, atol=1e-12)
    v[:, :, -1] += 100
    changed = sdpa(q, k, v, is_causal=True)
    assert torch.equal(implicit[:, :, :-1], changed[:, :, :-1])


@pytest.mark.parametrize("batch", [1, 2, 8, 32])
def test_explicit_rope_axes_equal_native_default_without_packing_axis(batch):
    torch = pytest.importorskip("torch")
    ids = torch.arange(16).reshape(1, 16).expand(batch, -1)
    native = torch.arange(16).view(1, 1, -1).expand(4, batch, -1)[1:]
    explicit = probe.fixed_positions(ids)
    assert torch.equal(explicit, native)
    assert explicit.shape == native.shape and explicit.stride() == native.stride()


@pytest.mark.parametrize("shape", [(0, 16), (1, 1), (16,)])
def test_explicit_rope_rejects_other_layouts(shape):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="fixed full sequences"):
        probe.fixed_positions(torch.ones(shape, dtype=torch.long))
