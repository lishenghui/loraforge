# SPDX-License-Identifier: Apache-2.0
"""Real microbatch inputs and capacity reports must not hide failures."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distinct_microbatch_and_legacy_input():
    torch = pytest.importorskip("torch")
    module = load("benchmark_qwen35_training")
    seed = list(range(51))
    single = module.make_input_ids(seed, batch_size=1, tokens=128, device="cpu")
    assert torch.equal(single[0], torch.tensor((seed * 3)[:128]))
    batch = module.make_input_ids(seed, batch_size=8, tokens=128, device="cpu")
    assert torch.unique(batch, dim=0).shape[0] == 8
    assert torch.equal(batch[0], single[0])
    batch[-1].zero_()
    assert torch.equal(batch[0], single[0])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("torch.OutOfMemoryError: HIP out of memory", "oom"),
        ("RuntimeError: shape mismatch", "failed"),
        ("MemoryError: host mmap failed", "failed"),
    ],
)
def test_capacity_error_is_explicit(text, expected):
    assert load("run_qwen35_training_batches").failed_state(text) == expected
