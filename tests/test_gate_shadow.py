# SPDX-License-Identifier: Apache-2.0
"""Packing ownership, rank changes and clearing are independent of the HIP ABI."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
path = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "sglang_gate_expand_experiment.py"
)
spec = importlib.util.spec_from_file_location("gate_shadow", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rank_changes_clear_padding_and_keep_addresses():
    weights = torch.arange(2 * 10 * 64).reshape(2, 10, 64).to(torch.bfloat16)
    original = weights.clone()
    entry = module.GateShadow(weights)
    addresses = entry.addresses()
    for slot, rank in ((0, 64), (1, 32), (0, 17), (0, 0), (0, 16)):
        entry.refresh(slot, rank)
        bucket = int(entry.bucket[slot])
        region = entry.store[slot * 640 : (slot + 1) * 640]
        if rank:
            view = region[: 10 * bucket].view(10, bucket)
            torch.testing.assert_close(view[:, :rank], weights[slot, :, :rank])
            assert torch.count_nonzero(view[:, rank:]) == 0
            assert torch.count_nonzero(region[10 * bucket :]) == 0
        else:
            assert torch.count_nonzero(region) == 0
        assert entry.addresses() == addresses
    torch.testing.assert_close(weights, original)
    saved_other = entry.store[640:].clone()
    weights[0].fill_(7)
    entry.refresh(0, 32)
    torch.testing.assert_close(entry.store[640:], saved_other)
    assert entry.store.data_ptr() != weights.data_ptr()


def test_invalid_capacity_or_rank_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        module.GateShadow(torch.zeros(2, 10, 48, dtype=torch.bfloat16))
    entry = module.GateShadow(torch.zeros(2, 10, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="invalid"):
        entry.refresh(0, 65)
