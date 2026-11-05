# SPDX-License-Identifier: Apache-2.0
"""Parameter identity, exact values, alias ownership, and stale-cache rejection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as N

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "qwen35_gdn_storage",
    Path(__file__).resolve().parents[1] / "benchmarks/qwen35_gdn_storage.py",
)
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def fixture():
    q = torch.nn.Parameter(torch.ones(8192, 1024, dtype=torch.bfloat16), False)
    b = torch.nn.Parameter(torch.full((32, 1024), 2, dtype=torch.bfloat16), False)
    q.weight_loader = object()
    return (
        N(
            in_proj_qkvz=N(base_layer=N(weight=q)),
            in_proj_ba=N(base_layer=N(weight=b)),
            _lf_packed=torch.cat((q, b)),
        ),
        q,
        b,
    )


def test_storage_consolidation_preserves_parameter_identity_and_values():
    layer, q, b = fixture()
    metadata = q.weight_loader
    expected_q, expected_b = q.clone(), b.clone()
    assert storage.consolidate(layer)
    assert layer.in_proj_qkvz.base_layer.weight is q
    assert layer.in_proj_ba.base_layer.weight is b
    assert q.weight_loader is metadata
    assert torch.equal(q, expected_q) and torch.equal(b, expected_b)
    assert q.is_contiguous() and b.is_contiguous()
    packed = layer._lf_packed
    assert q.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert b.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert b.storage_offset() == 8192 * 1024
    assert storage.consolidate(layer) is False


def test_stale_packed_storage_is_rejected_before_rebinding():
    layer, q, b = fixture()
    old_q, old_b = q.data_ptr(), b.data_ptr()
    layer._lf_packed[0, 0] = 3
    with pytest.raises(ValueError, match="differs"):
        storage.consolidate(layer)
    assert (q.data_ptr(), b.data_ptr()) == (old_q, old_b)


def test_trainable_weights_are_rejected():
    layer, q, _ = fixture()
    q.requires_grad_(True)
    with pytest.raises(ValueError, match="immutable"):
        storage.consolidate(layer)


def test_no_packed_weight_keeps_native_layer():
    assert storage.consolidate(N()) is None


def test_closure_based_lazy_finalizer_cannot_bypass_storage_consolidation():
    layer, q, b = fixture()
    del layer._lf_packed
    calls = []

    def lazy_finalize(self):
        calls.append("finalize")
        self._lf_packed = torch.cat((q.detach(), b.detach()))

    class Layer:
        finalize_fused_in_proj = lazy_finalize

        def __init__(self):
            self.in_proj_qkvz = layer.in_proj_qkvz
            self.in_proj_ba = layer.in_proj_ba

        def _forward_input_proj(self):
            if not hasattr(self, "_lf_packed"):
                lazy_finalize(self)
            return self._lf_packed

    storage.wrap_class(Layer)
    obj = Layer()
    packed = obj._forward_input_proj()
    assert q.data_ptr() == packed.data_ptr()
    assert b.data_ptr() == packed[8192:].data_ptr()
    assert obj._forward_input_proj() is packed
    assert calls == ["finalize"]
