# SPDX-License-Identifier: Apache-2.0
"""Standalone GDN initialization, adapter preservation and prefill fallback."""

import sys
from types import ModuleType
from types import SimpleNamespace as N

import pytest

torch = pytest.importorskip("torch")


def modules(monkeypatch):
    for path in (
        "sglang.srt.models.qwen3_5",
        "sglang.srt.lora.backend.chunked_backend",
    ):
        for length in range(1, len(path.split(".")) + 1):
            name = ".".join(path.split(".")[:length])
            if name not in sys.modules:
                module = ModuleType(name)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, name, module)
                parent, _, child = name.rpartition(".")
                if parent:
                    monkeypatch.setattr(
                        sys.modules[parent], child, module, raising=False
                    )
    return sys.modules["sglang.srt.models.qwen3_5"], sys.modules[
        "sglang.srt.lora.backend.chunked_backend"
    ]


def test_installed_gdn_prepares_once_and_preserves_active_adapter_updates(monkeypatch):
    from loraforge.integrations._sglang_gdn import install

    model, backend_module = modules(monkeypatch)

    class Backend:
        def prepare_lora_batch(self, batch):
            return 42

    backend_module.ChunkedSgmvLoRABackend = Backend
    backend = Backend()
    q = torch.nn.Parameter(torch.ones(8192, 1024, dtype=torch.bfloat16), False)
    b = torch.nn.Parameter(torch.full((32, 1024), 2, dtype=torch.bfloat16), False)

    class Layer:
        attn_tp_size = 1
        _fused_in_proj_weight = None

        def __init__(self):
            def projection(weight):
                return N(
                    base_layer=N(weight=weight, bias=None),
                    lora_backend=backend,
                    lora_active=True,
                    apply_lora=lambda base, x: base + 1,
                )

            self.in_proj_qkvz = projection(q)
            self.in_proj_ba = projection(b)

        def _forward_input_proj(self, x):
            return "native-q", "native-b"

    Layer._forward_input_proj.__module__ = "sglang.srt.models.qwen3_5"
    model.Qwen3_5GatedDeltaNet = Layer
    model.bf16_gemm_dispatch = torch.nn.functional.linear
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    install()
    install()
    layer = Layer()
    x = torch.full((1, 1024), 1 / 1024, dtype=torch.bfloat16)
    batch = N(forward_mode=N(is_decode=lambda: False))
    assert backend.prepare_lora_batch(batch) == 42
    assert layer._forward_input_proj(x) == ("native-q", "native-b")
    owner = layer._lf_packed
    assert q.data_ptr() == owner.data_ptr()
    assert b.data_ptr() == owner[8192:].data_ptr()
    batch.forward_mode.is_decode = lambda: True
    backend.prepare_lora_batch(batch)
    qout, bout = layer._forward_input_proj(x)
    assert torch.equal(qout, torch.full_like(qout, 2))
    assert torch.equal(bout, torch.full_like(bout, 3))
    layer.in_proj_ba.lora_active = False
    qout, bout = layer._forward_input_proj(x)
    assert torch.equal(bout, torch.full_like(bout, 2))
    assert layer._lf_packed is owner
    assert Layer._loraforge_gdn_stats == dict(
        prepared=1, fused_calls=2, fallback_calls=1
    )


def test_gdn_refuses_a_conflicting_projection_patch(monkeypatch):
    from loraforge.integrations._sglang_gdn import install

    model, backend_module = modules(monkeypatch)
    backend_module.ChunkedSgmvLoRABackend = type("Backend", (), {})

    class Layer:
        def _forward_input_proj(self, x):
            return x

    model.Qwen3_5GatedDeltaNet = Layer
    original = Layer._forward_input_proj
    with pytest.raises(RuntimeError, match="another patch"):
        install()
    assert Layer._forward_input_proj is original
