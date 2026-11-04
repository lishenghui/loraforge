# SPDX-License-Identifier: Apache-2.0
"""Registration contracts without a serving-framework dependency."""

import sys
from types import ModuleType

import pytest

from loraforge.integrations import sglang


def test_disabled_plugin_is_inert(monkeypatch):
    monkeypatch.delenv("LORAFORGE_SGLANG", raising=False)
    monkeypatch.delenv("LORAFORGE_SGLANG_QWEN35", raising=False)
    monkeypatch.setattr(
        sglang, "install_sglang_backend", lambda: pytest.fail("unexpected activation")
    )
    sglang.install()


def test_unsupported_version(monkeypatch):
    monkeypatch.setattr(sglang, "version", lambda name: "0.5.13")
    with pytest.raises(RuntimeError, match="0.5.22"):
        sglang.install_sglang_backend()


def test_registration_idempotence_and_conflict(monkeypatch):
    path = "sglang.srt.lora.backend.lora_registry"
    for length in range(1, 6):
        name = ".".join(path.split(".")[:length])
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    registry = ModuleType(path)

    def native():
        pass

    native.__name__ = "create_triton_csgmv_backend"
    native.__module__ = path
    registry.LORA_SUPPORTED_BACKENDS = {"csgmv": native}
    monkeypatch.setitem(sys.modules, path, registry)
    monkeypatch.setattr(sglang, "version", lambda name: "0.5.22")
    sglang.install_sglang_backend()
    assert registry.LORA_SUPPORTED_BACKENDS["csgmv"] is sglang.get_backend
    sglang.install_sglang_backend()
    registry.LORA_SUPPORTED_BACKENDS["csgmv"] = lambda: None
    with pytest.raises(RuntimeError, match="another plugin"):
        sglang.install_sglang_backend()


def test_enabled_entrypoint(monkeypatch):
    calls = []
    monkeypatch.delenv("LORAFORGE_SGLANG_QWEN35", raising=False)
    monkeypatch.setenv("LORAFORGE_SGLANG", "1")
    monkeypatch.setattr(sglang, "install_sglang_backend", lambda: calls.append(True))
    sglang.install()
    assert calls == [True]


def test_hybrid_bridge_can_keep_native_backend(monkeypatch):
    from loraforge.integrations import _sglang_qwen35

    calls = []
    monkeypatch.setenv("LORAFORGE_SGLANG_QWEN35", "1")
    monkeypatch.setenv("LORAFORGE_SGLANG", "0")
    monkeypatch.setattr(
        _sglang_qwen35, "install_qwen35_lora", lambda: calls.append("bridge")
    )
    monkeypatch.setattr(sglang, "install_sglang_backend", lambda: calls.append("fleet"))
    sglang.install()
    assert calls == ["bridge"]


@pytest.mark.parametrize("policy", ["compatible", "rank-aware", "fleet"])
def test_decode_policy_selection(monkeypatch, policy):
    monkeypatch.setenv("LORAFORGE_SGLANG_POLICY", policy)
    assert sglang._decode_policy() == policy


def test_unknown_decode_policy_rejected(monkeypatch):
    monkeypatch.setenv("LORAFORGE_SGLANG_POLICY", "unknown")
    with pytest.raises(ValueError, match="POLICY"):
        sglang._decode_policy()
