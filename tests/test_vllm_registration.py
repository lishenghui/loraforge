# SPDX-License-Identifier: Apache-2.0
"""CPU registration contracts; real framework execution is tested on GPU."""

import subprocess
import sys
from types import ModuleType

import pytest

from loraforge.integrations import vllm as integration


def test_disabled_plugin_is_dependency_free(monkeypatch):
    monkeypatch.delenv("LORAFORGE_VLLM", raising=False)
    code = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'vllm'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
from loraforge.integrations.vllm import register
from loraforge.ops.vllm_gemv import shrink, expand
register()
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_unsupported_version(monkeypatch):
    monkeypatch.setattr(integration, "version", lambda _: "0.25.0")
    with pytest.raises(RuntimeError, match="exactly"):
        integration.install_vllm_backend()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_threshold(value):
    with pytest.raises(ValueError, match="positive integer"):
        integration.install_vllm_backend(tokens_per_lora=value)


def test_opt_in(monkeypatch):
    calls = []
    monkeypatch.setenv("LORAFORGE_VLLM", "1")
    monkeypatch.setattr(integration, "install_vllm_backend", lambda: calls.append(1))
    integration.register()
    assert calls == [1]


def test_registration_conflicts(monkeypatch):
    name = "vllm.lora.punica_wrapper.punica_gpu"
    module = ModuleType(name)

    class Wrapper:
        def add_lora_linear(self):
            pass

    Wrapper.add_lora_linear.__module__ = name
    module.PunicaWrapperGPU = Wrapper
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(integration, "version", lambda _: "0.26.0")
    monkeypatch.setattr(integration, "_installed", None)
    monkeypatch.setattr(integration, "_operator", lambda: None)
    integration.install_vllm_backend()
    first = Wrapper.add_lora_linear
    integration.install_vllm_backend()
    assert Wrapper.add_lora_linear is first
    Wrapper.add_lora_linear = lambda self: None
    with pytest.raises(RuntimeError, match="conflict"):
        integration.install_vllm_backend()
    monkeypatch.setattr(integration, "_installed", None)
    with pytest.raises(RuntimeError, match="foreign"):
        integration.install_vllm_backend()
