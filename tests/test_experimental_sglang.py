# SPDX-License-Identifier: Apache-2.0
"""Explicit worker opt-ins and rejected configurations without a GPU runtime."""

import pytest

from loraforge.integrations import sglang
from loraforge.integrations import sglang_experimental as experimental


@pytest.fixture(autouse=True)
def clear_environment(monkeypatch):
    for key in (
        "LORAFORGE_EXPERIMENTAL_QWEN35_NF4",
        "LORAFORGE_EXPERIMENTAL_QWEN35_GDN",
        "LORAFORGE_SGLANG",
        "LORAFORGE_SGLANG_QWEN35",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("mode", ["stock", "cached-scales", "byte-pairs"])
def test_nf4_worker_entrypoint_dispatches_explicit_mode(monkeypatch, mode):
    calls = []
    monkeypatch.setenv("LORAFORGE_EXPERIMENTAL_QWEN35_NF4", mode)
    monkeypatch.setattr(
        experimental, "install_qwen35_nf4", lambda **kw: calls.append(kw)
    )
    sglang.install()
    assert calls == [{"mode": mode}]


@pytest.mark.parametrize("mode", ["1", "packed"])
def test_gdn_worker_entrypoint_dispatches_without_nf4(monkeypatch, mode):
    calls = []
    monkeypatch.setenv("LORAFORGE_EXPERIMENTAL_QWEN35_GDN", mode)
    monkeypatch.setattr(
        experimental, "install_qwen35_gdn", lambda **kw: calls.append(kw)
    )
    sglang.install()
    assert calls == [{"packed_checkpoint": mode == "packed"}]


@pytest.mark.parametrize("gdn,nf4", [("unexpected", ""), ("1", "byte-pairs")])
def test_invalid_worker_options_fail_before_installation(monkeypatch, gdn, nf4):
    monkeypatch.setenv("LORAFORGE_EXPERIMENTAL_QWEN35_GDN", gdn)
    monkeypatch.setenv("LORAFORGE_EXPERIMENTAL_QWEN35_NF4", nf4)
    monkeypatch.setattr(
        experimental, "install_qwen35_gdn", lambda: pytest.fail("unexpected mutation")
    )
    monkeypatch.setattr(
        experimental,
        "install_qwen35_nf4",
        lambda **kw: pytest.fail("unexpected mutation"),
    )
    with pytest.raises(ValueError):
        sglang.install()


@pytest.mark.filterwarnings("ignore::loraforge.annotations.ExperimentalAPIWarning")
def test_public_installation_rejects_mode_changes_and_incompatible_version(monkeypatch):
    with pytest.raises(ValueError, match="mode"):
        experimental.install_qwen35_nf4(mode="unknown")
    monkeypatch.setattr(experimental, "_NF4_MODE", "cached-scales")
    with pytest.raises(RuntimeError, match="different options"):
        experimental.install_qwen35_nf4(mode="byte-pairs")
    monkeypatch.setattr(experimental, "_NF4_MODE", None)
    monkeypatch.setattr(experimental, "version", lambda name: "unsupported")
    with pytest.raises(RuntimeError, match="0.5.22"):
        experimental.install_qwen35_nf4()


@pytest.mark.filterwarnings("ignore::loraforge.annotations.ExperimentalAPIWarning")
def test_gdn_cannot_replace_an_active_nf4_installation(monkeypatch):
    monkeypatch.setattr(experimental, "_NF4_MODE", "stock")
    with pytest.raises(RuntimeError, match="combined"):
        experimental.install_qwen35_gdn()
