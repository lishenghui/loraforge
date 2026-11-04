# SPDX-License-Identifier: Apache-2.0
"""Do not turn divergent, nonfinite, or mismatched model outputs into a pass."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "benchmarks" / "qwen35_hybrid.py"
spec = importlib.util.spec_from_file_location("qwen35_protocol", path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def record(tokens, *, adapter=0):
    return dict(serial=[dict(adapter=adapter, length=17, tokens=tokens)], mixed=[])


def test_model_gate_and_common_prefix():
    native = record([[-0.1, 1], [-0.2, 2], [-0.3, 3]])
    assert protocol.numerical_check(native, native)["numerical_pass"]
    drift = record([[-0.11, 1], [-0.2, 7], [-9.0, 3]])
    report = protocol.numerical_check(native, drift)
    assert not report["numerical_pass"]
    assert report["max_common_prefix_logprob_difference"] == pytest.approx(0.01)
    drift = record([[-0.11, 1], [-0.26, 2], [-0.3, 3]])
    assert not protocol.numerical_check(native, drift)["numerical_pass"]


def test_model_gate_rejects_nonfinite_and_changed_cases():
    native = record([[-0.1, 1], [-0.2, 2]])
    with pytest.raises(AssertionError, match="nonfinite"):
        protocol.numerical_check(native, record([[-0.1, 1], [float("nan"), 2]]))
    with pytest.raises(AssertionError, match="cases differ"):
        protocol.numerical_check(native, record([[-0.1, 1], [-0.2, 2]], adapter=1))


def test_base_payload_omits_adapter_and_preserves_work():
    manifest = dict(
        prompt=list(range(128)), adapters=[dict(name=f"a{i}") for i in range(4)]
    )
    native = protocol.payload(manifest, "native", 5)
    fleet = protocol.payload(manifest, "fleet", 5)
    base = protocol.payload(manifest, "base", 5)
    assert native == fleet
    assert native["lora_path"] == "a1"
    assert "lora_path" not in base
    assert base == {k: v for k, v in native.items() if k != "lora_path"}
    assert len(base["input_ids"]) == 128
    assert base["sampling_params"]["max_new_tokens"] == 64


def test_model_gate_requires_matching_large_batch_validation():
    native = record([[-0.1, 1]])
    candidate = dict(native, mixed32=[[[-0.1, 1]]] * 32)
    with pytest.raises(AssertionError, match="cases differ"):
        protocol.numerical_check(native, candidate)
    native = dict(candidate)
    candidate = dict(candidate, mixed32=[[[-0.1, 2]]] * 32)
    assert not protocol.numerical_check(native, candidate)["numerical_pass"]
