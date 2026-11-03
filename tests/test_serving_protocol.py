# SPDX-License-Identifier: Apache-2.0
"""Validation protocol must reject missing execution evidence and output drift."""

import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "benchmarks" / "validate_sglang_serving.py"
spec = importlib.util.spec_from_file_location("serving_validation", path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_output_comparison():
    left = {"tokens": [1, 2], "logprobs": [-0.1, -0.2]}
    close = {"tokens": [1, 2], "logprobs": [-0.11, -0.22]}
    assert protocol.difference(left, close, 0.05) == pytest.approx(0.02)
    with pytest.raises(AssertionError, match="token IDs"):
        protocol.difference(left, {**left, "tokens": [1, 3]}, 0.05)
    with pytest.raises(AssertionError, match="logprob"):
        protocol.difference(left, {**left, "logprobs": [-0.1, -0.3]}, 0.05)


def test_completion_metadata():
    response = {
        "text": "hello",
        "meta_info": {
            "completion_tokens": 1,
            "output_token_logprobs": [[-0.1, 42, "hello"]],
        },
    }
    assert protocol.normalize(response, 1)["tokens"] == [42]
    with pytest.raises(AssertionError, match="incomplete"):
        protocol.normalize(response, 2)


def report(tmp_path, counts, loads):
    (tmp_path / "audit-1.json").write_text(
        json.dumps({"counts": counts, "loads": loads})
    )


def test_observer_evidence(tmp_path):
    loads = [
        {"uid": "a", "slot": 1, "rank": 32, "addresses_stable": True},
        {"uid": "b", "slot": 1, "rank": 16, "addresses_stable": True},
    ]
    report(tmp_path, {"DECODE/eager": 3}, loads)
    assert protocol.audit_summary(tmp_path, "native-eager")["rank_changes"] == 1
    with pytest.raises(AssertionError, match="did not execute"):
        protocol.audit_summary(tmp_path, "loraforge-eager")
    with pytest.raises(AssertionError, match="graph replay"):
        protocol.audit_summary(tmp_path, "native-graph")
    report(tmp_path, {}, loads)
    with pytest.raises(AssertionError, match="eager decode"):
        protocol.audit_summary(tmp_path, "native-eager")
    report(tmp_path, {"DECODE/graph": 3, "loraforge_shrink_capture": 2}, loads)
    assert protocol.audit_summary(tmp_path, "loraforge-graph")["slot_reuses"] == 1


def test_no_rank_changing_reuse_is_rejected(tmp_path):
    report(
        tmp_path,
        {"DECODE/eager": 3},
        [
            {"uid": "a", "slot": 1, "rank": 32, "addresses_stable": True},
            {"uid": "b", "slot": 1, "rank": 32, "addresses_stable": True},
        ],
    )
    with pytest.raises(AssertionError, match="different ranks"):
        protocol.audit_summary(tmp_path, "native-eager")
