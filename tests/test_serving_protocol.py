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


def test_matched_batch_comparison_keeps_threshold():
    single = {"tokens": [1], "logprobs": [-0.1]}
    mixed = {"tokens": [1], "logprobs": [-0.1573]}
    assert protocol.observed_difference(single, mixed)["max_logprob_delta"] > 0.05
    baseline = {
        "passed": True,
        "manifest_sha256": "same",
        "protocol": {"version": 2, "new_tokens": 1, "tolerance": 0.05},
        "rows": {"case": single},
        "mixed": [mixed, mixed],
    }
    candidate = {
        **baseline,
        "rows": {"case": dict(single)},
        "mixed": [dict(mixed), dict(mixed)],
    }
    assert protocol.compare_reports(
        {"native-eager": baseline, "loraforge-eager": candidate}, 0.05
    )["loraforge-eager"]["mixed"] == [0, 0]
    candidate["mixed"][0] = {"tokens": [1], "logprobs": [-0.25]}
    with pytest.raises(AssertionError, match="logprob"):
        protocol.compare_reports(
            {"native-eager": baseline, "loraforge-eager": candidate}, 0.05
        )


def test_stale_protocol_and_nonfinite_rejected():
    row = {"tokens": [1], "logprobs": [-0.1]}
    with pytest.raises(AssertionError, match="nonfinite"):
        protocol.difference(row, {"tokens": [1], "logprobs": [float("nan")]}, 0.05)
    stale = {
        "passed": True,
        "protocol": {"new_tokens": 1, "tolerance": 0.05},
        "rows": {"case": row},
        "mixed": [row, row],
    }
    with pytest.raises(AssertionError, match="version"):
        protocol.compare_reports({"native-eager": stale}, 0.05)
