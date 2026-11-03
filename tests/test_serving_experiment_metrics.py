# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "serving_experiment_metrics.py"
)
spec = importlib.util.spec_from_file_location("experiment_metrics", path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_exact_answer_format_does_not_score_reasoning_digits():
    assert metrics.answer_digit("There are 2 options left.\nAnswer: 3") == "3"
    assert metrics.answer_digit("The answer is probably 2") is None
    assert metrics.answer_digit("Answer: 20") is None
    assert metrics.answer_digit("Answer: 2 or 3") is None


def test_all_ties_in_small_pilot_do_not_prove_quality_equivalence():
    cases = {str(i): dict(answer="0", prediction="0") for i in range(32)}
    result = metrics.paired_quality(cases, cases)
    lo, hi = result["conservative_95pct_degradation_interval"]
    assert result["degradation"] == 0
    assert lo < -0.02 < 0.02 < hi


def test_pairing_and_degradation_direction():
    native = {
        "a": dict(answer="1", prediction="1"),
        "b": dict(answer="2", prediction=None),
    }
    fleet = {
        "a": dict(answer="1", prediction="0"),
        "b": dict(answer="2", prediction=None),
    }
    result = metrics.paired_quality(native, fleet)
    assert result["lost"] == 1 and result["gained"] == 0
    assert result["degradation"] == 0.5
    assert result["fleet_invalid"] == 1
    with pytest.raises(ValueError, match="case sets"):
        metrics.paired_quality(native, {"a": fleet["a"]})
    with pytest.raises(ValueError, match="labels"):
        metrics.paired_quality(native, {**fleet, "a": dict(answer="3", prediction="3")})
