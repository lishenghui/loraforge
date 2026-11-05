# SPDX-License-Identifier: Apache-2.0
"""The training comparison must reject loss, logit, gradient or update drift."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safe = pytest.importorskip("safetensors.torch")
spec = importlib.util.spec_from_file_location(
    "analyze_qwen35_training",
    Path(__file__).resolve().parents[1] / "benchmarks/analyze_qwen35_training.py",
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.mark.parametrize(
    "drift",
    [
        None,
        "loss",
        "grad",
        "updated",
        "logits",
        "timed-loss",
        "changed-loss",
        "changed-grad",
        "changed-logits",
    ],
)
def test_cpu_comparison_gates(tmp_path, monkeypatch, drift):
    for name in ("peft-before", "packed"):
        path = tmp_path / name
        path.mkdir()
        losses = [1.0, 0.9, 0.8]
        if name == "packed" and drift == "loss":
            losses[2] += 0.02
        (path / "result.json").write_text(
            json.dumps(
                dict(
                    state="complete",
                    input_receipt={"text_sha256": "same"},
                    initial_adapter_sha256="same",
                    cases=[
                        dict(
                            tokens=128,
                            checkpointing=False,
                            losses=losses,
                            sample_losses=[
                                1.0
                                + (
                                    0.02
                                    if name == "packed" and drift == "timed-loss"
                                    else 0.0
                                )
                            ],
                            changed_input_loss=2.0
                            + (
                                0.02
                                if name == "packed" and drift == "changed-loss"
                                else 0.0
                            ),
                            median_wall_ms=2 if name == "peft-before" else 1,
                        )
                    ],
                )
            )
        )
        for family in ("grad", "updated", "logits", "changed-grad", "changed-logits"):
            value = torch.ones(2, 3)
            if name == "packed" and family == drift:
                value += {
                    "grad": 0.1,
                    "updated": 0.0002,
                    "logits": 0.03,
                    "changed-grad": 0.1,
                    "changed-logits": 0.03,
                }[family]
            safe.save_file({"weight": value}, path / f"{family}-gc0-t128.safetensors")
    (tmp_path / "run.json").write_text(json.dumps({"state": "complete"}))
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys, "argv", ["analyze_qwen35_training", str(tmp_path), "--output", str(output)]
    )
    analysis.main()
    result = json.loads(output.read_text())["comparisons"]
    assert len(result) == 1
    assert result[0]["speedup"] == 2
    assert result[0]["numerics_passed"] is (drift is None)
