# SPDX-License-Identifier: Apache-2.0
"""CPU comparison of complete training workers; no GPU or model loading."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    gates = dict(
        loss_max_abs=0.01,
        sampled_logits_relative_l2=0.02,
        sampled_logits_per_sequence_relative_l2=0.02,
        per_parameter_gradient_relative_l2=0.05,
        updated_parameter_max_abs=0.0001,
    )
    report = dict(
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Whole BF16 Qwen3.5 text training step; fixed text, no convergence claim",
        gates=gates,
        workers={},
        comparisons=[],
    )
    if (args.results / "run.json").exists():
        report["run"] = json.loads((args.results / "run.json").read_text())
    for path in args.results.glob("*/result.json"):
        value = json.loads(path.read_text())
        receipt = value.pop("input_receipt")
        report.setdefault("checkpoint", receipt)
        value["raw_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["workers"][path.parent.name] = value

    def tensors(candidate, reference, family, gc, tokens):
        name = f"{family}-gc{gc}-t{tokens}.safetensors"
        summary = dict(
            tensors=0,
            exact_tensors=0,
            max_abs=0.0,
            max_relative_l2=0.0,
            failed=[],
            passed=True,
        )
        with (
            safe_open(
                args.results / candidate / name, framework="pt", device="cpu"
            ) as cf,
            safe_open(
                args.results / reference / name, framework="pt", device="cpu"
            ) as rf,
        ):
            assert set(cf.keys()) == set(rf.keys()), (candidate, reference, family)
            error_norm, reference_norm = 0.0, 0.0
            for key in cf.keys():
                a, b = cf.get_tensor(key).float(), rf.get_tensor(key).float()
                assert (
                    a.shape == b.shape
                    and torch.isfinite(a).all()
                    and torch.isfinite(b).all()
                ), key
                diff = a - b
                maximum = diff.abs().max().item()
                en, rn = (
                    diff.double().square().sum().item(),
                    b.double().square().sum().item(),
                )
                relative = math.sqrt(en / max(rn, 1e-40))
                sequence_relative = None
                if (
                    family.endswith("logits")
                    and report["workers"][candidate].get("batch", 1) > 1
                ):
                    assert a.ndim == 3, "audit batch dimension mismatch"
                if family.endswith("logits") and a.ndim == 3:
                    assert a.shape[0] == report["workers"][candidate].get("batch", 1), (
                        "audit batch dimension mismatch"
                    )
                    sequence_relative = (
                        (
                            diff.double().flatten(1).square().sum(1)
                            / b.double().flatten(1).square().sum(1).clamp_min(1e-40)
                        )
                        .sqrt()
                        .max()
                        .item()
                    )
                    summary["max_sequence_relative_l2"] = max(
                        summary.get("max_sequence_relative_l2", 0), sequence_relative
                    )
                error_norm += en
                reference_norm += rn
                summary["tensors"] += 1
                summary["exact_tensors"] += int(torch.equal(a, b))
                summary["max_abs"] = max(summary["max_abs"], maximum)
                summary["max_relative_l2"] = max(summary["max_relative_l2"], relative)
                passed = (
                    maximum <= gates["updated_parameter_max_abs"]
                    if family == "updated"
                    else relative
                    <= gates[
                        "per_parameter_gradient_relative_l2"
                        if family.endswith("grad")
                        else "sampled_logits_relative_l2"
                    ]
                )
                if sequence_relative is not None:
                    passed &= (
                        sequence_relative
                        <= gates["sampled_logits_per_sequence_relative_l2"]
                    )
                if not passed:
                    summary["failed"].append(
                        dict(name=key, max_abs=maximum, relative_l2=relative)
                    )
            summary["aggregate_relative_l2"] = math.sqrt(
                error_norm / max(reference_norm, 1e-40)
            )
        summary["passed"] = not summary["failed"]
        return summary

    pairs = [
        ("unsloth-ce", "peft-before"),
        ("unsloth-ce-graph", "peft-before"),
        ("unsloth-ce-graph", "unsloth-ce"),
        ("dynamo-eager", "peft-before"),
        ("aot-eager", "peft-before"),
        ("graph-backend", "peft-before"),
        ("native-graph", "peft-before"),
        ("native-graph", "dynamo-eager"),
        ("peft-implicit", "peft-before"),
        ("native-graph", "peft-implicit"),
        ("graph-backend", "aot-eager"),
        ("aot-eager", "dynamo-eager"),
        ("native-activation", "peft-before"),
        ("native-consumer", "peft-before"),
        ("native-expand", "peft-before"),
        ("native-consumer", "native-activation"),
        ("packed", "peft-before"),
        ("compiled-mlp", "packed"),
        ("fused", "packed"),
        ("compact", "packed"),
        ("fused", "peft-before"),
        ("compact", "peft-before"),
        ("compiled-model", "peft-before"),
        ("peft-after", "peft-before"),
    ]
    for candidate, reference in pairs:
        if candidate not in report["workers"] or reference not in report["workers"]:
            continue
        c, r = report["workers"][candidate], report["workers"][reference]
        if not c.get("cases") or not r.get("cases"):
            continue
        assert c["initial_adapter_sha256"] == r["initial_adapter_sha256"], (
            candidate,
            reference,
        )
        assert c.get("batch", 1) == r.get("batch", 1), "microbatch mismatch"
        reference_cases = {(v["checkpointing"], v["tokens"]): v for v in r["cases"]}
        for value in c["cases"]:
            key = value["checkpointing"], value["tokens"]
            if key not in reference_cases:
                continue
            rc = reference_cases[key]
            assert value.get("batch", c.get("batch", 1)) == rc.get(
                "batch", r.get("batch", 1)
            ), "microbatch mismatch"
            assert value.get("input_sha256") == rc.get("input_sha256"), (
                "original inputs differ"
            )
            loss_error = max(
                abs(a - b) for a, b in zip(value["losses"], rc["losses"], strict=True)
            )
            comparison = dict(
                candidate=candidate,
                reference=reference,
                checkpointing=key[0],
                tokens=key[1],
                batch=c.get("batch", 1),
                speedup=rc["median_wall_ms"] / value["median_wall_ms"],
                loss_max_abs=loss_error,
            )
            for family in ("grad", "updated", "logits"):
                comparison[family] = tensors(
                    candidate, reference, family, int(key[0]), key[1]
                )
            comparison["numerics_passed"] = loss_error <= gates["loss_max_abs"] and all(
                comparison[family]["passed"] for family in ("grad", "updated", "logits")
            )
            if "sample_losses" in value and "sample_losses" in rc:
                comparison["timed_loss_max_abs"] = max(
                    abs(a - b)
                    for a, b in zip(
                        value["sample_losses"], rc["sample_losses"], strict=True
                    )
                )
                comparison["numerics_passed"] &= (
                    comparison["timed_loss_max_abs"] <= gates["loss_max_abs"]
                )
            if "changed_input_loss" in value or "changed_input_loss" in rc:
                assert value.get("changed_input_sha256") == rc.get(
                    "changed_input_sha256"
                ), "changed inputs differ"
                comparison["changed_input_loss_max_abs"] = abs(
                    value["changed_input_loss"] - rc["changed_input_loss"]
                )
                for family in ("changed-grad", "changed-logits"):
                    comparison[family] = tensors(
                        candidate, reference, family, int(key[0]), key[1]
                    )
                comparison["numerics_passed"] &= comparison[
                    "changed_input_loss_max_abs"
                ] <= gates["loss_max_abs"] and all(
                    comparison[f]["passed"] for f in ("changed-grad", "changed-logits")
                )
            report["comparisons"].append(comparison)
            print(
                candidate,
                reference,
                int(key[0]),
                key[1],
                "speed",
                round(comparison["speedup"], 4),
                "numerics",
                comparison["numerics_passed"],
                "loss",
                loss_error,
                "gradient_l2",
                comparison["grad"]["max_relative_l2"],
                flush=True,
            )
    report["state"] = (
        "complete" if report.get("run", {}).get("state") == "complete" else "partial"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
