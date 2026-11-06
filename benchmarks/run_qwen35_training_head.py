# SPDX-License-Identifier: Apache-2.0
"""GPU chunked-head sweep with separate CPU numerical comparisons."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from run_qwen35_training_batches import shape
from run_qwen35_training_loss import save, worker

METHODS = [
    ("peft-before", "peft", "native"),
    ("head-full-reuse", "peft", "head-full-reuse"),
    ("head-chunked", "peft", "head-chunked"),
    ("head-full-reuse-graph", "native-graph", "head-full-reuse"),
    ("head-chunked-graph", "native-graph", "head-chunked"),
    ("peft-after", "peft", "native"),
]


def run(args):
    args.output.mkdir(exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    report = dict(state="running", shapes=[])
    report["check"] = worker(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("check_qwen35_training_head.py")),
            str(args.checkpoint),
            "--output",
            str(args.output / "head-check.json"),
        ],
        args.output / "head-check.log",
        180,
    )
    save(args.output / "sweep.json", report)
    if report["check"]["state"] != "complete":
        report["state"] = "failed_operator_check"
        save(args.output / "sweep.json", report)
        raise SystemExit(1)
    for batch, tokens in args.shapes:
        directory = args.output / f"b{batch}-t{tokens}"
        directory.mkdir()
        case = dict(
            name=directory.name, batch=batch, tokens=tokens, state="running", workers=[]
        )
        report["shapes"].append(case)
        for name, backend, loss in METHODS:
            status = dict(name=name, backend=backend, loss=loss)
            remaining = deadline - time.monotonic()
            if name == "peft-after" and case["workers"][0]["state"] != "complete":
                status["state"] = "skipped_missing_reference"
            elif remaining < 30:
                status["state"] = "skipped_time_budget"
            else:
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("benchmark_qwen35_training.py")),
                    str(args.checkpoint),
                    "--output",
                    str(directory / name),
                    "--backend",
                    backend,
                    "--loss-backend",
                    loss,
                    "--batch-size",
                    str(batch),
                    "--tokens",
                    str(tokens),
                    "--checkpointing",
                    "0",
                    "--check-input-change",
                    "--head-chunk-rows",
                    "512",
                ]
                if backend == "native-graph":
                    command += ["--implicit-causal"]
                elif name != "peft-after":
                    command += ["--audit-initial-step"]
                print("START_HEAD_WORKER", directory.name, name, flush=True)
                status.update(
                    worker(command, directory / (name + ".log"), min(240, remaining))
                )
                print(
                    "END_HEAD_WORKER", directory.name, name, status["state"], flush=True
                )
            case["workers"].append(status)
            save(directory / "run.json", case)
            save(args.output / "sweep.json", report)
        states = {w["state"] for w in case["workers"]}
        case["state"] = (
            "complete"
            if states == {"complete"}
            else "capacity_limit"
            if "oom" in states
            and states <= {"complete", "oom", "skipped_missing_reference"}
            else "failed"
        )
        save(directory / "run.json", case)
    report["state"] = (
        "failed"
        if any(s["state"] == "failed" for s in report["shapes"])
        else "complete_with_capacity_limits"
        if any(s["state"] == "capacity_limit" for s in report["shapes"])
        else "complete"
    )
    save(args.output / "sweep.json", report)
    if report["state"] == "failed":
        raise SystemExit(1)


def compare(args):
    from diagnose_unsloth_training_loss import parameters

    sweep = json.loads((args.results / "sweep.json").read_text())
    report = dict(
        run=sweep,
        operator=json.loads((args.results / "head-check.json").read_text()),
        shapes=[],
    )
    success = (
        sweep["state"] in {"complete", "complete_with_capacity_limits"}
        and report["operator"]["state"] == "passed"
    )
    native_checks = 0
    for case in sweep["shapes"]:
        directory = args.results / case["name"]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("analyze_qwen35_training.py")),
                str(directory),
                "--output",
                str(directory / "comparison.json"),
            ],
            check=True,
        )
        data = json.loads((directory / "comparison.json").read_text())
        data.update(batch=case["batch"], tokens=case["tokens"], initial_steps={})
        reference_ok = case["workers"][0]["state"] == "complete"
        data["native_reference_available"] = reference_ok
        pairs = {(v["candidate"], v["reference"]): v for v in data["comparisons"]}
        for status in case["workers"]:
            name = status["name"]
            if not name.startswith("head-") or status["state"] != "complete":
                continue
            reference = "peft-before" if reference_ok else name.removesuffix("-graph")
            if name != reference:
                success &= (name, reference) in pairs
            if reference_ok and not name.endswith("-graph"):
                data["initial_steps"][name] = parameters(
                    argparse.Namespace(
                        reference=directory / "peft-before", candidate=directory / name
                    )
                )
        native = [
            c
            for c in data["comparisons"]
            if c["reference"] == "peft-before" and c["candidate"].startswith("head-")
        ]
        native_checks += len(native)
        data["native_candidate_numerics_passed"] = (
            all(c["numerics_passed"] for c in native) if native else None
        )
        success &= all(c["numerics_passed"] for c in data["comparisons"])
        report["shapes"].append(data)
        print(
            "HEAD_COMPARE",
            case["name"],
            data["native_candidate_numerics_passed"],
            flush=True,
        )
    success &= native_checks > 0
    report["state"] = "passed" if success else "failed"
    report["controller_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    save(args.output, report)
    if not success:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    gpu = modes.add_parser("run")
    gpu.add_argument("checkpoint", type=Path)
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument("--shapes", type=shape, nargs="+", default=[(4, 1024), (8, 1024)])
    gpu.add_argument("--max-seconds", type=int, default=1100)
    cpu = modes.add_parser("compare")
    cpu.add_argument("results", type=Path)
    cpu.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        if len(set(args.shapes)) != len(args.shapes):
            parser.error("duplicate shapes")
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
