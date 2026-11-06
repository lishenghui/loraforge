# SPDX-License-Identifier: Apache-2.0
"""Isolated Unsloth-CE ablation: GPU workers, then separate CPU comparisons."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from run_qwen35_training_batches import failed_state, os_kill_group, shape


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(command, log, timeout):
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
            state = "complete" if code == 0 else failed_state(log.read_text())
        except subprocess.TimeoutExpired:
            os_kill_group(process)
            code, state = process.returncode, "timed_out"
    return dict(state=state, returncode=code, seconds=time.monotonic() - started)


def run(args):
    args.output.mkdir(exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    report = dict(state="running", checks={}, shapes=[])
    for launch in ("upstream", "hip"):
        print("START_CE_CHECK", launch, flush=True)
        report["checks"][launch] = worker(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("check_unsloth_training_loss.py")),
                str(args.source),
                "--launch",
                launch,
                "--output",
                str(args.output / ("check-" + launch + ".json")),
            ],
            args.output / ("check-" + launch + ".log"),
            150,
        )
        print("END_CE_CHECK", launch, report["checks"][launch]["state"], flush=True)
        save(args.output / "sweep.json", report)
    if report["checks"]["hip"]["state"] != "complete":
        report["state"] = "failed_operator_check"
        save(args.output / "sweep.json", report)
        raise SystemExit(1)
    for batch, tokens in args.shapes:
        name = f"b{batch}-t{tokens}"
        directory = args.output / name
        directory.mkdir()
        case = dict(name=name, batch=batch, tokens=tokens, workers=[], state="running")
        report["shapes"].append(case)
        for name, backend, loss in (
            ("peft-before", "peft", "native"),
            ("unsloth-ce", "peft", "unsloth-ce"),
            ("unsloth-ce-graph", "native-graph", "unsloth-ce"),
            ("peft-after", "peft", "native"),
        ):
            remaining = deadline - time.monotonic()
            status = dict(name=name, backend=backend, loss=loss)
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
                ]
                if loss != "native":
                    command += ["--loss-source", str(args.source)]
                if backend == "native-graph":
                    command += ["--implicit-causal"]
                print("START_LOSS_WORKER", case["name"], name, flush=True)
                status.update(
                    worker(command, directory / (name + ".log"), min(240, remaining))
                )
                print(
                    "END_LOSS_WORKER", case["name"], name, status["state"], flush=True
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
        save(args.output / "sweep.json", report)
    report["state"] = (
        "failed"
        if any(c["state"] == "failed" for c in report["shapes"])
        else "complete_with_capacity_limits"
        if any(c["state"] == "capacity_limit" for c in report["shapes"])
        else "complete"
    )
    save(args.output / "sweep.json", report)
    if report["state"] == "failed":
        raise SystemExit(1)


def compare(args):
    sweep = json.loads((args.results / "sweep.json").read_text())
    report = dict(run=sweep, shapes=[], operator_checks={})
    for path in args.results.glob("check-*.json"):
        report["operator_checks"][path.stem] = json.loads(path.read_text())
    all_passed = sweep["state"] in {"complete", "complete_with_capacity_limits"}
    all_passed &= (
        report["operator_checks"].get("check-hip", {}).get("state") == "passed"
    )
    native_candidate_checks = 0
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
        result = json.loads((directory / "comparison.json").read_text())
        result["batch"], result["tokens"] = case["batch"], case["tokens"]
        result["native_reference_available"] = case["workers"][0]["state"] == "complete"
        required = {(c["candidate"], c["reference"]): c for c in result["comparisons"]}
        for name in ("unsloth-ce", "unsloth-ce-graph"):
            if (
                next(w["state"] for w in case["workers"] if w["name"] == name)
                == "complete"
            ):
                reference = (
                    "peft-before"
                    if result["native_reference_available"]
                    else "unsloth-ce"
                )
                if name != reference:
                    all_passed &= (name, reference) in required
        all_passed &= all(c["numerics_passed"] for c in result["comparisons"])
        native_checks = [
            c
            for c in result["comparisons"]
            if c["reference"] == "peft-before"
            and c["candidate"].startswith("unsloth-ce")
        ]
        native_candidate_checks += len(native_checks)
        result["native_candidate_numerics_passed"] = (
            all(c["numerics_passed"] for c in native_checks) if native_checks else None
        )
        report["shapes"].append(result)
        print(
            "LOSS_COMPARE",
            case["name"],
            result["native_reference_available"],
            [
                (c["candidate"], c["reference"], c["numerics_passed"])
                for c in result["comparisons"]
            ],
            flush=True,
        )
    all_passed &= native_candidate_checks > 0
    report["state"] = "passed" if all_passed else "failed"
    report["controller_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    save(args.output, report)
    if not all_passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    gpu = modes.add_parser("run")
    gpu.add_argument("checkpoint", type=Path)
    gpu.add_argument("source", type=Path)
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument("--max-seconds", type=int, default=1100)
    gpu.add_argument("--shapes", type=shape, nargs="+", default=[(4, 1024), (8, 1024)])
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
