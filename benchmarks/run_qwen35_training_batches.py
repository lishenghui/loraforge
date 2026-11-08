# SPDX-License-Identifier: Apache-2.0
"""Isolated microbatch training sweep; GPU run, then separate CPU comparison."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def shape(value):
    try:
        batch, tokens = map(int, value.lower().split("x"))
        if batch < 1 or tokens < 2:
            raise ValueError
        return batch, tokens
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shape must be positive batch x length>=2"
        ) from exc


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def failed_state(log):
    text = log.lower()
    return (
        "oom"
        if any(
            s in text
            for s in (
                "torch.outofmemoryerror",
                "torch.cuda.outofmemoryerror",
                "hip out of memory",
                "cuda out of memory",
            )
        )
        else "failed"
    )


def run(args):
    args.output.mkdir(exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    report = dict(
        state="running",
        gradient_accumulation=1,
        shapes=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for batch, tokens in args.shapes:
        name = f"b{batch}-t{tokens}"
        directory = args.output / name
        directory.mkdir()
        case = dict(name=name, batch=batch, tokens=tokens, state="running", workers=[])
        report["shapes"].append(case)
        for worker, backend in (
            ("peft-before", "peft"),
            ("native-graph", "native-graph"),
            ("peft-after", "peft"),
        ):
            if worker == "peft-after" and case["workers"][0]["state"] != "complete":
                case["workers"].append(
                    dict(name=worker, state="skipped_missing_reference")
                )
                continue
            remaining = deadline - time.monotonic()
            if remaining < 30:
                case["workers"].append(dict(name=worker, state="skipped_time_budget"))
                continue
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("benchmark_qwen35_training.py")),
                str(args.checkpoint),
                "--output",
                str(directory / worker),
                "--backend",
                backend,
                "--batch-size",
                str(batch),
                "--tokens",
                str(tokens),
                "--checkpointing",
                "0",
                "--check-input-change",
            ]
            if backend == "native-graph":
                command.append("--implicit-causal")
            started = time.monotonic()
            log = directory / (worker + ".log")
            print("START_BATCH_WORKER", name, worker, flush=True)
            with log.open("w") as output:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=min(240, remaining))
                    state = "complete" if code == 0 else failed_state(log.read_text())
                except subprocess.TimeoutExpired:
                    os_kill_group(process)
                    code, state = process.returncode, "timed_out"
            case["workers"].append(
                dict(
                    name=worker,
                    backend=backend,
                    state=state,
                    returncode=code,
                    seconds=time.monotonic() - started,
                )
            )
            save(directory / "run.json", case)
            save(args.output / "sweep.json", report)
            print("END_BATCH_WORKER", name, worker, state, flush=True)
        states = {w["state"] for w in case["workers"]}
        case["state"] = (
            "complete"
            if states == {"complete"}
            else "oom"
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
        if any(c["state"] == "oom" for c in report["shapes"])
        else "complete"
    )
    save(args.output / "sweep.json", report)
    if report["state"] == "failed":
        raise SystemExit(1)


def os_kill_group(process):
    import os

    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def compare(args):
    sweep = json.loads((args.results / "sweep.json").read_text())
    report = dict(
        run=sweep,
        gradient_accumulation=1,
        rows=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for case in sweep["shapes"]:
        directory = args.results / case["name"]
        row = dict(
            batch=case["batch"],
            tokens=case["tokens"],
            input_tokens_per_step=case["batch"] * case["tokens"],
            run_state=case["state"],
        )
        report["rows"].append(row)
        if case["state"] != "complete":
            row["numerics_passed"] = None
            row["workers"] = case["workers"]
            continue
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
        if "checkpoint" not in report:
            report["checkpoint"] = data["checkpoint"]
        else:
            for key in ("text_sha256", "input_tokens_sha256"):
                assert report["checkpoint"][key] == data["checkpoint"][key], (
                    "sweep fixture mismatch"
                )
        row["comparisons"] = data["comparisons"]
        row["workers"] = data["workers"]
        row["numerics_passed"] = len(data["comparisons"]) == 2 and all(
            v["numerics_passed"] for v in data["comparisons"]
        )
        graph = data["workers"]["native-graph"]["cases"][0]
        assert all(
            w["batch"] == case["batch"]
            and len(w["cases"]) == 1
            and w["cases"][0]["tokens"] == case["tokens"]
            for w in data["workers"].values()
        )
        controls = [
            data["workers"][n]["cases"][0] for n in ("peft-before", "peft-after")
        ]
        row.update(
            graph_wall_ms=graph["median_wall_ms"],
            graph_tokens_per_second=graph["tokens_per_second"],
            native_wall_ms=[v["median_wall_ms"] for v in controls],
            native_tokens_per_second=[v["tokens_per_second"] for v in controls],
            speedup_range=sorted(
                v["median_wall_ms"] / graph["median_wall_ms"] for v in controls
            ),
        )
        print(
            "BATCH_RESULT",
            row["batch"],
            row["tokens"],
            row["graph_tokens_per_second"],
            row["speedup_range"],
            row["numerics_passed"],
            flush=True,
        )
    report["state"] = (
        "failed"
        if sweep["state"] == "failed"
        or any(v["numerics_passed"] is False for v in report["rows"])
        else sweep["state"]
    )
    save(args.output, report)
    if report["state"] == "failed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    gpu = modes.add_parser("run")
    gpu.add_argument("checkpoint", type=Path)
    gpu.add_argument("--output", type=Path, required=True)
    gpu.add_argument(
        "--shapes",
        type=shape,
        nargs="+",
        default=list(
            map(
                shape,
                [
                    "1x128",
                    "8x128",
                    "32x128",
                    "1x1024",
                    "2x1024",
                    "4x1024",
                    "8x512",
                    "8x1024",
                ],
            )
        ),
    )
    gpu.add_argument("--max-seconds", type=int, default=1100)
    cpu = modes.add_parser("compare")
    cpu.add_argument("results", type=Path)
    cpu.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        if len(set(args.shapes)) != len(args.shapes):
            parser.error("duplicate batch shapes")
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
