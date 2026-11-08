# SPDX-License-Identifier: Apache-2.0
"""Run isolated training workers on one already allocated GPU; never submit jobs."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 1024, 4096])
    parser.add_argument("--max-seconds", type=int, default=1100)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--skip-whole-compile", action="store_true")
    selection.add_argument("--whole-compile-only", action="store_true")
    selection.add_argument("--native-consumer-only", action="store_true")
    selection.add_argument("--autograd-isolation-only", action="store_true")
    selection.add_argument("--causal-policy-only", action="store_true")
    selection.add_argument("--native-graph-only", action="store_true")
    parser.add_argument("--emulate-precision-casts", action="store_true")
    parser.add_argument("--recompile-limit", type=int)
    parser.add_argument(
        "--compile-backend",
        choices=["inductor", "cudagraphs", "eager", "aot_eager"],
        default="inductor",
    )
    parser.add_argument("--check-input-change", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    report = dict(state="running", workers=[])
    modes = [
        ("peft-before", "peft"),
        ("packed", "packed"),
        ("compiled-mlp", "compiled-mlp"),
        ("fused", "fused"),
        ("compact", "compact"),
        ("compiled-model", "compiled-model"),
        ("peft-after", "peft"),
    ]
    if args.skip_whole_compile:
        modes = [
            (name, backend) for name, backend in modes if backend != "compiled-model"
        ]
    elif args.whole_compile_only:
        modes = [
            ("peft-before", "peft"),
            ("compiled-model", "compiled-model"),
            ("peft-after", "peft"),
        ]
    elif args.native_consumer_only:
        modes = [
            ("peft-before", "peft"),
            ("native-activation", "native-activation"),
            ("native-consumer", "native-consumer"),
            ("native-expand", "native-expand"),
            ("peft-after", "peft"),
        ]
    elif args.autograd_isolation_only:
        modes = [
            ("peft-before", "peft"),
            ("dynamo-eager", "compiled-model"),
            ("aot-eager", "compiled-model"),
            ("graph-backend", "compiled-model"),
            ("native-graph", "native-graph"),
            ("peft-after", "peft"),
        ]
    elif args.causal_policy_only:
        modes = [
            ("peft-before", "peft"),
            ("peft-implicit", "peft"),
            ("dynamo-eager", "compiled-model"),
            ("native-graph", "native-graph"),
            ("peft-after", "peft"),
        ]
    elif args.native_graph_only:
        modes = [
            ("peft-before", "peft"),
            ("peft-implicit", "peft"),
            ("native-graph", "native-graph"),
            ("peft-after", "peft"),
        ]
    for name, backend in modes:
        remaining = deadline - time.monotonic()
        if remaining < 30:
            report["workers"].append(dict(name=name, state="skipped_time_budget"))
            continue
        if backend == "compiled-model" and remaining < 240:
            report["workers"].append(dict(name=name, state="skipped_time_budget"))
            continue
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("benchmark_qwen35_training.py")),
            str(args.checkpoint),
            "--output",
            str(args.output / name),
            "--backend",
            backend,
            "--tokens",
            *map(str, args.tokens),
            "--checkpointing",
            *(
                ["0"]
                if backend in ("compiled-model", "native-graph")
                or args.whole_compile_only
                or args.autograd_isolation_only
                or args.causal_policy_only
                or args.native_graph_only
                else ["0", "1"]
            ),
        ]
        if args.emulate_precision_casts:
            command.append("--emulate-precision-casts")
        if args.recompile_limit is not None:
            command += ["--recompile-limit", str(args.recompile_limit)]
        compiler = {
            "dynamo-eager": "eager",
            "aot-eager": "aot_eager",
            "graph-backend": "cudagraphs",
        }.get(name, args.compile_backend)
        command += ["--compile-backend", compiler]
        if args.check_input_change:
            command.append("--check-input-change")
        if (args.causal_policy_only or args.native_graph_only) and name not in (
            "peft-before",
            "peft-after",
        ):
            command.append("--implicit-causal")
        timeout = min(
            remaining, 300 if name in ("peft-before", "compiled-model") else 240
        )
        if backend == "compiled-model":
            if args.whole_compile_only:
                # Measured native controls take ~20s; reserve a full minute.
                timeout = min(420, remaining - 60)
            elif not (args.autograd_isolation_only or args.causal_policy_only):
                timeout = min(timeout, remaining - 150)
        started = time.monotonic()
        print("START_WORKER", name, flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=timeout)
                state = "complete" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code, state = process.returncode, "timed_out"
        report["workers"].append(
            dict(
                name=name,
                backend=backend,
                state=state,
                returncode=code,
                seconds=time.monotonic() - started,
            )
        )
        (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        print("END_WORKER", name, state, flush=True)
        if name == "peft-before" and state != "complete":
            break
    required = {
        name for name, _ in modes if name != "compiled-model" or args.whole_compile_only
    }
    completed = {
        worker["name"] for worker in report["workers"] if worker["state"] == "complete"
    }
    report["state"] = "complete" if required <= completed else "failed"
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
