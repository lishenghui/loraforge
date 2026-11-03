# SPDX-License-Identifier: Apache-2.0
"""Pilot: full-model HTTP timing, labeled quality, and forced-prefix numerics.

Run only on an allocated GPU. Prepare tokenized inputs and analyze logits on CPU.
An explicit external HIP library is required; this does not enable a production
backend or weaken validate_sglang_serving.py's strict compatibility contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from serving_experiment_metrics import answer_digit, paired_quality
from validate_sglang_serving import dump, http

if os.environ.get("FLEET_EXPERIMENT_MODE") in ("native", "fleet"):
    from sglang_fleet_experiment_hook import install, observe_graph_execution

    observe_graph_execution(Path(os.environ["FLEET_EXPERIMENT_OUTPUT"]))
    if os.environ["FLEET_EXPERIMENT_MODE"] == "fleet":
        install(
            Path(os.environ["FLEET_EXPERIMENT_LIBRARY"]),
            Path(os.environ["FLEET_EXPERIMENT_OUTPUT"]),
        )


def serve(args, manifest):
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    options = dict(
        model_path=manifest["model"],
        host="127.0.0.1",
        port=args.port,
        dtype="bfloat16",
        tp_size=1,
        random_seed=0,
        context_length=1024,
        attention_backend="triton",
        disable_radix_cache=True,
        disable_prefill_cuda_graph=True,
        chunked_prefill_size=512,
        max_prefill_tokens=512,
        max_total_tokens=2048,
        max_running_requests=2,
        mem_fraction_static=0.65,
        enable_lora=True,
        lora_backend="csgmv",
        max_lora_chunk_size=16,
        max_lora_rank=max(a["rank"] for a in manifest["adapters"]),
        lora_target_modules=manifest["target_modules"],
        max_loras_per_batch=2,
        max_loaded_loras=len(manifest["adapters"]) + 2,
        enable_lora_overlap_loading=False,
        lora_paths=[
            dict(lora_name=a["name"], lora_path=a["path"], pinned=False)
            for a in manifest["adapters"]
        ],
        enable_custom_logit_processor=True,
        cuda_graph_config={"decode": {"backend": "full", "bs": [1, 2], "max_bs": 2}},
        log_level="warning",
    )
    dump(args.output / "server-args.json", options)
    launch_server(ServerArgs(**options))


def generate(url, ids, adapters, tokens, fixed=True, extra=None):
    body = dict(
        input_ids=ids,
        lora_path=adapters,
        return_logprob=False,
        sampling_params=dict(temperature=0, ignore_eos=fixed, max_new_tokens=tokens),
    )
    if extra:
        body.update(extra)
    start = time.perf_counter()
    response = http(url, "/generate", body)
    elapsed = time.perf_counter() - start
    rows = response if isinstance(response, list) else [response]
    completed = sum(r["meta_info"]["completion_tokens"] for r in rows)
    if fixed and completed != tokens * len(rows):
        raise AssertionError("fixed-work request returned an unexpected token count")
    return rows, elapsed, completed


def run_requests(args, manifest, quality, mode, root, shared):
    url = f"http://127.0.0.1:{args.port}"
    result = dict(mode=mode, timing={}, quality={}, trace={}, completed=False)
    # Warm each adapter set before measuring: loading/paging is deliberately excluded.
    workloads = [("base-b1", [manifest["prompts"][0]], [None])]
    for adapter in manifest["adapters"]:
        for batch in (1, 2):
            workloads.append(
                (
                    f"{adapter['name']}-b{batch}",
                    manifest["prompts"][:batch],
                    [adapter["name"]] * batch,
                )
            )
    workloads.append(
        (
            "mixed-b2",
            manifest["prompts"][:2],
            [a["name"] for a in manifest["adapters"][:2]],
        )
    )
    random.Random(20261103).shuffle(workloads)
    for key, ids, adapters in workloads:
        for _ in range(2):
            generate(url, ids, adapters, 16)
        trials = []
        for _ in range(args.trials):
            _, seconds, tokens = generate(url, ids, adapters, args.timing_tokens)
            trials.append(
                dict(
                    seconds=seconds,
                    completion_tokens=tokens,
                    tokens_per_second=tokens / seconds,
                )
            )
        result["timing"][key] = trials
        dump(root / "result.json", result)
    # Normal free generation with labels; no custom processor or forced tokens.
    for case in quality["cases"]:
        rows, _, _ = generate(
            url, case["input_ids"], quality["adapter"], args.quality_tokens, fixed=False
        )
        text = rows[0]["text"]
        result["quality"][case["id"]] = dict(
            answer=case["answer"],
            prediction=answer_digit(text),
            text=text,
            meta=rows[0]["meta_info"],
        )
        dump(root / "result.json", result)
    # Full-vocabulary logits before intervention. Force native token IDs at every step.
    from sglang_forced_trace import ForcedTrace

    processor = ForcedTrace.to_str()
    routes = [("base", None)] + [(a["name"], a["name"]) for a in manifest["adapters"]]
    for label, adapter in routes:
        for prompt, ids in enumerate(manifest["prompts"]):
            case = f"{label}-p{prompt}"
            extra = dict(
                custom_logit_processor=processor,
                sampling_params=dict(
                    temperature=0,
                    ignore_eos=True,
                    max_new_tokens=args.trace_tokens,
                    custom_params=dict(
                        trace_case=case,
                        trace_mode=mode,
                        trace_root=str(shared / "traces"),
                    ),
                ),
            )
            rows, _, _ = generate(url, ids, adapter, args.trace_tokens, extra=extra)
            result["trace"][case] = rows[0]["text"]
            dump(root / "result.json", result)
    result["completed"] = True
    dump(root / "result.json", result)
    return result


def run(args, manifest):
    if args.output.exists():
        raise ValueError("use a new output directory; existing attempts are preserved")
    args.output.mkdir(parents=True)
    quality = json.loads(args.quality.read_text())
    if quality["split"] != "validation" or not quality["cases"]:
        raise ValueError("a nonempty labeled validation split is required")
    protocol = dict(
        version=1,
        source_sha256={
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "benchmark_sglang_service.py",
                "serving_experiment_metrics.py",
                "sglang_forced_trace.py",
                "sglang_fleet_experiment_hook.py",
                "validate_sglang_serving.py",
            )
        },
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        quality_sha256=hashlib.sha256(args.quality.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.fleet_library.read_bytes()).hexdigest(),
        trials=args.trials,
        timing_tokens=args.timing_tokens,
        quality_tokens=args.quality_tokens,
        trace_tokens=args.trace_tokens,
        quality_source=quality["source"],
        timing_scope="single-client HTTP wall time incl base model, prefill and decode; warm resident adapters; excludes startup/paging; no trace processor",
        mode_order=["native", "fleet"],
        limits="pilot, fixed mode order; no general quality or performance certification",
    )
    dump(args.output / "protocol.json", protocol)
    for mode in ("native", "fleet"):
        root = args.output / mode
        root.mkdir()
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PROBE_", "LORAFORGE_SERVING_AUDIT"))
        }
        env.update(
            FLEET_EXPERIMENT_MODE=mode,
            FLEET_EXPERIMENT_LIBRARY=str(args.fleet_library),
            FLEET_EXPERIMENT_OUTPUT=str(root),
            LORAFORGE_SGLANG="1" if mode == "fleet" else "0",
            LORAFORGE_SGLANG_POLICY="rank-aware",
            SGLANG_PLUGINS="loraforge",
        )
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "serve",
            "--manifest",
            str(args.manifest),
            "--output",
            str(root),
            "--port",
            str(args.port),
        ]
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + args.startup_timeout
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"{mode} server exited; see {root / 'server.log'}"
                        )
                    try:
                        http(f"http://127.0.0.1:{args.port}", "/health", timeout=3)
                        break
                    except (OSError, RuntimeError):
                        time.sleep(2)
                else:
                    raise TimeoutError("server startup timed out")
                run_requests(args, manifest, quality, mode, root, args.output)
                print(
                    f"{mode}: timing, quality and forced-prefix traces complete",
                    flush=True,
                )
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
    print("GPU phases finished; run analyze on CPU", flush=True)


def analyze(args):
    import torch

    torch.set_num_threads(2)
    root = args.output
    reports = {
        m: json.loads((root / m / "result.json").read_text())
        for m in ("native", "fleet")
    }
    if not all(r["completed"] for r in reports.values()):
        raise ValueError("both modes must complete")
    native, fleet = reports["native"], reports["fleet"]
    if (
        native["timing"].keys() != fleet["timing"].keys()
        or native["trace"] != fleet["trace"]
    ):
        raise ValueError("workloads differ or forced continuations did not match")
    evidence = [json.loads(p.read_text()) for p in (root / "fleet").glob("hip-*.json")]
    if not any(e["supported"] and e["capture"] for e in evidence):
        raise ValueError("missing HIP graph capture evidence")
    for mode in ("native", "fleet"):
        execution = [
            json.loads(p.read_text()) for p in (root / mode).glob("execution-*.json")
        ]
        if not any(
            e["phase"] == "DECODE" and e["graph"] and not e["trace"] for e in execution
        ):
            raise ValueError(f"missing actual graph execution evidence for {mode}")
    timing = {}
    for case, rows in native["timing"].items():
        other = fleet["timing"][case]
        if [r["completion_tokens"] for r in rows] != [
            r["completion_tokens"] for r in other
        ]:
            raise ValueError("timed work differs")
        a, b = (
            statistics.median(r["seconds"] for r in rows),
            statistics.median(r["seconds"] for r in other),
        )
        timing[case] = dict(native_seconds=a, fleet_seconds=b, speedup=a / b)
    traces = []
    native_files = sorted((root / "traces/native").glob("*/*.pt"))
    fleet_files = sorted((root / "traces/fleet").glob("*/*.pt"))
    protocol = json.loads((root / "protocol.json").read_text())
    if len(native_files) != len(native["trace"]) * protocol["trace_tokens"]:
        raise ValueError("missing native trace positions")
    if [p.relative_to(root / "traces/native") for p in native_files] != [
        p.relative_to(root / "traces/fleet") for p in fleet_files
    ]:
        raise ValueError("trace position sets differ")
    for a_path, b_path in zip(native_files, fleet_files, strict=True):
        a, b = [
            torch.load(p, map_location="cpu", weights_only=True)
            for p in (a_path, b_path)
        ]
        if a["forced_token"] != b["forced_token"] or a["step"] != b["step"]:
            raise ValueError("trace conditioning differs")
        x, y = a["logits"].double(), b["logits"].double()
        if (
            x.shape != y.shape
            or not torch.isfinite(x).all()
            or not torch.isfinite(y).all()
        ):
            raise ValueError("invalid trace logits")
        logp, logq = x.log_softmax(0), y.log_softmax(0)
        p, q = logp.exp(), logq.exp()
        top = torch.topk(logp, 2).values
        traces.append(
            dict(
                case=a_path.parent.name,
                step=a["step"],
                phase="prefill" if a["step"] == 0 else "decode",
                kl_native_to_fleet=float((p * (logp - logq)).sum()),
                total_variation=float((p - q).abs().sum() / 2),
                max_abs_logprob_delta=float((logp - logq).abs().max()),
                native_token_logprob_delta=float(
                    logq[a["forced_token"]] - logp[a["forced_token"]]
                ),
                top1_matches=int(x.argmax()) == int(y.argmax()),
                native_top2_margin=float(top[0] - top[1]),
            )
        )
    summary = dict(
        protocol=protocol,
        quality=paired_quality(native["quality"], fleet["quality"]),
        timing=timing,
        traces=traces,
        decision="pilot results only; strict compatibility result remains failed",
    )
    dump(root / "summary.json", summary)
    decode = [t for t in traces if t["phase"] == "decode"]
    print(
        json.dumps(
            dict(
                quality=summary["quality"],
                timing=timing,
                decode_positions=len(decode),
                max_kl=max(t["kl_native_to_fleet"] for t in decode),
                max_tv=max(t["total_variation"] for t in decode),
                top1_flips=sum(not t["top1_matches"] for t in decode),
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "serve", "analyze"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--fleet-library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timing-tokens", type=int, default=64)
    parser.add_argument("--quality-tokens", type=int, default=96)
    parser.add_argument("--trace-tokens", type=int, default=16)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.action == "analyze":
        analyze(args)
    else:
        args.manifest = args.manifest.resolve()
        manifest = json.loads(args.manifest.read_text())
        if args.action == "serve":
            serve(args, manifest)
        else:
            if (
                min(
                    args.trials,
                    args.timing_tokens,
                    args.quality_tokens,
                    args.trace_tokens - 1,
                )
                < 1
            ):
                parser.error(
                    "positive trials/token counts and at least two trace tokens required"
                )
            args.quality = args.quality.resolve()
            args.fleet_library = args.fleet_library.resolve()
            run(args, manifest)


if __name__ == "__main__":
    main()
