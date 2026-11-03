# SPDX-License-Identifier: Apache-2.0
"""Offline full-model correctness validation; requires an allocated GPU for run.

The manifest supplies existing model/adapter paths and prepared prompt token IDs.
CPU preparation, model downloads and performance measurement are outside this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import wraps
from pathlib import Path


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n"
    )
    temporary.replace(path)


def install_observer():
    """Validation-only observation, installed again when multiprocessing spawns."""
    import torch
    from sglang.srt.lora.mem_pool import LoRAMemoryPool
    from sglang.srt.model_executor.model_runner import ModelRunner

    root = Path(os.environ["LORAFORGE_SERVING_AUDIT"])
    root.mkdir(parents=True, exist_ok=True)
    counts, loads = Counter(), []
    report = {"pid": os.getpid(), "scope": "correctness observation; no timing claims"}

    def write():
        dump(
            root / f"audit-{os.getpid()}.json",
            {**report, "counts": dict(counts), "loads": loads},
        )

    def buffers(pool):
        values = []
        for kind in ("A_buffer", "B_buffer"):
            for name, layers in sorted(getattr(pool, kind).items()):
                for index, tensor in enumerate(layers):
                    if tensor is not None:
                        values.append(
                            (kind, name, index, tensor.data_ptr(), tuple(tensor.shape))
                        )
        return hashlib.sha256(json.dumps(values).encode()).hexdigest()

    previous_load = LoRAMemoryPool.load_lora_weight_to_buffer

    @wraps(previous_load)
    def load(pool, uid, buffer_id, lora_adapter, *args, **kwargs):
        before = buffers(pool)
        result = previous_load(pool, uid, buffer_id, lora_adapter, *args, **kwargs)
        loads.append(
            {
                "uid": str(uid),
                "slot": buffer_id,
                "rank": int(lora_adapter.config.r) if lora_adapter is not None else 0,
                "addresses_stable": buffers(pool) == before,
            }
        )
        write()
        return result

    LoRAMemoryPool.load_lora_weight_to_buffer = load
    previous_forward = ModelRunner._forward_raw

    @wraps(previous_forward)
    def forward(runner, batch, *args, **kwargs):
        result = previous_forward(runner, batch, *args, **kwargs)
        phase = batch.forward_mode.name + (
            "/graph" if result.can_run_graph else "/eager"
        )
        counts[phase] += 1
        write()
        return result

    ModelRunner._forward_raw = forward
    if os.environ.get("LORAFORGE_SGLANG") == "1":
        from loraforge.integrations.sglang import get_backend

        backend = get_backend()
        previous_shrink = backend._shrink

        @wraps(previous_shrink)
        def shrink(self, *args, **kwargs):
            counts[
                "loraforge_shrink_capture"
                if torch.cuda.is_current_stream_capturing()
                else "loraforge_shrink_eager"
            ] += 1
            return previous_shrink(self, *args, **kwargs)

        backend._shrink = shrink
    report["observer_installed"] = True
    write()


# Spawned SGLang workers re-import the entrypoint and inherit the observation path.
if os.environ.get("LORAFORGE_SERVING_AUDIT"):
    install_observer()


def http(url, path, body=None, timeout=240):
    request = urllib.request.Request(
        url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if path == "/health":
                return {"status": response.status}
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"{path}: HTTP {error.code}: {error.read().decode()}"
        ) from error


def normalize(response, tokens):
    meta = response["meta_info"]
    values = meta.get("output_token_logprobs")
    if not values or len(values) != tokens or meta["completion_tokens"] != tokens:
        raise AssertionError(f"incomplete generation metadata: {meta}")
    return {
        "tokens": [row[1] for row in values],
        "logprobs": [row[0] for row in values],
        "text": response.get("text"),
    }


PROTOCOL_VERSION = 2


def observed_difference(left, right):
    """Describe cross-batch drift without claiming identical conditioning."""
    equal = left["tokens"] == right["tokens"]
    delta = None
    if equal:
        deltas = [
            abs(a - b) for a, b in zip(left["logprobs"], right["logprobs"], strict=True)
        ]
        if not deltas or not all(math.isfinite(value) for value in deltas):
            raise AssertionError("invalid or nonfinite output logprobs")
        delta = max(deltas)
    return {"tokens_equal": equal, "max_logprob_delta": delta}


def difference(left, right, tolerance):
    observed = observed_difference(left, right)
    if not observed["tokens_equal"]:
        raise AssertionError("generated token IDs differ")
    delta = observed["max_logprob_delta"]
    if delta > tolerance:
        raise AssertionError(f"logprob difference {delta} exceeds {tolerance}")
    return delta


def compare_reports(results, tolerance):
    """Compare each workload against the same native-eager batch shape."""
    baseline = results["native-eager"]
    expected_protocol = baseline.get("protocol")
    if not expected_protocol or expected_protocol.get("version") != PROTOCOL_VERSION:
        raise AssertionError("unsupported service protocol version")
    if expected_protocol.get("tolerance") != tolerance:
        raise AssertionError("comparison tolerance differs from recorded protocol")
    comparisons = {}
    for mode, report in results.items():
        if not report.get("passed"):
            raise AssertionError("all service modes must pass before comparison")
        if (
            report.get("manifest_sha256") != baseline.get("manifest_sha256")
            or report.get("protocol") != expected_protocol
        ):
            raise AssertionError("service inputs or protocols do not match")
        if set(report["rows"]) != set(baseline["rows"]):
            raise AssertionError("service case sets differ")
        if len(report.get("mixed", [])) != 2:
            raise AssertionError("two mixed-batch outputs required")
        if mode == "native-eager":
            continue
        comparisons[mode] = {
            "single": {
                key: difference(value, report["rows"][key], tolerance)
                for key, value in baseline["rows"].items()
            },
            "mixed": [
                difference(left, right, tolerance)
                for left, right in zip(baseline["mixed"], report["mixed"], strict=True)
            ],
        }
    return comparisons


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
        log_level="info",
    )
    if args.mode.endswith("eager"):
        options["disable_cuda_graph"] = True
    else:
        options["cuda_graph_config"] = {
            "decode": {"backend": "full", "bs": [1, 2], "max_bs": 2}
        }
    config = ServerArgs(**options)
    dump(args.output / "server-args.json", options)
    launch_server(config)


def audit_summary(root, mode):
    counts, loads = Counter(), []
    for path in root.glob("audit-*.json"):
        report = json.loads(path.read_text())
        counts.update(report["counts"])
        loads.extend(report["loads"])
    if not loads or not all(row["addresses_stable"] for row in loads):
        raise AssertionError("native loader missing or buffer addresses changed")
    occupants, reuse, rank_changes = {}, 0, 0
    for row in loads:
        previous = occupants.get(row["slot"])
        if (
            previous
            and row["rank"]
            and previous["rank"]
            and previous["uid"] != row["uid"]
        ):
            reuse += 1
            rank_changes += previous["rank"] != row["rank"]
        occupants[row["slot"]] = row
    if not reuse or not rank_changes:
        raise AssertionError("did not observe actual slot reuse with different ranks")
    graph_calls = counts["DECODE/graph"]
    if mode.endswith("graph") and not graph_calls:
        raise AssertionError("decode graph replay was not observed")
    if mode.endswith("eager") and not counts["DECODE/eager"]:
        raise AssertionError("eager decode execution was not observed")
    if mode.endswith("eager") and graph_calls:
        raise AssertionError("eager validation unexpectedly used decode graphs")
    custom = counts["loraforge_shrink_eager"] + counts["loraforge_shrink_capture"]
    if mode.startswith("loraforge") and not custom:
        raise AssertionError("LoRAForge decode did not execute")
    if mode.startswith("native") and custom:
        raise AssertionError("native baseline unexpectedly used LoRAForge")
    return {
        "counts": dict(counts),
        "load_events": len(loads),
        "slot_reuses": reuse,
        "rank_changes": rank_changes,
    }


def run_mode(args, manifest, mode):
    root = args.output / mode
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "result.json"
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    protocol = {
        "version": PROTOCOL_VERSION,
        "new_tokens": args.new_tokens,
        "tolerance": args.tolerance,
    }
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if (
            saved.get("passed")
            and saved.get("manifest_sha256") == manifest_hash
            and saved.get("protocol") == protocol
        ):
            return saved
    root = root / f"attempt-{time.time_ns()}"
    root.mkdir()
    attempt_report = root / "result.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROBE_")}
    env.update(
        LORAFORGE_SGLANG="1" if mode.startswith("loraforge") else "0",
        SGLANG_PLUGINS="loraforge",
        LORAFORGE_SERVING_AUDIT=str(root),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
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
        "--mode",
        mode,
        "--port",
        str(args.port),
    ]
    report = {
        "mode": mode,
        "manifest_sha256": manifest_hash,
        "protocol": protocol,
        "attempt": str(root),
        "rows": {},
        "passed": False,
    }

    def save_report():
        dump(report_path, report)
        dump(attempt_report, report)

    save_report()
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited {process.returncode}; see {root / 'server.log'}"
                    )
                try:
                    http(url, "/health", timeout=3)
                    break
                except (OSError, RuntimeError):
                    time.sleep(2)
            else:
                raise TimeoutError("server startup timed out")
            report["server_info"] = http(url, "/server_info")

            def generate(ids, name):
                body = dict(
                    input_ids=ids,
                    lora_path=name,
                    return_logprob=True,
                    logprob_start_len=-1,
                    sampling_params=dict(
                        temperature=0, ignore_eos=True, max_new_tokens=args.new_tokens
                    ),
                )
                response = http(url, "/generate", body)
                return (
                    [normalize(row, args.new_tokens) for row in response]
                    if isinstance(response, list)
                    else normalize(response, args.new_tokens)
                )

            routes = [("base", None)] + [
                (a["name"], a["name"]) for a in manifest["adapters"]
            ]
            for cycle in range(2):
                for label, name in routes:
                    for index, ids in enumerate(manifest["prompts"]):
                        value = generate(ids, name)
                        key = f"{label}/{index}"
                        if cycle:
                            difference(report["rows"][key], value, 1e-5)
                        else:
                            report["rows"][key] = value
                        save_report()
            first, second = manifest["adapters"][:2]
            mixed = generate(manifest["prompts"], [first["name"], second["name"]])
            report["mixed"] = mixed
            save_report()
            report["mixed_vs_single_diagnostic"] = [
                observed_difference(report["rows"][f"{adapter['name']}/{index}"], value)
                for index, (adapter, value) in enumerate(
                    zip((first, second), mixed, strict=True)
                )
            ]
            save_report()
            repeated_mixed = generate(
                manifest["prompts"], [first["name"], second["name"]]
            )
            report["mixed_repeat"] = repeated_mixed
            save_report()
            for left, right in zip(mixed, repeated_mixed, strict=True):
                difference(left, right, 1e-5)
            http(url, "/unload_lora_adapter", {"lora_name": first["name"]})
            http(
                url,
                "/load_lora_adapter",
                {"lora_name": "reloaded_first", "lora_path": first["path"]},
            )
            for index, ids in enumerate(manifest["prompts"]):
                difference(
                    report["rows"][f"{first['name']}/{index}"],
                    generate(ids, "reloaded_first"),
                    1e-5,
                )
            if not any(
                report["rows"][f"{a['name']}/0"] != report["rows"]["base/0"]
                for a in manifest["adapters"]
            ):
                raise AssertionError("all adapter outputs identical to the base model")
            report["audit"] = audit_summary(root, mode)
            report["passed"] = True
            save_report()
            print(mode, "passed", report["audit"], flush=True)
        except BaseException as error:
            report["error"] = repr(error)
            save_report()
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "serve", "compare"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="native-eager")
    parser.add_argument("--port", type=int, default=39175)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if args.action == "serve":
        serve(args, manifest)
        return
    modes = ["native-eager", "native-graph", "loraforge-eager", "loraforge-graph"]
    if args.action == "run":
        for mode in modes:
            run_mode(args, manifest, mode)
        print(
            "All service modes completed; run compare on a CPU/login node.", flush=True
        )
        return
    results = {
        mode: json.loads((args.output / mode / "result.json").read_text())
        for mode in modes
    }
    comparisons = compare_reports(results, args.tolerance)
    summary = {
        "passed": True,
        "modes": modes,
        "cases_per_mode": len(results[modes[0]]["rows"]),
        "max_logprob_delta": {
            mode: max(list(rows["single"].values()) + rows["mixed"])
            for mode, rows in comparisons.items()
        },
        "tolerance": args.tolerance,
        "protocol_version": PROTOCOL_VERSION,
        "scope": "matched single/mixed batch shapes and real adapter paging; correctness, not accuracy or throughput",
    }
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
