# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 dense text: matched synthetic-adapter csgmv/Fleet/base pilot.

Prepare on CPU; run on an allocated GPU. Nonzero random adapters exercise HF
GDN packing and heterogeneous retained ranks, NOT trained-adapter quality or a
compression quality/speed tradeoff. Every arm uses the same prepared inputs.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def prepare(root, model, rank=None):
    if rank not in (None, 4, 8):
        raise ValueError("fixed synthetic rank must be 4 or 8")
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    config = read(model / "config.json")["text_config"]
    assert config["model_type"] == "qwen3_5_text"
    hidden, inter = config["hidden_size"], config["intermediate_size"]
    key = config["linear_key_head_dim"] * config["linear_num_key_heads"]
    value = config["linear_value_head_dim"] * config["linear_num_value_heads"]
    head, heads = config["head_dim"], config["linear_num_value_heads"]
    adapters, histogram = [], Counter()
    physical_rank = 64 if rank is None else rank
    physical_parameters = retained_parameters = rank64_parameters = 0
    for index in range(4):
        path = root / "adapters" / f"a{index}"
        path.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(1937 + index)
        weights, ranks, targets = {}, {}, set()
        for layer, kind in enumerate(config["layer_types"]):
            dims = [
                ("mlp.gate_proj", hidden, inter),
                ("mlp.up_proj", hidden, inter),
                ("mlp.down_proj", inter, hidden),
            ]
            if kind == "linear_attention":
                dims += [
                    ("linear_attn.in_proj_qkv", hidden, 2 * key + value),
                    ("linear_attn.in_proj_z", hidden, value),
                    ("linear_attn.in_proj_b", hidden, heads),
                    ("linear_attn.in_proj_a", hidden, heads),
                    ("linear_attn.out_proj", value, hidden),
                ]
            else:
                dims += [
                    (
                        "self_attn.q_proj",
                        hidden,
                        config["num_attention_heads"] * head * 2,
                    ),
                    ("self_attn.k_proj", hidden, config["num_key_value_heads"] * head),
                    ("self_attn.v_proj", hidden, config["num_key_value_heads"] * head),
                    ("self_attn.o_proj", config["num_attention_heads"] * head, hidden),
                ]
            for pos, (name, nin, nout) in enumerate(dims):
                k = (
                    rank
                    if rank is not None
                    else [8, 16, 32, 64][(layer + pos + index) % 4]
                )
                if name.endswith(("in_proj_a", "in_proj_b")):
                    k = min(k, 8)
                prefix = f"base_model.model.model.language_model.layers.{layer}.{name}"
                a = torch.zeros(physical_rank, nin, dtype=torch.bfloat16)
                b = torch.zeros(nout, physical_rank, dtype=torch.bfloat16)
                a[:k] = torch.randn(k, nin) * 0.02
                b[:, :k] = torch.randn(nout, k) * (0.04 / k**0.5)
                weights[prefix + ".lora_A.weight"] = a
                weights[prefix + ".lora_B.weight"] = b
                ranks[prefix + ".lora_A.weight"] = k
                targets.add(name.rsplit(".", 1)[1])
                histogram[k] += 1
                physical_parameters += physical_rank * (nin + nout)
                retained_parameters += k * (nin + nout)
                rank64_parameters += 64 * (nin + nout)
        save_file(weights, str(path / "adapter_model.safetensors"))
        dump(
            path / "adapter_config.json",
            dict(
                peft_type="LORA",
                task_type="CAUSAL_LM",
                base_model_name_or_path=str(model),
                r=physical_rank,
                lora_alpha=physical_rank,
                lora_dropout=0.0,
                bias="none",
                target_modules=sorted(targets),
                inference_mode=True,
            ),
        )
        dump(
            path / "compression.json",
            dict(
                module_ranks=ranks,
                scope="Synthetic retained factors; no trained adapter was compressed",
            ),
        )
        adapters.append(
            dict(
                name=f"a{index}",
                path=str(path),
                sha256=hashlib.sha256(
                    (path / "adapter_model.safetensors").read_bytes()
                ).hexdigest(),
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(model)
    text = tokenizer.apply_chat_template(
        [
            dict(
                role="user",
                content="List the integers from 1 to 100 in order, separated by commas. "
                "Return only the numbers. " * 8,
            )
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    manifest = dict(
        model=str(model),
        adapters=adapters,
        max_lora_rank=physical_rank,
        fixture=dict(
            kind="mixed_padded" if rank is None else "fixed_compact",
            physical_rank=physical_rank,
            fixed_retained_rank=rank,
            physical_parameters=physical_parameters,
            retained_parameters=retained_parameters,
            rank64_parameters=rank64_parameters,
            retained_fraction_of_rank64=retained_parameters / rank64_parameters,
            scaling=1.0,
        ),
        target_modules=[
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
        ],
        retained_rank_histogram=dict(histogram),
        prompt=tokens[-128:],
        requests=96,
        output_tokens=64,
        concurrencies=[1, 8, 32],
        repeats=3,
        scope="BF16 TP1 text; synthetic nonzero "
        + ("padded" if rank is None else f"compact rank-{rank}")
        + " adapters; no trained compression or quality claim",
    )
    dump(root / "manifest.json", manifest)


def observe(root):
    """Same observer in all arms; no per-token file access during timed work."""
    from sglang.srt.lora.mem_pool import LoRAMemoryPool
    from sglang.srt.model_executor.model_runner import ModelRunner

    counts, batches, loads = Counter(), Counter(), []
    previous_load = LoRAMemoryPool.load_lora_weight_to_buffer

    def load(pool, uid, buffer_id, adapter, *args, **kwargs):
        buffers = [
            v
            for group in (pool.A_buffer, pool.B_buffer)
            for layers in group.values()
            for v in layers
            if v is not None
        ]
        pointers = [v.data_ptr() for v in buffers]
        result = previous_load(pool, uid, buffer_id, adapter, *args, **kwargs)
        loads.append(
            dict(
                slot=buffer_id,
                uid=str(uid),
                fixed_addresses=pointers == [v.data_ptr() for v in buffers],
            )
        )
        return result

    LoRAMemoryPool.load_lora_weight_to_buffer = load
    original = ModelRunner._forward_raw

    def forward(self, batch, *args, **kwargs):
        result = original(self, batch, *args, **kwargs)
        phase = batch.forward_mode.name + (
            "/graph" if result.can_run_graph else "/eager"
        )
        counts[phase] += 1
        batches[f"{phase}/{batch.batch_size}"] += 1
        # Validation asks for a 9-token sentinel; timed requests are 128 tokens.
        if batch.forward_mode.is_extend() and len(batch.input_ids) == 9:
            report = dict(counts=dict(counts), batches=dict(batches), loads=loads)
            manager = getattr(self, "lora_manager", None)
            if manager is not None:
                report["backend"] = type(manager.lora_backend).__name__
                if hasattr(manager.lora_backend, "fleet_snapshot"):
                    report["fleet"] = manager.lora_backend.fleet_snapshot()
            dump(root / f"audit-{os.getpid()}.json", report)
        return result

    ModelRunner._forward_raw = forward


def serve(root, mode, attempt, port):
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    from loraforge.integrations._sglang_qwen35 import install_qwen35_lora

    install_qwen35_lora()
    if mode == "fleet":
        from loraforge.integrations.sglang import install_sglang_backend

        install_sglang_backend()
    observe(attempt)
    m = read(root / "manifest.json")
    options = server_options(m, mode, port)
    dump(attempt / "server-args.json", options)
    launch_server(ServerArgs(**options))


def server_options(m, mode, port):
    """Use the prepared capacity in both backends, including compact tiny ranks."""
    options = dict(
        model_path=m["model"],
        host="127.0.0.1",
        port=port,
        dtype="bfloat16",
        tp_size=1,
        random_seed=0,
        context_length=2048,
        attention_backend="triton",
        disable_radix_cache=True,
        disable_prefill_cuda_graph=True,
        chunked_prefill_size=1024,
        max_prefill_tokens=2048,
        max_total_tokens=16384,
        max_running_requests=32,
        mem_fraction_static=0.70,
        weight_loader_prefetch_checkpoints=True,
        enable_lora=mode != "base",
        log_level="info",
        cuda_graph_config={
            "decode": {"backend": "full", "bs": [1, 2, 4, 8, 16, 32], "max_bs": 32}
        },
    )
    if mode != "base":
        options.update(
            lora_backend="csgmv",
            max_lora_chunk_size=16,
            max_lora_rank=m["max_lora_rank"],
            lora_target_modules=m["target_modules"],
            max_loras_per_batch=5,
            max_loaded_loras=5,
            enable_lora_overlap_loading=False,
            lora_paths=[
                dict(lora_name=a["name"], lora_path=a["path"], pinned=True)
                for a in m["adapters"]
            ],
        )
    return options


def request(url, body=None, path="/generate"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.load(response)


def payload(m, mode, index, length=128, output=64, logprobs=False):
    ids = (m["prompt"] * 5)[-length:]
    result = dict(
        input_ids=ids,
        sampling_params=dict(temperature=0, ignore_eos=True, max_new_tokens=output),
        return_logprob=logprobs,
        logprob_start_len=-1,
    )
    if mode != "base":
        result["lora_path"] = m["adapters"][index % 4]["name"]
    return result


def numerical_check(reference, candidate):
    assert [(r["adapter"], r["length"]) for r in reference["serial"]] == [
        (r["adapter"], r["length"]) for r in candidate["serial"]
    ], "validation cases differ"
    pairs = [
        (a["tokens"], b["tokens"])
        for a, b in zip(reference["serial"], candidate["serial"], strict=True)
    ]
    pairs += list(zip(reference["mixed"], candidate["mixed"], strict=True))
    if "mixed32" in reference or "mixed32" in candidate:
        assert "mixed32" in reference and "mixed32" in candidate, (
            "validation cases differ"
        )
        pairs += list(zip(reference["mixed32"], candidate["mixed32"], strict=True))
    assert all(math.isfinite(x[0]) for a, b in pairs for seq in (a, b) for x in seq), (
        "nonfinite logprob"
    )
    same_tokens = all([x[1] for x in a] == [x[1] for x in b] for a, b in pairs)
    differences = []
    for a, b in pairs:
        for x, y in zip(a, b, strict=True):
            if x[1] != y[1]:
                break
            differences.append(abs(x[0] - y[0]))
    maximum = max(differences, default=None)
    return dict(
        cases=len(pairs),
        same_token_cases=sum([x[1] for x in a] == [x[1] for x in b] for a, b in pairs),
        same_tokens=same_tokens,
        max_common_prefix_logprob_difference=maximum,
        numerical_pass=same_tokens and maximum is not None and maximum <= 0.05,
    )


def timing_allowed(check, *, allow_logprob_drift=False):
    """A timing opt-in must never relabel the strict numerical result as passed."""
    if allow_logprob_drift:
        return (
            check["same_tokens"]
            and check["max_common_prefix_logprob_difference"] is not None
        )
    return check["numerical_pass"]


def run(
    root, modes, reference=None, validation_only=False, *, allow_logprob_drift=False
):
    m = read(root / "manifest.json")
    for mode in modes:
        attempt = (
            root
            / "results"
            / f"{mode}-{os.environ.get('SLURM_JOB_ID', 'local')}-{time.time_ns()}"
        )
        attempt.mkdir(parents=True)
        status = dict(mode=mode, state="starting", attempt=str(attempt))
        dump(root / f"{mode}-latest.json", status)
        env = dict(
            os.environ,
            LORAFORGE_SGLANG_QWEN35="1",
            LORAFORGE_SGLANG="1" if mode == "fleet" else "0",
            LORAFORGE_SGLANG_POLICY="fleet",
            LORAFORGE_SGLANG_MANIFEST=str(root / "manifest.json"),
            SGLANG_PLUGINS="loraforge",
            HYBRID_AUDIT_DIR=str(attempt),
        )
        port = 47653
        url, server = f"http://127.0.0.1:{port}", None
        try:
            with (attempt / "server.log").open("w") as log:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "serve",
                        str(root),
                        "--mode",
                        mode,
                        "--attempt",
                        str(attempt),
                        "--port",
                        str(port),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                for _ in range(120):
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}")
                    try:
                        info = request(url, path="/server_info")
                        break
                    except Exception:
                        time.sleep(3)
                else:
                    raise TimeoutError("server startup exceeded six minutes")
                dump(attempt / "server-info.json", info)
                status["state"] = "validation"
                dump(root / f"{mode}-latest.json", status)
                cases = []
                for index in range(4):
                    for length in (17, 65, 129):
                        body = payload(m, mode, index, length, 8, True)
                        first = request(url, body)
                        request(url, payload(m, mode, index + 1, length, 8, True))
                        again = request(url, body)
                        a, b = (
                            first["meta_info"]["output_token_logprobs"],
                            again["meta_info"]["output_token_logprobs"],
                        )
                        assert a == b, "same-request state was not reset"
                        cases.append(dict(adapter=index, length=length, tokens=a))
                correctness = dict(serial=cases)
                for count, key in ((8, "mixed"), (32, "mixed32")):
                    mixed = [payload(m, mode, i, 65, 8, True) for i in range(count)]
                    batched = dict(
                        mixed[0], input_ids=[body["input_ids"] for body in mixed]
                    )
                    if mode != "base":
                        batched["lora_path"] = [body["lora_path"] for body in mixed]
                    outputs = request(url, batched)
                    repeat = request(url, batched)
                    values = [o["meta_info"]["output_token_logprobs"] for o in outputs]
                    assert values == [
                        o["meta_info"]["output_token_logprobs"] for o in repeat
                    ], "mixed batch was not repeatable"
                    correctness[key] = values
                dump(attempt / "correctness.json", correctness)
                if reference is not None and mode == "fleet":
                    check = numerical_check(
                        read(reference), read(attempt / "correctness.json")
                    )
                    check["allow_logprob_drift_for_timing"] = allow_logprob_drift
                    check["timing_allowed"] = timing_allowed(
                        check, allow_logprob_drift=allow_logprob_drift
                    )
                    dump(attempt / "native-comparison.json", check)
                    assert check["timing_allowed"], check
                if validation_only:
                    request(url, payload(m, mode, 0, 9, 1))
                    status["state"] = "validated"
                    continue
                measurements = []
                for concurrency in m["concurrencies"]:

                    def measure(count):
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=concurrency
                        ) as executor:
                            start = time.perf_counter()
                            outputs = list(
                                executor.map(
                                    lambda i: request(url, payload(m, mode, i)),
                                    range(count),
                                )
                            )
                            elapsed = time.perf_counter() - start
                        assert all(
                            o["meta_info"]["completion_tokens"] == m["output_tokens"]
                            for o in outputs
                        )
                        return dict(
                            concurrency=concurrency,
                            seconds=elapsed,
                            requests=count,
                            output_tokens=count * m["output_tokens"],
                            tokens_per_second=count * m["output_tokens"] / elapsed,
                        )

                    status.update(state="measuring", concurrency=concurrency)
                    dump(root / f"{mode}-latest.json", status)
                    measure(32)
                    count = m.get("requests_at_concurrency", {}).get(
                        str(concurrency), m["requests"]
                    )
                    for repeat in range(m["repeats"]):
                        row = dict(measure(count), repeat=repeat)
                        measurements.append(row)
                        dump(attempt / "measurements.json", measurements)
                        print(mode, row, flush=True)
                request(url, payload(m, mode, 0, 9, 1))
                status["state"] = "complete"
        except BaseException as exc:
            status.update(state="failed", error=repr(exc))
            raise
        finally:
            dump(root / f"{mode}-latest.json", status)
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def compare(root, modes=("native", "fleet", "base")):
    if not {"native", "fleet"}.issubset(modes):
        raise ValueError("comparison requires native and fleet")
    reports = {mode: read(root / f"{mode}-latest.json") for mode in modes}
    assert all(r["state"] == "complete" for r in reports.values())
    # Attempt names are unique within this result bundle. Resolve beneath root
    # so a copied immutable bundle can be compared on a CPU-only machine too.
    paths = {k: root / "results" / Path(v["attempt"]).name for k, v in reports.items()}
    native, fleet = (read(paths[k] / "correctness.json") for k in ("native", "fleet"))
    numerics = numerical_check(native, fleet)
    rows = []
    for concurrency in read(root / "manifest.json")["concurrencies"]:
        medians = {
            mode: statistics.median(
                r["tokens_per_second"]
                for r in read(path / "measurements.json")
                if r["concurrency"] == concurrency
            )
            for mode, path in paths.items()
        }
        rows.append(
            dict(
                concurrency=concurrency,
                tokens_per_second=medians,
                fleet_over_native=medians["fleet"] / medians["native"],
                **(
                    dict(fleet_over_base=medians["fleet"] / medians["base"])
                    if "base" in medians
                    else {}
                ),
            )
        )
    adapter_effect_observed = None
    if "base" in paths:
        base = read(paths["base"] / "correctness.json")
        adapter_effect_observed = any(
            a["tokens"] != b["tokens"]
            for a, b in zip(native["serial"], base["serial"], strict=True)
        )
    audits = {
        mode: [read(p) for p in path.glob("audit-*.json")]
        for mode, path in paths.items()
    }
    audit_pass = all(
        rows and any(row["counts"].get("DECODE/graph", 0) > 0 for row in rows)
        for rows in audits.values()
    )
    for mode in ("native", "fleet"):
        audit_pass &= any(row["loads"] for row in audits[mode]) and all(
            load["fixed_addresses"] for row in audits[mode] for load in row["loads"]
        )
    audit_pass &= any(
        row.get("fleet", {})
        .get("python_capture_eager_calls", {})
        .get("rowexpand_decode", 0)
        > 0
        for row in audits["fleet"]
    )
    native_args = read(paths["native"] / "server-args.json")
    fleet_args = read(paths["fleet"] / "server-args.json")
    matched_server_options = (
        native_args == fleet_args
        and native_args["max_lora_rank"]
        == read(root / "manifest.json")["max_lora_rank"]
    )
    audit_pass &= matched_server_options
    report = dict(
        scope=read(root / "manifest.json")["scope"],
        **numerics,
        adapter_effect_observed=adapter_effect_observed,
        execution_audit_pass=audit_pass,
        matched_server_options=matched_server_options,
        max_lora_rank=native_args["max_lora_rank"],
        performance=rows,
        attempts=reports,
    )
    dump(root / "comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__mp_main__" and os.environ.get("HYBRID_AUDIT_DIR"):
    # SGLang workers spawn by re-importing this executable.
    from loraforge.integrations.sglang import install

    install()
    observe(Path(os.environ["HYBRID_AUDIT_DIR"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "serve", "run", "compare"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--rank",
        type=int,
        choices=[4, 8],
        help="Prepare compact fixed-rank synthetic adapters (default: mixed rank-64 fixture)",
    )
    parser.add_argument("--mode", choices=["native", "fleet", "base"])
    parser.add_argument("--modes", nargs="+", default=["native", "fleet", "base"])
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--port", type=int, default=47653)
    parser.add_argument(
        "--reference", type=Path, help="Require native parity before timing"
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--allow-logprob-drift",
        action="store_true",
        help="Time matching token sequences even if the 0.05 logprob gate fails; record both results",
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.action == "prepare":
        prepare(args.root.resolve(), args.model.resolve(), args.rank)
    elif args.action == "serve":
        serve(args.root, args.mode, args.attempt, args.port)
    elif args.action == "run":
        run(
            args.root,
            args.modes,
            args.reference,
            args.validation_only,
            allow_logprob_drift=args.allow_logprob_drift,
        )
    else:
        compare(args.root, args.modes)
