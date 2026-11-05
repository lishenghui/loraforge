# SPDX-License-Identifier: Apache-2.0
"""Same-device Qwen3.5 study: stage/peak VRAM, unprofiled throughput, GPU traces.

ROOT/study.json supplies arms with manifest paths, BF16/NF4 kind, native/fleet/
base mode and an explicit worker environment. Existing private research hooks
are not installed by this runner. Shared framework files remain unchanged.
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import qwen35_hybrid as q


def post_profile(url, body):
    request = urllib.request.Request(
        url + "/start_profile",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read().decode()


def batched(m, mode, count, output=64, *, logprobs=False):
    bodies = [q.payload(m, mode, i, 128, output, logprobs) for i in range(count)]
    body = dict(bodies[0], input_ids=[v["input_ids"] for v in bodies])
    if mode != "base":
        body["lora_path"] = [v["lora_path"] for v in bodies]
    return body


def control(attempt, url, m, mode, label, action):
    q.dump(
        attempt / "resource-control.json",
        dict(id=time.time_ns(), label=label, action=action),
    )
    q.request(url, q.payload(m, mode, 0, 9, 1))


def run(root):
    study = q.read(root / "study.json")
    for arm in study["arms"]:
        name, mode = arm["name"], arm["mode"]
        directory = root / name
        directory.mkdir()
        m = q.read(Path(arm["manifest"]))
        m.update(
            model=arm["model"],
            requests_at_concurrency={"1": 8, "8": 32, "32": 96},
            repeats=3,
        )
        q.dump(directory / "manifest.json", m)
        attempt = (
            directory
            / f"attempt-{os.environ.get('SLURM_JOB_ID', 'local')}-{time.time_ns()}"
        )
        attempt.mkdir()
        q.dump(
            root / "status.json", dict(arm=name, state="starting", attempt=str(attempt))
        )
        env = dict(
            os.environ,
            **arm["environment"],
            LF_RESOURCE_AUDIT="1",
            LORAFORGE_SGLANG_QWEN35="1",
            LORAFORGE_SGLANG="1" if mode == "fleet" else "0",
            LORAFORGE_SGLANG_POLICY="fleet",
            LORAFORGE_SGLANG_MANIFEST=str(directory / "manifest.json"),
            SGLANG_PLUGINS="loraforge",
            HYBRID_AUDIT_DIR=str(attempt),
        )
        server = None
        url = "http://127.0.0.1:47653"
        try:
            with (attempt / "server.log").open("w") as log:
                server = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "serve",
                        str(directory),
                        "--mode",
                        mode,
                        "--attempt",
                        str(attempt),
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
                        info = q.request(url, path="/server_info")
                        break
                    except Exception:
                        time.sleep(3)
                else:
                    raise TimeoutError("startup exceeded six minutes")
                q.dump(attempt / "server-info.json", info)
                correctness = dict(serial=[])
                for i in range(4):
                    body = q.payload(m, mode, i, 17, 8, True)
                    a, b = q.request(url, body), q.request(url, body)
                    values = a["meta_info"]["output_token_logprobs"]
                    assert values == b["meta_info"]["output_token_logprobs"]
                    correctness["serial"].append(
                        dict(adapter=i, length=17, tokens=values)
                    )
                for count, key in [(8, "mixed"), (32, "long32")]:
                    body = batched(m, mode, count, logprobs=True)
                    a, b = q.request(url, body), q.request(url, body)
                    values = [x["meta_info"]["output_token_logprobs"] for x in a]
                    assert values == [
                        x["meta_info"]["output_token_logprobs"] for x in b
                    ]
                    correctness[key] = values
                q.numerical_check(correctness, correctness)
                q.dump(attempt / "correctness.json", correctness)
                if arm.get("expect_gdn_shared_storage", False):
                    captured = [
                        stage["components"]
                        for path in attempt.glob("resource-audit-*.json")
                        for stage in q.read(path)["stages"]
                        if stage["label"] == "graphs_captured"
                    ]
                    assert captured and all(
                        row["gdn_packed_base_bytes"] == 303169536
                        and row["gdn_extra_storage_bytes"] == 0
                        for row in captured
                    ), "shared GDN storage did not take effect before graph capture"
                if arm.get("expect_nf4_byte_pairs", False):
                    checked = [
                        q.read(path) for path in attempt.glob("nf4-linear-audit-*.json")
                    ]
                    assert checked and all(
                        row["byte_pairs"] and row["pair_exact_checks"] == 96
                        for row in checked
                    ), "NF4 byte-pair path did not verify all 96 selected projections"
                measurements = []

                def measure(concurrency, count):
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=concurrency
                    ) as executor:
                        begin = time.perf_counter()
                        values = list(
                            executor.map(
                                lambda i: q.request(url, q.payload(m, mode, i)),
                                range(count),
                            )
                        )
                        seconds = time.perf_counter() - begin
                    assert all(
                        v["meta_info"]["completion_tokens"] == 64 for v in values
                    )
                    return dict(
                        concurrency=concurrency,
                        requests=count,
                        output_tokens=count * 64,
                        seconds=seconds,
                        tokens_per_second=count * 64 / seconds,
                    )

                q.dump(
                    root / "status.json",
                    dict(arm=name, state="measuring", attempt=str(attempt)),
                )
                for c in (1, 8, 32):
                    measure(c, min(c, 32))
                    control(attempt, url, m, mode, f"C{c}-begin", "reset")
                    for repeat in range(study.get("repeats", 3)):
                        row = dict(
                            measure(c, m["requests_at_concurrency"][str(c)]),
                            repeat=repeat,
                        )
                        measurements.append(row)
                        q.dump(attempt / "measurements.json", measurements)
                        print("MEASURE", name, row, flush=True)
                    control(attempt, url, m, mode, f"C{c}-end", "snapshot")
                q.dump(
                    root / "status.json",
                    dict(arm=name, state="profiling", attempt=str(attempt)),
                )
                for c in study.get("profile_concurrencies", (1, 8, 32)):
                    target = attempt / f"profile-c{c}"
                    target.mkdir()
                    q.request(url, batched(m, mode, c, logprobs=False))
                    body = dict(
                        output_dir=str(target),
                        profile_id=f"{name}-c{c}",
                        profile_prefix="qwen35-resources",
                        activities=["CPU", "GPU"],
                        with_stack=False,
                        record_shapes=False,
                        profile_by_stage=False,
                        detailed_annotations=False,
                        num_steps=12,
                        start_step=1,
                    )
                    q.dump(target / "request.json", body)
                    (target / "response.txt").write_text(post_profile(url, body))
                    q.request(url, batched(m, mode, c, logprobs=False))
                    for _ in range(60):
                        if list(target.glob("*.trace.json*")):
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError(f"profile missing at C{c}")
                q.request(url, q.payload(m, mode, 0, 9, 1))
                resources = [q.read(p) for p in attempt.glob("resource-audit-*.json")]
                audits = [q.read(p) for p in attempt.glob("audit-*.json")]
                models = [q.read(p) for p in attempt.glob("model-audit-*.json")]
                assert resources and models and audits
                assert all(
                    any(v["label"] == "graphs_captured" for v in a["stages"])
                    and len(a["windows"]) == 6
                    for a in resources
                )
                assert all(
                    len(a["checked_checkpoint_storage"]) == 114
                    and all(
                        x["exact"] for x in a["checked_checkpoint_storage"].values()
                    )
                    for a in models
                )
                assert all(
                    a["counts"].get("DECODE/graph", 0) > 0
                    and a["counts"].get("EXTEND/graph", 0) > 0
                    for a in audits
                )
                row = dict(
                    name=name,
                    kind=arm["kind"],
                    mode=mode,
                    attempt=str(attempt),
                    measurements=measurements,
                    correctness=correctness,
                    resources=resources,
                    model_audit=models,
                    audit=audits,
                    server_args=q.read(attempt / "server-args.json"),
                )
                q.dump(root / f"{name}-summary.json", row)
                print("ARM_COMPLETE", name, flush=True)
        finally:
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    q.dump(
        root / "status.json", dict(state="complete", job=os.environ.get("SLURM_JOB_ID"))
    )
    print("RESOURCE_STUDY_COMPLETE", flush=True)


def options():
    old = q.server_options

    def configured(m, mode, port):
        out = old(m, mode, port)
        out["max_mamba_cache_size"] = 64
        return out

    q.server_options = configured


def worker():
    from qwen35_resource_audit import install

    install()
    q.install_packed_experiment()
    q.install_nf4_experiment()
    from loraforge.integrations.sglang import install as plugin

    plugin()
    q.install_cold_prefill_experiment()
    q.observe(Path(os.environ["HYBRID_AUDIT_DIR"]))


if __name__ == "__mp_main__" and os.environ.get("LF_RESOURCE_AUDIT") == "1":
    worker()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "serve"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args.root.resolve())
    else:
        from qwen35_resource_audit import install

        install()
        options()
        q.serve(args.root.resolve(), args.mode, args.attempt, 47653)
