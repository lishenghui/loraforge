# SPDX-License-Identifier: Apache-2.0
"""Allocated-MI250 rank-four specialization checks against generic fused A+B."""

import argparse
import json
import os
import platform
import random
import statistics
from pathlib import Path

import torch
from qwen35_fused_lora import load_variant
from qwen35_rank4_fused import extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old = load_variant("narrow")
    new = extension()
    kw = dict(dtype=torch.bfloat16, device="cuda")
    report = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", platform.node()),
        state="running",
        cases=[],
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def times(fs):
        graphs = {}
        samples = {n: [] for n in fs}
        for n, f in fs.items():
            for _ in range(3):
                f()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(16):
                    f()
            graphs[n] = g
        for rep in range(3):
            names = list(fs)
            random.Random(731 + rep).shuffle(names)
            for n in names:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                graphs[n].replay()
                s.record()
                for _ in range(8):
                    graphs[n].replay()
                e.record()
                e.synchronize()
                samples[n].append(s.elapsed_time(e) * 1000 / 128)
        return {
            n: dict(median_us=statistics.median(v), samples_us=v)
            for n, v in samples.items()
        }

    for k in (2048, 3584):
        configs = [
            (t, g, False)
            for t in (1, 2, 4, 8, 16, 32)
            for g in sorted(set((1, min(4, t), t)))
        ] + [(8, 4, True)]
        for t, groups, mixed in configs:
            torch.manual_seed(2491)
            slots = groups + 1
            routes = torch.arange(t, device="cuda", dtype=torch.int32) % groups + 1
            if t > 1:
                routes[0] = 0
            ranks = torch.tensor(
                [0] + (list(range(1, 5)) if mixed else [4] * groups),
                device="cuda",
                dtype=torch.int32,
            )
            scales = torch.tensor(
                [0] + [0.75] * groups, device="cuda", dtype=torch.float32
            )
            x = torch.randn(t, k, **kw) * 0.1
            a = torch.randn(slots, 4, k, **kw) * 0.02
            b = torch.randn(slots, 1024, 4, **kw) * 0.02
            base = torch.randn(t, 1024, **kw) * 0.1
            y = base.clone()

            def generic():
                old.fused_ab(x, a, b, y, routes, ranks, scales, 256)
                return y

            def specialized():
                new.update(x, a, b, y, routes, ranks, scales)
                return y

            y.copy_(base)
            ref = generic().clone()
            y.copy_(base)
            got = specialized().clone()
            diff = float((got.float() - ref.float()).abs().max())
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
            graphs = []
            for f in (generic, specialized):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    f()
                graphs.append(graph)
            routes.copy_(routes.flip(0))
            ranks[1:].copy_(ranks[1:].flip(0))
            y.copy_(base)
            graphs[0].replay()
            ref = y.clone()
            y.copy_(base)
            graphs[1].replay()
            got = y.clone()
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
            routes.copy_(routes.flip(0))
            ranks[1:].copy_(ranks[1:].flip(0))
            timing = times(dict(generic=generic, specialized=specialized))
            report["cases"].append(
                dict(
                    input=k,
                    tokens=t,
                    adapters=groups,
                    mixed_rank=mixed,
                    max_abs=diff,
                    changed_metadata_graph_pass=True,
                    timing=timing,
                )
            )
            save()
    policy = {str(k): [] for k in (2048, 3584)}
    for k in (2048, 3584):
        for t in (1, 2, 4, 8, 16, 32):
            cases = [
                c
                for c in report["cases"]
                if c["input"] == k and c["tokens"] == t and not c["mixed_rank"]
            ]
            ratio = min(
                c["timing"]["generic"]["median_us"]
                / c["timing"]["specialized"]["median_us"]
                for c in cases
            )
            if ratio >= 1.01:
                policy[str(k)].append(t)
    report.update(
        state="complete",
        policy=policy,
        selection="At least1percent device-event gain at every sampled adapter count, rank4 capacity; partial ranks1/2/3/4 and route/rank changes require bitwise equality.",
    )
    save()
    print("RANK4_SPEC_MICRO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
