# SPDX-License-Identifier: Apache-2.0
"""MI250 rank-four per-row B-at-SiLU with live graph metadata checks."""

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from qwen35_decode_silu import row_act
from qwen35_fused_lora import load_variant
from sglang.srt.layers.activation import silu_and_mul
from sglang.srt.lora.backend import chunked_backend as native
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from loraforge.ops.rowexpand import direct_shrink, row_expand, rowexpand_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    native.get_lora = lambda: SimpleNamespace(max_lora_chunk_size=16)
    rowexpand_extension()
    wide = load_variant("wide")
    report = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", platform.node()),
        cases=[],
        state="running",
        source_sha256=hashlib.sha256(
            Path(__file__).with_name("qwen35_decode_silu.py").read_bytes()
        ).hexdigest(),
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def times(fs):
        graphs = {}
        samples = {n: [] for n in fs}
        for n, f in fs.items():
            for _ in range(3):
                f()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(16):
                    f()
            graphs[n] = graph
        for rep in range(3):
            names = list(fs)
            random.Random(203 + rep).shuffle(names)
            for n in names:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                graphs[n].replay()
                start.record()
                for _ in range(8):
                    graphs[n].replay()
                end.record()
                end.synchronize()
                samples[n].append(start.elapsed_time(end) * 1000 / 128)
        return {
            n: dict(median_us=statistics.median(v), samples_us=v)
            for n, v in samples.items()
        }

    for t, groups, mixed in [
        (t, g, False) for t in (1, 8, 32) for g in sorted(set((1, min(4, t), t)))
    ] + [(32, 4, True)]:
        torch.manual_seed(9312)
        rr = [i % groups + 1 for i in range(t)]
        if t > 1:
            rr[0] = 0
        ranks = torch.tensor(
            [0] + (list(range(1, 5)) if mixed else [4] * groups),
            device="cuda",
            dtype=torch.int32,
        )
        scales = torch.tensor([0] + [0.75] * groups, device="cuda", dtype=torch.float32)
        backend = native.ChunkedSgmvLoRABackend(groups + 1, torch.device("cuda"), None)
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, batch_size=t, return_logprob=False
        )
        backend.prepare_lora_batch(
            fb, rr, [0] + [4] * groups, [0] + [0.75] * groups, False
        )
        info = backend.batch_info
        routes = torch.tensor(rr, device="cuda", dtype=torch.int32)
        x = torch.randn(t, 1024, **kw) * 0.1
        a = torch.randn(groups + 1, 8, 1024, **kw) * 0.02
        b = torch.randn(groups + 1, 7168, 4, **kw) * 0.02
        base = torch.randn(t, 7168, **kw) * 2
        y = base.clone()
        u = torch.empty(t, 8, **kw)
        out = torch.empty(t, 3584, **kw)
        boff = torch.arange(groups + 1, device="cuda", dtype=torch.int64) * 7168 * 4
        bucket = torch.full((groups + 1,), 4, device="cuda", dtype=torch.int32)
        segments = torch.tensor(info.num_segments, device="cuda", dtype=torch.int32)
        off = torch.tensor([0, 3584, 7168], device="cuda", dtype=torch.int32)

        def shrink():
            direct_shrink(x, a, u, routes, ranks, 2, 4)

        def baseline():
            shrink()
            row_expand(
                u,
                b.flatten(),
                boff,
                bucket,
                y,
                info.seg_indptr,
                info.weight_indices,
                info.permutation,
                ranks,
                scales,
                off,
                segments,
                4,
                blocks=880,
                threads=256,
                row_tile=16,
            )
            silu_and_mul(y, out)
            return out

        # Consumer graph changes routes/ranks directly. A/B reference uses original
        # static segmented routing only for the initial metadata; rebuild later.
        def current():
            if t == 32:
                return baseline()
            wide.wide_ab(x, a, b, y, routes, ranks, scales, 2, 3584, 0)
            silu_and_mul(y, out)
            return out

        y.copy_(base)
        ref = baseline().clone()
        y.copy_(base)
        torch.testing.assert_close(current(), ref, rtol=0, atol=0)
        fs = {"separate": baseline, "current": current}
        checks = {}
        for tile in (64, 128, 256, 512):

            def f(tile=tile):
                shrink()
                return row_act(base, u, b, routes, ranks, scales, tile=tile, out=out)

            got = f().clone()
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                f()
            routes.copy_(routes.flip(0))
            ranks[1:].copy_(ranks[1:].flip(0))
            scales[1:].mul_(0.5)
            graph.replay()
            got = out.clone()
            changed_routes = routes.cpu().tolist()
            backend.prepare_lora_batch(
                fb, changed_routes, [0] + [4] * groups, [0] + [0.75] * groups, False
            )
            changed_info = backend.batch_info
            original_info = info
            info = changed_info
            y.copy_(base)
            changed_ref = baseline().clone()
            info = original_info
            torch.testing.assert_close(got, changed_ref, rtol=0, atol=0)
            routes.copy_(routes.flip(0))
            ranks[1:].copy_(ranks[1:].flip(0))
            scales[1:].mul_(2)
            checks[str(tile)] = dict(
                eager_max_abs=float((f().float() - ref.float()).abs().max()),
                changed_metadata_graph_exact=True,
            )
            fs["row" + str(tile)] = f
        tm = times(fs)
        report["cases"].append(
            dict(tokens=t, adapters=groups, mixed_rank=mixed, checks=checks, timing=tm)
        )
        save()
    report["state"] = "complete"
    save()
    print("DECODE_ROW_ACT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
