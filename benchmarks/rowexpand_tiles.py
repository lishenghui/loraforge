# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 B-only work decomposition, with fixed A metadata and exact checks.

Run on one allocated GPU. Warm and flushed operator timings are separate;
neither is a full-model speedup. Each slot owns a distinct checkpoint region,
including slots with equal ranks. Native B receives the same effective ranks.
"""

import argparse
import copy
import json
import os
import random
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from types import SimpleNamespace

    import torch
    from sglang.srt.lora.backend import chunked_backend as native
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from loraforge.ops.rowexpand import row_expand, rowexpand_extension
    from loraforge.runtime.retained import RetainedLayout

    native.get_lora = lambda: SimpleNamespace(max_lora_chunk_size=16)
    backend = native.ChunkedSgmvLoRABackend(5, torch.device("cuda:0"), None)
    backend.init_cuda_graph_batch_info(64, 1)
    rowexpand_extension()
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    kw = dict(device="cuda", dtype=torch.bfloat16)
    shapes = [
        ("in_proj_qkvz", [2048] * 4, [16, 32, 64, 64]),
        ("gate_up_proj", [3584] * 2, [16, 32, 64, 64]),
        ("out_proj", [1024], [8, 16, 32, 64]),
        ("qkv_proj", [4096, 512, 512], [32, 64, 64, 64]),
    ]
    rows = []
    report = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME"),
        device=str(torch.cuda.get_device_properties(0)),
        torch=torch.__version__,
        hip=torch.version.hip,
        protocol="32 shape/batch/rank cases; randomized variant order in three rounds. Warm: 160 calls. Flushed: median of three single calls after 128MiB flush/reset outside timing. A metadata unchanged; row_tile splits inside B only.",
        scope="B-only cache-regime study, not serving throughput or model quality",
        state="running",
        rows=rows,
    )

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(args.output)

    def capture(fn, count):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(count):
                fn()
        return graph

    for name, widths, mixed_ranks in shapes:
        cases = [(1, [r] * 4, [1]) for r in (8, 16, 32, 64)]
        cases += [(n, mixed_ranks, [1, 2, 3, 4]) for n in (8, 32, 64)]
        cases += [(32, [32] * 4, [1])]
        for tokens, retained, pattern in cases:
            torch.manual_seed(411)
            slices, width = len(widths), sum(widths)
            a = torch.randn(5, slices * 64, 64, **kw) / 20
            b = torch.randn(5, width, 64, **kw) / 20
            catalog = {
                f"slot{slot}": {f"model.layers.0.block.{name}.lora_A.weight": rank}
                for slot, rank in enumerate(retained, 1)
            }
            layout = RetainedLayout(a, b, name, 0, catalog)
            layout.load(None, 0)
            for slot in range(1, 5):
                layout.load(f"slot{slot}", slot, 64)
            route = (pattern * tokens)[:tokens]
            fb = SimpleNamespace(
                forward_mode=ForwardMode.DECODE, batch_size=tokens, return_logprob=False
            )
            backend.prepare_lora_batch(
                fb, route, [0, 64, 64, 64, 64], [0, 1, 0.75, 1.25, 0.5], True
            )
            info = copy.copy(backend.batch_info)
            info.lora_ranks = layout.rank
            nseg = torch.tensor(info.num_segments, device="cuda", dtype=torch.int32)
            cpu_offsets = [0]
            for n in widths:
                cpu_offsets.append(cpu_offsets[-1] + n)
            cpu_offsets = torch.tensor(cpu_offsets, dtype=torch.int32)
            offsets = cpu_offsets.cuda()
            z = torch.randn(tokens, slices * 64, **kw) / 10
            base = torch.randn(tokens, width, **kw) / 10
            y = base.clone()

            def custom(tile, threads):
                return row_expand(
                    z,
                    layout.store,
                    layout.boff,
                    layout.bucket,
                    y,
                    info.seg_indptr,
                    info.weight_indices,
                    info.permutation,
                    layout.rank,
                    info.scalings,
                    offsets,
                    nseg,
                    64,
                    row_tile=tile,
                    threads=threads,
                )

            fns = {
                f"t{tile}_c{threads}": lambda t=tile, c=threads: custom(t, c)
                for tile in (2, 4, 8, 16)
                for threads in (128, 256)
            }
            fns["native_rank"] = lambda: native.chunked_sgmv_lora_expand_forward(
                z, b, info, offsets, max(widths), y, slice_offsets_cpu=cpu_offsets
            )
            y.copy_(base)
            reference = custom(16, 256).clone()
            checks, graphs, samples = {}, {}, {}
            for label, fn in fns.items():
                y.copy_(base)
                actual = fn()
                error = actual.float() - reference.float()
                exact = torch.equal(actual, reference)
                relative = float(
                    error.norm() / reference.float().norm().clamp_min(1e-8)
                )
                peak = float(error.abs().max())
                assert torch.isfinite(actual).all()
                if label != "native_rank":
                    assert exact, (name, tokens, retained, label, peak)
                else:
                    assert relative < 0.01 and peak < 0.025 * max(
                        0.1, float(reference.abs().max())
                    )
                checks[label] = dict(
                    exact_to_row16=exact, relative_l2=relative, max_abs=peak
                )
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                graphs[label] = (capture(fn, 1), capture(fn, 16))
                samples[label] = dict(warm=[], flushed=[])
            labels = list(fns)
            for repeat in range(3):
                random.Random(912 + repeat).shuffle(labels)
                for label in labels:
                    single, many = graphs[label]
                    y.copy_(base)
                    many.replay()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(10):
                        many.replay()
                    end.record()
                    end.synchronize()
                    samples[label]["warm"].append(start.elapsed_time(end) * 1000 / 160)
                    cold = []
                    for _ in range(3):
                        flush.fill_(repeat)
                        y.copy_(base)
                        start.record()
                        single.replay()
                        end.record()
                        end.synchronize()
                        cold.append(start.elapsed_time(end) * 1000)
                    samples[label]["flushed"].append(statistics.median(cold))
            timing = {
                label: {
                    cache: dict(median_us=statistics.median(values), samples_us=values)
                    for cache, values in caches.items()
                }
                for label, caches in samples.items()
            }
            rows.append(
                dict(
                    projection=name,
                    widths=widths,
                    tokens=tokens,
                    ranks=retained,
                    route_pattern=pattern,
                    segments=info.num_segments,
                    checks=checks,
                    timing=timing,
                )
            )
            save()
            print(
                name,
                tokens,
                retained,
                pattern,
                {k: round(v["warm"]["median_us"], 2) for k, v in timing.items()},
                flush=True,
            )
    report["state"] = "complete"
    save()


if __name__ == "__main__":
    main()
