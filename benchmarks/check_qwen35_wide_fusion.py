# SPDX-License-Identifier: Apache-2.0
"""Allocated-MI250 operator screen for wide A+B and GDN A+B+unpack.

Run with the pinned SGLang 0.5.22 environment. This process-local benchmark
constructs synthetic factors and routing; it does not launch a model service.
Prebuild load_variant("wide") and RowExpand on CPU before GPU allocation.
"""

import argparse
import json
import os
import platform
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from qwen35_fused_lora import load_variant
from qwen35_gdn_consumer import kernel as consumer_kernel
from sglang.srt.lora.backend import chunked_backend as native
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from loraforge.ops.rowexpand import direct_shrink, row_expand, rowexpand_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ex = load_variant("wide")
    rowexpand_extension()
    kw = dict(device="cuda", dtype=torch.bfloat16)
    native.get_lora = lambda: SimpleNamespace(max_lora_chunk_size=16)
    report = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", platform.node()),
        torch=torch.__version__,
        hip=torch.version.hip,
        device=torch.cuda.get_device_name(),
        wide=[],
        gdn=[],
        state="running",
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

    def setup(r, t, groups):
        rr = [i % groups + 1 for i in range(t)]
        if t > 1:
            rr[0] = 0
        backend = native.ChunkedSgmvLoRABackend(groups + 1, torch.device("cuda"), None)
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=t,
            extend_num_tokens=t,
            extend_seq_lens_cpu=[1] * t,
            extend_seq_lens=torch.ones(t, device="cuda", dtype=torch.int32),
            return_logprob=False,
        )
        backend.prepare_lora_batch(
            fb, rr, [0] + [r] * groups, [0] + [0.75] * groups, False
        )
        return backend.batch_info, torch.tensor(rr, device="cuda", dtype=torch.int32)

    for slices, n, offsets in (
        (2, 7168, [0, 3584, 7168]),
        (3, 5120, [0, 4096, 4608, 5120]),
    ):
        for r in (4, 8):
            for t in (1, 2, 4, 8, 16, 32):
                for groups in sorted(set((1, min(4, t), t))):
                    torch.manual_seed(1447)
                    info, routes = setup(r, t, groups)
                    slots = groups + 1
                    a = torch.randn(slots, slices * r, 1024, **kw) * 0.02
                    b = torch.randn(slots, n, r, **kw) * 0.02
                    x = torch.randn(t, 1024, **kw) * 0.1
                    u = torch.zeros(t, slices * r, **kw)
                    base = torch.randn(t, n, **kw) * 0.1
                    y = base.clone()
                    boff = torch.arange(slots, device="cuda", dtype=torch.int64) * n * r
                    bucket = torch.full((slots,), r, device="cuda", dtype=torch.int32)
                    off = torch.tensor(offsets, device="cuda", dtype=torch.int32)
                    segments = torch.tensor(
                        info.num_segments, device="cuda", dtype=torch.int32
                    )

                    def separate():
                        direct_shrink(x, a, u, routes, info.lora_ranks, slices, r)
                        return row_expand(
                            u,
                            b.flatten(),
                            boff,
                            bucket,
                            y,
                            info.seg_indptr,
                            info.weight_indices,
                            info.permutation,
                            info.lora_ranks,
                            info.scalings,
                            off,
                            segments,
                            r,
                            blocks=880,
                            threads=256,
                            row_tile=4 if slices == 3 and t == 32 else 16,
                        )

                    def fused():
                        ex.wide_ab(
                            x,
                            a,
                            b,
                            y,
                            routes,
                            info.lora_ranks,
                            info.scalings,
                            slices,
                            offsets[1],
                            offsets[2] if slices == 3 else 0,
                        )
                        return y

                    y.copy_(base)
                    ref = separate().clone()
                    y.copy_(base)
                    got = fused().clone()
                    torch.testing.assert_close(got, ref, rtol=0.03, atol=0.002)
                    diff = float((got.float() - ref.float()).abs().max())
                    tm = times(dict(separate=separate, fused=fused))
                    report["wide"].append(
                        dict(
                            slices=slices,
                            output=n,
                            rank=r,
                            tokens=t,
                            adapters=groups,
                            max_abs=diff,
                            timing=tm,
                        )
                    )
                    save()
    for r in (4, 8):
        for t in (1, 2, 4, 8, 16, 32):
            for groups in sorted(set((1, min(4, t), t))):
                torch.manual_seed(2337)
                info, routes = setup(r, t, groups)
                slots = groups + 1
                x = torch.randn(t, 1024, **kw) * 0.1
                aq = torch.randn(slots, 4 * r, 1024, **kw) * 0.02
                ab = torch.randn(slots, 2 * r, 1024, **kw) * 0.02
                bq = torch.randn(slots, 8192, r, **kw) * 0.02
                bb = torch.randn(slots, 32, r, **kw) * 0.02
                base = torch.randn(t, 8224, **kw) * 0.1
                q, ba = base[:, :8192], base[:, 8192:]
                u = torch.zeros(t, 4 * r, **kw)
                v = torch.zeros(t, 2 * r, **kw)
                out = torch.empty(t, 6144, **kw)
                z = torch.empty(t, 16, 128, **kw)
                bo = torch.empty(t, 16, **kw)
                ao = torch.empty_like(bo)

                def separate():
                    direct_shrink(x, aq, u, routes, info.lora_ranks, 4, r)
                    direct_shrink(x, ab, v, routes, info.lora_ranks, 2, r)
                    consumer_kernel[(t, 33)](
                        q,
                        ba,
                        u,
                        v,
                        bq,
                        bb,
                        routes,
                        info.lora_ranks,
                        info.lora_ranks,
                        info.scalings,
                        out,
                        z,
                        bo,
                        ao,
                        q.stride(0),
                        ba.stride(0),
                        r,
                        256,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                    return out, z, bo, ao

                def fused():
                    ex.gdn_ab_unpack(
                        x,
                        aq,
                        ab,
                        bq,
                        bb,
                        q,
                        ba,
                        out,
                        z,
                        bo,
                        ao,
                        routes,
                        info.lora_ranks,
                        info.lora_ranks,
                        info.scalings,
                    )
                    return out, z, bo, ao

                ref = tuple(a.clone() for a in separate())
                got = fused()
                diff = max(
                    float((a.float() - b.float()).abs().max()) for a, b in zip(ref, got)
                )
                for a, b in zip(ref, got):
                    torch.testing.assert_close(a, b, rtol=0.03, atol=0.002)
                # Replay a one-call captured kernel after route changes, including null row.
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    captured = fused()
                routes.copy_(routes.flip(0))
                ref = tuple(a.clone() for a in separate())
                g.replay()
                for a, b in zip(ref, captured):
                    torch.testing.assert_close(a, b, rtol=0.03, atol=0.002)
                # Restore the segment-consistent routes before the timing baseline.
                routes.copy_(routes.flip(0))
                tm = times(dict(separate=separate, fused=fused))
                report["gdn"].append(
                    dict(
                        rank=r,
                        tokens=t,
                        adapters=groups,
                        max_abs=diff,
                        changed_route_graph_pass=True,
                        timing=tm,
                    )
                )
                save()
    report["state"] = "complete"
    save()
    print("WIDE_MICRO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
