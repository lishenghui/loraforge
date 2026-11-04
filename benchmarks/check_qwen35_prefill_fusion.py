# SPDX-License-Identifier: Apache-2.0
"""MI250 cap4 operator screen: segmented A+B and B-at-SiLU, not service ratios."""

import argparse
import json
import os
import platform
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from qwen35_prefill_fused import expand_act, segmented_ab
from sglang.srt.layers.activation import SiluAndMul, silu_and_mul
from sglang.srt.lora.backend import chunked_backend as native
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from loraforge.integrations._sglang_fleet import FleetPolicy
from loraforge.integrations._sglang_rank_aware import shrink
from loraforge.ops.rowexpand import direct_shrink, row_expand, rowexpand_extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    native.get_lora = lambda: SimpleNamespace(max_lora_chunk_size=16)
    rowexpand_extension()
    act = SiluAndMul()
    policy = FleetPolicy(hybrid=True, hybrid_tiny=True)
    report = dict(
        job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", platform.node()),
        activation=[],
        prefill=[],
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
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(16):
                    f()
            graphs[n] = graph
        for rep in range(3):
            names = list(fs)
            random.Random(918 + rep).shuffle(names)
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

    def setup(t, groups, k, n, slices, offsets):
        torch.manual_seed(4198)
        prefill = t > 32
        reqs = 8 if prefill else t
        seq = t // reqs
        rr = [i % groups + 1 for i in range(reqs)]
        rr[0] = 0
        backend = native.ChunkedSgmvLoRABackend(groups + 1, torch.device("cuda"), None)
        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
            batch_size=reqs,
            extend_num_tokens=t,
            extend_seq_lens_cpu=[seq] * reqs,
            extend_seq_lens=torch.full((reqs,), seq, device="cuda", dtype=torch.int32),
            return_logprob=False,
        )
        backend.prepare_lora_batch(
            fb, rr, [0] + [4] * groups, [0] + [0.75] * groups, False
        )
        info = backend.batch_info
        routes = torch.tensor(
            [g for g in rr for _ in range(seq)], device="cuda", dtype=torch.int32
        )
        x = torch.randn(t, k, **kw) * 0.1
        a = torch.randn(groups + 1, slices * 4, k, **kw) * 0.02
        b = torch.randn(groups + 1, n, 4, **kw) * 0.02
        base = torch.randn(t, n, **kw) * 0.1
        y = base.clone()
        u = torch.zeros(t, slices * 4, **kw)
        off_cpu = torch.tensor(offsets, dtype=torch.int32)
        off = off_cpu.cuda()
        boff = torch.arange(groups + 1, device="cuda", dtype=torch.int64) * n * 4
        bucket = torch.full((groups + 1,), 4, device="cuda", dtype=torch.int32)
        segments = torch.tensor(info.num_segments, device="cuda", dtype=torch.int32)
        return locals()

    def old_a(d):
        if d["prefill"]:
            d["u"] = shrink(d["x"], d["a"], d["info"], d["slices"])
        else:
            direct_shrink(
                d["x"],
                d["a"],
                d["u"],
                d["routes"],
                d["info"].lora_ranks,
                d["slices"],
                4,
            )

    def old_b(d, name):
        info = d["info"]
        cfg = policy.select(
            name,
            "prefill" if d["prefill"] else "decode",
            d["t"],
            input_width=d["k"],
            output_width=d["n"],
            rank_capacity=4,
            slices=d["slices"],
        )
        if cfg["b"] == "native":
            return native.chunked_sgmv_lora_expand_forward(
                d["u"],
                d["b"],
                info,
                d["off"],
                max(b - a for a, b in zip(d["offsets"], d["offsets"][1:])),
                d["y"],
                slice_offsets_cpu=d["off_cpu"],
            )
        return row_expand(
            d["u"],
            d["b"].flatten(),
            d["boff"],
            d["bucket"],
            d["y"],
            info.seg_indptr,
            info.weight_indices,
            info.permutation,
            info.lora_ranks,
            info.scalings,
            d["off"],
            d["segments"],
            4,
            blocks=880,
            threads=256,
            row_tile=16,
        )

    diagnostic = setup(128, 4, 1024, 7168, 2, [0, 3584, 7168])
    diagnostic["base"].mul_(20)
    reference = act(diagnostic["base"]).clone()
    diagnostic["u"].zero_()
    round_checks = {}
    for flag in (False, True):
        got = expand_act(
            diagnostic["base"],
            diagnostic["u"],
            diagnostic["b"],
            diagnostic["info"],
            diagnostic["info"].lora_ranks,
            round_act=flag,
        )
        diff = (got.float() - reference.float()).abs()
        round_checks[str(flag)] = dict(
            max_abs=float(diff.max()), mismatches=int(torch.count_nonzero(diff))
        )
    round_act = min((False, True), key=lambda f: round_checks[str(f)]["mismatches"])
    report.update(activation_rounding=round_checks, round_act=round_act)
    save()
    for t, groups in ((32, 1), (32, 4), (32, 32), (128, 4), (1024, 1), (1024, 4)):
        d = setup(t, groups, 1024, 7168, 2, [0, 3584, 7168])
        old_a(d)
        d["y"].copy_(d["base"])
        old_b(d, "gate_up_proj")
        reference = act(d["y"]).clone()
        out = torch.full((t, 3584), float("nan"), **kw)
        checks = {}
        fs = {}

        def baseline():
            old_a(d)
            old_b(d, "gate_up_proj")
            silu_and_mul(d["y"], out)
            return out

        fs["separate"] = baseline
        for tile in (64, 128, 256):

            def f(tile=tile):
                old_a(d)
                return expand_act(
                    d["base"],
                    d["u"],
                    d["b"],
                    d["info"],
                    d["info"].lora_ranks,
                    tile=tile,
                    round_act=round_act,
                    out=out,
                )

            got = f().clone()
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, reference, rtol=0.03, atol=0.001)
            checks[str(tile)] = dict(
                max_abs=float((got.float() - reference.float()).abs().max())
            )
            fs["fused" + str(tile)] = f
        tm = times(fs)
        report["activation"].append(
            dict(tokens=t, adapters=groups, checks=checks, timing=tm)
        )
        save()
    for name, k, n, slices, offsets in [
        ("gate_up_proj", 1024, 7168, 2, [0, 3584, 7168]),
        ("in_proj_qkvz", 1024, 8192, 4, [0, 2048, 4096, 6144, 8192]),
        ("qkv_proj", 1024, 5120, 3, [0, 4096, 4608, 5120]),
        ("out_proj", 2048, 1024, 1, [0, 1024]),
        ("down_proj", 3584, 1024, 1, [0, 1024]),
    ]:
        for t in (128, 1024):
            d = setup(t, 4, k, n, slices, offsets)

            def baseline():
                old_a(d)
                return old_b(d, name)

            d["y"].copy_(d["base"])
            reference = baseline().clone()
            checks = {}
            fs = {"separate": baseline}
            for tile in (128, 256):

                def f(tile=tile):
                    return segmented_ab(
                        d["x"],
                        d["a"],
                        d["b"],
                        d["y"],
                        d["info"],
                        d["info"].lora_ranks,
                        slices,
                        offsets,
                        tile=tile,
                    )

                d["y"].copy_(d["base"])
                got = f().clone()
                torch.testing.assert_close(got, reference, rtol=0.03, atol=0.002)
                checks[str(tile)] = dict(
                    max_abs=float((got.float() - reference.float()).abs().max())
                )
                fs["fused" + str(tile)] = f
            tm = times(fs)
            report["prefill"].append(
                dict(
                    name=name,
                    input=k,
                    output=n,
                    slices=slices,
                    tokens=t,
                    adapters=4,
                    checks=checks,
                    timing=tm,
                )
            )
            save()
    report["state"] = "complete"
    save()
    print("HC_MICRO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
