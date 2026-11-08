# SPDX-License-Identifier: Apache-2.0
"""Allocated-GPU feasibility checks; projection training, not whole-model speed."""

import argparse
import gc
import hashlib
import json
import random
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import bitsandbytes as bnb
    import lora_training as train
    import torch
    from bitsandbytes.functional import QuantState
    from safetensors import safe_open

    torch.manual_seed(671)
    report = dict(
        state="running",
        scope="Frozen actual NF4 decoder projections;BF16compute;FP32LoRA master parameters;dropout0;singleadapter;forward+backward graph timings;no wholemodel or Unsloth benchmark claim",
        torch=torch.__version__,
        bnb=bnb.__version__,
        device=torch.cuda.get_device_name(),
        cases=[],
        optimizer_checks=[],
    )
    report["source_sha256"] = {
        n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
        for n in ("lora_training.py", "qwen35_nf4_ops.py", "check_lora_training.py")
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def clones(values):
        return [v.detach().clone().requires_grad_(True) for v in values]

    def compare(got, ref):
        diffs = {}
        for name, a, b in zip(("output", "dX", "dA", "dB"), got, ref, strict=True):
            torch.testing.assert_close(a, b, rtol=0.02, atol=0.003)
            diffs[name] = dict(
                max_abs=(a.float() - b.float()).abs().max().item(),
                exact=torch.equal(a, b),
            )
        return diffs

    def eager(values, base, backend, dy):
        x, a, b = clones(values)
        y = train.linear(x, a, b, base, backend=backend, scale=0.7)
        dx, da, db = torch.autograd.grad(y, (x, a, b), dy)
        return y.detach(), dx, da, db

    shape_names = {}
    receipt = json.loads((args.fixture / "packed-bf16/packed-receipt.json").read_text())
    wanted = {(8192, 1024), (7168, 1024), (1024, 3584), (1024, 2048)}
    for name, shape in receipt["targets"].items():
        if tuple(shape) in wanted:
            shape_names.setdefault(tuple(shape), name)
    assert set(shape_names) == wanted
    with safe_open(
        args.fixture / "nf4/model.safetensors", framework="pt", device="cpu"
    ) as nf:
        for (n, k), name in shape_names.items():
            qw = nf.get_tensor(name).cuda()
            state = QuantState.from_dict(
                {
                    key: nf.get_tensor(key).cuda()
                    if "quant_state.bitsandbytes" not in key
                    else nf.get_tensor(key)
                    for key in nf.keys()
                    if key.startswith(name + ".")
                },
                device="cuda",
            )
            base = train.FrozenBase.nf4(qw, state)
            for rank in (4, 32):
                for rows in (128, 1024, 4096):
                    gc.collect()
                    torch.cuda.empty_cache()
                    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.1
                    a = torch.randn(rank, k, device="cuda", dtype=torch.float32) * 0.02
                    b = torch.randn(n, rank, device="cuda", dtype=torch.float32) * (
                        0.04 / rank**0.5
                    )
                    values = [x, a, b]
                    # Noncontiguous upstream gradient catches view-only backward errors.
                    dy = (
                        torch.randn(n, rows, device="cuda", dtype=torch.bfloat16) * 0.1
                    ).t()
                    ref = eager(values, base, "bnb", dy)
                    row = dict(
                        name=name,
                        shape=[n, k],
                        rank=rank,
                        rows=rows,
                        numerics={},
                        memory={},
                        timing={},
                    )
                    graphs = {}
                    objects = {}
                    samples = {}
                    for backend in ("bnb", "cached", "naive", "recompute"):
                        row["numerics"][backend] = compare(
                            eager(values, base, backend, dy), ref
                        )
                        px, pa, pb = clones(values)
                        retained = []
                        input_storage = px.untyped_storage().data_ptr()

                        def pack(t):
                            if (
                                t.dtype == torch.bfloat16
                                and tuple(t.shape) in ((n, k), (k, n))
                                and t.untyped_storage().data_ptr() != input_storage
                            ):
                                retained.append(
                                    (
                                        t.untyped_storage().data_ptr(),
                                        t.untyped_storage().nbytes(),
                                    )
                                )
                            return t

                        for _ in range(2):
                            out = train.linear(
                                px, pa, pb, base, backend=backend, scale=0.7
                            )
                            out.backward(dy)
                            px.grad = pa.grad = pb.grad = None
                            del out
                        torch.cuda.synchronize()
                        gc.collect()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        with torch.autograd.graph.saved_tensors_hooks(
                            pack, lambda t: t
                        ):
                            out = train.linear(
                                px, pa, pb, base, backend=backend, scale=0.7
                            )
                        torch.cuda.synchronize()
                        live = torch.cuda.memory_allocated() - before
                        out.backward(dy)
                        torch.cuda.synchronize()
                        row["memory"][backend] = dict(
                            forward_live_delta_bytes=live,
                            peak_step_delta_bytes=torch.cuda.max_memory_allocated()
                            - before,
                            retained_bf16_base_weight_bytes=sum(
                                dict(retained).values()
                            ),
                        )
                        px.grad = pa.grad = pb.grad = None
                        del out
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            # Fresh leaves avoid carrying default-stream gradient
                            # accumulator nodes from the eager memory probe.
                            px, pa, pb = clones(values)
                            for _ in range(2):
                                out = train.linear(
                                    px, pa, pb, base, backend=backend, scale=0.7
                                )
                                out.backward(dy)
                                px.grad = pa.grad = pb.grad = None
                                del out
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            out = train.linear(
                                px, pa, pb, base, backend=backend, scale=0.7
                            )
                            out.backward(dy)
                        torch.cuda.current_stream().wait_stream(stream)
                        # Updated activations and trainable factors must affect replay.
                        with torch.no_grad():
                            px.mul_(0.9)
                            pa.add_(0.001)
                            pb.sub_(0.0001)
                        graph.replay()
                        torch.cuda.synchronize()
                        replay = (out.detach(), px.grad, pa.grad, pb.grad)
                        row["numerics"][backend]["updated_graph"] = compare(
                            replay, eager([px, pa, pb], base, "bnb", dy)
                        )
                        graphs[backend] = graph
                        objects[backend] = (px, pa, pb, out, stream)
                        samples[backend] = []
                    for repeat in range(3):
                        order = list(graphs)
                        random.Random(671 + repeat).shuffle(order)
                        for backend in order:
                            graph = graphs[backend]
                            graph.replay()
                            begin, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            begin.record()
                            for _ in range(8):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            samples[backend].append(begin.elapsed_time(end) * 1000 / 8)
                    row["timing"] = {
                        b: dict(median_us=statistics.median(v), samples_us=v)
                        for b, v in samples.items()
                    }
                    row["cached_over_bnb_speed"] = (
                        row["timing"]["bnb"]["median_us"]
                        / row["timing"]["cached"]["median_us"]
                    )
                    row["recompute_over_bnb_speed"] = (
                        row["timing"]["bnb"]["median_us"]
                        / row["timing"]["recompute"]["median_us"]
                    )
                    assert (
                        row["memory"]["naive"]["retained_bf16_base_weight_bytes"]
                        == n * k * 2
                    )
                    assert all(
                        row["memory"][b]["retained_bf16_base_weight_bytes"] == 0
                        for b in ("bnb", "cached", "recompute")
                    )
                    report["cases"].append(row)
                    save()
                    print(
                        "CASE",
                        n,
                        k,
                        rank,
                        rows,
                        {b: round(t["median_us"], 2) for b, t in row["timing"].items()},
                        "cached_speed",
                        row["cached_over_bnb_speed"],
                        flush=True,
                    )
                    # End graph lifetimes before the next shape's memory measurement.
                    del (
                        graphs,
                        objects,
                        graph,
                        stream,
                        out,
                        replay,
                        px,
                        pa,
                        pb,
                        ref,
                        values,
                        dy,
                        x,
                        a,
                        b,
                    )
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            # Multi-step AdamW probe, including conventional zero-B initialization.
            for backend in ("bnb", "cached", "recompute"):
                torch.manual_seed(672)
                x = torch.randn(128, k, device="cuda", dtype=torch.bfloat16) * 0.1
                pa = torch.nn.Parameter(torch.randn(4, k, device="cuda") * 0.02)
                pb = torch.nn.Parameter(torch.zeros(n, 4, device="cuda"))
                opt = torch.optim.AdamW([pa, pb], lr=0.001, foreach=False)
                target = torch.randn(128, n, device="cuda") * 0.1
                losses = []
                for step in range(3):
                    opt.zero_grad(set_to_none=True)
                    z = x.detach().requires_grad_(True)
                    loss = (
                        (
                            train.linear(
                                z, pa, pb, base, backend=backend, scale=0.7
                            ).float()
                            - target
                        )
                        .square()
                        .mean()
                    )
                    loss.backward()
                    assert (
                        pa.grad is not None
                        and pb.grad is not None
                        and z.grad is not None
                    )
                    assert (
                        torch.isfinite(pa.grad).all() and torch.isfinite(pb.grad).all()
                    )
                    opt.step()
                    losses.append(loss.item())
                result = dict(name=name, backend=backend, losses=losses)
                if backend == "bnb":
                    reference = (
                        pa.detach().clone(),
                        pb.detach().clone(),
                        torch.tensor(losses),
                    )
                else:
                    for got, want in zip(
                        (pa, pb, torch.tensor(losses)), reference, strict=True
                    ):
                        torch.testing.assert_close(got, want, rtol=0.02, atol=0.0001)
                report["optimizer_checks"].append(result)
                save()
            del base, qw, state, reference, opt, pa, pb, x, z, loss, target
    report["state"] = "complete"
    save()
    print("TRAINING_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
