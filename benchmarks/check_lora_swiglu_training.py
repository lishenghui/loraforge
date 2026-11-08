# SPDX-License-Identifier: Apache-2.0
"""Allocated-GPU BF16 LoRA MLP checks and complete forward/backward timings."""

import argparse
import gc
import hashlib
import json
import random
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 1024, 4096])
    args = parser.parse_args()
    import lora_swiglu_training as probe
    import torch
    from safetensors import safe_open

    report = dict(
        state="running",
        scope="Frozen fused BF16 Qwen3.5 MLP; FP32 master LoRA; BF16 compute; one adapter; dropout=0; forward+backward timing, not whole model",
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        cases=[],
        optimizer=[],
        compiled_failures=[],
        tolerance=dict(rtol=0.02, atol=0.003, max_relative_l2=0.02),
        timing_protocol=dict(shuffled_samples=3, replays_per_sample=8),
    )
    report["source_sha256"] = {
        n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
        for n in (
            "lora_swiglu_training.py",
            "lora_swiglu_kernels.py",
            "check_lora_swiglu_training.py",
        )
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
        wgu = f.get_tensor("model.layers.0.mlp.gate_up_proj.weight").cuda()
        wd = f.get_tensor("model.layers.0.mlp.down_proj.weight").cuda()
    assert wgu.shape == (7168, 1024) and wd.shape == (1024, 3584)
    base = (wgu, wd)

    def copies(values):
        return [v.detach().clone().requires_grad_(True) for v in values]

    def compare(got, ref):
        out = {}
        for name, g, r in zip(
            ("output", "dX", "dAg", "dBg", "dAu", "dBu", "dAd", "dBd"),
            got,
            ref,
            strict=True,
        ):
            torch.testing.assert_close(g, r, rtol=0.02, atol=0.003)
            diff = g.float() - r.float()
            relative_l2 = (diff.norm() / r.float().norm().clamp_min(1e-20)).item()
            assert relative_l2 <= 0.02, (name, relative_l2)
            out[name] = dict(
                max_abs=diff.abs().max().item(),
                relative_l2=relative_l2,
                exact=torch.equal(g, r),
            )
        return out

    def native(*v):
        return probe.mlp(*v, *base, backend="native")

    def eager(fn, values, dy):
        v = copies(values)
        y = fn(*v)
        grads = torch.autograd.grad(y, v, dy)
        return (y.detach(), *grads)

    for rank in (4, 32):
        for rows in args.tokens:
            torch.manual_seed(771)
            values = [
                torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
            ]
            values += [
                torch.randn(*shape, device="cuda") * 0.02
                for shape in (
                    (rank, 1024),
                    (3584, rank),
                    (rank, 1024),
                    (3584, rank),
                    (rank, 3584),
                    (1024, rank),
                )
            ]
            dy = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
            ref = eager(native, values, dy)
            functions = dict(
                native=native,
                activation=lambda *v: probe.mlp(*v, *base, backend="activation"),
                block=lambda *v: probe.mlp(*v, *base, backend="block"),
                fused=lambda *v: probe.mlp(*v, *base, backend="fused"),
                compact=lambda *v: probe.mlp(*v, *base, backend="compact"),
            )
            row = dict(rank=rank, rows=rows, numerics={}, memory={}, timing={})
            # Reset only this isolated benchmark's compilation cache between shapes.
            torch._dynamo.reset()
            compiled = torch.compile(native, fullgraph=True, dynamic=False)
            try:
                row["numerics"]["compiled"] = compare(eager(compiled, values, dy), ref)
                functions["compiled"] = compiled
            except Exception as exc:
                error = dict(rank=rank, rows=rows, error=repr(exc))
                report["compiled_failures"].append(error)
                print("COMPILED_FAILED", error, flush=True)
            graphs, objects, samples = {}, {}, {}
            for kind, fn in functions.items():
                row["numerics"][kind] = compare(eager(fn, values, dy), ref)
                v = copies(values)
                for _ in range(2):
                    y = fn(*v)
                    y.backward(dy)
                    for p in v:
                        p.grad = None
                    del y
                torch.cuda.synchronize()
                gc.collect()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                known = {p.untyped_storage().data_ptr() for p in (*v, *base)}
                saved = {}
                wide = {}

                def pack(t):
                    ptr = t.untyped_storage().data_ptr()
                    if ptr not in known:
                        saved[ptr] = t.untyped_storage().nbytes()
                        if t.dtype == torch.bfloat16 and t.numel() == rows * 3584:
                            wide[ptr] = t.untyped_storage().nbytes()
                    return t

                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    y = fn(*v)
                torch.cuda.synchronize()
                live = torch.cuda.memory_allocated() - before
                y.backward(dy)
                torch.cuda.synchronize()
                row["memory"][kind] = dict(
                    forward_live_delta_bytes=live,
                    peak_step_delta_bytes=torch.cuda.max_memory_allocated() - before,
                    saved_extra_storage_bytes=sum(saved.values()),
                    saved_wide_activation_bytes=sum(wide.values()),
                )
                for p in v:
                    p.grad = None
                del y
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    v = copies(values)
                    for _ in range(2):
                        y = fn(*v)
                        y.backward(dy)
                        for p in v:
                            p.grad = None
                        del y
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    y = fn(*v)
                    y.backward(dy)
                torch.cuda.current_stream().wait_stream(stream)
                with torch.no_grad():
                    v[0].mul_(0.9)
                    for p in v[1:]:
                        p.add_(0.0001)
                graph.replay()
                torch.cuda.synchronize()
                row["numerics"][kind]["updated_graph"] = compare(
                    (y.detach(), *[p.grad for p in v]), eager(native, v, dy)
                )
                graphs[kind] = graph
                objects[kind] = (v, y, stream)
                samples[kind] = []
            for repeat in range(3):
                order = list(graphs)
                random.Random(771 + repeat).shuffle(order)
                for kind in order:
                    graph = graphs[kind]
                    graph.replay()
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    for _ in range(8):
                        graph.replay()
                    b.record()
                    b.synchronize()
                    samples[kind].append(a.elapsed_time(b) * 1000 / 8)
            row["timing"] = {
                k: dict(median_us=statistics.median(v), samples_us=v)
                for k, v in samples.items()
            }
            for kind in ("activation", "block", "fused", "compact"):
                row[kind + "_over_native"] = (
                    row["timing"]["native"]["median_us"]
                    / row["timing"][kind]["median_us"]
                )
                if "compiled" in graphs:
                    row[kind + "_over_compiled"] = (
                        row["timing"]["compiled"]["median_us"]
                        / row["timing"][kind]["median_us"]
                    )
            report["cases"].append(row)
            save()
            print(
                "CASE",
                rank,
                rows,
                {k: round(v["median_us"], 2) for k, v in row["timing"].items()},
                flush=True,
            )
            del (
                graphs,
                objects,
                graph,
                stream,
                y,
                v,
                values,
                ref,
                compiled,
                functions,
                dy,
            )
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    # Multiple optimizer updates from standard zero-B initialization.
    for kind in ("native", "activation", "block", "fused", "compact"):
        torch.manual_seed(772)
        x = torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
        shapes = ((4, 1024), (3584, 4), (4, 1024), (3584, 4), (4, 3584), (1024, 4))
        params = [
            torch.nn.Parameter(
                torch.randn(*s, device="cuda") * 0.02
                if i % 2 == 0
                else torch.zeros(s, device="cuda")
            )
            for i, s in enumerate(shapes)
        ]
        optim = torch.optim.AdamW(params, lr=0.001, foreach=False)
        target = torch.randn_like(x).float() * 0.1
        losses = []
        for step in range(3):
            optim.zero_grad(set_to_none=True)
            z = x.detach().requires_grad_(True)
            y = probe.mlp(z, *params, *base, backend=kind)
            loss = (y.float() - target).square().mean()
            loss.backward()
            optim.step()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in params
            )
            losses.append(loss.item())
        if kind == "native":
            reference = [p.detach().clone() for p in params] + [torch.tensor(losses)]
        else:
            for a, b in zip([*params, torch.tensor(losses)], reference, strict=True):
                torch.testing.assert_close(a, b, rtol=0.02, atol=0.0001)
        report["optimizer"].append(dict(kind=kind, losses=losses))
        save()
    report["state"] = "complete"
    save()
    print("SWIGLU_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
