# SPDX-License-Identifier: Apache-2.0
"""Compare BF16 routed LoRA A+B latency; synthetic model shapes, not serving throughput."""

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--fleet-library", type=Path)
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.trials) < 1:
        parser.error("iterations and trials must be positive")
    import sglang.srt.lora.backend.chunked_backend as native
    import torch
    import triton
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    from loraforge.ops import gemv

    spec = importlib.util.spec_from_file_location(
        "loraforge.ops._benchmark_legacy", args.legacy_module
    )
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    baseline = None
    if args.baseline_package:
        package_spec = importlib.util.spec_from_file_location(
            "loraforge_benchmark_baseline",
            args.baseline_package / "__init__.py",
            submodule_search_locations=[str(args.baseline_package)],
        )
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_spec.name] = package
        package_spec.loader.exec_module(package)
        baseline = importlib.import_module("loraforge_benchmark_baseline.gemv")
    native.get_lora = lambda: SimpleNamespace(max_lora_chunk_size=16)
    backend = native.ChunkedSgmvLoRABackend(4, torch.device("cuda:0"), None)
    policy_backend = None
    if args.policy:
        from loraforge.integrations.sglang import get_backend

        policy_backend = get_backend()(4, torch.device("cuda:0"), None)
    fleet = None
    if args.fleet_library:
        fleet_spec = importlib.util.spec_from_file_location(
            "loraforge_fleet_probe", args.fleet_library
        )
        fleet = importlib.util.module_from_spec(fleet_spec)
        fleet_spec.loader.exec_module(fleet)
    torch.manual_seed(741)
    rng = random.Random(741)
    report = dict(
        sglang=version("sglang"),
        source_sha256=hashlib.sha256(Path(gemv.__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
        hip=torch.version.hip,
        triton=triton.__version__,
        device=str(torch.cuda.get_device_properties(0)),
        dtype="bfloat16",
        iterations=args.iterations,
        trials=args.trials,
        legacy_sha256=hashlib.sha256(args.legacy_module.read_bytes()).hexdigest(),
        scope="Synthetic model-shape A+B with base.clone included; no paging, model forward or throughput claim. Warmup/compilation excluded. Trial distributions; interleaved randomized backend order. Repeated resident inputs/weights (warm cache). Eager event intervals include host submission gaps.",
        rows=[],
    )
    shapes = [
        ("qkv", 4096, [4096, 1024, 1024]),
        ("o", 4096, [4096]),
        ("gate_up", 4096, [14336, 14336]),
        ("down", 14336, [4096]),
    ]
    for tokens in (1, 2, 8):
        for rank in (16, 32, 64):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                batch_size=tokens,
                extend_num_tokens=tokens,
                extend_seq_lens_cpu=[1] * tokens,
                extend_seq_lens=torch.ones(tokens, dtype=torch.int32, device="cuda"),
                return_logprob=False,
            )
            backend.prepare_lora_batch(
                batch, [1] * tokens, [0, rank, rank, rank], [0, 0.5, 1, 1], False
            )
            if policy_backend is not None:
                policy_backend.prepare_lora_batch(
                    batch, [1] * tokens, [0, rank, rank, rank], [0, 0.5, 1, 1], False
                )
            capacity = args.capacity or rank
            if capacity < rank:
                raise ValueError("capacity must cover every tested rank")
            info = backend.batch_info
            for name, width, sizes in shapes:
                slices = len(sizes)
                x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16) / 10
                a = (
                    torch.randn(
                        4, slices * capacity, width, device="cuda", dtype=x.dtype
                    )
                    / 100
                )
                b = (
                    torch.randn(4, sum(sizes), capacity, device="cuda", dtype=x.dtype)
                    / 100
                )
                base = torch.randn(tokens, sum(sizes), device="cuda", dtype=x.dtype)
                boundaries = [0]
                for size in sizes:
                    boundaries.append(boundaries[-1] + size)
                offsets_cpu = torch.tensor(boundaries, dtype=torch.int64)
                offsets = offsets_cpu.cuda()

                def native_call():
                    out = base.clone()
                    z = backend.run_lora_a_sgemm(x, a, stack_num=slices)
                    return backend.run_lora_b_sgemm(
                        z, b, offsets, out, output_offset_cpu=offsets_cpu
                    )

                def custom_call(module):
                    out = base.clone()
                    z = module.lora_shrink(
                        x, a, info.lora_ranks, info.req_weight_indices, slices
                    )
                    return module.lora_expand(
                        z,
                        b,
                        info.lora_ranks,
                        info.scalings,
                        info.req_weight_indices,
                        offsets,
                        max(sizes),
                        out,
                    )

                functions = {
                    "native": native_call,
                    "legacy": lambda: custom_call(legacy),
                    "compatible": lambda: custom_call(gemv),
                }
                if baseline is not None:
                    functions["previous_compatible"] = lambda: custom_call(baseline)
                if policy_backend is not None:

                    def policy_call():
                        out = base.clone()
                        z = policy_backend.run_lora_a_sgemm(x, a, stack_num=slices)
                        return policy_backend.run_lora_b_sgemm(
                            z, b, offsets, out, output_offset_cpu=offsets_cpu
                        )

                    functions["probe_policy"] = policy_call
                if fleet is not None:
                    packed = b[:, :, :rank].contiguous().flatten()
                    boff = (
                        torch.arange(4, device=x.device, dtype=torch.int64)
                        * sum(sizes)
                        * rank
                    )
                    buckets = torch.full((4,), rank, device=x.device, dtype=torch.int32)
                    nseg = torch.tensor(
                        [info.num_segments], device=x.device, dtype=torch.int32
                    )
                    offs32 = offsets.to(torch.int32)

                    def fleet_call():
                        out = base.clone()
                        z = torch.empty(
                            (tokens, slices * capacity), device=x.device, dtype=x.dtype
                        )
                        fleet.shrink(
                            x,
                            a,
                            z,
                            info.req_weight_indices,
                            info.lora_ranks,
                            slices,
                            rank,
                        )
                        fleet.expand(
                            z,
                            packed,
                            boff,
                            buckets,
                            out,
                            info.seg_indptr,
                            info.weight_indices,
                            info.permutation,
                            info.lora_ranks,
                            info.scalings,
                            offs32,
                            nseg,
                            sum(sizes),
                            capacity,
                            880,
                            256,
                        )
                        return out

                    def hip_a_native_b():
                        out = base.clone()
                        z = torch.empty(
                            (tokens, slices * capacity), device=x.device, dtype=x.dtype
                        )
                        fleet.shrink(
                            x,
                            a,
                            z,
                            info.req_weight_indices,
                            info.lora_ranks,
                            slices,
                            rank,
                        )
                        return backend.run_lora_b_sgemm(
                            z, b, offsets, out, output_offset_cpu=offsets_cpu
                        )

                    def native_a_hip_b():
                        out = base.clone()
                        z = backend.run_lora_a_sgemm(x, a, stack_num=slices)
                        fleet.expand(
                            z,
                            packed,
                            boff,
                            buckets,
                            out,
                            info.seg_indptr,
                            info.weight_indices,
                            info.permutation,
                            info.lora_ranks,
                            info.scalings,
                            offs32,
                            nseg,
                            sum(sizes),
                            capacity,
                            880,
                            256,
                        )
                        return out

                    functions["probe_fleet_hip"] = fleet_call
                    functions["probe_hip_a_native_b"] = hip_a_native_b
                    functions["probe_native_a_hip_b"] = native_a_hip_b
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graphs = {}
                outputs = {}
                with torch.cuda.stream(stream):
                    for label, fn in functions.items():
                        for _ in range(10):
                            outputs[label] = fn()
                stream.synchronize()
                drift = {
                    label: float(
                        (value.float() - outputs["native"].float()).abs().max()
                    )
                    for label, value in outputs.items()
                }
                for label, fn in functions.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs[label] = fn()
                    graphs[label] = graph
                torch.cuda.synchronize()
                for mode in ("eager", "graph"):
                    timings = {label: [] for label in functions}
                    for _ in range(args.trials):
                        order = list(functions)
                        rng.shuffle(order)
                        for label in order:
                            fn = (
                                functions[label]
                                if mode == "eager"
                                else graphs[label].replay
                            )
                            for _ in range(5):
                                fn()
                            torch.cuda.synchronize()
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            begin = time.perf_counter()
                            start.record()
                            for _ in range(args.iterations):
                                fn()
                            end.record()
                            end.synchronize()
                            timings[label].append(
                                dict(
                                    event_us=start.elapsed_time(end)
                                    * 1000
                                    / args.iterations,
                                    wall_us=(time.perf_counter() - begin)
                                    * 1e6
                                    / args.iterations,
                                )
                            )
                    row = dict(
                        projection=name,
                        tokens=tokens,
                        rank=rank,
                        capacity=capacity,
                        width=width,
                        sizes=sizes,
                        mode=mode,
                        max_abs_drift=drift,
                        timings=timings,
                        median_event_us={
                            k: statistics.median(t["event_us"] for t in v)
                            for k, v in timings.items()
                        },
                    )
                    report["rows"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "timings"}),
                        flush=True,
                    )
                del graphs, outputs
    print("BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
