# SPDX-License-Identifier: Apache-2.0
"""Allocated-GPU lifecycle checks and graph timings for compact gate/up B."""

import argparse
import importlib.util
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sglang.srt.lora.backend.chunked_backend as native
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_gate_expand_experiment import GateShadow

    spec = importlib.util.spec_from_file_location(
        args.fleet_library.stem, args.fleet_library
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native.get_lora = lambda: SimpleNamespace(max_lora_chunk_size=16)
    torch.manual_seed(741)
    randomizer = random.Random(741)
    width, capacity = 28672, 64
    b = torch.randn(4, width, capacity, device="cuda", dtype=torch.bfloat16) / 10
    shadow = GateShadow(b)
    addresses = shadow.addresses()
    ranks, scales = [0, 16, 32, 64], [0, 0.5, 1.0, 1.5]
    for slot, rank in enumerate(ranks):
        shadow.refresh(slot, rank)
    offsets_cpu = torch.tensor([0, width // 2, width], dtype=torch.int64)
    offsets = offsets_cpu.cuda()

    def batch(tokens):
        return SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=tokens,
            extend_num_tokens=tokens,
            extend_seq_lens_cpu=[1] * tokens,
            extend_seq_lens=torch.ones(tokens, dtype=torch.int32, device="cuda"),
            return_logprob=False,
        )

    def setup(tokens, routes):
        backend = native.ChunkedSgmvLoRABackend(4, torch.device("cuda:0"), None)
        backend.init_cuda_graph_batch_info(tokens, 1)
        forward = batch(tokens)
        backend.prepare_lora_batch(forward, routes, ranks, scales, True)
        info = backend.batch_info
        nseg = torch.tensor(info.num_segments, device="cuda", dtype=torch.int32)
        z = torch.randn(tokens, 2 * capacity, device="cuda", dtype=torch.bfloat16) / 10
        base = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16) / 10

        def hip(blocks=880):
            out = base.clone()
            module.expand(
                z,
                shadow.store,
                shadow.boff,
                shadow.bucket,
                out,
                info.seg_indptr,
                info.weight_indices,
                info.permutation,
                info.lora_ranks,
                info.scalings,
                shadow.offsets,
                nseg,
                width,
                capacity,
                blocks,
                256,
            )
            return out

        def reference():
            return backend.run_lora_b_sgemm(
                z, b, offsets, base.clone(), output_offset_cpu=offsets_cpu
            )

        return backend, forward, info, nseg, hip, reference

    def capture(function):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                function()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = function()
        return graph, result

    backend, forward, info, nseg, hip, reference = setup(8, [1, 2, 3, 0, 1, 2, 3, 0])
    graph, result = capture(hip)
    lifecycle = []
    for rank, routes in (
        (32, [0, 3, 3, 1, 2, 0, 3, 2]),
        (16, [3] * 8),
        (0, [3, 0] * 4),
        (64, [1, 2, 3, 0] * 2),
    ):
        b[3].normal_(std=0.1)
        shadow.refresh(3, rank)
        ranks[3] = rank
        backend.prepare_lora_batch(forward, routes, ranks, scales, True)
        nseg.fill_(info.num_segments)
        graph.replay()
        torch.cuda.synchronize()
        expected = reference()
        torch.testing.assert_close(result, expected, rtol=0.03, atol=0.001)
        assert shadow.addresses() == addresses
        lifecycle.append(
            dict(
                rank=rank,
                routes=routes,
                max_abs=float((result.float() - expected.float()).abs().max()),
            )
        )
    del graph, result, hip, reference, backend, info, forward, nseg
    timing = []
    for tokens in (1, 2, 8):
        for rank in (16, 32, 64):
            ranks[1] = rank
            shadow.refresh(1, rank)
            backend, forward, info, nseg, hip, reference = setup(tokens, [1] * tokens)
            functions = {"native": reference}
            functions.update(
                {
                    str(blocks): (
                        lambda blocks=blocks, operation=hip: operation(blocks)
                    )
                    for blocks in (64, 110, 220, 440, 880)
                }
            )
            graphs, values, durations = {}, {}, {k: [] for k in functions}
            expected = reference()
            for name, function in functions.items():
                graphs[name], values[name] = capture(function)
                graphs[name].replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    values[name], expected, rtol=0.03, atol=0.001
                )
            for _ in range(5):
                order = list(graphs)
                randomizer.shuffle(order)
                for name in order:
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(50):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    durations[name].append(start.elapsed_time(end) * 1000 / 50)
            row = dict(
                tokens=tokens,
                rank=rank,
                median_us={k: statistics.median(v) for k, v in durations.items()},
                trials_us=durations,
            )
            timing.append(row)
            print(json.dumps(row), flush=True)
            del graphs, values, functions, hip, reference, backend, info, forward, nseg
    report = dict(
        scope="compact gate/up B only, base.clone included; synthetic numerics, not quality certification",
        lifecycle=lifecycle,
        timing=timing,
        extra_bytes=shadow.store.numel() * shadow.store.element_size(),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("GATE_CHECK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
