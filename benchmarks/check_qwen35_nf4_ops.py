# SPDX-License-Identifier: Apache-2.0
"""Screen quantized projection candidates on an allocated gfx90a GPU."""

import argparse
import json
import random
import statistics
from pathlib import Path


def measure(functions):
    import torch

    graphs, samples = {}, {k: [] for k in functions}
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(8):
                fn()
        graphs[name] = graph
    for repeat in range(3):
        names = list(functions)
        random.Random(981 + repeat).shuffle(names)
        for name in names:
            graphs[name].replay()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(8):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / 64)
    return {
        k: dict(median_us=statistics.median(v), samples_us=v)
        for k, v in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import bitsandbytes as bnb
    import qwen35_nf4_ops as ops
    import torch
    import torch.nn.functional as functional
    from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
    from bitsandbytes.functional import QuantState, dequantize_4bit
    from safetensors import safe_open

    torch.manual_seed(671)
    report = dict(
        state="running",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        bnb=bnb.__version__,
        cases=[],
    )
    configs = [(16, 32, 64, 4), (16, 64, 64, 4), (32, 64, 64, 4), (16, 64, 128, 4)]
    fixture = args.fixture
    receipt = json.loads((fixture / "packed-bf16/packed-receipt.json").read_text())
    with safe_open(
        fixture / "nf4/model.safetensors", framework="pt", device="cpu"
    ) as nf:
        seen = set()
        for name, shape in receipt["targets"].items():
            if tuple(shape) in seen:
                continue
            seen.add(tuple(shape))
            qw = nf.get_tensor(name).cuda()
            state = QuantState.from_dict(
                {
                    k: nf.get_tensor(k).cuda()
                    if "quant_state.bitsandbytes" not in k
                    else nf.get_tensor(k)
                    for k in nf.keys()
                    if k.startswith(name + ".")
                },
                device="cuda",
            )
            scales = ops.prepare_scales(state)
            dw = dequantize_4bit(qw, quant_state=state)
            reconstructed = ops.dequantize(qw, scales, state.code, shape)
            assert torch.equal(reconstructed, dw), (
                name,
                (reconstructed.float() - dw.float()).abs().max().item(),
            )
            for batch in (1, 8, 32, 1024):
                x = (
                    torch.randn(batch, shape[1], device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                ref = functional.linear(x, dw)
                functions = dict(
                    stock=lambda: bnb.matmul_4bit(x, qw.t(), state),
                    dequant_linear=lambda: functional.linear(
                        x, ops.dequantize(qw, scales, state.code, shape)
                    ),
                    resident_bf16_ceiling=lambda: functional.linear(x, dw),
                )
                if batch <= 32:
                    functions["forced_bnb_simt"] = lambda: _gemm_4bit_kernel_impl(
                        x,
                        qw,
                        state.shape,
                        state.state2.absmax,
                        state.blocksize,
                        state.quant_type,
                        absmax_8bit=state.absmax,
                        absmax_code=state.state2.code,
                        absmax_offset=state.offset,
                    )
                failures, numerics = {}, {}
                for cfg in configs:
                    key = "fused_" + "_".join(map(str, cfg))
                    functions[key] = lambda cfg=cfg: ops.matmul(
                        x, qw, scales, state.code, shape, config=cfg
                    )
                for key, fn in list(functions.items()):
                    try:
                        value = fn()
                        torch.testing.assert_close(value, ref, rtol=0.03, atol=0.03)
                        numerics[key] = dict(
                            max_abs=(value.float() - ref.float()).abs().max().item(),
                            exact=torch.equal(value, ref),
                        )
                    except Exception as exc:
                        failures[key] = repr(exc)
                        del functions[key]
                assert "stock" in functions and "dequant_linear" in functions
                row = dict(
                    name=name,
                    shape=shape,
                    batch=batch,
                    dequant_exact=True,
                    scale_bytes=scales.numel() * scales.element_size(),
                    failures=failures,
                    numerics=numerics,
                    timing=measure(functions),
                )
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    "CASE",
                    shape,
                    batch,
                    {k: round(v["median_us"], 2) for k, v in row["timing"].items()},
                    "FAIL",
                    failures,
                    flush=True,
                )
    report["state"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("NF4_OPT_SCREEN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
