# SPDX-License-Identifier: Apache-2.0
"""Direct NF4/BF16 projection checks; run from a prepared fixture on a GPU."""


def main():
    import json
    import random
    import statistics
    from pathlib import Path

    import bitsandbytes as bnb
    import torch
    import torch.nn.functional as F
    from bitsandbytes.functional import QuantState, dequantize_4bit
    from safetensors import safe_open

    root = Path(".")
    report = dict(
        cases=[],
        state="running",
        torch=torch.__version__,
        bnb=bnb.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    torch.manual_seed(671)

    def save():
        (root / "nf4-micro.json").write_text(json.dumps(report, indent=2) + "\n")

    def measure(fs):
        graphs = {}
        samples = {k: [] for k in fs}
        for name, fn in fs.items():
            for _ in range(3):
                fn()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(8):
                    fn()
            graphs[name] = g
        for rep in range(3):
            names = list(fs)
            random.Random(rep + 617).shuffle(names)
            for name in names:
                graphs[name].replay()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(8):
                    graphs[name].replay()
                b.record()
                b.synchronize()
                samples[name].append(a.elapsed_time(b) * 1000 / 64)
        return {
            name: dict(median_us=statistics.median(v), samples_us=v)
            for name, v in samples.items()
        }

    with (
        safe_open(
            root / "packed-bf16/model.safetensors", framework="pt", device="cpu"
        ) as bf,
        safe_open(root / "nf4/model.safetensors", framework="pt", device="cpu") as nf,
    ):
        shapes = set()
        for name, shape in json.loads(
            (root / "packed-bf16/packed-receipt.json").read_text()
        )["targets"].items():
            if tuple(shape) in shapes:
                continue
            shapes.add(tuple(shape))
            w = bf.get_tensor(name).cuda()
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
            dw = dequantize_4bit(qw, quant_state=state)
            for batch in (1, 8, 32, 1024):
                x = (
                    torch.randn(batch, shape[1], device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                got = bnb.matmul_4bit(x, qw.t(), state)
                ref = F.linear(x, dw)
                torch.testing.assert_close(got, ref, rtol=0.03, atol=0.03)
                fs = {
                    "bf16": lambda x=x, w=w: F.linear(x, w),
                    "nf4": lambda x=x, qw=qw, state=state: bnb.matmul_4bit(
                        x, qw.t(), state
                    ),
                }
                timing = measure(fs)
                report["cases"].append(
                    dict(
                        name=name,
                        shape=shape,
                        batch=batch,
                        max_abs_vs_dequant_ref=(got.float() - ref.float())
                        .abs()
                        .max()
                        .item(),
                        timing=timing,
                        nf4_over_bf16_speed=timing["bf16"]["median_us"]
                        / timing["nf4"]["median_us"],
                    )
                )
                save()
            del w, qw, state, dw
    report["state"] = "complete"
    save()
    print("NF4_MICRO_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
