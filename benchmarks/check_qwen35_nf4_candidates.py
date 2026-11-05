# SPDX-License-Identifier: Apache-2.0
"""Screen exact byte-pair reconstruction and rejected split-K NF4 candidates.

Uses six unique fixture shapes, three shuffled eight-call graph timing samples,
rows 8/32/1024, and the same BF16 matmul baseline as the prepared linear path.
Resident BF16 is a diagnostic ceiling only; it is not a quantized candidate.
"""

import argparse
import itertools
import json
from pathlib import Path


def main():
    import qwen35_nf4_ops as ops
    import torch
    import torch.nn.functional as F
    from bitsandbytes.functional import QuantState, dequantize_4bit
    from check_qwen35_nf4_ops import measure
    from safetensors import safe_open

    torch.manual_seed(671)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = args.fixture
    receipt = json.loads((fixture / "packed-bf16/packed-receipt.json").read_text())
    report = {"state": "running", "torch": torch.__version__, "cases": []}
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
            pairs = {}
            for block, warps, lookup in itertools.product(
                (256, 1024, 4096), (4,), (False, True)
            ):
                key = f"pairs_{block}_{warps}_{int(lookup)}"

                def fn(block=block, warps=warps, lookup=lookup):
                    return ops.dequantize_pairs(
                        qw,
                        scales,
                        state.code,
                        shape,
                        block=block,
                        warps=warps,
                        lookup=lookup,
                    )

                assert torch.equal(fn(), dw), key
                pairs[key] = fn
            dtime = measure(
                dict(old=lambda: ops.dequantize(qw, scales, state.code, shape), **pairs)
            )
            best = min(dtime, key=lambda k: dtime[k]["median_us"])
            for batch in (8, 32, 1024):
                x = (
                    torch.randn(batch, shape[1], device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                ref = F.linear(x, dw)
                fns = {
                    "old": lambda: F.linear(
                        x, ops.dequantize(qw, scales, state.code, shape)
                    ),
                    "resident": lambda: F.linear(x, dw),
                }
                if best != "old":
                    fns[best] = lambda: F.linear(x, pairs[best]())
                if batch <= 32:
                    for cfg in (
                        (16, 32, 128, 4, 4),
                        (16, 32, 128, 8, 4),
                        (16, 32, 64, 16, 4),
                        (16, 64, 128, 8, 4),
                    ):
                        key = "split_" + "_".join(map(str, cfg))
                        fns[key] = lambda cfg=cfg: ops.split_matmul(
                            x, qw, scales, state.code, shape, config=cfg
                        )
                numerics = {}
                failures = {}
                for key, fn in list(fns.items()):
                    try:
                        y = fn()
                        torch.testing.assert_close(y, ref, rtol=0.03, atol=0.03)
                        numerics[key] = {
                            "exact": torch.equal(y, ref),
                            "max_abs": (y.float() - ref.float()).abs().max().item(),
                        }
                    except Exception as e:
                        failures[key] = repr(e)
                        del fns[key]
                assert "old" in fns
                row = dict(
                    shape=shape,
                    batch=batch,
                    dequant_timings=dtime,
                    timings=measure(fns),
                    numerics=numerics,
                    failures=failures,
                )
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    "CASE",
                    shape,
                    batch,
                    {k: round(v["median_us"], 2) for k, v in row["timings"].items()},
                    failures,
                    flush=True,
                )
    report["state"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
