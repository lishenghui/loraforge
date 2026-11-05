# SPDX-License-Identifier: Apache-2.0
"""Screen ROCm SwiGLU math policies against native Torch on an allocated GPU."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def candidate(G, U, DY, H, DG, DU, SIZE: tl.constexpr, POLICY: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    g = tl.load(G + i, i < SIZE, 0).to(tl.float32)
    u = tl.load(U + i, i < SIZE, 0).to(tl.float32)
    dy = tl.load(DY + i, i < SIZE, 0).to(tl.float32)
    if POLICY == 0:
        sig = tl.sigmoid(g)
    else:
        if POLICY == 1 or POLICY == 3:
            e = libdevice.exp(-g)
        else:
            e = libdevice.fast_expf(-g)
        if POLICY >= 3:
            sig = tl.div_rn(1.0, 1.0 + e)
        else:
            sig = 1.0 / (1.0 + e)
    act = (g * sig).to(tl.bfloat16).to(tl.float32)
    da = (dy * u).to(tl.bfloat16).to(tl.float32)
    dg = da * sig * (1.0 + g * (1.0 - sig))
    tl.store(H + i, act * u, i < SIZE)
    tl.store(DG + i, dg, i < SIZE)
    tl.store(DU + i, dy * act, i < SIZE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(778)
    # Exhaust every BF16 bit pattern in a stated finite activation range.
    g = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    g = g[torch.isfinite(g) & (g.abs() <= 80)].cuda()
    u, dy = [torch.randn_like(g) for _ in range(2)]
    g.requires_grad_(True)
    u.requires_grad_(True)
    reference = F.silu(g) * u
    grads = torch.autograd.grad(reference, (g, u), dy)
    refs = (reference, *grads)
    report = dict(
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="All finite BF16 gate patterns in [-80,80], seeded BF16 up/upstream gradient; no whole-model claim",
        elements=g.numel(),
        cases=[],
    )
    for policy in range(5):
        for fusion in (False, True):
            values = [torch.empty_like(g) for _ in range(3)]
            candidate[(triton.cdiv(g.numel(), 256),)](
                g, u, dy, *values, g.numel(), policy, enable_fp_fusion=fusion
            )
            result = dict(policy=policy, enable_fp_fusion=fusion, differences={})
            for name, value, ref in zip(
                ("output", "dG", "dU"), values, refs, strict=True
            ):
                result["differences"][name] = dict(
                    unequal_elements=int((value != ref).sum().item()),
                    max_abs=(value.float() - ref.float()).abs().max().item(),
                )
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
