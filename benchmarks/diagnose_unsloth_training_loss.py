# SPDX-License-Identifier: Apache-2.0
"""Compare first-step gradients on CPU and actual-head CE derivatives on GPU."""

import argparse
import json
from pathlib import Path


def parameters(args):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    report = {}
    for family in ("grad", "updated"):
        rows = []
        with (
            safe_open(
                args.reference / f"initial-{family}.safetensors", framework="pt"
            ) as ref,
            safe_open(
                args.candidate / f"initial-{family}.safetensors", framework="pt"
            ) as candidate,
        ):
            assert set(ref.keys()) == set(candidate.keys())
            error, total = 0.0, 0.0
            for name in ref.keys():
                a, b = (
                    candidate.get_tensor(name).double(),
                    ref.get_tensor(name).double(),
                )
                en, rn = (a - b).square().sum().item(), b.square().sum().item()
                error += en
                total += rn
                rows.append(
                    dict(
                        name=name,
                        relative_l2=(en / max(rn, 1e-40)) ** 0.5,
                        max_abs=(a - b).abs().max().item(),
                        exact=torch.equal(a, b),
                    )
                )
        report[family] = dict(
            aggregate_relative_l2=(error / max(total, 1e-40)) ** 0.5,
            exact_tensors=sum(v["exact"] for v in rows),
            tensors=len(rows),
            worst=sorted(rows, key=lambda x: x["relative_l2"], reverse=True)[:8],
        )
    return report


def kernels(args):
    import torch
    import torch.nn.functional as functional
    from qwen35_training_loss import load_unsloth_ce
    from safetensors.torch import load_file

    ref = load_file(args.reference / "initial-head-logits.safetensors", device="cuda")
    candidate = load_file(
        args.candidate / "initial-head-logits.safetensors", device="cuda"
    )
    assert all(torch.equal(ref[k], candidate[k]) for k in ref), (
        "initial head outputs differ"
    )
    logits = ref["logits"].detach().requires_grad_()
    labels = ref["targets"]
    fast_ce, source = load_unsloth_ce(args.source)
    values, grads = {}, {}
    for method in ("native", "unsloth-ce", "fp64"):
        if method == "unsloth-ce":
            loss = fast_ce(logits, labels, n_items=4092)
            gradient = torch.autograd.grad(loss, logits)[0]
        else:
            x = (
                logits.detach()
                .to(torch.float64 if method == "fp64" else torch.float32)
                .requires_grad_()
            )
            loss = (
                functional.cross_entropy(
                    x.flatten(0, 1), labels.flatten(), reduction="sum"
                )
                / 4092
            )
            gradient = torch.autograd.grad(loss, x)[0].to(logits.dtype)
        values[method] = loss.item()
        grads[method] = gradient
    oracle = grads["fp64"].float()
    native = grads["native"].float()
    actual = grads["unsloth-ce"].float()
    target = labels.unsqueeze(-1)
    return dict(
        source=source,
        initial_logits_exact=True,
        losses=values,
        native_vs_unsloth_relative_l2=((native - actual).norm() / native.norm()).item(),
        native_vs_fp64_relative_l2=((native - oracle).norm() / oracle.norm()).item(),
        unsloth_vs_fp64_relative_l2=((actual - oracle).norm() / oracle.norm()).item(),
        native_target_gradients=grads["native"].gather(-1, target).flatten().tolist(),
        unsloth_target_gradients=grads["unsloth-ce"]
        .gather(-1, target)
        .flatten()
        .tolist(),
        fp64_target_gradients=grads["fp64"].gather(-1, target).flatten().tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["parameters", "kernels"])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "kernels" and args.source is None:
        parser.error("kernel diagnosis requires upstream sources")
    report = parameters(args) if args.mode == "parameters" else kernels(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
