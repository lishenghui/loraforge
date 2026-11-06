# SPDX-License-Identifier: Apache-2.0
"""Allocated-GPU native frozen-head derivative/ownership check."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as functional
    from qwen35_training_graph import sample_logits
    from qwen35_training_head import FrozenHeadLoss, _targets
    from safetensors import safe_open

    torch.manual_seed(791)
    with safe_open(
        args.checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        weight = source.get_tensor("model.embed_tokens.weight").cuda()
    assert weight.dtype == torch.bfloat16 and not weight.requires_grad
    report = dict(
        state="running", vocab=weight.shape[0], hidden=weight.shape[1], cases=[]
    )
    for batch, tokens in ((2, 64), (4, 1024)):
        hidden = torch.randn(
            batch, tokens, weight.shape[1], device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        labels = torch.randint(weight.shape[0], (batch, tokens), device="cuda")
        original = hidden.detach().clone()
        version = weight._version
        logits = functional.linear(hidden, weight)
        reference = functional.cross_entropy(
            logits.float().flatten(0, 1), _targets(labels)
        )
        audit = sample_logits(logits.detach())
        expected = torch.autograd.grad(reference, hidden)[0]
        reference_loss = reference.item()
        del logits, reference
        torch.cuda.empty_cache()
        for strategy in ("full-reuse", "chunked"):
            loss, actual_audit, _ = FrozenHeadLoss.apply(
                hidden, weight, labels, 512, strategy, False
            )
            assert hidden.grad is None and weight.grad is None
            actual = torch.autograd.grad(loss, hidden, retain_graph=True)[0]
            repeated = torch.autograd.grad(loss * 0.5, hidden)[0]
            error = (
                (actual.float() - expected.float()).norm() / expected.float().norm()
            ).item()
            ae = (
                (actual_audit.float() - audit.float()).norm() / audit.float().norm()
            ).item()
            loss_error = abs(loss.item() - reference_loss)
            assert error < 0.05 and ae < 0.02 and loss_error < 0.01
            assert torch.equal(repeated, actual * 0.5)
            assert torch.equal(hidden, original) and weight._version == version
            case = dict(
                batch=batch,
                tokens=tokens,
                strategy=strategy,
                chunk_rows=512,
                loss_max_abs=loss_error,
                hidden_gradient_relative_l2=error,
                sampled_logits_relative_l2=ae,
                exact_hidden_gradient=torch.equal(actual, expected),
                caller_storage_unchanged=True,
                repeated_backward_scaled=True,
            )
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("HEAD_CHECK", case, flush=True)
            del loss, actual_audit, actual, repeated
            torch.cuda.empty_cache()
        del hidden, labels, original, expected, audit
    report["state"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
