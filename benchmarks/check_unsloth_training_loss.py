# SPDX-License-Identifier: Apache-2.0
"""Allocated-GPU derivative/lifetime check of isolated upstream Unsloth CE."""

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--launch", choices=["upstream", "hip"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as functional
    from qwen35_training_loss import load_unsloth_ce

    fast_ce, source = load_unsloth_ce(args.source, launch=args.launch)
    torch.manual_seed(784)
    report = dict(state="running", source=source, cases=[])
    for batch, length, vocab in ((2, 17, 1003), (4, 32, 248320), (8, 64, 248320)):
        logits = (torch.randn(batch, length, vocab, device="cuda") * 3).bfloat16()
        labels = torch.randint(vocab, (batch, length), device="cuda")
        labels[:, -1] = -100
        labels[0, 3] = -100
        logits.requires_grad_()
        original = logits.detach().clone()
        reference = functional.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100
        )
        reference_grad = torch.autograd.grad(reference, logits)[0]
        actual = fast_ce(logits, labels)
        gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        second = torch.autograd.grad(actual * 0.5, logits)[0]
        loss_error = abs(actual.item() - reference.item())
        gradient_error = (
            (gradient.float() - reference_grad.float()).norm()
            / reference_grad.float().norm().clamp_min(1e-20)
        ).item()
        assert loss_error < 1e-4 and gradient_error < 0.02
        torch.testing.assert_close(second, gradient * 0.5, rtol=0, atol=0)
        assert torch.equal(original, logits), "upstream CE changed saved input"
        assert torch.count_nonzero(gradient[:, -1]).item() == 0
        timings = {}
        del original, gradient, second, reference_grad, reference, actual
        for backend in ("native", "unsloth-ce"):
            times = []
            for iteration in range(7):
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss = (
                    fast_ce(logits, labels)
                    if backend == "unsloth-ce"
                    else functional.cross_entropy(
                        logits.float().flatten(0, 1), labels.flatten()
                    )
                )
                grad = torch.autograd.grad(loss, logits)[0]
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if iteration >= 2:
                    times.append(elapsed)
                del grad, loss
            timings[backend] = dict(
                samples_ms=times, median_ms=statistics.median(times)
            )
        case = dict(
            batch=batch,
            tokens=length,
            vocab=vocab,
            loss_max_abs=loss_error,
            gradient_relative_l2=gradient_error,
            caller_storage_unchanged=True,
            repeated_backward_scaled=True,
            timings=timings,
        )
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            "CE_CHECK",
            args.launch,
            batch,
            length,
            loss_error,
            gradient_error,
            flush=True,
        )
        del logits, labels
        torch.cuda.empty_cache()
    report["state"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
