# SPDX-License-Identifier: Apache-2.0
"""First-order frozen LM-head/CE experiments using native Torch derivatives.

Precompute dHidden in forward, retaining only that small owned buffer. The
original frozen weight and caller hidden/labels are never modified. This is
standalone benchmark code, not a Trainer hook or a trainable-head backend.
"""

import torch
import torch.nn.functional as functional
from torch import nn
from torch.autograd.function import once_differentiable


def _targets(labels):
    targets = torch.empty_like(labels)
    targets[:, :-1].copy_(labels[:, 1:])
    targets[:, -1].fill_(-100)
    return targets.flatten()


def _audit_storage(hidden, weight, full_audit):
    batch, tokens, _ = hidden.shape
    positions = (0, tokens // 2, tokens - 2)
    sampled = hidden.new_empty((batch, 3, (weight.shape[0] + 256) // 257))
    full = (
        hidden.new_empty((batch, 3, weight.shape[0]))
        if full_audit
        else hidden.new_empty((0,))
    )
    rows = [
        (b * tokens + p, b, j) for b in range(batch) for j, p in enumerate(positions)
    ]
    return sampled, full, rows


def _collect_audit(logits, start, sampled, full, rows):
    for row, batch, position in rows:
        if start <= row < start + logits.shape[0]:
            value = logits.detach()[row - start]
            sampled[batch, position].copy_(value[::257])
            if full.numel():
                full[batch, position].copy_(value)


class FrozenHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_rows, strategy, full_audit):
        if (
            hidden.ndim != 3
            or labels.shape != hidden.shape[:2]
            or weight.ndim != 2
            or hidden.shape[-1] != weight.shape[1]
            or hidden.shape[0] < 1
            or hidden.shape[1] < 2
            or labels.dtype != torch.long
            or hidden.device != weight.device
            or hidden.device != labels.device
            or hidden.dtype != weight.dtype
        ):
            raise ValueError(
                "matching rectangular hidden/labels and frozen head required"
            )
        if weight.requires_grad or chunk_rows < 1:
            raise ValueError("frozen head and positive chunk size required")
        if strategy not in {"chunked", "full-reuse"}:
            raise ValueError("strategy must be chunked or full-reuse")
        flat = hidden.detach().reshape(-1, hidden.shape[-1])
        targets = _targets(labels)
        denominator = (targets != -100).sum()
        loss_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        losses = torch.empty(flat.shape[0], dtype=loss_dtype, device=hidden.device)
        grad_hidden = torch.empty_like(flat)
        sampled, full, rows = _audit_storage(hidden, weight, full_audit)
        owned = functional.linear(flat, weight) if strategy == "full-reuse" else None
        if owned is not None:
            _collect_audit(owned, 0, sampled, full, rows)
        device = hidden.device.type
        amp_enabled = torch.is_autocast_enabled(device)
        amp_dtype = torch.get_autocast_dtype(device)
        for start in range(0, flat.shape[0], chunk_rows):
            end = min(start + chunk_rows, flat.shape[0])
            with (
                torch.enable_grad(),
                torch.autocast(
                    device, enabled=amp_enabled, dtype=amp_dtype, cache_enabled=False
                ),
            ):
                source = (
                    (owned[start:end] if owned is not None else flat[start:end])
                    .detach()
                    .requires_grad_(True)
                )
                logits = (
                    source if owned is not None else functional.linear(source, weight)
                )
                if owned is None:
                    _collect_audit(logits, start, sampled, full, rows)
                per_row = functional.cross_entropy(
                    logits.to(loss_dtype), targets[start:end], reduction="none"
                )
                scales = torch.ones_like(per_row) / denominator
                gradient = torch.autograd.grad(per_row, source, grad_outputs=scales)[0]
            losses[start:end].copy_(per_row.detach())
            if owned is None:
                grad_hidden[start:end].copy_(gradient)
            else:
                # owned is private scratch. No saved graph/consumer retains
                # these original logits after this chunk's autograd.grad.
                owned[start:end].copy_(gradient)
            del source, logits, per_row, scales, gradient
        if owned is not None:
            grad_hidden.copy_(owned.matmul(weight))
            del owned
        ctx.save_for_backward(grad_hidden.reshape_as(hidden))
        ctx.mark_non_differentiable(sampled, full)
        ctx.set_materialize_grads(False)
        return losses.sum() / denominator, sampled, full

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, grad_sampled, grad_full):
        (gradient,) = ctx.saved_tensors
        # Never scale the saved buffer in place: retained backward calls and
        # later graph replays must see its original contents. BF16 scaled-loss
        # rounding can differ; this experiment uses unit upstream loss grad.
        result = None if grad_loss is None else gradient * grad_loss
        return result, None, None, None, None, None


class HeadLossAndSamples(nn.Module):
    """Keep original PEFT parameters; call its already-adapted text backbone."""

    def __init__(
        self, model, *, chunk_rows, strategy, implicit_causal=False, full_audit=False
    ):
        super().__init__()
        self.model = model
        self.chunk_rows = chunk_rows
        self.strategy = strategy
        self.implicit_causal = implicit_causal
        self.full_audit = full_audit
        self.last_full_samples = None

    def forward(self, ids):
        base = self.model.get_base_model()
        kwargs = {}
        if self.implicit_causal:
            from qwen35_training_mask import fixed_positions

            kwargs["position_ids"] = fixed_positions(ids)
        output = base.model(input_ids=ids, use_cache=False, **kwargs)
        loss, sampled, full = FrozenHeadLoss.apply(
            output.last_hidden_state,
            base.lm_head.weight,
            ids,
            self.chunk_rows,
            self.strategy,
            self.full_audit,
        )
        self.last_full_samples = full if self.full_audit else None
        return loss, sampled[0] if ids.shape[0] == 1 else sampled
