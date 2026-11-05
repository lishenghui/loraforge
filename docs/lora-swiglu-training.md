# BF16 LoRA MLP training fusion

Tiny-rank expansion can be fused into SwiGLU and gradient consumers with a
measurable MLP forward/backward benefit. Activation-only fusion does not beat
the measured `torch.compile` baseline. The broader tiny-rank fusion does at
rank 4, but its advantage over compilation shrinks to a few percent at long
token batches. The same scalar rank-unrolled schedule is substantially slower
at rank 32 and must not be selected as a general LoRA training backend.

This is a standalone benchmark, not an installed training integration, full
Qwen3.5 training result or comparison against actual Unsloth. It builds on
[the earlier NF4 training probe](lora-training-transfer.md), but uses a **BF16
frozen base**. These experiments' speed ratios cannot be multiplied together.
Both candidates and baselines use the same rank; no rank-32-to-4 compression
or training-quality claim is involved.

## Implementation

`benchmarks/lora_swiglu_training.py` implements a first-order LoRA MLP using
frozen, already-fused gate/up base weights `[7168,1024]`, down weights
`[1024,3584]`, BF16 compute/activations, and six independently trainable FP32
A/B master tensors. Gate and up do **not** share A. Scaling is 0.7, dropout
zero, one adapter, no bias or DoRA. The input and all six parameter gradients
are included in the timing and numerical checks.

| Path | Behavior |
| --- | --- |
| `native` | Ordinary Torch LoRA plus native SiLU/multiply autograd |
| `compiled` | The same reference with `torch.compile(fullgraph=True, dynamic=False)` |
| `activation` | Only the SwiGLU forward/backward elementwise work is fused |
| `block` | Custom whole-MLP autograd, saving small shrink results and rematerializing wide H |
| `fused` | Also merges gate/up LoRA B expansion, residual and SwiGLU; down expansion/residual; backward low-rank correction/activation gradients; and LoRA input-gradient corrections |
| `compact` | Same consumer fusion, with a shorter-lived, reused gradient buffer |

Large base and gradient GEMMs remain native. A custom autograd function is not
one GPU kernel. The new Triton kernels explicitly retain BF16 rounding at the
reference's GEMM, scaling, addition and SiLU boundaries. They never cache stale
trainable A/B casts or modify saved inputs. Small rank-sized shrink results are
saved, because the preceding experiment found indiscriminate U recomputation
slower. Only the wide SwiGLU output is reconstructed in backward.

The compact backward first forms the unscaled projection-gradient buffer and
the rematerialized H. After dA consumes H, H is released. The frozen-base dX
GEMM consumes the gradient buffer on the same stream, then that **new gradient
buffer** is scaled in place for LoRA gradient GEMMs. Saved activations and
parameter storage are never overwritten. This first-order-only contract is
marked with `once_differentiable`; higher-order differentiation is unsupported.
The dense CPU/float64 fallback exists solely for derivative/checkpoint tests.

## Measurement protocol

One MI250 GCD, small-g, 7 CPUs, 40GB, 20-minute cap per job. Torch 2.11.0 ROCm
7.2 and Triton 3.6.0. Actual first-layer Qwen3.5-0.8B packed BF16 base weights;
synthetic activations, upstream gradients and LoRA factors. The checkpoint SHA256
is `2e273d1ae568bb24263c1b2633244b80d3428644473c85e98ad36ac00d84942c`.

All paths are warmed and CUDA-graph captured, with fresh leaves on a consistent
warmup/capture stream. Three shuffled samples of eight replays time the whole
**MLP forward plus backward**. Compilation and optimizer execution are excluded
from timing. Memory deltas exclude already-resident inputs/base/master weights;
they are not total-model VRAM. No attention, GDN recurrence, language-model
loss, data loading, multi-layer activation checkpointing, distributed training
or full-model optimizer time is measured.

Job `23488906` established that activation-only and whole-block rematerializing
paths are slower than compilation. Job `23489184` tested consumer fusion and
exposed rank-32 regressions. Job `23489357` adds the compact backward and an
8192-token case. Earlier payloads/results are retained separately; the receipts
record each source hash rather than treating the latest code as every run's
implementation.

## Rank-4 results

Final job `23489357` completed in 70 seconds on `nid005033`. Median microseconds
for matched rank-4 MLP forward plus backward:

| Tokens | Eager | Compiled | Fused | Compact | Eager / fused | Compiled / fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 613.93 | 570.05 | 446.10 | 456.78 | 1.376x | 1.278x |
| 1024 | 1415.31 | 1281.79 | 1115.85 | 1131.25 | 1.268x | 1.149x |
| 4096 | 3742.82 | 3357.51 | 3234.77 | 3305.93 | 1.157x | 1.038x |
| 8192 | 6894.01 | 6194.08 | 5987.30 | 6143.04 | 1.151x | 1.035x |

The preceding job on a different GCD measured fused speed relative to eager of
1.360x/1.264x/1.148x and relative to compiled of 1.261x/1.145x/1.039x at the first
three token counts. Each ratio above uses paths in the same job and shape;
these are repeated local measurements, not a statistical confidence interval.
The long-token fused gain over compilation is modest. Compact/compiled is
1.248x/1.133x/1.016x/1.008x: essentially parity at the longest case, rather
than evidence of a substantial long-context win over an optimized baseline.

At rank 32 the scalar rank-unrolled consumer kernels are **rejected**:
fused is 1.67x to 5.04x slower than compiled across these token counts. For
8192 tokens it takes 30903 us versus compiled's 6128 us. Tiny-rank consumer
fusion must not be unconditionally applied to ordinary ranks. This standalone
experiment keeps the unsuccessful candidate explicit and installs no dispatcher
or default training hook. Ranks 1/2 and other GPUs are not measured here.

### Saved activations versus peak allocation

Both whole-block paths save two wide BF16 gate/up activations instead of the
eager path's four; compiled saves three in this probe. At 4096 tokens, that is
56 MiB of wide saved storage versus 112 MiB eager or 84 MiB compiled. Total
forward-live deltas are 64.09 MiB versus 120.20 MiB eager and 92.20 MiB compiled.
These numbers do not by themselves establish a lower step peak.

| Rank-4 tokens | Eager peak MiB | Compiled peak MiB | Fused peak MiB | Compact peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 128 | 6.11 | 7.43 | 7.38 | 5.63 |
| 1024 | 48.13 | 44.16 | 58.15 | 44.15 |
| 4096 | 192.20 | 176.23 | 232.25 | 176.25 |
| 8192 | 384.29 | 352.29 | 464.37 | 352.37 |

The fastest fused path stores separate scaled and unscaled full gradient
buffers. Compact removes that duplicate and drops peak about 24% relative to
fused at the longest cases, at a 2–3% timing cost. It is about 8% below eager's
step peak and approximately equal to compiled's peak. No total-model memory
advantage over compiled training or Unsloth is established.

## Validation and limits

The check program compares output, dX and all six parameter gradients in eager
execution, then again after graph capture and changing X and every A/B master.
The final run requires elementwise `rtol=0.02, atol=0.003` and relative L2 at
most 0.02. Observed exactness/error is recorded separately from tolerance.
Three AdamW updates from zero-B initialization also compare loss and updated
parameters; this is a short optimizer sanity check, not a convergence study.

All eight shape/rank cases and five optimizer paths completed. Rank-4 fused
and compact output/all gradients are bitwise equal to eager at
128/1024/4096 tokens, before and after changed-parameter graph replay. At 8192
tokens the initial check is exact, but changed-parameter replay has maximum
absolute errors 4.768e-7 in output and 9.537e-7 in dX, with maximum relative L2
9.559e-7; all six parameter gradients remain exact. Do not describe the entire
rank-4 result as universally bitwise identical. The rejected rank-32 paths
also have nonzero differences, recorded in the receipts. Compiled has nonzero
BF16 differences too; all measured paths satisfy the declared numerical gate.
Loss trajectories in the three-step rank-4 optimizer check are identical, and
updated parameters satisfy the stated optimizer-check tolerance.

[The curated receipt](lora-swiglu-training-mi250.json) includes raw final timing
samples, saved-storage/peak measurements, eager and changed-graph numerical
errors, optimizer losses, earlier-run summaries and source/input hashes.

CPU: 190 passed, 3 skipped, 49 GPU tests deselected. The 12 new cases include
float64 gradcheck and multi-step AdamW with non-reentrant checkpointing and
with/without input gradients. Ruff and wheel/sdist checks accompany the change.

Reproduce explicitly on an allocated gfx90a GPU with the packed BF16 fixture:

```bash
PYTHONPATH=/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/check_lora_swiglu_training.py PACKED_BF16/model.safetensors \
  --output swiglu.json --tokens 128 1024 4096 8192
```

The next model-level step is to attach a rank-4 wrapper with correct parameter
ownership to a BF16 Qwen3.5 training baseline, then compare a complete step
including GDN/attention/loss/checkpointing against PEFT, compiled PEFT and
actual Unsloth under matched settings. Neither this microbenchmark nor its
three optimizer updates establish full-model speed, task quality or superiority
over Unsloth. No training hook is installed and serving PRs are unchanged.
