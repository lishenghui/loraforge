# Transferring serving optimizations to LoRA training

The NF4 scale/dequantization optimization transfers to a frozen-base training
linear with a proper input-gradient path. It is a projection-level feasibility
result, not a whole-model training speedup or a comparison against Unsloth.
Training code stays in standalone benchmarks and changes no installed serving
integration. The experiment uses a separate checkout/branch from the NF4 PR.

## What the Unsloth source establishes

Reviewed Unsloth revision `157305f6596d6196189bbc3e393bb8fe8ed604a0`:

- [`LoRA_W` and `LoRA_MLP`](https://github.com/unslothai/unsloth/blob/157305f6596d6196189bbc3e393bb8fe8ed604a0/unsloth/kernels/fast_lora.py)
  use custom autograd boundaries, retain frozen quantized weights rather than
  a persistent dequantized base, compute input/A/B derivatives, and rematerialize
  selected intermediates. One autograd function is not necessarily one GPU
  kernel. The current source also guards configurations requiring other paths.
- [`swiglu.py`](https://github.com/unslothai/unsloth/blob/157305f6596d6196189bbc3e393bb8fe8ed604a0/unsloth/kernels/swiglu.py)
  recomputes the activation from projection results in backward and combines
  elementwise gradient work. This targets wide activation traffic, beyond merely
  accelerating a small rank-sized temporary.

These are established upstream ideas, not a novelty claim. The local prototype
is new code using ordinary LoRA derivatives and the existing LoRAForge NF4
dequantizer; no Unsloth implementation was copied.

For Qwen3.5 specifically, the [current official training guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
warns about greater quantization differences in 4-bit training and presents
BF16 LoRA examples. This NF4 operator probe does not validate Qwen3.5 QLoRA
quality. A subsequent whole-model training study should first establish a BF16
reference and compare with a properly configured Unsloth baseline.

## Training changes the contract

With row-major inputs and factors:

\[
U=XA^T,\qquad Y=XW^T+sUB^T
\]

For upstream gradient \(G\):

\[
G_U=sGB,\quad dB=sG^TU,\quad dA=G_U^TX,\quad dX=GW+G_UA.
\]

Frozen W eliminates its parameter gradient, not the large `GW` input-gradient
GEMM. The implementation also preserves native BF16 rounding of the scaled
upstream gradient rather than freely reassociating mixed-precision products.

| Serving optimization | Training transfer | Required additional work |
| --- | --- | --- |
| Immutable NF4 scale reuse and exact decode | Implemented in this probe | Custom base backward must produce dX without retaining full BF16 W |
| A+B fusion | Conditional | dA/dB token reductions and dX; choose saved versus recomputed U |
| B-at-SiLU/SwiGLU | Promising next candidate | Save/recompute corrected activations and implement their backward, preserving rounding |
| GDN B-at-unpack | Conditional, not implemented | Preserve convolution/recurrent gradient flow and actual parameter sharing |
| Decode RowExpand schedule | Not a direct training port | Training has long token batches and global dB reductions; compare against dense/grouped GEMM |
| Cached inference A/B copies or pool layouts | Unsafe to reuse unchanged | Trainable factors change every step; copies must refresh and gradients reach master parameters |
| Spectral rank compression during training | Separate algorithmic study | Coordinate changes/rank changes interact with Adam moments and the optimization problem |

Shared shrink can be reused only when projections share the actual trainable A
parameter. Independent A tensors that happen to start equal are not equivalent
to a tied parameter. Inference state/KV cache reuse is not activation
checkpointing and must not replace recurrent training state.

## Implemented probe and reference

[`benchmarks/lora_training.py`](../benchmarks/lora_training.py) exposes four
explicit experiment paths:

- `bnb`: native bitsandbytes frozen 4-bit base plus ordinary PyTorch LoRA.
- `cached`: the existing exact dequantizer and FP32 scale cache, wrapped in a
  custom base autograd function. The base is decoded independently for forward
  and input-gradient backward; no full BF16 base is saved for backward.
- `naive`: direct dequantize-plus-Linear under ordinary autograd. Its derivatives
  are correct, but it retains full BF16 weights when dX is required. It exposes
  a real speed/memory tradeoff rather than an additional correctness failure.
- `recompute`: custom complete LoRA backward that reconstructs U and refreshes
  casts of A/B. It saves less forward state but performs extra GEMM work.

The base bytes and quantization state must be immutable. Only the explicitly
named dense reference supports CPU/float64 derivative checks. The GPU probe has
BF16 activations/compute, FP32 trainable LoRA master factors, dropout zero, one
adapter, no bias or DoRA, and first-order gradients only. There is no DDP/FSDP,
mixed-adapter optimizer, full GDN model, SFT convergence or torch.compile claim.

## MI250 measurements

Job `23487644`, node `nid005060`, completed in 41 seconds on one MI250 GCD with
7 CPUs, 40GB and a 20-minute cap. Torch 2.11.0 ROCm 7.2, Triton 3.6.0,
bitsandbytes 0.50.2. Four actual NF4 checkpoint projections:
8192x1024, 7168x1024, 1024x2048, 1024x3584. Ranks 4/32 and token counts
128/1024/4096 give 24 cases. Activations and LoRA factors are synthetic; base
weights are the previously checked Qwen3.5 fixture.

Timings are **one projection's forward plus backward**, captured and replayed
with three shuffled samples of eight iterations. They exclude optimizer time,
data loading, attention, recurrence, full-model checkpointing and language-model
loss. Both reference and candidates use the same precision and graph regime.

| Tokens per projection | Cached/BNB minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| 128 | 1.3280x | 1.3657x | 1.4721x |
| 1024 | 1.1396x | 1.1526x | 1.1711x |
| 4096 | 1.0521x | 1.0643x | 1.0789x |

These ranges span shapes/ranks; they are not confidence intervals. The maximum
observed absolute difference is **zero** for output, dX, dA and dB across all
24 cases, both eager and captured replay after changing X/A/B. Declared tensor
tolerances are retained in the check program, alongside the observed exactness.
Four projection shapes also pass three-step AdamW tests from zero-B
initialization in the BNB/cached/recompute paths: the saved loss trajectories
are identical, and updated parameters satisfy the check tolerance. Optimizer
correctness checks are not optimizer timing benchmarks.

The rematerialized-U variant does not beat the cached-base/ordinary-LoRA path
in any of these 24 cases. Its median speed relative to cached is
0.9360x/0.9566x/0.9716x at 128/1024/4096 tokens. Saving the tiny rank-sized U
is less useful here than avoiding another GEMM. This rejects a blanket
"recompute all LoRA intermediates" policy; it does not reject rematerializing
wide activation tensors or tuned full-MLP backward implementations.

## What the memory check proves

For the 8192x1024 projection, `naive` retains an additional **16 MiB** BF16 base
weight for backward. The BNB/cached/recompute paths retain zero BF16 base-weight
bytes. Cached scales add a resident **0.5 MiB** for this projection; across the
114 decoder projections the corresponding cache is about 31 MB.

Memory deltas in the report are measured after the common base/cache preparation.
They are incremental operator allocations, not total-model or total-device
VRAM. In particular, they must not hide the scale-cache overhead or be presented
as a memory reduction over Unsloth. The cached path broadly preserves the
native BNB weight-lifetime behavior; its benefit over BNB is the measured
projection speed, not a new fourfold training-memory saving.

The full [receipts and raw samples](lora-training-transfer-mi250.json) include
live-forward allocations, peak-step deltas, backward-saved weight bytes,
gradient comparisons, changed-input graph checks and AdamW losses.

## Validation and rejected attempts

CPU: 178 passed, 3 skipped, 49 GPU tests deselected. The added tests include
float64 finite-difference gradcheck, no-input-gradient cases, zero-B
initialization, multiple AdamW updates and non-reentrant checkpointing.
Ruff and wheel/sdist build checks accompany the standalone experiment.

Jobs `23487434` and `23487557` failed in the probe's memory inventory: when the
input activation shape equaled the base-weight shape, a shape-only filter
counted that activation as BF16 weight storage. The second attempt still had
the incorrect filter. No partial timing is used in the final aggregate. The
completed run excludes the input storage identity and gives fresh captured
leaves a consistent warmup/capture stream. Numerical training code was unchanged.

To reproduce on an allocated gfx90a GPU with the prepared fixture:

```bash
PYTHONPATH=/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/check_lora_training.py FIXTURE --output training.json
```

The BF16 LoRA MLP follow-up is now recorded in
[the SwiGLU training experiment](lora-swiglu-training.md), including an actual
`torch.compile` baseline. It is a separate BF16 experiment; its speed ratios
must not be multiplied by these NF4 ratios to invent a combined result.

A subsequent model-level experiment should validate a GDN projection block:
validate gradients first, measure saved activations and forward/backward costs,
then attach to a small Qwen3.5 SFT run. Compare against native PEFT and actual
Unsloth with matched checkpointing, loss fusion, precision and batch tokens.
The present result cannot establish superiority over Unsloth or complete-model
training acceleration.
