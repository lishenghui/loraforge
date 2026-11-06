# Qwen3.5 frozen LM-head memory study

The native-compatible strategy now runs **batch eight × 1024** on one MI250X
GCD. It retains the original full-head GEMM shapes and reuses one private BF16
output buffer for its gradient, processing CE in 512-row chunks. At batch four,
all saved adapter gradients, updates and sampled logits match original PEFT
exactly. This is a capacity improvement with nearly unchanged same-batch time.

A separate fully chunked head implementation avoids the full output buffer,
but fails native-training numerical gates and is slower. The full study's CPU
comparison therefore exits nonzero; that failure remains visible in the
[receipt](qwen35-training-head-mi250.json). Only the full-reuse strategy is
qualified within the measured scope below.

## Complete-step result

LUMI job `23518127` completes in **9m43s** on `nid005034`: small-g, one MI250X
GCD, seven requested CPUs, 40GB host memory and a 20-minute cap. Ten successful
workers perform **100 real AdamW updates and ten additional changed-input
backwards**. Original PEFT at batch eight encounters GPU OOM; its second
control is skipped. This study's GPU allocation is released.

The original 4B text checkpoint is unchanged. BF16 base/compute, FP32 LoRA
masters, all-linear rank four, alpha four, dropout zero, no checkpointing,
gradient accumulation one, no padding/packing or KV cache. Each complete
worker includes the full 32-layer text backbone, frozen LM head, causal loss,
backward, zero_grad and actual foreach AdamW. The sequences are the existing
deterministic synthetic microbatch fixture, not a dataset/convergence test.

Median synchronized complete-step wall time; reserved memory is the PyTorch
allocator peak, not total-device VRAM:

| Batch × length | Method | Step ms | Input tokens/s | Peak reserved GiB | Numerical evidence |
| --- | --- | ---: | ---: | ---: | --- |
| 4 × 1024 | PEFT before / after | 1313.300 / 1314.629 | 3119 / 3116 | 46.033 / 46.104 | Exact controls |
| 4 × 1024 | Full-head scratch reuse | 1315.538 | 3114 | 36.635 | Passes native gates |
| 4 × 1024 | Full-head reuse + graph | 1318.284 | 3107 | 36.350 | Passes native gates |
| 4 × 1024 | Fully chunked head | 1352.214 | 3029 | 34.979 | **Fails native gates** |
| 4 × 1024 | Fully chunked + graph | 1358.730 | 3015 | 34.693 | **Fails native gates** |
| 8 × 1024 | Original PEFT | GPU OOM | — | — | No native reference |
| 8 × 1024 | Full-head scratch reuse | 2548.190 | 3215 | 62.805 | Eager/graph exact consistency |
| 8 × 1024 | Full-head reuse + graph | 2550.826 | 3211 | 62.912 | Eager/graph exact consistency |
| 8 × 1024 | Fully chunked head | 2626.131 | 3119 | 59.494 | Differs from full-reuse reference |
| 8 × 1024 | Fully chunked + graph | 2630.842 | 3114 | 59.186 | Same as chunked eager |

Same-batch full reuse is essentially unchanged in time: **0.9983–0.9993x**
PEFT/worker at batch four; graph gives **0.9962–0.9972x**. Before/after native
control medians differ by about 0.10%. There are five samples after two warmup
updates; these ranges bracket controls, not confidence intervals.

Increasing capacity from original PEFT batch four to full-reuse batch eight
raises measured throughput **3.08–3.18%**. This compares different microbatches;
there is no runnable original-PEFT batch-eight timing and no same-batch 1.03x
claim. It is not a time-to-quality result. Reducing loss memory does not remove
the frozen backbone's forward/input-gradient work, and no stage profile here
identifies the dominant remaining kernel.

## How memory is handled

Both paths compute the frozen-head derivative with native Torch CE/autograd
inside the custom function's forward. They retain only owned dHidden for the
outer model backward, which still updates the actual original LoRA Parameters.
The optimizer stays outside capture and no model parameter gradient is
accumulated early.

- **Full reuse:** compute the original full BF16 head projection once. Copy
  detached audit samples before reuse. For each 512-row CE chunk, use native
  FP32 CE and copy its BF16 dLogits back into that private output slice after
  the chunk's autograd has consumed the original values. Then execute the
  original full-shape dLogits × W GEMM. There is one full private BF16 scratch
  matrix, not simultaneous full BF16 logits and gradient matrices. Caller
  hidden/labels/weights and exposed audit samples remain unchanged.
- **Fully chunked:** project each 512-row hidden chunk, perform native CE and
  its head backward immediately, and retain just that chunk's dHidden. No
  full-vocabulary output/gradient matrix is materialized. Both large GEMM
  shapes change, which can change rounding.

The retained strategy therefore does **not** eliminate every full-size head
buffer. It eliminates duplication and full-size FP32 loss intermediates while
preserving the native large GEMM shapes. The distinction matters: presenting
it as a completely chunked LM head would overstate the result.

This follows established chunked-loss/precomputed-gradient ideas discussed in
the [Unsloth review](qwen35-training-loss.md). The new implementation calls
native Torch operations; no Unsloth/Zoo kernel or source is imported. See
[provenance](provenance.md#frozen-lm-head-and-native-ce-chunking).

## Numerical and lifecycle evidence

Four selected full-reuse comparisons pass: eager/native and graph/native at
batch four, graph/eager at batch four, and graph/eager at batch eight.

At batch four, both full-reuse modes have exact saved third-step gradients
and updates for **all 496 trainable tensors**, exact sampled logits from every
sequence, exact changed-input loss/logits/all gradients, and exact first-step
gradients/updates. Correctness losses are exact. The maximum timed-loss scalar
difference from native is only **2.98e-8**; no bitwise timed-loss claim is made.
All original loss/logit/per-parameter gradient/update gates remain unchanged.

Batch eight has no original native whole-model reference because native OOMs
at its 7.58 GiB FP32-logits allocation. Full-reuse eager and graph are exact
for every saved tensor and loss, including all 496 gradients/updates,
changed-input checks and timed losses. This establishes consistency of the
new paths, not a direct native batch-eight comparison.

Each graph case has two captures, 22 total replays and ten timed replays over
five measured updates. It precomputes fresh head gradients in each forward
replay from current backbone output/labels; there is no stale cached gradient.
All workers verify 427 loaded state entries, the original adapter identities,
and real isolated FLA GDN/causal-convolution kernels.

The fully chunked variant fails native equivalence. At batch four its initial
model-gradient aggregate relative L2 is **0.02232**, with only 248/496 initial
gradients and updates exact. At step three its maximum per-factor gradient
relative L2 is **1.7255** and loss differs by 0.00605. Eager/graph agree with
each other, so graph replay is not the source of that difference.

The isolated actual-weight operator check is exact for full reuse at 128 and
4096 rows. Fully chunked at 4096 rows has dHidden relative L2 **1.335e-4** and
sampled-logit relative L2 **2.764e-5**, while loss is equal. Combined with the
full-shape control, this supports GEMM-shape numerical changes as a source of
drift. It does not isolate a particular reduction instruction or establish
training quality. The failed variant remains experimental and unqualified.

## Scope and reproduction

This is first-order standalone benchmark code for the plain frozen, bias-free
Qwen3.5 head and original single-adapter backbone. It bypasses the outer
CausalLM head wrapper to call that already-adapted backbone, preserving every
trainable Parameter object. Arbitrary PEFT head adapters, custom forward
hooks, trainable heads, higher derivatives, padding/packing and Trainer
integration are outside scope. The benchmark uses unit upstream loss gradients
with BF16 and no GradScaler. Scaling a precomputed BF16 derivative can round
differently; scaled-loss/gradient-accumulation configurations are not validated.

Use the unchanged [4B checkpoint/environment](qwen35-training-4b.md). On an
allocated gfx90a GCD:

```bash
PYTHONPATH=/absolute/path/to/deps:/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/run_qwen35_training_head.py run BF16_TEXT_4B \
  --output RESULTS --max-seconds 1100 --shapes 4x1024 8x1024
```

Analyze on CPU after GPU workers finish:

```bash
python benchmarks/run_qwen35_training_head.py compare RESULTS \
  --output comparison.json
```

This includes the rejected fully chunked ablation; analysis intentionally exits
one when those unchanged numerical gates fail. Retain the complete report.
To run the retained eager path alone, use
`benchmark_qwen35_training.py BF16_TEXT_4B --backend peft --loss-backend head-full-reuse --head-chunk-rows 512 --batch-size 8 --tokens 1024 --checkpointing 0 --check-input-change --output NEW_RESULT`.
Keep actual attempt directories and audit tensors outside the package.

CPU: **253 passed**, three skipped, 49 GPU unit cases deselected; Ruff and
wheel/source-archive checks pass. The curated receipt retains both qualified
and failed comparisons, initial-step summaries, all samples, checkpoint/source
hashes, graph counters and native-OOM evidence. No installed framework or
research/paper source is modified.
