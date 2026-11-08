# Unsloth loss ideas for Qwen3.5 LoRA training

The later [native-CE/head-buffer study](qwen35-training-head.md) qualifies a
different full-shape scratch-reuse path at batch four and runs batch eight.
The isolated Unsloth CE results and failures below remain unchanged.

The first CE-only ablation **fails the existing whole-model native numerical
gates** and does not make batch eight fit. Its roughly 3% raw timing gain is
not an accepted equivalent-training speedup. The separate first-step
diagnostic `23515537` is also complete. Both jobs have released their GPUs.

## First measured result

Job `23515154` completes in 5m08s on `nid005036`, small-g, one MI250X GCD,
seven requested CPUs, 40GB host memory and a 20-minute cap. The unmodified
launch fails at the large vocabulary: it requests **2048 threads** against a
**1024-thread hardware limit**. The HIP configuration passes all three
operator loss/gradient/lifetime cases, with maximum loss absolute error
1.91e-6 and gradient relative L2 1.47e-6. The 512-row/248320-vocabulary
forward/backward operator measures 5.473 ms native versus 1.017 ms fused
(5.38x); the small-vocabulary case regresses. These are not model timings.

Complete model at batch four × 1024, BF16 rank four, no checkpointing:

| Method | Step ms | Input tokens/s | Peak reserved GiB | Native numerical gates |
| --- | ---: | ---: | ---: | --- |
| Original PEFT before | 1282.725 | 3193 | 46.104 | Reference |
| Isolated CE + HIP launch | 1244.337 | 3292 | 38.525 | **Failed** |
| Same CE + native graph | 1248.230 | 3281 | 36.773 | **Failed** |
| Original PEFT after | 1282.202 | 3195 | 46.104 | Exact to before |

The raw CE timing ratio is 1.0304–1.0309x; graph is 1.0272–1.0276x.
Memory figures are PyTorch allocator reserved peaks, not total-device VRAM.
The two CE methods are exactly equal in their saved losses, sampled logits,
all factor gradients/updates and changed-input audits. Both differ from
original PEFT after updates: at the third step, aggregate gradient relative
L2 is **1.1145**, 426/496 gradients fail the existing per-parameter gate,
495/496 updates fail the 1e-4 absolute gate, and sampled-logit relative L2
is **0.0930**. Third-step loss differs by 0.00324; checking loss alone would
miss the failure. The original before/after controls are exact. This is not a
graph-only discrepancy and the result does not establish training quality.

At batch eight × 1024, native again fails to allocate **7.58 GiB** at the
FP32 logits upcast. Both CE methods instead reach backward and fail at
`dlogits = torch.empty_like(logits)`, requesting **3.79 GiB** with only
3.00/3.06 GiB device memory free. Avoiding the forward FP32 matrix alone
does not remove the large BF16 logits and gradient buffers. There is no valid
native batch-eight numerical reference, no completed candidate step, and no
batch-eight throughput claim. A chunked LM-head/loss path must address these
remaining buffers separately.

## First-step diagnostic

Job `23515537` completes in 2m31s on `nid005071`, one small-g GCD, seven
requested CPUs, 40GB host memory and a seven-minute cap. It runs native and
CE eager workers from identical initial adapters, saving gradients and updates
after the first actual backward/AdamW step. It also saves the full vocabulary
of three sampled positions from every sequence: **12 head rows** in total.

The initial sampled logits and targets are exact between methods. On these
same rows, isolated CE and native CE produce exactly equal BF16-rounded
gradients. Their gradient relative L2 against a BF16-rounded FP64 oracle is
4.48e-9. This is a sampled-head derivative check, not a comparison of all 4096
head rows or proof that all backward intermediates agree.

Across all 496 trainable factors, the first model backward already differs:
aggregate gradient relative L2 **0.02259**, with 248 exact tensors (the initial
zero A gradients) and 248 non-exact B gradients. The largest per-factor
relative L2 is **0.06523**. The first update differs by up to approximately
**0.0002000** and only 248 tensors remain exact. The largest B-update differences
are near twice the 1e-4 learning rate, consistent with opposite update signs.
AdamW sensitivity to small gradient changes is a plausible amplification
mechanism; the exact first differing full-head/layer intermediate has not been
isolated. The later multi-step drift must not be attributed to a single kernel
or declared harmless rounding from this evidence.

The initial shared forward and sampled derivative results do not show a simple
causal-label shift mistake. They also do not establish that fused CE harms
training quality: that would require convergence/task-quality evidence. The
present outcome is narrower: the CE-only path does not meet our existing
native-training equivalence gates, and its 3% timing gain cannot be presented
as a validated equivalent optimization. A next ablation should combine
chunked LM-head/loss memory handling with fixed-state derivative comparisons
before interpreting optimizer trajectories.

The [curated receipt](qwen35-training-loss-mi250.json) preserves timing samples,
unchanged gates, failed-tensor counts with bounded examples, full initial-step
summaries, both actual source manifests and OOM/thread-limit evidence. Original
large audit tensors and full failed lists remain outside the package.

## What the upstream implementation changes

Reviewed Unsloth revision `a0197ca266e14d22f66c0dcffc452fe1ce42bb0f`
and Unsloth Zoo revision `771dcfb9d9da713db32cc2abda73299ff3db33cd`.

The [Apache-2.0 CE implementation](https://github.com/unslothai/unsloth/blob/a0197ca266e14d22f66c0dcffc452fe1ce42bb0f/unsloth/kernels/cross_entropy_loss.py)
loads BF16 logits into FP32 registers, computes large-vocabulary reductions
in chunks, and writes a fresh BF16 gradient in backward. It saves the original
logits instead of overwriting them, allowing another loss or a retained-graph
backward to use the original values. It does not allocate the complete logits
matrix as FP32. This directly targets the previous experiment's failure at
`ForCausalLMLoss: logits = logits.float()`.

Zoo's [loss registration](https://github.com/unslothai/unsloth-zoo/blob/771dcfb9d9da713db32cc2abda73299ff3db33cd/unsloth_zoo/loss_utils.py)
redirects causal-loss aliases, including Qwen3.5 conditional-generation aliases
that otherwise retain the stock upcast. Its separate
[chunked LM-head/CE path](https://github.com/unslothai/unsloth-zoo/blob/771dcfb9d9da713db32cc2abda73299ff3db33cd/unsloth_zoo/fused_losses/cross_entropy_loss.py)
operates on chunks of hidden states and projects each chunk, limiting logits
materialization further. These are distinct optimizations: replacing CE alone
still creates the full BF16 LM-head output.

For our actual 248320-token vocabulary, batch eight times 1024 positions has a
full FP32-logits size of **7.58 GiB**, plus BF16 logits and backward/workspace
storage. The [previous measured OOM](qwen35-training-batches.md) requested
exactly this allocation with less than 3 GiB free. Avoiding it is a well-founded
capacity hypothesis; it does not establish a measured speedup.

## Standalone first ablation

Keep the unchanged original 4B checkpoint, all-linear rank-four adapters,
BF16 base/compute, FP32 LoRA masters, dropout zero and actual microbatches.
Gradient accumulation stays one; checkpointing stays off. Preserve the native
LM-head GEMM, native FLA GDN layers, full attention and real foreach AdamW.
Only the model instance's loss function is replaced; the installed framework
and global loss registries are unchanged.

`qwen35_training_loss.py` requires the three original upstream files with
fixed SHA256 hashes. It selects unchanged kernel/autograd definitions with
AST while supplying their small Torch/Triton dependency bindings. It does
not import or benchmark the complete Unsloth trainer. See
[provenance](provenance.md#isolated-unsloth-cross-entropy-experiment).

On gfx90a the pinned upstream helper chooses a forward chunk of 65536
elements and 32 warps; its `is_cdna` architecture list does not include gfx90a.
The separate `hip` launch explicitly selects 16384 elements and eight warps,
without changing the kernel equations. Both settings receive an operator
check; only the HIP setting proceeds to model training, and only after passing
its loss/gradient checks. No performance equivalence to an unchanged upstream
launch is implied.

Each operator check compares BF16-input loss/gradients to native FP32 CE,
checks ignored targets, verifies unchanged input storage and performs a second
scaled backward through the retained graph. It tests vocabularies 1003 and
248320 and records five timed forward/backward samples after two warmups.
These are operator timings, not complete-step results.

The model sweep uses **4 × 1024** and **8 × 1024**. Each shape runs, in fresh
sequential GPU processes:

1. Original PEFT/native loss before.
2. Original PEFT with isolated Unsloth CE and the reported HIP launch.
3. Native-autograd graph with the same CE and fixed causal policy.
4. Original PEFT/native loss after, if the first reference fits.

Each successful worker retains the previous three correctness optimizer
updates, one changed-input backward, reset, two warmup updates and five timed
updates. Loading, full base-weight auditing, compilation, capture and audit
copies stay outside step timing. The original loss, sampled-logit,
per-parameter gradient/update and per-sequence gates are unchanged. A native
reference OOM cannot become a passing whole-model native numerical comparison:
candidate eager/graph consistency is reported separately when possible.

## Reproduction and checks

Download the original `cross_entropy_loss.py`, `utils.py` and
`rope_embedding.py` from `unsloth/kernels/` at the pinned revision into an
external `UPSTREAM_KERNELS` directory, retaining their Apache notices. With
the previously prepared 4B text checkpoint and isolated FLA environment, on an
allocated gfx90a GPU:

```bash
PYTHONPATH=/absolute/path/to/deps:/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/run_qwen35_training_loss.py run BF16_TEXT_4B UPSTREAM_KERNELS \
  --output RESULTS --max-seconds 1100 --shapes 4x1024 8x1024
```

Run cross-method analysis on CPU after GPU workers finish:

```bash
python benchmarks/run_qwen35_training_loss.py compare RESULTS \
  --output comparison.json
```

For the separate diagnostic, run the two eager workers with
`--audit-initial-step`, one length and one checkpointing setting, then use
`diagnose_unsloth_training_loss.py parameters REFERENCE CANDIDATE --output JSON`
on CPU. Its `kernels` mode needs an allocated GPU and `--source UPSTREAM_KERNELS`;
the current oracle experiment is scoped to batch four × 1024 and normalizes
the sampled rows by the full batch's 4092 supervised tokens. Diagnostic copies
are outside timing. Original failed attempts remain intact.

CPU regression checks pass: **245 passed**, three skipped, 49 GPU unit cases
deselected. Tests cover causal shifts within each sequence, ignored labels,
actual loss normalization, gradient ownership and rejection of modified
external source before execution. CPU tests do not establish GPU support or
training speed. The full fused/chunked LM-head path is not implemented here.
