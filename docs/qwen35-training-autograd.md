# Native-autograd graph isolation

This standalone BF16 Qwen3.5-0.8B training follow-up separates Dynamo, AOT
autograd and graph scheduling. The [preceding experiment](qwen35-training.md)
shows short-sequence speed with Torch's `cudagraphs` backend but fails native
PEFT gradient/update gates. Inspection of the installed Torch backend confirms
that it invokes AOT autograd before constructing graphs; those two effects
therefore require separate controls.

## Verified fixed-causal graph result

Job `23502375` completes all five required workers in 3m22s on `nid005038`:
15 cases, 150 real optimizer updates and 15 changed-input forward/backward
checks. **All 18 CPU comparisons are exact**, not merely within tolerance:
three correctness losses, sampled logits, every one of 372 gradients and 372
updated Parameter tensors, five timed-update losses, and changed-input
loss/gradients/logits. All three candidate paths match the unmodified native
PEFT controls. No acceptance threshold is relaxed.

The repaired native-autograd graph path retains the original implicit causal
SDPA/GQA execution. Rank remains four in both candidate and reference; this is
scheduling/attention-path optimization, not rank-compression speedup.

| Tokens | PEFT before / after ms | Fixed-causal graph ms | Speed range | Numerical result |
| --- | ---: | ---: | ---: | --- |
| 128 | 225.07 / 225.19 | 65.63 | **3.429–3.431x** | exact |
| 1024 | 224.75 / 238.15 | 154.60 | **1.454–1.540x** | exact |
| 4096 | 460.81 / 461.23 | 460.81 | **1.000–1.001x** | exact |

Ratios bracket both same-job native controls. Median control-time drift is
0.05% / **5.96%** / 0.09%; the 1024-token range preserves that uncertainty.
There are five measured samples after two real warmup updates, so these ranges
are bracketing-control observations, not statistical confidence intervals.
At 4096 tokens there is **no demonstrated stable speedup**. Ordinary PEFT with
only the fixed mask/position policy stays essentially at native timing.
Dynamo-eager also becomes exact but is not faster. The gain comes from native
forward/backward graph execution while preserving the attention path.

Each shape records two successful captures and 22 actual replays across
correctness, changed-input, warmup and measured updates. Exactly ten replays
occur in the five timed updates: forward plus backward each time. AdamW is
outside graph capture and updates the actual FP32 master Parameters; reset
and changed-input results rule out a frozen-input or stale-weight replay.
There are no empty-graph warnings in the native graph log. AccumulateGrad
stream-mismatch warnings remain recorded; they are not suppressed. They do not
change the observed exact numerical comparisons in this fixture.

PyTorch allocated peaks alone do not describe graph residency:

| Tokens | Native / graph allocated GiB | Native / graph reserved GiB |
| --- | ---: | ---: |
| 128 | 2.191 / 1.749 | 2.291 / 2.412 |
| 1024 | 6.455 / 1.897 | 6.969 / 8.252 |
| 4096 | 20.862 / 2.046 | 23.104 / 28.244 |

The graph path reserves **more**, so no total-VRAM saving is claimed. This
remains a standalone, first-order, fixed-shape BF16 experiment: batch one,
rank four, dropout zero, checkpointing off, unpadded/unpacked full sequence and
no KV cache. Padding/packing, checkpointing, other batch sizes, quantized-base
training and other Torch/Transformers/hardware combinations are not validated.

[The verified receipt](qwen35-training-causal-mi250.json) retains every wall/GPU
sample, all exact-tensor counts, original source/analysis hashes, native
framework-source hashes, live-input checks, graph counts and allocator totals.
CPU checks: **225 passed**, three skipped, 49 GPU unit cases deselected; those
unit-test skips are separate from the completed whole-model GPU experiments.
Ruff and wheel/sdist build checks accompany this change. All GPU jobs for this
round have finished; no recurring follow-up or queued job remains.

## Larger-model check

The [Qwen3.5-4B follow-up](qwen35-training-4b.md) now passes exact native PEFT
checks for all 496 factor tensors at the same three lengths. Its gains are
2.466–2.621x at 128 tokens, essentially none at 1024, and none at 4096. The
short-sequence benefit survives, but the 0.8B scaling must not be extrapolated
to larger models. This is a separately controlled single-GCD job.

## Isolation results before the mask fix

Job `23501814`, `nid005045`, completes all six workers in 5m27s: 18 cases,
180 real optimizer updates and 18 additional changed-input backwards.
All candidate paths fail the original PEFT gates. Median complete-step ms:

| Tokens | PEFT before / after | Dynamo eager | AOT eager | AOT graph backend | Native autograd graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 218.77 / 213.47 | 227.69 | 280.59 | 122.00 | 65.95 |
| 1024 | 221.67 / 222.28 | 226.52 | 285.78 | 183.27 | 156.62 |
| 4096 | 460.02 / 462.34 | 499.50 | 520.62 | 541.58 | 499.78 |

The native graph's experimental speed range is 3.237–3.317x / 1.415–1.419x /
0.920–0.925x against the same-job controls. These are **failed native-parity
measurements**, not a selected training replacement. The native controls remain
exact across saved and changed-input results; time drift is at most 2.42%.

Two direct comparisons are exact, including all losses, audit logits, 372
parameter gradients, updated parameters, timed-loss trajectories and
changed-input gradients/logits:

- Native-autograd graph against Dynamo eager, at all three lengths.
- AOT graph backend against AOT eager, at all three lengths.

Thus graph execution adds no observed numerical error against each path's
matched reference, while AOT changes the result relative to Dynamo. Yet AOT is
not the only source of the original discrepancy. At 1024 tokens, before the
first update, ordinary PEFT loss is 0.2921494246 while **all** traced/captured
paths report 0.2922703028. At 4096 these are 0.0732563287 and 0.0732276887.
This requires investigating a forward-path change, not only backward sums.

Each native graph case records two captures, 22 actual graph replays including
all checks/warmups, and exactly ten replays in five timed updates. There are no
empty-graph warnings for this path. Its log contains AccumulateGrad stream
mismatch warnings; these are retained. AOT graph-backend capture remains
fragmented. PyTorch allocated peaks for native graph are
1.749/1.897/2.046 GiB, but reserved peaks are 2.416/8.285/28.529 GiB. Allocated
peaks alone cannot establish total-VRAM savings.

[The isolation receipt](qwen35-training-autograd-mi250.json) records every
comparison, source/analysis hashes, graph counts and numerical failures. The
initial analysis and extended direct-pair analysis are both retained externally.

## Forward-path cause and scoped repair

The installed Transformers 5.10.1 `masking_utils.is_tracing` includes both
Dynamo compilation **and CUDA stream capture**. Its `_ignore_causal_mask_sdpa`
therefore refuses the implicit-causal shortcut while tracing/capturing. Normal
unpadded full-sequence eager training uses no explicit attention mask and
`is_causal=True`; the traced/captured path materializes a causal mask instead.

The installed `use_gqa_in_sdpa` also permits GQA only when the mask is `None`.
With the explicit mask, the same SDPA wrapper repeats K/V instead of passing
`enable_gqa=True`. This provides a concrete common forward-path change for
Dynamo and native capture, consistent with their exact mutual comparison.
It is not evidence that every forward difference is harmless rounding.

`qwen35_training_mask.py` registers a **new** AttentionInterface name pointing
to the unchanged native `sdpa_attention_forward`, paired with a guarded mask
factory returning `None`. It preserves the module's native causal flag and GQA
eligibility. It rejects padding, packing/custom patterns, cached offsets,
unequal Q/KV lengths and batches other than one. All six actual Qwen3.5 full
attention modules must have `is_causal=True`; installed Transformers must be
5.10.1. The original `sdpa` registry entry and installed files are unchanged.
CPU meta-model preflight confirms the actual registry/config interface without
loading weights or allocating a GPU. GPU numerical acceptance is separate.

## Rejected packing guard attempt and explicit positions

Job `23502195` (`nid005026`, 2m38s) is **FAILED**, not a successful graph run.
Both native PEFT controls and `peft-implicit` complete all three lengths. The
implicit-policy ordinary PEFT losses, all gradients, updates, timed losses and
changed-input checks are exact against default PEFT. However, Dynamo/native
capture fail before timing: the factory rejects a nontrivial mask-function
wrapper. [The rejected receipt](qwen35-training-mask-rejected-mi250.json) retains
that failure and the valid ordinary-PEFT controls; graph timings are not used.

The installed `find_packed_sequence_indices` returns `None` for a contiguous
single sequence only outside tracing. During capture/tracing it returns a
zero-valued sequence-index tensor, then wraps the causal mask with an all-true
packing predicate and disables the skip. Thus ordinary contiguous positions
still appear as an arbitrary wrapped mask to the protective factory.

The targeted follow-up constructs explicit `(3,1,T)` RoPE position axes. Native
Qwen3.5 constructs `(4,1,T)` axes and uses `[1:]` for exactly these three rotary
axes; its first axis is separately supplied to automatic packing detection.
Providing the three rotary axes directly keeps the exact original values,
shape and strides of rotary inputs, while declaring this known single sequence
without the automatic packing axis. No token positions are removed from RoPE.
Both ordinary-PEFT and graph candidates use this same position construction.
The original PEFT before/after controls retain their original default inputs.

The factory continues rejecting custom/packed mask functions, padding, nonzero
cached offsets, unequal lengths and other batches. CPU tests check the rotary
axes against native defaults, layout rejection and causal future-token masking.
CPU meta/config preflight additionally calls the actual installed mask factory.
The full-model checks after this correction are completed in the verified result above.

## Matched controls

All workers use one allocated MI250 GCD, the same unmerged text-only checkpoint,
rank-four all-linear LoRA, FP32 masters, BF16 autocast, batch one and identical
AdamW updates. Checkpointing is off; token lengths are 128, 1024 and 4096.

| Worker | Execution | Purpose |
| --- | --- | --- |
| `peft-before` / `peft-after` | Ordinary HF/PEFT/autograd | Bracketing native controls |
| `dynamo-eager` | `torch.compile(backend="eager")` | Dynamo without AOT or graph replay |
| `aot-eager` | `torch.compile(backend="aot_eager")` | AOT without graph replay or Inductor fusion |
| `graph-backend` | `torch.compile(backend="cudagraphs")` | AOT plus backend-managed graph execution |
| `native-graph` | `torch.cuda.make_graphed_callables` | Native autograd capture without Dynamo/AOT tracing |

The native-graph wrapper owns a static copy of input IDs, retains the original
module Parameters, and uses Torch's installed implementation. AMP weight
caching is disabled during capture to refresh casts of live LoRA masters.
The graph factory is temporarily instrumented; constructed graph objects keep
their capture/replay counters after the original factory is restored. Setup
and capture are outside timing. Each timed step includes zeroing gradients,
forward/loss, backward and a real external foreach AdamW update.

All workers now collect the same small detached audit-logit sample on each
forward. Slices plus stack avoid host index Tensor creation during capture.
This common audit work is included in the timed steps; compare only same-job
controls, not absolute times from older payloads or different nodes.

## Acceptance and input refresh

Every case saves three correctness-update losses, all 372 third-backward LoRA
gradients, all updated LoRA tensors and audit logits. After the third update,
one additional forward/backward uses IDs rolled by one position, saves its
loss/gradients/logits, and verifies that the caller's original input storage
was not overwritten. This check does not perform an optimizer update.

Parameters and AdamW are then reset for two warmup updates and five timed
updates. The same declared gates are retained: loss max absolute error 0.01,
sampled-logit relative L2 0.02, each parameter-gradient relative L2 0.05 and
updated-parameter max absolute error 1e-4. The analyzer additionally checks
timed-loss trajectories and changed-input loss/gradients/logits using these
same thresholds. None is relaxed to admit a faster result.

Only successful GPU execution plus CPU comparison establishes a numerical
outcome. CPU checks validate wrapper parameter ownership, loss derivatives,
graph-factory restoration and rejection of changed-input/timed-loss drift;
they do not establish GPU graph support.

## Reproduction

On an allocated gfx90a GPU with the isolated FLA dependency directory:

```bash
PYTHONPATH=/absolute/path/to/deps:/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/run_qwen35_training.py BF16_TEXT --output RESULTS \
  --autograd-isolation-only --check-input-change --recompile-limit 64 \
  --max-seconds 560
```

After GPU execution, compare on CPU without holding the allocation:

```bash
python benchmarks/analyze_qwen35_training.py RESULTS --output comparison.json
```

This experiment changes no installed framework, serving integration, public
training default or paper source. It is fixed-text feasibility evidence,
not task-quality convergence or a matched Unsloth/QLoRA benchmark.

The targeted causal-policy control uses `--causal-policy-only` instead of
`--autograd-isolation-only`, with the same changed-input and time-budget flags.
It requires default PEFT before/after, ordinary PEFT with the fixed policy,
Dynamo eager with that policy and native graph with that policy. Only native
graph is intended to reduce complete-step dispatch overhead.
