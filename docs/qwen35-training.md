# Whole-model BF16 LoRA training experiment

This experiment attaches the [MLP training consumer fusion](lora-swiglu-training.md)
to the complete Qwen3.5-0.8B text model. It measures real optimizer steps, not
manual CUDA-graph replay. It is a standalone first-order training feasibility
study with fixed text batches, not a convergence or Unsloth comparison.

The main full-model run has completed, but **native PEFT numerical equivalence
has not passed**. Consumer fusion passes its matched packed-projection reference;
the packed projection itself fails the original PEFT gradient/update gates.
The timings below are experimental, not a validated drop-in training speedup.
CPU checks pass (204 passed, three skipped, 49 GPU cases deselected), including
actual PEFT parameter ownership, checkpointed updates and comparison failure
gates. The independent complete-model compiler study is recorded separately. A later
separate-native-projection ablation passes all native gates only when retaining
native activation, with no demonstrated complete-step speedup.

## Complete training step measurements

Job `23492718`, `nid007897`, completed all six required workers and all 36
shape/checkpoint cases in 6m14s. Each case has three correctness updates, two
warmups and five timed updates: 360 real full-model optimizer steps in total.
Job `23491457` was cancelled before starting, with zero GPU time, while waiting
for priority; the successful replacement requested a 10-minute window.

Median synchronized wall-clock milliseconds, same GCD within the job:

| Checkpoint | Tokens | PEFT before / after | Fused | PEFT / fused range | Compiled MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 128 | 239.16 / 238.53 | 221.25 | 1.078–1.081x | 222.05 |
| off | 1024 | 238.69 / 235.60 | 220.81 | 1.067–1.081x | 220.60 |
| off | 4096 | 468.10 / 465.59 | 442.36 | 1.053–1.058x | 444.62 |
| on | 128 | 395.86 / 391.86 | 364.90 | 1.074–1.085x | 362.58 |
| on | 1024 | 395.51 / 392.35 | 360.28 | 1.089–1.098x | 362.38 |
| on | 4096 | 603.00 / 602.53 | 574.90 | 1.048–1.049x | 573.13 |

These are complete training steps, including GDN, attention, the LM head, loss,
backward and AdamW. They are not the earlier MLP-only ratios. The two native
controls differ by at most 1.31% in median step time and have **exactly equal
losses, all saved gradients and updated parameters** in every case.
Relative to the packed reference, consumer fusion improves time by only
0.2–3.1%; compared with compiling the MLP the differences are within about 0.7%
and change direction. The large isolated MLP win does not become a large
whole-model training win here.

At 4096 tokens without checkpointing, peak PyTorch allocated memory is
20.86 GiB for original PEFT, 20.67 GiB packed, 20.02 GiB compiled-MLP and
19.36 GiB for both fused/compact. With checkpointing it is about 13.15 GiB for
all paths. The compact backward's operator-level memory benefit does not lower
the complete-step peak beyond fused in this fixture.

### Numerical outcome

Fused and compact pass **all six** packed-reference comparisons. Their three
losses and sampled logits are identical to packed in each case. At 128 tokens,
every saved gradient and updated tensor is also exact. At 1024/4096 tokens,
the maximum per-parameter gradient relative L2 is 0.00758/0.00575, and the
maximum updated-parameter absolute error is 4.59e-5/2.38e-5. Checkpointed
and non-checkpointed numerical results agree.

Original PEFT equivalence fails for packed and hence for both consumer variants:

| Tokens | Max loss drift | All-gradient relative L2 | Worst parameter gradient relative L2 | Max updated parameter error |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.002890 | 0.02695 | 0.14684 | 5.978e-4 |
| 1024 | 0.000267 | 0.02914 | 0.38297 | 5.828e-4 |
| 4096 | 0.0000181 | 0.02661 | 0.18475 | 5.696e-4 |

The worst gradient ratios are often small GDN b/a-branch A gradients; they are
not 15–38% error in the entire gradient vector. Nevertheless, both the declared
per-parameter gradient gate and update gate fail. The compiled-MLP reference
also fails its packed-reference gates: aggregate gradient relative L2 is about
5.4–5.7%, and 128-token loss drift reaches 0.04244. No strict native training
parity or SFT quality conclusion is inferred from these performance numbers.

There is a concrete precision-boundary change in the packed base backward.
Separate projections round `dG @ Wg` and `dU @ Wu` to BF16 independently before
adding them. A concatenated projection accumulates the longer dot product and
rounds once. These are different finite-precision operations. For example,
representable BF16 operands can yield separate exact real subtotals
`1 + 1/256` and `1/256`: separate BF16 GEMM rounds followed by addition give
`1.0`, while one combined GEMM gives `1.0078125`. This small counterexample
isolates the backward rounding change; it is not itself a GPU/model measurement.
Row concatenation also allows a different forward GEMM schedule; the first
zero-B loss already differs at 128/1024 tokens. Thus attribution must include
the projection geometry, not only the new consumer kernel.

The next drop-in candidate should preserve the native projection and gradient
rounding boundaries while fusing the LoRA consumers. A statistical SFT quality
study is a separate acceptance path; the present repeated-text three-update
probe establishes neither equivalent convergence nor degraded task quality.

[The curated receipt](qwen35-training-mi250.json) includes all step samples,
controls, peak allocations, hashes and numerical summaries with failed-tensor
counts and worst examples. Full CPU comparison output and raw gradient/update
artifacts are retained in the experiment directory, not copied into the package.

## Default whole-model compiler control

Job `23493295` completed in 3m34s on `nid005038`, testing the complete PEFT model
with `torch.compile(fullgraph=False, dynamic=False)` without checkpointing.
The before/after PEFT controls again have exactly equal saved numerical results.
Ratios use this job's own controls; absolute times must not be compared directly
to the preceding job on a different node.

| Tokens | PEFT before / after ms | Whole-model compiled ms | Speed range | PEFT / compiled peak GiB |
| --- | ---: | ---: | ---: | ---: |
| 128 | 223.99 / 225.97 | 186.22 | 1.203–1.213x | 2.191 / 1.958 |
| 1024 | 224.60 / 226.15 | 182.80 | 1.229–1.237x | 6.455 / 4.614 |
| 4096 | 460.75 / 460.49 | 423.30 | 1.088–1.089x | 20.862 / 13.980 |

This compiler control is faster than the earlier consumer-only results relative
to their respective native controls. Its default numerical policy also fails
all three native gates: aggregate gradient relative L2 is 0.0838/0.0904/0.0710,
maximum parameter-gradient relative L2 is 0.590/0.619/0.731, and maximum updated
parameter error is approximately 5.9e-4. Maximum loss drift is
0.07574/0.00980/0.00173. These are timing observations, not demonstrated native
training equivalence or inferior/superior SFT quality.

The 4096-token worker log reports Torch's default `recompile_limit=8` being
reached in `transformers/modeling_layers.py.__call__`, with eager fallback.
Thus that row is not an unrestricted compiler optimum. The separate targeted
experiment enables Inductor's `emulate_precision_casts` option and raises the
process-local recompile limit to 64. The installed Torch 2.11 source documents
the former as preserving downcast/upcast boundaries to emulate eager numerics;
it does not promise complete tensor parity. Acceptance still uses the same
predeclared gates. No installed Torch/Transformers/SGLang file is edited.
The explicit experimental CLI settings are `--emulate-precision-casts` and
`--recompile-limit 64`; they are recorded per worker. They alter only the
current process and do not silently change the default benchmark policy.

[The compiler receipt](qwen35-training-compile-mi250.json) preserves samples,
errors, control measurements and the fallback warning. Earlier worker payloads
are retained with their hashes; later compiler-setting options must not be
attributed retroactively to the default runs.

## Precision-preserving compiler configuration

The single targeted follow-up, job `23494172`, completed all three workers in
3m33s on `nid005067`. Worker receipts confirm
`emulate_precision_casts=True`, `recompile_limit=64`, and float32 matmul
precision `highest`. No recompile-limit fallback warning appears in this run.
Graph breaks remain permitted by `fullgraph=False`; this is not a single static
graph claim. Checkpointing is off, batch one, and rank remains four.

| Tokens | PEFT before / after ms | Compiled ms | Speed range | PEFT / compiled peak GiB |
| --- | ---: | ---: | ---: | ---: |
| 128 | 208.84 / 213.72 | 176.22 | 1.185–1.213x | 2.191 / 1.958 |
| 1024 | 212.03 / 219.70 | 172.07 | 1.232–1.277x | 6.455 / 4.614 |
| 4096 | 462.28 / 462.64 | 405.98 | 1.139–1.140x | 20.862 / 13.652 |

The native controls drift 2.34%/3.62%/0.08% in step time, while their saved
loss/gradient/update tensors are exactly equal. The ranges above preserve that
timing uncertainty. Both compiler options changed together, so this experiment
does not isolate the performance contribution of either option.

Loss drift improves substantially, but **all three native gradient/update gates
still fail**:

| Tokens | Max loss drift | All-gradient relative L2 | Worst parameter gradient relative L2 | Max updated parameter error |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.001741 | 0.02718 | 0.19430 | 5.900e-4 |
| 1024 | 0.000733 | 0.02606 | 0.59052 | 5.932e-4 |
| 4096 | 0.0000654 | 0.02574 | 0.51283 | 5.861e-4 |

Preserving downcast/upcast boundaries is not a complete fix for a deep model's
numerical execution differences. Other reduction/gradient accumulation and
elementwise fusion changes still need isolation; this run does not establish
their individual causes. No additional blind retry or relaxed acceptance
threshold is used. [The precision receipt](qwen35-training-precision-mi250.json)
records settings, samples, drift, failed-tensor counts and original payload
hashes.

The three completed GPU jobs execute 54 worker/shape/checkpoint cases and 540
real optimizer steps. They establish experimental timing and memory savings,
not a native-equivalent training backend or a trained-quality result. The
consumer implementation's matched-reference success remains distinct from
the original PEFT failures. GPU work for this study is finished; no new jobs or
recurring follow-up are required for the recorded experiment.

## Preserving separate native base projections

Job `23501083` completed in 4m17s on `nid005053`: five workers, six cases
per worker, 300 real optimizer steps. It reuses the same BF16 model, rank-four
adapters, optimizer, input and acceptance gates. Each candidate retains the
original PEFT module and Parameter objects, separate gate/up base GEMMs and
native base-input-gradient autograd nodes. There is no concatenated base
weight, trainable shadow copy or installed-environment modification.

Three ablations isolate the consumer operations:

- `native-expand` fuses only B expansion, scaling and the base residual; native
  SiLU/multiplication and the original forward evaluation order are retained.
- `native-activation` retains PEFT projections and fuses only SwiGLU activation
  forward/backward.
- `native-consumer` fuses gate/up B expansion, residual and SwiGLU, while keeping
  separate original base and shrink projections. Down projection remains PEFT.

Median complete-step milliseconds; speed ranges bracket both same-job PEFT
controls:

| Checkpoint | Tokens | PEFT before / after | Expand | Activation | Consumer | PEFT / consumer range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 128 | 223.40 / 221.18 | 226.51 | 225.71 | 221.50 | 0.999–1.009x |
| off | 1024 | 223.53 / 221.08 | 225.54 | 226.24 | 221.42 | 0.998–1.010x |
| off | 4096 | 462.71 / 466.83 | 464.74 | 458.98 | 460.47 | 1.005–1.014x |
| on | 128 | 374.28 / 371.72 | 374.28 | 376.45 | 365.15 | 1.018–1.025x |
| on | 1024 | 374.11 / 371.05 | 374.17 | 377.69 | 365.15 | 1.016–1.025x |
| on | 4096 | 600.47 / 605.08 | 604.33 | 596.43 | 595.00 | 1.009–1.017x |

Before/after controls are exact numerically and drift at most 1.10% in median
time. `native-expand` is **exact in all saved losses, logits, 372 gradients and
372 updated tensors in all six cases**, but has no robust speed benefit.
`native-activation` and `native-consumer` are exact at 128/4096 tokens and exact
against each other in every case. Both fail the 1024-token gradient/update
gates, identically with checkpointing on/off: maximum loss drift 5.636e-5,
aggregate gradient relative L2 0.02886, worst parameter gradient 0.27288, and
maximum parameter-update difference 5.861e-4. Eleven parameter gradients and
329 updated tensors fail their respective gates. Sampled-logit relative L2
0.01077 passes its own gate; this does not override failed gradient gates.

This narrows the remaining discrepancy to the activation-fusion path; it does
not prove every approximate sigmoid intrinsic produces unacceptable training.
Preserving separate base projections removes the earlier packed-projection
failure in the expansion-only control. It also removes most of that path's
apparent speedup. The consumer's modest complete-step benefit must not be
reported as a native-equivalent result at 1024 tokens. Peak allocation at 4096
without checkpointing changes from 20.862 GiB (PEFT) to 20.204 GiB (consumer);
with checkpointing it remains 13.152 GiB.

[The native-projection receipt](qwen35-training-native-consumer-mi250.json)
retains samples, hashes, controls, gate outcomes and worst failing tensors.
Raw CPU comparisons and all saved gradient/update artifacts remain in the
separate experiment directory.

### SiLU math-policy screen

A targeted diagnostic, job `23501232` (`nid005030`, 17 seconds), exhausts every
finite BF16 gate bit pattern in [-80,80], with seeded BF16 up values and
upstream gradients. It compares the existing sigmoid implementation, ROCm
libdevice exp/fast-exp and rounded-division variants, with compiler FMA off/on.
The existing policy is exact against native Torch for output, gate gradient and
up gradient throughout this stated input fixture. Two exp variants with FMA
on differ in one gradient element by 1.14e-13; the other tested policies are
exact. This does **not** identify approximate sigmoid as the cause of the
1024-token whole-model drift. A single up/gradient pair per gate value does not
exhaust all finite triples or model autograd accumulation orders. No math-policy
change is selected based on this screen. [The diagnostic receipt](qwen35-training-silu-mi250.json)
records the input count, source hash and every policy result.

## Native-operator graph backend

Job `23501330`, `nid005030`, completes in 3m01s: native PEFT before/after controls
and `torch.compile(..., backend="cudagraphs", fullgraph=False, dynamic=False)`,
all without checkpointing. The process-local recompile limit is 64. This tests
reducing dispatch overhead through Torch's backend, without selecting Inductor
fusion or the packed/custom MLP. AOT autograd still transforms the execution;
this configuration is **not** a guarantee of native gradient accumulation order.
The optimizer is outside compilation. Each measured call performs forward,
backward and a real foreach AdamW update; no manually replayed frozen update is
timed. All nine worker/shape cases and 90 real updates complete.

| Tokens | PEFT before / after ms | Graph backend ms | Speed range | Native numerical gate |
| --- | ---: | ---: | ---: | --- |
| 128 | 217.59 / 222.84 | 119.00 | 1.828–1.873x | fail |
| 1024 | 219.38 / 223.25 | 181.14 | 1.211–1.232x | fail |
| 4096 | 467.65 / 467.80 | 546.71 | 0.855–0.856x | fail |

Bracketing native controls remain exactly equal numerically; time drift is
2.42%/1.77%/0.03%. The short-sequence result is consistent with substantial
avoidable dispatch overhead, but the experiment does not isolate graph replay
from every AOT transformation. It rejects using this backend indiscriminately
for long sequences. No demonstrated native-equivalent training speedup is
inferred: all three gradient/update comparisons fail their unchanged gates.

| Tokens | Loss max abs | All-gradient relative L2 | Worst parameter gradient relative L2 | Updated parameter max abs |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.002315 | 0.02822 | 0.21265 | 5.796e-4 |
| 1024 | 0.000405 | 0.02753 | 0.32436 | 5.824e-4 |
| 4096 | 0.0000362 | 0.02786 | 0.44464 | 5.707e-4 |

Loss and sampled-logit gates pass individually. Maximum timed-loss drift after
the separate reset/warmup is 0.00656/0.000276/0.000143; losses continue changing
with real updates, but this does not establish equivalent convergence. The
failures affect 11/16/16 parameter gradients and 343/331/334 updated tensors.

The worker records 57 successful AOT graphs and 1188 captured calls. There are
324 empty-graph warnings, explicit graph breaks at causal-convolution
non-contiguous `out=` arguments and at FLA's compiler-disabled GDN/gated-norm
functions. Some surrounding model frames fall back to eager. No recompile-limit
warning occurs. These facts preclude a whole-model single-static-graph claim;
`cuda_graph_replay` is deliberately null in the receipt for this requested
backend, rather than asserting that every region is captured. The process
records the requested backend and complete compiler counters.

Allocated-memory peaks alone are misleading here. At 128/1024/4096 tokens,
PyTorch allocated peaks fall from 2.191/6.455/20.862 GiB to
1.532/2.235/4.640 GiB, while **reserved** peaks rise from
2.291/6.969/23.104 GiB to 3.693/9.633/29.361 GiB. This is not a demonstrated
total-VRAM saving. Both allocator measures are retained in
[the graph-backend receipt](qwen35-training-graphs-mi250.json), alongside source
hashes, warnings, gates and every timing sample. These three new GPU jobs are
finished; none remains queued or running for this optimization round.

The next useful investigation is the GDN/causal-convolution graph boundaries
and AOT backward accumulation order, with the same native gates. The measured
benefit of short-sequence scheduling is larger than the separate-base consumer
benefit; another tiny MLP fusion alone is not supported as the path to 1.3x.
This remains a fixed-text BF16 training experiment, not QLoRA, serving throughput,
trained compression quality or a matched Unsloth benchmark.

## Model and parameter contract

The original local Hugging Face checkpoint is extracted without numerical
transformation by `benchmarks/prepare_qwen35_training.py`. Only the known
`model.language_model.` prefix changes to `model.`. Vision/MTP tensors are
explicitly excluded for text-only training; all remaining names/shapes must
match a meta-instantiated `Qwen3_5ForCausalLM`. Every one of the 320 serialized
text tensors is checked against its original value. The original shard SHA256
is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`;
the text-only checkpoint is
`aa364f2a0e1d96b7ffb69e52bbdd4f1f07634ce27d10d3686ffff0c90824f7dd`.
These are original unmerged HF weights, not the serving checkpoint's fused
names. Each GPU worker additionally audits the actual loaded tensors, missing
and unexpected keys, and the tied embedding/output-head storage.

The complete decoder has 24 layers: 18 GDN and six full-attention layers. Its
native GDN chunk operation must resolve to the installed isolated FLA training
implementation, with an available causal convolution; the benchmark rejects a
silent Torch fallback. SDPA remains the configured full-attention interface.

PEFT wraps all text linear projections except the output head with ordinary
rank-4 LoRA, alpha=4, dropout=0, FP32 master factors and BF16 autocast compute.
The base stays frozen. `qwen35_training_mlp.py` replaces only the 24 MLPs,
preserving the actual six PEFT Parameter objects per MLP and their scaling.
GDN and attention adapters retain ordinary PEFT execution. Stable canonical
names and a complete initial adapter hash must agree across workers.
For the original packed variants, concatenated gate/up base weights belong to the replacement; discarded modules
are not retained, and no inactive baseline model occupies the same process.

## Comparisons and timing

Each path runs in a separate process on the same allocated GCD:

| Worker | MLP path | Other model execution |
| --- | --- | --- |
| `peft-before` / `peft-after` | Unmodified PEFT MLP | Native HF/PEFT; bracketing controls |
| `packed` | Frozen gate/up concatenation plus ordinary Torch LoRA/autograd | Native HF/PEFT |
| `compiled-mlp` | Same packed MLP compiled with `fullgraph=True, dynamic=False` | Native HF/PEFT |
| `fused` | Consumer fusion, separate scaled/unscaled backward buffers | Native HF/PEFT |
| `compact` | Consumer fusion with fresh gradient-buffer reuse | Native HF/PEFT |
| `native-expand` | Separate PEFT base/shrink nodes; B expansion/residual fusion | Native HF/PEFT |
| `native-activation` | PEFT projections; activation-only fusion | Native HF/PEFT |
| `native-consumer` | Separate PEFT base/shrink nodes; gate/up expand/residual/activation fusion | Native HF/PEFT |
| `compiled-model` | Ordinary PEFT; complete model compiled with `fullgraph=False` | Record graph breaks and any compiler failure |

The `packed` control separates base-projection concatenation and PEFT casting
overheads from consumer-kernel gains. A `compiled-mlp` result is **not** a
fully compiled model result. The full-model compiler worker has a bounded
timeout and tests without checkpointing; a failed or timed-out compiler cannot
be presented as a successfully measured optimized baseline.

Batch one, sequence lengths 128/1024/4096, gradient accumulation one. Each
required worker tests both no checkpointing and non-reentrant decoder-layer
activation checkpointing. Cache reuse is disabled. Timing includes
`zero_grad(set_to_none=True)`, all decoder/GDN/attention operations, the tied
full-vocabulary LM head, native causal language-model loss, backward, and a
foreach FP32 AdamW update at learning rate 1e-4. Two warmups precede five real
timed updates. The primary time is synchronized wall-clock time; CUDA events
are recorded too. Tokenization, on-device batch preparation, model loading,
compilation, audits and correctness artifact copies are excluded from timing.

Peak allocated/reserved values are PyTorch allocator measurements, not total
device memory. Each process has only its active model, parameters and optimizer.
The same authored repeated text and labels are used everywhere; neither random
text repetition nor three optimizer updates establish SFT task quality.

## Numerical checks

Each shape/checkpoint pair first performs three updates from identical
zero-B adapter initialization with a fresh AdamW optimizer. It saves every
LoRA parameter gradient from the third backward, the updated parameters, the
three losses, and sampled logits (positions 0/middle/penultimate, vocabulary
stride 257). All trainable gradients and timed losses must be finite. Parameters
are reset before the separate warmup and timed-update trajectory.

`analyze_qwen35_training.py` compares on CPU using predeclared gates:

- Maximum absolute loss drift across the three steps: 0.01.
- Relative L2 drift of sampled logits: 0.02.
- Relative L2 drift of **each** LoRA parameter gradient: 0.05.
- Maximum absolute drift of updated parameters: 1e-4.

Observed error, exact tensor counts, failed tensor names and timing are
recorded separately. A speed result does not override a failed numerical gate.
Direct PEFT-versus-fused checks and packed-versus-fused checks are both needed;
the latter alone cannot establish equivalence to the ordinary PEFT model.

## Environment and reproduction

LUMI small-g, one MI250 GCD, seven CPUs, 40GB host memory; the completed runs request a 10-minute cap.
Torch 2.11.0 ROCm 7.2, Transformers 5.10.1, PEFT 0.21.1. FLA 0.6.0 pure-Python
wheels are extracted only into the test directory and prepended to PYTHONPATH;
the existing Torch/Triton/SGLang installation is unchanged. Dependency receipts
pin both wheels:

- `fla_core-0.6.0-py3-none-any.whl`:
  `b0708b70e0b119cd58a0a5146ea56cb175cef6ed969d8ac60267e3e077ce3ebd`
- `flash_linear_attention-0.6.0-py3-none-any.whl`:
  `6dec9735f8454523c13553622e98aff30152dde6586b4ac7d88010fbb7a06b5f`

CPU preparation, without exposing the FLA GPU-only constructor:

```bash
python benchmarks/prepare_qwen35_training.py ORIGINAL_HF_MODEL BF16_TEXT
```

Then on an allocated gfx90a GPU with the isolated dependencies available:

```bash
PYTHONPATH=/absolute/path/to/deps:/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/run_qwen35_training.py BF16_TEXT --output RESULTS
```

When a shorter allocation is needed, `--skip-whole-compile --max-seconds 520`
keeps all six required workers, token lengths and checkpoint settings, and
omits the separate optional whole-model compilation worker. Its measurement
still includes the compiled-MLP baseline; it does not establish a comparison
against compiling the complete model. That comparison needs a separate run.
`--whole-compile-only` runs ordinary PEFT before/after controls around the
complete-model compiler, all without checkpointing, and requires the compiler
worker to complete for that run to count as complete. These workers still run
in separate processes and measure identical full optimizer steps.

`--native-consumer-only --max-seconds 520` runs the three separate-base
consumer ablations with native before/after controls, token lengths 128/1024/4096
and checkpointing off/on. It requires all five workers to complete.

The explicit `--whole-compile-only --compile-backend cudagraphs
--recompile-limit 64 --max-seconds 520` selection runs the native-operator graph
backend and before/after controls without checkpointing. The default compiler
backend remains Inductor. Backend-managed captures are distinct from manually
replaying an isolated forward/backward graph; real AdamW updates remain timed.

CPU comparison after collection:

```bash
python benchmarks/analyze_qwen35_training.py RESULTS --output comparison.json
```

No training default or public serving integration is changed by these tools.
