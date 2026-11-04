# Tiny LoRA fusion experiments on MI250

This follows the [GDN packing/consumer study](qwen35-gdn.md), using the same
Qwen3.5-0.8B BF16 model, TP=1, four synthetic physical-rank-4 adapters and
the opt-in tiny profile. The changes below are research prototypes, not
installed by the library's tiny-profile flag. The maintained runtime remains
at the preceding implementation; the benchmark kernels and measurements make
the next optimization steps reviewable.

## What was tested

- **Shared QKV A:** contract the structurally shared Q/K/V factor once, copy its
  BF16 result into the three logical slices, and contract the independent Z
  factor separately. Validate structural sharing at each adapter load.
- **Narrow A+B:** combine shrink and expand for the 48 `out_proj`, `o_proj` and
  `down_proj` calls per model step. Each block contracts A, retains the BF16-rounded
  intermediate in shared memory, then adds the BF16-rounded scaled B update to
  the BF16 base output. Output tiles deliberately duplicate A work to expose
  enough independent blocks.
- **Wide A+B:** use output-tile slice ownership for gate/up and QKV, retaining
  the separate A factors and effective-rank-packed slice layout.
- **GDN A+B+unpack:** produce corrected QKV, Z, b and a directly from base
  projections and the small factors, eliminating both standalone shrinks and
  the separate B/unpack call. Corrections still precede the original convolution,
  cache update, nonlinear gates and recurrence. This does not move a correction
  across any of those operations.

The standalone sources are [narrow/shared A](../benchmarks/qwen35_tiny_fused.cu),
[wide/GDN fusion](../benchmarks/qwen35_wide_fused.cu), and the
[lazy benchmark loader](../benchmarks/qwen35_fused_lora.py). They retain the
attributed HIP shrink's reduction and BF16 intermediate rounding. These are
trusted-input, 64-lane gfx90a prototypes, not general CUDA operators.

## Initial operator screen

Job **23473128** on nid005030 completed 12 shared-A cases and 28 narrow-A+B
cases, the latter with four output tiles each. Ranks are 1/2/4/8 for shared A
and 4/8 for narrow A+B; batches are 1/8/32. Multiple-row cases include a null
route. Narrow cases vary active adapters between one, four and up to one per row.

Every tested shared-A result is bitwise equal to the original HIP shrink.
Every tested narrow-A+B output has zero maximum absolute difference from HIP A
plus native B. At rank 4 with four adapters, tile256 reduces the narrow segment
from 6.33 to 4.73 microseconds (K2048, T8) and from 7.16 to 5.60 (K3584, T8).
At T32 the corresponding ratios are 1.20x and 1.19x. These are operator results.

Shared A is not a general latency win: rank-4 timings remain around 3 microseconds,
with ratios 0.960x/0.987x/1.034x at T1/8/32. It was excluded from the full-model
candidate. Rank-8 narrow fusion also regresses for K3584 with one/four adapters;
the full-model candidate below is restricted to rank 4. Fewer FLOPs and fewer
launches each need separate measurement.

Graph timing captures 16 calls per graph, replays eight graphs per sample,
and reports the median of three shuffled rounds using device events. Cache,
working-set and launch behavior differ from a complete model. Operator ratios
are not multiplied into serving speedups.

## Narrow A+B whole-model comparison

Job **23473650** ran candidate/control/native/base on **nid005106** in one
allocation. Control is the previous combined GDN prototype, already containing
packed base projection and B-at-unpack. All three LoRA arms have identical server
arguments, weights and physical/pool rank 4. Each concurrency has three repeats,
128 input/64 forced output tokens and 32/192/384 requests at C1/8/32.

| Concurrency | Native tokens/s | Previous combined | Narrow A+B | Candidate/native | Candidate/control | Base tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 121.84 | 149.87 | 154.50 | 1.2680x | 1.0309x | 184.50 |
| 8 | 836.65 | 981.29 | 997.95 | 1.1928x | 1.0170x | 1219.01 |
| 32 | 2255.38 | 2482.73 | 2535.89 | 1.1244x | 1.0214x | 3136.93 |

Throughput includes the entire base model and serving overhead. The candidate
reaches 83.7%/81.9%/80.8% of the matched unfused, no-AITER base configuration.
These are fixed-order, single-allocation pilot measurements, not across-run
confidence intervals or a reproduction of the paper workload. Ratios should
be compared within this allocation, rather than dividing measurements from
different nodes or runs.

Candidate and control match all 52 native token sequences. Candidate versus
control has **zero saved logprob difference**. Both retain the previous native
maximum difference **0.1187558**, failing the strict 0.05 gate; that result was
recorded without blocking the authorized timing study. Numerical cases generate
eight tokens, timing cases 64. These synthetic adapters do not establish trained
adapter quality or long-generation stability. Fixed-address loads, actual decode
graphs and all 18 GDN layers' consumer paths are audited.

## Wide/GDN operator screen and dispatch

Job **23474106** completed 84 cases on **nid005064**. The published check program
was executed on the allocated MI250 GCD. All eager comparisons have zero maximum
absolute difference; all 28 GDN cases also pass captured replay after reversing
routes, including the null row. Ranks are 4/8; batches are 1/2/4/8/16/32.

For rank 4, require at least 3% benefit for every tested adapter count at each
batch size. This selects:

| Path | Decode batches selected | Result supporting the boundary |
| --- | --- | --- |
| GDN A+B+unpack | 1/2/4/8/16/32 | Worst adapter-count ratio falls from 1.51x at T1 to 1.18x at T32 |
| Gate/up A+B | 1/2/4/8/16 | T1 is 1.60x; T32/four adapters is 0.982x and is excluded |
| QKV A+B | 1/2/4/8/16/32 | Worst adapter-count ratio is 1.62x at T1 and 1.12x at T32 |

The full-model candidate is restricted to rank-4 decode; prefill keeps its
preceding path. In particular, the selected batch-32 gate/up path retains
separate HIP A and RowExpand B. These
operator decisions require complete-model validation; they are not serving
speedup predictions.

The combined-versus-base profile in job **23473128** identifies, per decode
step, 114 shrink calls, 48 native B calls, 30 RowExpand B calls and 18 GDN
B/unpack calls: **210** calls in this part of the previous combined model.
The proposed dispatch reduces this structurally to **96** at batches 1–16
and **120** at batch 32. Base GEMMs, convolution/state, normalization and other
work remain separate. Profile fractions use GPU kernel-duration sums, not
service wall time; prefill and decode are attributed separately and unknown
kernels remain explicit in the evidence.

## Selected fusion whole-model comparison

Job **23474251** completed in **19m12s** on **nid005044**, one MI250X GCD,
7 requested CPUs and 40GB host memory. It ran old-combined control, selected
fusion, native, base, and the old-combined control again. Every point has three
repeats and the same request protocol as the narrow study. All LoRA server
arguments and weights match. The ending control quantifies drift within this
allocation; it is not an independent across-run confidence interval.

| Concurrency | Native tokens/s | Control before | Selected fusion | Control after | Fusion/native | Fusion/base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 121.96 | 149.50 | 158.46 | 153.41 | 1.2993x | 86.1% |
| 8 | 839.24 | 974.95 | 1014.85 | 987.58 | 1.2093x | 82.9% |
| 32 | 2270.05 | 2490.36 | 2536.85 | 2509.91 | 1.1175x | 81.2% |

The base-only medians are 184.03/1224.15/3122.46 tokens/s, using the existing
unfused no-AITER configuration. This is not an optimized base ceiling.
The ending controls improve by **2.61%/1.30%/0.79%** over the starting controls.
Candidate gains over the two controls therefore range from **3.30–5.99%** at
C1, **2.76–4.09%** at C8, and **1.07–1.87%** at C32. Cross-run comparisons to the
narrow study must account for these conditions. The C1 ratio rounds to 1.30x
but is **1.2993x**, not a demonstrated result reliably above 1.3x.

All three Fleet arms retain all 52 native token sequences and the native
maximum logprob difference 0.1187558. Selected fusion versus its control has
**zero** saved logprob difference. The strict native gate still fails.
Actual decode graphs, fixed addresses, all 18 GDN direct consumers, and both
wide projection hooks are audited. Rank-4 fixed shapes, synthetic factors and
eight-token numerical samples remain the validation boundary.

[Complete measurements, controls, source checks and receipts](qwen35-fused-lora-mi250.json)
retain the initial operator screen, separate narrow comparison, wide/GDN screen,
and selected full-model result.

## Rank-capacity-four specialization

The [specialized narrow kernel](../benchmarks/qwen35_rank4_fused.cu) fixes
capacity 4, K2048/K3584, N1024 and tile256. It unrolls input chunks in the
original accumulation order and reads each four-element BF16 B row with one
aligned 64-bit vector load. Effective rank and route are still live GPU
metadata; capacity specialization does not freeze the current adapter rank.
The [loader](../benchmarks/qwen35_rank4_fused.py) is lazy and benchmark-only.
Its trusted input contract requires 16-byte-aligned input/A rows and
8-byte-aligned B rows, as provided by the tested native pool and allocations.

The [portable check](../benchmarks/check_qwen35_rank4_fusion.py) completed
30 cases with bitwise-equal eager outputs and captured replay after route/rank
updates. Two cases mix retained ranks 1/2/3/4 in a capacity-4 pool. Regular cases
cover batch 1/2/4/8/16/32 and 1/min(4,T)/T adapters, including null routes.
Every tested batch is at least 1% faster for every sampled adapter count.
Typical timings relative to the generic fused narrow kernel are:

| K | Tokens | Adapters | Generic microseconds | Specialized microseconds | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1 | 1 | 4.619 | 3.429 | 1.347x |
| 3584 | 1 | 1 | 5.479 | 3.685 | 1.487x |
| 2048 | 32 | 4 | 5.589 | 4.064 | 1.375x |
| 3584 | 32 | 4 | 6.248 | 4.661 | 1.340x |

These are warmed operator results, not complete-model speedups. The same
inputs, factors, output buffer and BF16 rounding are used by both arms.

Prebuild `qwen35_rank4_fused.extension()` in the same CPU preparation step,
then run `python benchmarks/check_qwen35_rank4_fusion.py --output rank4.json`
on the allocated GCD. This operator check needs ROCm PyTorch and the benchmark
sources; it does not import SGLang or launch a service.

Job **23474915** completed the operator screen and four whole-model arms in
**14m36s** on **nid005033**, one GCD, 7 requested CPUs and 40GB host memory.
The controls use all selected GDN/wide/narrow fusions; the candidate changes
only the narrow kernel. Order is control-before/specialized/native/control-after.
Every serving point has the same three-repeat request protocol above.

| Concurrency | Native tokens/s | Fused control before | Rank-4 specialized | Fused control after | Specialized/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 125.10 | 160.45 | 162.32 | 159.84 | 1.2975x |
| 8 | 846.41 | 1025.33 | 1058.27 | 1032.71 | 1.2503x |
| 32 | 2279.40 | 2550.88 | 2586.05 | 2551.94 | 1.1345x |

Specialization adds **1.17–1.55% / 2.48–3.21% / 1.34–1.38%** over its two
controls. Control-after/before changes by **-0.38% / +0.72% / +0.04%**.
All Fleet arms retain 52/52 native token sequences; the specialized versus
fused-control saved logprob difference is zero. Native maximum logprob drift
remains 0.1187558, so the strict native gate still fails. Audits confirm actual
decode graphs, fixed-address loads and the specialized projection calls.

These are whole-model service measurements including base computation,
prefill and request handling. The C1 ratios 1.2975x and the preceding run's
1.2993x both sit near 1.30x. Different native/control throughputs across the
two allocations prevent treating their difference as a specialization effect;
the within-allocation controls show the incremental gain. Neither study
demonstrates a reliable result above 1.3x. This specialization study has no
same-card base-only arm. The rank-4 synthetic/short numerical-sample boundary
and lack of across-run confidence intervals remain.

## Remaining experiments

After fusion, further rank arithmetic reductions have limited measured leverage:
shared A already showed little latency benefit, while specialization keeps the
same launch count. Potential next experiments are correction-at-consumer for
residual/RMSNorm and gate/up activation, preserving the BF16 correction before
the nonlinear operation. Their benefits have not been measured here.

At batch32, gate/up retains RowExpand because duplicating A per output tile
regresses in the operator screen. A grouped fusion that preserves B reuse across
an adapter's token rows is a separate candidate. Prefill is also outside the
new decode fusions. These are reasons to target those paths in another study,
not a prediction that the current C32 speedup can reach 1.3x.

## Reproducing the operator screen

Use the pinned ROCm/SGLang 0.5.22 environment described in [qwen35.md](qwen35.md).
Build on CPU before reserving the GPU, with an explicit gfx90a build target:

```bash
PYTORCH_ROCM_ARCH=gfx90a PYTHONPATH=benchmarks python - <<'PY'
from qwen35_fused_lora import load_variant
from loraforge.ops.rowexpand import rowexpand_extension
load_variant("narrow")
load_variant("wide")
rowexpand_extension()
PY
```

Then on one allocated MI250 GCD:

```bash
python benchmarks/check_qwen35_wide_fusion.py --output wide-fusion.json
```

The wide/GDN screen covers ranks 4/8, batch sizes 1/2/4/8/16/32, adapter counts
1/min(4,T)/T, null routes, and changed-route GDN graph replay. Both GDN timing
paths reuse fixed persistent output buffers. The script runs only operators;
it does not install serving hooks or measure complete-model throughput.
