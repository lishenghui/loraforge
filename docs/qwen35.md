# Qwen3.5 hybrid LoRA

This experimental integration targets the **dense Qwen3.5 text decoder**, BF16,
TP=1, synchronous adapter loading, and the inspected SGLang 0.5.22 implementation.
Vision adapters, MoE, quantization, tensor parallelism and recurrent-state cache
reuse across different adapters are outside the validated scope. Current checks
pass 133 CPU cases and 93 MI250X operator/framework GPU cases (jobs 23463508 and
23464050),
including four slices, unequal output widths, strided output, rank/slot changes
and graph replay. The conservative full-model synthetic-adapter pilot passes the declared
native/Fleet token/logprob gate; measured throughput and its limits are below.
This does not establish trained-adapter task quality or general hybrid support.

## What changes

The official Qwen3.5-0.8B config has 24 decoder layers: 18 GatedDeltaNet linear
attention layers and six gated full-attention layers, one per group of four.
The text hidden size is 1024. LoRA still computes `scale * B @ A`; the changed
boundary is how the model's projections are fused and how their factors are
packed, rather than a different LoRA equation.

| HF projection | SGLang projection | Output widths in 0.8B | LoRA A slices |
| --- | --- | --- | --- |
| `in_proj_qkv`, `in_proj_z` | `in_proj_qkvz` | 2048 / 2048 / 2048 / 2048 | shared qkv A repeated three times, separate z A |
| `in_proj_b`, `in_proj_a` | `in_proj_ba` | 16 / 16 | two independent A blocks, b before a |
| `out_proj` | `out_proj` | 1024 from 2048 inputs | one |
| `q_proj`, `k_proj`, `v_proj` | `qkv_proj` | 4096 / 512 / 512 | three; Q includes its attention gate |
| `gate_proj`, `up_proj` | `gate_up_proj` | 3584 / 3584 | two |
| `o_proj`, `down_proj` | unchanged | 1024 | one each |

The gate `z` is part of the four-slice input projection. The small `b/a`
projections control the DeltaNet update and decay; dropping them is a correctness
error when the adapter contains their weights. Convolution and recurrent-state
updates remain native SGLang operations, with no LoRAForge replacement.

## Native batching and compatibility bridge

SGLang's LoRA manager maps each request's adapter ID to a fixed pool slot and
refreshes slot ranks, scales and request routes. Its `csgmv` backend groups token
rows by adapter and splits long segments into chunks (16 rows in this setup).
Shrink computes A products per slice; expand adds the corresponding B product to
the base projection. Decode graphs consume updated device metadata at fixed
addresses. GDN layers use the same adapter routing as attention and MLP layers;
their separate convolution/recurrent states still follow SGLang's request-state
indices. Correct LoRA packing does not by itself prove recurrent-state isolation.

The inspected runtime already has a generic four-slice merged-linear wrapper,
but its LoRA normalizer handles independent q/k/v/z or fused qkvz, not HF's shared
qkv plus z factors. It also lacks the complete b/a LoRA target/dimension/stacking
path. The explicit `LORAFORGE_SGLANG_QWEN35=1` plugin switch installs a process-local
bridge for these gaps. Both the **native csgmv baseline and Fleet** must use this
same bridge when comparing HF-format adapters. Shared SGLang files stay intact.

The bridge validates factor shapes, rejects overlapping formats, zero-fills
missing projections, preserves global checkpoint scaling and adds b/a to target
resolution. It supports fused and independently split factors too. This is an
internal-version integration: it refuses a version other than 0.5.22 and refuses
to overwrite a runtime already advertising b/a support. The tested framework
file hashes must accompany measurements; a version string alone is insufficient.

## Fleet changes

The retained layout and RowExpand now accept four slices. The existing 1–3-slice
kernel keeps its original shared-memory footprint; a separate instantiation
stages four slices for GDN. Rank lookup includes both HF and fused aliases.
By default, hybrid execution uses rank-aware native shrink and native prefill expansion for
all projections. RowExpand handles supported decode ranks, including the new
`in_proj_qkvz` and `out_proj`; the narrow b/a output stays native. Native shrink
and prefill reductions avoid carrying the dense-model HIP-A/prefill-B choices
into the hybrid's recurrent state. This is a separate dispatch policy, not a
claim that the dense-model tuning table was measured on the new shapes.

The hybrid B profile also checks output width and decode batch shape. For
1024-wide out/down/o, batches 1–8 use 128 threads with the original 16-row
kernel; batches 16–32 use 128 threads and two-row work units. The 5120-wide
QKV output uses four-row work units at batches 16–32. Other shapes/batches
retain the original mapping. The measured qkvz/gate-up shapes remain unchanged.
These choices are local to the opted-in hybrid policy; low-level `row_expand`
still defaults to the original 16-row kernel. Full-model results are below.

The separate `LORAFORGE_SGLANG_QWEN35_TINY=1` opt-in enables a tiny-rank
profile. It checks physical pool capacity **4 or 8**, input/output widths and
slice count against the seven 0.8B projection shapes above. Decode batches
1–32 use HIP A; 1024-wide out/down/o B uses native expansion. Wide qkvz,
gate-up and QKV prefill uses RowExpand at 128–1024 tokens, while 1024-wide
prefill uses it only at 1024 tokens. b/a B always remains native. Other shapes,
pool capacities and batch ranges retain the conservative profile.

This opt-in changes A/prefill reduction order and does **not** pass the strict
0.05 logprob gate in this pilot. The diagnostic rank-4 full profile retained
all 52 token sequences but reached 0.1187558 maximum logprob difference.
The switch defaults to zero and leaves dense-model dispatch unchanged.

Use both switches in each worker for Fleet:

```sh
export SGLANG_PLUGINS=loraforge
export LORAFORGE_SGLANG_QWEN35=1
export LORAFORGE_SGLANG=1
export LORAFORGE_SGLANG_POLICY=fleet
export LORAFORGE_SGLANG_MANIFEST=/absolute/path/manifest.json
```

For native csgmv with the same format bridge, set `LORAFORGE_SGLANG=0`.
Do not enable the Qwen3-specific stable-prefill hook for Qwen3.5.

## Reproducible performance pilot

`benchmarks/qwen35_hybrid.py prepare OUTPUT --model LOCAL_MODEL` prepares four
deterministic, nonzero synthetic adapters. Their physical global rank is 64;
their retained module ranks are 8/16/32/64 (b/a capped at 8). These are test
factors with zero padding, **not trained adapters compressed at a measured quality
budget**. The exact same padded factors and scaling go to native and Fleet.

On an allocated GPU, `run OUTPUT` runs native, Fleet and base-only serially,
validates repeated and mixed-adapter requests, warms each concurrency, and times
96 requests with 128 input / 64 output tokens at concurrency 1/8/32, three repeats
each. Decode uses graphs; prefill is eager. Prefix caching is disabled. Model
loading, compilation and conditioning are untimed. `compare OUTPUT` is CPU-only.
Every attempt is retained; the latest pointer identifies the selected run.
An optional `requests_at_concurrency` manifest mapping overrides the request
count for each concurrency, allowing longer high-concurrency measurements
without lengthening the serial point. Use identical counts in every arm.

The numerical gate requires identical native/Fleet generated token IDs and at
most 0.05 logprob difference along their common prefix. Fixed-address load audits,
actual graph execution and Fleet dispatch evidence must also be inspected.
Base-only measures the available model throughput; it is not a quality baseline
for a real fine-tuned adapter. A failed numerical gate must remain visible even
if the throughput measurement is useful diagnostically.

For a speed study that permits logprob drift, explicitly pass
`--allow-logprob-drift` to `run` alongside `--reference`. Timing still requires
identical generated token IDs and finite logprobs. The saved comparison keeps
`numerical_pass` at its original strict meaning and records `timing_allowed`
separately; `compare` does not relabel a strict failure as numerical success.
Enable `LORAFORGE_SGLANG_QWEN35_TINY=1` for the fast Fleet worker when studying
the tiny-rank profile.

For a compact tiny-rank study, add `--rank 4` or `--rank 8` to `prepare` in a
new output directory. Every A/B tensor then has that physical and retained rank,
with `lora_alpha / r = 1`; both engines use the same weights and matching
`max_lora_rank`. The manifest records actual parameter counts and their ratio
to the same shapes at rank 64 (6.25% and 12.5%, respectively). These fixtures
are generated directly at small rank, not compressed trained checkpoints.
Omitting `--rank` preserves the original mixed-rank/padded fixture.

```sh
python benchmarks/qwen35_hybrid.py prepare OUTPUT --model LOCAL_MODEL --rank 4
python benchmarks/qwen35_hybrid.py run OUTPUT --modes native
python benchmarks/qwen35_hybrid.py run OUTPUT --modes fleet \
  --reference OUTPUT/results/NATIVE_ATTEMPT/correctness.json
python benchmarks/qwen35_hybrid.py compare OUTPUT --modes native fleet
```

Use the native attempt from `native-latest.json` as the reference. The Fleet
run requires native parity before timing. Comparison checks equal native/Fleet
server options and agreement with the prepared pool capacity. The optional
two-arm comparison reports no base-only ratio or adapter-effect validation.
The runtime policy stays unchanged, isolating rank from dispatch changes.

The first pilot (23457086, completed in 14m47s) used the dense profile on the
existing projections and RowExpand on the new GDN projections. Its median
native/Fleet/base output throughput was 118.03/127.08/183.54 at concurrency 1,
770.03/828.77/1196.49 at 8, and 2127.46/2126.57/3137.19 at 32. These are
**diagnostic speeds, not an accepted speedup**: 19/20 native/Fleet token sequences
matched, but one diverged and the maximum common-prefix logprob difference was
0.075245, exceeding the declared 0.05 gate. Adapter effects, graph execution and
fixed-address loads were observed. The unchanged native and base results remain
available; the conservative hybrid policy must pass this same gate before timing.

The conservative policy completed the matched repeat in **job 23457660**,
14m38s on one MI250X GCD. All **20/20** native/Fleet sequences (12 single-request
cases and eight mixed requests, eight output tokens each) matched. The maximum
common-prefix logprob difference was **0.0159646**, within the unchanged 0.05
gate. Repeating requests after another adapter and repeating mixed batches were
exact within each engine. All arms replayed decode graphs; adapter loads kept
fixed buffer addresses; Fleet's RowExpand decode calls and nonzero ranks for
all seven projection types were observed. The bridge's CPU factor-product
checks independently cover packing; model parity is between the two engines
using the same bridge, not a comparison against Transformers/PEFT generation.

Three measurements per point, median output tokens/s, including the base model:

| Concurrency | Native csgmv + bridge | Fleet + bridge | Base only | Fleet/native |
| --- | ---: | ---: | ---: | ---: |
| 1 | 119.54 | 123.34 | 185.24 | 1.032x |
| 8 | 784.11 | 811.09 | 1224.81 | 1.034x |
| 32 | 2145.60 | 2165.62 | 3162.24 | 1.009x |

This is a short single-device pilot, not a broad workload/seed confidence
interval. The high-concurrency difference is effectively near parity; Fleet
retains about 66–68% of base-only throughput. The earlier 7.6% low-concurrency
gain was rejected by the numerical gate. The accepted policy improves this
workload only modestly and does not justify a 1.3x performance claim.

The model is `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`. Framework/library fingerprints,
adapter hashes, raw durations, numerical results and the rejected first pilot
are in [qwen35-mi250.json](qwen35-mi250.json). A final installer-selection guard
also enables the hybrid policy for explicit installer calls; the measured
workers already selected that same policy through the environment switch.

## Where the initial gain went

The earlier 1.17–1.22x result is a different workload: Qwen3-4B-Instruct-2507,
50 real compressed adapters, and a fixed 1,000-request arrival trace. This pilot
uses Qwen3.5-0.8B, four synthetic adapters, and closed-loop concurrency 1/8/32.
Model size, architecture, rank distribution and batching all changed. With only
four adapters, larger batches give native grouped matrix operations more rows
per adapter; the operator study below directly observes that advantage on some
outputs. These two studies do not isolate an architecture-only slowdown or a
regression in the already validated dense-model path.

The mixed-rank fixture also differs substantially from the paper's retained
budget. Its executed RowExpand module/adapter entries have mean rank **37.75**;
**39.0625%** are still rank 64 after fused projections take their maximum rank.
These are counts of positive entries, excluding native b/a, not latency- or
parameter-weighted averages. The paper's Qwen50 spectral checkpoint contains
457,866,752 / 5,946,605,568 parameters (**7.70%** of its own original fleet).
The fixed rank-4/rank-8 protocol above tests much smaller factors while keeping
native pool capacity equal to the checkpoint rank in both engines. It does not
reproduce that heterogeneous, trained fleet or its task-quality budget.

Profiling job 23459047 sampled the accepted native/Fleet/base implementations
at batch sizes 1, 8 and 32 on one MI250X GCD. Each separate 12-forward trace
includes prefill; the following numbers select **decode GPU kernel durations**
only and must not be read as end-to-end service latency:

| Batch | Native A, ms/step | Fleet A, ms/step | Native B, ms/step | Fleet B, ms/step |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.324 | 1.351 | 1.037 | 0.876 |
| 8 | 1.355 | 1.358 | 1.467 | 1.066 |
| 32 | 1.346 | 1.356 | 1.563 | 1.525 |

There are 114 A and 114 B calls per decode step. LoRA still accounts for about
30–31% of Fleet's summed decode kernel time. Preserving native A reductions
leaves that cost essentially unchanged; at batch 32, B alone saves very little.
This explains the modest initial speedup more directly than assuming that all
remaining time belongs to the base model. These samples include profiler
overhead and do not predict the speed of an unprofiled optimization.

The four-slice representation repeats the shared HF qkv A product three times,
but the entire qkvz A projection accounts for only about 12% of A kernel time;
deduplicating qkv alone can save less than that. Down-projection A takes about
31–33%, and the two output-projection types together take about 23–24%. This
projection breakdown is inferred from the verified serial layer order after
checking exactly 114 A kernels in every decode span, not from explicit kernel
projection labels. It motivates first removing wasted launch work across all
projections. Shared-A reuse and small GDN b/a fusion remain possible follow-ups,
without a measured speedup claim here.

A bounded-launch experiment removed provably empty chunks using
`G + floor((N - G) / 16)`, where `G = min(N, number_of_slots)`, and capped
RowExpand's fixed 880-block grid at the corresponding output-tile bound.
Job 23459326 passed 97 GPU tests and all 20 full-model numerical cases; the
candidate's tokens and logprobs were exactly equal to the previous Fleet.
However, matched service medians regressed by 0.97%, 1.30% and 1.62% at
concurrency 1/8/32. The change was reverted. Fewer empty thread blocks did not
translate into faster service; this is not an accepted optimization.
The same-card timings, numerical checks, sampled kernel costs and source hashes
are retained in [qwen35-optimization-mi250.json](qwen35-optimization-mi250.json).

The follow-up operator study (23460210) measured 42 cache-warm shape/rank/batch
cases. Direct HIP A was faster for MLP down and output projections, but sometimes
changed the last BF16 bits. At batch 32 with four adapters, grouped native B was
faster for several narrower outputs; wide qkvz and gate/up still favored
RowExpand. These are operator measurements, not service speedups. The small b/a
B comparison is excluded because its synthetic fixture incorrectly assigned
different weights to the same checkpoint key; no dispatch decision uses it.

Job 23461243 tested these ideas with a stronger **52-case model gate**, adding a
repeated mixed batch of 32 to the original 20 cases. The prefill reductions stayed
native. HIP A on down/output projections plus small-batch gate/QKV passed 51/52
token sequences; restricting HIP A to down/output projections passed 52/52.
Both exceeded the unchanged logprob limit: **0.0654472 > 0.05**. Each passed the
85 operator/framework tests, showing why those tests alone do not justify model
parity. Both candidates were rejected before throughput timing.

The native-A, B-only variant and conservative control passed all 52 cases, with
maximum logprob difference 0.0159646. B-only measured 1.032x/1.031x/1.003x versus
native at C1/8/32. Its gain over the conservative control was only
1.03%/0.29%/1.47%; even C1, whose dispatch did not change, differed by 1.03%.
That is insufficient evidence to adopt the B change independently. This trial
used identical 32/192/384 request counts at C1/8/32 in all arms, three repetitions,
on one card in fixed order. It did not measure a new base-only arm.

The MLP-only ablation (23461933) kept every GDN and full-attention A reduction
native. Enabling HIP A for down-projection through C32 and gate/up through C8
still passed only 51/52 token sequences, with maximum logprob difference
0.0654472. Enabling it for **down-projection alone** restored 52/52 token matches
but left the same 0.0654472 maximum. Both variants passed 85 GPU tests each;
neither passed the model gate, so neither was timed or adopted. Slurm reports
`FAILED` after 8m39s because the bounded driver deliberately stopped after both
numerical rejections; this is not a successful serving validation. Native A
remains necessary under the declared gate even outside the GDN projections.

One implementation target is a faster shrink path that preserves the native
matrix-dot accumulation behavior, rather than reusing the dense-model scalar
HIP reduction. B-side work decomposition also remains a target, as the paper
reconciliation below shows. Shared-qkv A reuse has limited profiled headroom.
The tested A tile/staging settings and empty-grid bounds did not solve this.
No further speedup, 1.3x result, or trained-adapter quality equivalence is claimed
by these investigations; none of the four rejected HIP-A profiles is enabled.

## Reconciliation with the paper's RowExpand ablation

The paper does not claim a universal decode-only RowExpand speedup. Recomputing
its 256 archived cases gives native/staged-RowExpand geometric means of 1.238x
overall and 1.540x for prefill at a 16-row limit, but only **0.995x for decode**.
The decode aggregate reaches 1.074x with an 8-row limit. The 1.83x token-reuse and
2.72x staging numbers compare RowExpand variants, not native CSGMV. These are
cache-flushed operator results; the current warm operator study and profiled
service traces are different measurement regimes. The installed native expand
and tuning-config file hashes match the paper's archived hashes exactly.

The Qwen3.5 batch-32 trace shows cancellation between shapes, not a universal
loss of RowExpand's benefit. The ratios below use profiled B-kernel duration
sums per decode step; values above one favor RowExpand:

| Projection | Output width | Calls/step | Native / RowExpand |
| --- | ---: | ---: | ---: |
| GDN qkvz | 8192 | 18 | 1.244x |
| MLP gate/up | 7168 | 24 | 1.368x |
| GDN out | 1024 | 18 | 0.666x |
| MLP down | 1024 | 24 | 0.704x |
| Attention QKV | 5120 | 6 | 0.949x |
| Attention o | 1024 | 6 | 0.629x |

The two wider winners save about 215 microseconds per sampled decode step;
the four losing projection types cost about 203 microseconds. Their aggregate
therefore falls from 1410.238 to 1399.123 microseconds, only **1.008x**. The
separate b/a path is native in both engines and must not be credited to
RowExpand. Projection labels are inferred from the inspected serial model order,
with exactly 114 alternating A/B pairs checked in each decode span. These are
instrumented kernel sums, not end-to-end latency or a new serving speedup.

There is a concrete parallelism concern in the narrow-output mapping. The
prepared batch has 32 rows over four adapters, hence four 8-row chunks. At 256
output threads, a 1024-wide output yields only `4 * ceil(1024 / 256) = 16` live
work items, versus 112 for gate/up and 128 for qkvz. Launching 880 persistent
blocks does not create additional live work. Native expand uses 64-output
matrix tiles and also reuses B across token rows. This explains a plausible
crossover between reuse, per-thread serial work and parallelism; no occupancy
or bandwidth counters were collected to assign a causal percentage.

Changing the maximum from 16 to 8 alone cannot split an already 8-row chunk.
The rejected empty-grid bound changed launch size, not the number of useful
work items. This motivated the smaller B-only work units tested below, while
leaving A segments and native prefill unchanged.

## B-only row subdivision

Job **23463508** completed in 1m41s on one MI250X GCD. The new
`row_expand(..., row_tile=2/4/8)` alternatives split only the B work inside
each existing segment; `row_tile=16` retains the original kernel. Each token
keeps the same scalar rank reduction and BF16 rounding. The 93 GPU tests passed,
including graph replay across changed ranks, routes, segment counts and null
slots. All eight custom tile/thread variants matched the original bitwise in
all 32 operator cases (four shapes, single/mixed adapters, C1/8/32/64).

`python benchmarks/rowexpand_tiles.py RESULTS.json` reproduces the study on an
allocated GPU with the inspected SGLang runtime. Each slot has its own checkpoint
region, including equal-rank slots. Native B receives the same effective ranks.
Three rounds randomize variant order. Warm timing averages 160 graph calls;
flushed timing uses single-call graphs after a 128 MiB flush and output reset,
both outside the timed interval. Flushed measurements include graph/event costs
and must not be interpreted as pure kernel duration or serving throughput.

At C32 with four mixed-rank adapters, median microseconds:

| B output | Variant | Warm | Flushed |
| --- | --- | ---: | ---: |
| 1024 | Original 16-row / 256-thread | 9.709 | 26.400 |
| 1024 | 2-row / 128-thread | 5.800 | 21.120 |
| 1024 | Native, same effective ranks | 6.853 | 21.920 |
| QKV 4096/512/512 | Original 16-row / 256-thread | 12.070 | 28.800 |
| QKV 4096/512/512 | 4-row / 256-thread | 11.049 | 26.721 |
| QKV 4096/512/512 | Native, same effective ranks | 12.661 | 27.360 |

The narrow-output warm ratio is **1.674x** versus the old RowExpand and
**1.182x** versus rank-aware native B. The QKV warm improvement over the old
RowExpand is **1.092x**. Wider qkvz/gate-up do not benefit from subdivision in
the tested four-adapter C8/C32 cases and retain the original choice. Single
adapter and C64 results differ, reinforcing that output width alone is not a
dispatch rule. Exact samples, framework/source hashes and numerical checks are
in [qwen35-rowtiles-mi250.json](qwen35-rowtiles-mi250.json).

The first model attempt, 23464050, passed 93 GPU tests and all 52 native model
cases; its saved candidate/control token-logprob records are exactly equal.
The experiment driver then failed before candidate timing because it passed a
string environment path to a Path-only reader. This failed attempt is preserved.
The path conversion was corrected and CPU-checked for both equal outputs and
rejection of different outputs; runtime and policy source hashes did not change.

The same-card repeat **23464674 completed in 13m07s**. Both old and retuned Fleet
matched native token IDs in **52/52** cases, with maximum common-prefix logprob
difference **0.0159646**. Retuned Fleet's token/logprob records were **exactly
equal to old Fleet**, checked before timing. Fixed-address adapter loads, actual
decode graph replay, two-row/four-row custom dispatch and native A were audited.
The final readable policy matches the measured rule table over 24,768 CPU policy
inputs; the rest of the integration module has an identical syntax tree.

Median output tokens/s for the **complete model with LoRA**, three repeats,
128 input / 64 output tokens, 32/192/384 requests at C1/8/32:

| Concurrency | Native | Old Fleet | Retuned Fleet | Retuned / old | Retuned / native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 119.147 | 120.478 | 121.208 | 1.0061x | 1.0173x |
| 8 | 773.855 | 791.387 | 796.385 | 1.0063x | 1.0291x |
| 32 | 2128.400 | 2086.737 | 2118.570 | 1.0153x | 0.9954x |

The retuned samples all exceed the corresponding old-Fleet sample range in this
run, but this remains one device and a fixed native/control/candidate order,
not a confidence interval across runs or workloads. The gain over old Fleet is
only **0.61%/0.63%/1.53%**; C32 remains effectively tied with native. Operator
speedup therefore does not translate into a large full-model gain, and these
results do not justify 1.3x. Native A, other projections and base-model work
remain in the measured cost. There is no new base-only arm and the synthetic,
eight-output-token numerical gate does not certify trained-adapter quality or
long-generation equivalence. The bounded hybrid B profile is enabled on this
evidence, with these limits; all raw timing samples and audits are in the linked
RowExpand evidence file.

## Compact tiny-rank results with the conservative policy

Jobs **23465767** (rank 4, 9m09s) and **23466166** (rank 8, 9m05s) completed
sequentially on one MI250X GCD each. They use the unchanged runtime from the
B-tuning study with compact fixtures and matching pool capacities.
Each rank's native/Fleet pair shares the exact checkpoint files, scaling,
server options, input lengths and request counts. Each checkpoint stores its
actual small rank; no rank-64 pool is retained in either engine.

Both ranks pass all **52/52** token-ID comparisons. Rank 4's saved token/logprob
records are exact; rank 8's maximum logprob difference is **0.0351560**, within
the unchanged 0.05 limit. Repeated requests remain exact within each engine.
Actual eager prefill, decode graph replay and fixed-address loads pass the
audits. All **456** positive fused module/adapter entries have the selected
rank and checkpoint rank. Thus the small gains are not caused by accidentally
executing the old rank-64 fixture.

Complete-model median output tokens/s, three repetitions per point:

| Rank | Concurrency | Native | Fleet | Fleet/native |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1 | 122.333 | 124.664 | 1.0191x |
| 4 | 8 | 833.826 | 833.144 | 0.9992x |
| 4 | 32 | 2207.800 | 2175.878 | 0.9855x |
| 8 | 1 | 122.795 | 122.801 | 1.0001x |
| 8 | 8 | 846.107 | 831.814 | 0.9831x |
| 8 | 32 | 2250.762 | 2185.386 | 0.9710x |

Reducing rank alone **does not restore the paper's runtime gain** in this
hybrid policy. At a fixed uniform rank with a matching native pool, rank-aware
A has no extra pool padding to remove; the hybrid still uses native A and
prefill B reductions. Internal native kernel tile padding is a separate cost
that this policy does not eliminate. Decode B is the main remaining custom
path, and its current mixed-rank-tuned dispatch is not a universal tiny-rank
speedup. These measurements do not isolate kernel launch cost, memory traffic,
or a particular B variant as the cause; that requires a separate tiny-rank
profile/dispatch experiment.

The rank-4 C1 improvement, both C32 regressions and rank-8 C8 regression have
disjoint within-run sample ranges, but each rank has one fixed native-before-Fleet
ordering, not an across-run confidence interval. The two ranks ran on different
nodes, so their absolute rates are not a paired rank-scaling experiment.
No new custom-A policy is enabled by this study. These four random adapters do
not measure trained compression quality, and the short numerical gate does not
establish long-generation equivalence. Raw timing samples, input/runtime/model
hashes, accounting and audit summaries are in
[qwen35-tiny-rank-mi250.json](qwen35-tiny-rank-mi250.json); original attempt
bundles and full correctness records are retained separately.

## Opt-in tiny-rank performance

The follow-up operator/profile job **23467018** explains why rank reduction
alone was insufficient. Rank-4 A with native reductions still takes about **1.30–1.38 ms** per
sampled decode step. At batch 32, the old Fleet B path takes **0.860 ms** versus
native's **0.757 ms**, including its native b/a part. These are profiled sums
of GPU kernel durations, not service latencies. Sixty rank/shape/phase operator
cases motivate enabling HIP A, using native narrow B, and selecting prefill B
by shape. A separate compact shared-Z prototype passed 64 exact cases and 192
graph replays but improved operator timing by only about 0–2%; it was not adopted.

Strict screening in **23469297** kept 52/52 generated token sequences for each
candidate, but full/decode-only/B-only policies reached logprob differences
**0.1187558 / 0.0563851 / 0.1187558**. All exceeded 0.05, so that run collected
no candidate throughput. The subsequent speed study explicitly permits this
logprob drift while retaining the token and execution checks. The strict
failures remain recorded.

Job **23469847** completed in **15m27s** on **nid005107**, one MI250X GCD. It
uses the actual packaged, opt-in policy, with matching native/Fleet weights,
pool capacity and server arguments. It ran rank-4 native/fast, then rank-8
native/fast, on the same allocation. Each point has three repeats, 128 input
tokens and 64 forced output tokens; C1/8/32 use 32/192/384 requests respectively.
The following throughput includes base-model computation and serving overhead:

| Rank | Concurrency | Native tokens/s | Tiny fast tokens/s | Fast/native |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1 | 122.810 | 140.352 | 1.1428x |
| 4 | 8 | 844.085 | 915.695 | 1.0848x |
| 4 | 32 | 2271.468 | 2404.562 | 1.0586x |
| 8 | 1 | 121.527 | 140.099 | 1.1528x |
| 8 | 8 | 843.420 | 912.898 | 1.0824x |
| 8 | 32 | 2251.456 | 2392.244 | 1.0625x |

Both ranks retain **52/52** native token sequences; their maximum logprob
differences are **0.1187558** and **0.0909806**. Consequently `numerical_pass`
remains **false** under the strict 0.05 definition, while the explicitly opted-in
`timing_allowed` is true. Each rank's 456 positive fused entries match its
physical/checkpoint rank. Audits confirm HIP A, RowExpand in prefill and decode,
actual decode graphs, repeated-request consistency and fixed-address loads.
The runtime source matches the measured wheel byte-for-byte.

To measure this policy against an already completed native reference:

```sh
LORAFORGE_SGLANG_QWEN35_TINY=1 python benchmarks/qwen35_hybrid.py run OUTPUT \
  --modes fleet --reference OUTPUT/results/NATIVE_ATTEMPT/correctness.json \
  --allow-logprob-drift
```

These are **5.9–15.3%** single-run gains over matched native serving, not 1.3x.
The numerical samples generate eight tokens; throughput samples generate 64.
Four synthetic adapters, one native-before-fast ordering per rank, and no fresh
base-only arm do not establish trained-adapter quality, long-generation parity,
or the paper's 50-adapter workload result. Default hybrid behavior remains
conservative. Timings, strict failures, hardware/source hashes and audits are in
[qwen35-tiny-speed-mi250.json](qwen35-tiny-speed-mi250.json).

The subsequent [GDN fusion experiments](qwen35-gdn.md) measure packed base
projection and expand-at-unpack separately, then together against matched
native/base controls. The rank-4 combined prototype adds 6.86%/6.44%/3.55% over
the existing tiny profile at C1/8/32. Its kernel is a standalone benchmark
prototype; the installed tiny-profile flag retains the behavior documented above.

The [tiny-LoRA fusion follow-up](qwen35-fused-lora.md) tests shared A, combined
A+B, and direct GDN A+B+unpack. Its first full-model rank-4 candidate reaches
1.2680x/1.1928x/1.1244x native at C1/8/32 on one matched allocation, with exact
saved outputs versus the preceding combined GDN prototype. These additional
changes remain standalone benchmark prototypes.

The final capacity-4 narrow specialization in that study adds 1.17–3.21%
over its matched fused controls. Its whole-model/native ratios are
1.2975x/1.2503x/1.1345x, with zero new saved logprob difference versus those
controls. Native strict drift remains; C1 sits near 1.30x, without a reliable
above-1.3x result. Both studies retain before/after controls and their drift.

## Sources

- [Official Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
- [SGLang Qwen3.5 model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5.py)
- [SGLang LoRA normalization](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py)
- [SGLang chunked LoRA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/chunked_backend.py)

The bridge is independent LoRAForge code written against these contracts; no
framework source is vendored. Existing RowExpand provenance remains in
[provenance.md](provenance.md).
