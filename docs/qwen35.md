# Qwen3.5 hybrid LoRA

This experimental integration targets the **dense Qwen3.5 text decoder**, BF16,
TP=1, synchronous adapter loading, and the inspected SGLang 0.5.22 implementation.
Vision adapters, MoE, quantization, tensor parallelism and recurrent-state cache
reuse across different adapters are outside the validated scope. Initial checks
pass 119 CPU cases and 85 MI250X operator/framework GPU cases (job 23457660),
including four slices, unequal output widths, strided output, rank/slot changes
and graph replay. The full-model synthetic-adapter pilot passes the declared
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
Hybrid execution uses rank-aware native shrink and native prefill expansion for
all projections. RowExpand handles supported decode ranks, including the new
`in_proj_qkvz` and `out_proj`; the narrow b/a output stays native. Native shrink
and prefill reductions avoid carrying the dense-model HIP-A/prefill-B choices
into the hybrid's recurrent state. This is a separate dispatch policy, not a
claim that the dense-model tuning table was measured on the new shapes.

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

The matched full-model trial is pending. These are accepted operator results,
not an additional full-model speedup or a trained-adapter quality claim.

## Sources

- [Official Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
- [SGLang Qwen3.5 model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5.py)
- [SGLang LoRA normalization](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py)
- [SGLang chunked LoRA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/backend/chunked_backend.py)

The bridge is independent LoRAForge code written against these contracts; no
framework source is vendored. Existing RowExpand provenance remains in
[provenance.md](provenance.md).
