# Qwen3.5 high-concurrency optimization

This follows the [selected decode fusions](qwen35-fused-lora.md). The scope is
BF16 Qwen3.5-0.8B, TP1, one MI250X GCD, four synthetic compact rank-4 adapters,
128 input tokens and 64 forced output tokens. C8/C32 are closed-loop client
concurrency limits, not fixed requests per second. These are research prototypes;
the installed library runtime is unchanged.

## Paired base gap and operator screen

Job **23476060**, nid005032, completed in **6m36s**. Its current policy includes
selected GDN/wide/narrow fusions and the capacity-four narrow specialization.
It measured base then current on the same allocation, three repeats per point:

| Concurrency | Base output tokens/s | Current output tokens/s | Current/base |
| --- | ---: | ---: | ---: |
| 8 | 1205.14 | 1029.23 | 85.40% |
| 32 | 3098.13 | 2522.74 | 81.43% |

This is the unfused, no-AITER base configuration from the earlier studies,
not an optimized base ceiling. These are diagnostic timings of the preceding
policy; this screen did not validate new full-model candidates. Do not combine
them with throughput from a different allocation to claim an optimization gain.

The 12-step profile windows contain prefill as well as decode. At C32,
1024-token prefill has about 1.58ms of LoRA A and 4.07ms of B per forward in
summed GPU kernel durations. The B sum includes RowExpand and native B.
Decode has about 0.93ms in the named LoRA/fused-consumer kernels; the latter
include base-output unpack work. These sums are not service wall time or a
prediction of the benefit from removing a kernel. Repeated CPU metadata work
was also observed; profiler annotations have nested/inclusive costs and must
not be added together as an end-to-end estimate.

The [prefill prototype](../benchmarks/qwen35_prefill_fused.py) implements:

- Two BF16 dot products for segmented A+B, using native 16-token chunk metadata.
  A's result, the scaled B delta and the final base addition retain BF16 rounding.
- B correction at gate/up's SiLU consumer, retaining B reuse across an adapter's
  token rows. It rounds the corrected gate/up to BF16, then SiLU to BF16 before
  multiplying by up, as the tested native implementation does.

The SiLU rounding boundary matters: a zero-correction diagnostic with wider
activation values has maximum difference 0.25 without that intermediate round,
and zero difference with it. The public helper therefore defaults to rounding;
the measured candidate explicitly enables the same behavior.

Both kernels are trusted-input, inference-only benchmarks, with physical
capacity 4. The consumer has fixed output width 3584. Segment size must be at
most 16; slices and output tiles must respect the tested partition boundaries.
No general-shape, trained-adapter quality or cross-device support is claimed.

| Path | Tokens | Adapters | Separate microseconds | Fused microseconds | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate/up A+B | 128 | 4 | 13.69 | 12.72 | 1.08x |
| GDN QKVZ A+B | 128 | 4 | 15.23 | 13.07 | 1.17x |
| QKV A+B | 128 | 4 | 13.65 | 10.44 | 1.31x |
| GDN QKVZ A+B | 1024 | 4 | 78.00 | 62.93 | 1.24x |
| Gate/up A+B+SiLU | 1024 | 4 | 85.01 | 67.40 | 1.26x |

Every tested eager result has zero maximum absolute difference against the
configured preceding A/B path. Operator timing uses graph replay, 128 calls
per sample and three shuffled rounds. It is not whole-model serving throughput.

The grouped consumer **regresses at decode batch32**: with four adapters the
separate path is 12.02 microseconds and the best tested fusion is 14.28; with
32 adapters it is 14.85 versus 35.20. The batch32 decode path keeps RowExpand.
Narrow prefill A+B also fails to win consistently and is excluded. Only the
measured wide 128/1024-token prefill geometries enter the next model candidate.

## Consecutive decode metadata reuse candidate

The [process-local installer](../benchmarks/qwen35_metadata_cache.py) avoids
rebuilding native segment/routing metadata on consecutive decode calls only
when the batch size, CPU routes/ranks/scales, weight-load generation, graph
buffer identity and chunk configuration are unchanged. Reset, prefill, capture,
non-graph execution, failed prepare and weight loading invalidate reuse.
Every input still goes through the native manager's active-LoRA update.

It caches no token, position, KV or recurrent-state values. Install once in an
isolated SGLang 0.5.22 Fleet worker **before backend construction**, after any
other experimental backend factories. The Fleet bridge enforces TP1 and
non-overlapping loads; this prototype additionally rejects attention DP and
waits until model initialization/capture has finished before enabling reuse.
This is not an installed library flag or a general SGLang cache.

The CPU test covers 14 invalidation transitions using a framework stub.
Job **23476373** then checks 64 full-model sequences: cache-only matches every
saved token/logprob record exactly, with 7341 observed cache hits and 574
metadata rebuilds. C32 medians are 2552.33 tokens/s for its preceding control
and 2578.86 for cache; C8 is 1041.59 versus 1039.83. These unbracketed results
suggest only about 1% C32 gain, not the much larger profiler-span fraction.

The same attempt stops at the new prefill candidate's strict incremental gate:
64/64 token sequences match, but maximum saved logprob difference is **0.086396**.
The largest delta occurs in serial128 samples, where only new A+B executes,
not the1024-token SiLU consumer. Original A+B uses K chunks of64; the actual
native shrink uses256. Aligning that partition is a targeted follow-up, not a
claim that the failure is already fixed. No prefill timings, base arm or ending
control were executed after the failure; the attempt and logs are retained.

Follow-up job **23476560** completes the C32 ablation in **14m05s** on
**nid005026**, one MI250X GCD, 7 requested CPUs and 40GB host memory. It repeats
three 384-request measurements per arm, with 128 input and 64 forced output
tokens. All LoRA arms have identical server arguments, physical/pool rank 4,
weights and scaling. The base is the same unfused no-AITER configuration.

| Arm, in execution order | Output tokens/s |
| --- | ---: |
| Previous fused/specialized policy, before | 2562.02 |
| Metadata cache + prefill SiLU consumer | 2539.69 |
| Above + prefill A+B, K chunk 256 | 2571.33 |
| Above + selected per-row decode consumer | **2611.61** |
| Native compact-rank-4 SGLang | 2260.27 |
| Base, without LoRA | 3168.47 |
| Previous fused/specialized policy, after | 2547.17 |

The ending control is 0.58% slower than the starting control. The combined row
variant improves those controls by **1.94–2.53%**, and improves the prefill-A+B
arm by 1.57%. It is **1.1554x native** and **82.42% of base**: throughput remains
17.58% below base. This is a modest complete-service gain, not the operator's
1.39–1.73x ratio and not a demonstrated 1.3x service result. These are pilot
medians with three repetitions in one allocation, not across-run confidence
intervals. C1/C8 complete-model gains of this new row policy were not measured.

Every Fleet arm matches 64/64 native token sequences and repeated-request
records. Cache plus prefill SiLU consumer has zero saved logprob difference
from the preceding policy, but complete throughput **regresses by 0.29–0.87%**
against its two controls. Operator savings alone do not justify selecting it.

Adding prefill A+B produces **0.086396** maximum incremental logprob difference;
aligning K chunks to 256 **does not remove it**. Row versus that prefill-A+B arm
has zero new saved logprob difference. The largest native difference is now
**0.120192**, including the four added serial 128 samples, and is shared by
all Fleet arms. Both native and incremental A+B strict gates remain failed.
The user's explicit nonblocking-logprob speed-study opt-in permits timing
while requiring matching token IDs and finite comparisons; it never relabels
those gates as passed. This is not evidence of trained quality or harmless
rounding over long recurrent generations.

Actual decode graphs, all 456 positive rank-4 entries, metadata cache hits,
new prefill/row hooks and fixed-address loading are audited. The per-row
combined arm is a research candidate; it is not enabled by a library flag.
The optional scalar sequential-B follow-up is only an unsubmitted sketch,
not another measured optimization.

## Per-row decode consumer

A second operator experiment avoids computing a padded16-row tile for each
small adapter group. It performs HIP shrink followed by a per-token B-at-SiLU
kernel; B reuse is left to the cache instead of explicitly grouping16 rows.
It preserves corrected gate/up and SiLU BF16 rounding before multiplication.
The [standalone operator](../benchmarks/qwen35_decode_silu.py) remains cap4,
width3584 and inference-only; it is not installed by a library flag.

Job **23476537**, nid005045, completes in1m03s, including a rerun of the public
prefill check program. The per-row check covers8 configurations with4 tiles:
32 eager comparisons and32 graph replays after changing routes, ranks and
scales are exact. It includes null routes and retained ranks1/2/3/4 inside a
capacity4 pool. References are the actual separate HIP-A/RowExpand-B/SiLU path
and, at batches1/8, the already selected wide-A+B/SiLU path.

| Tokens | Adapters | Current microseconds | Per-row256 microseconds | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 7.17 | 5.25 | 1.37x |
| 8 | 4 | 8.16 | 5.97 | 1.37x |
| 32 | 1 | 13.92 | 8.53 | 1.63x |
| 32 | 4 | 11.93 | 8.57 | 1.39x |
| 32 | 32 | 14.76 | 8.52 | 1.73x |

A minimum3% improvement over the actual current path at every tested adapter
count selects batches 1/8/32, tile 256. Untested 2/4/16 retain their preceding
path. These are operator ratios; the matched model result above is much smaller.

The earlier 52-case native maximum logprob difference 0.1187558 and the new
64-case maximum 0.120192 both fail the strict 0.05 gate. The authorized
token-matched timing opt-in does not turn either result into a numerical pass.

## Reproduce the operator screen

In the pinned ROCm/SGLang environment, prebuild RowExpand on CPU, then on an
allocated MI250 GCD run:

```bash
python benchmarks/check_qwen35_prefill_fusion.py --output prefill-k64.json --block-k 64
python benchmarks/check_qwen35_prefill_fusion.py --output prefill-k256.json --block-k 256
python benchmarks/check_qwen35_decode_silu.py --output decode-silu.json
```

The program covers six grouped-consumer configurations and ten segmented-A+B
configurations, each with multiple tiles and a null route. It runs operators,
not an installed service backend. The kernel function ASTs match the measured
screen after formatting; source checks, timings and limits are retained in
[the evidence](qwen35-high-concurrency-mi250.json).


## Follow-up: null routing and joint base/A projection

Job **23480237**, nid005035, measures the preceding combined row policy,
base, and a null-routing diagnostic on the same allocation. All retain C32,
128 input tokens, 64 forced output tokens, 384 requests and three repeats.
Medians are **3172.68** tokens/s for base, **2608.23** for current, and
**2734.67** with LoRA enabled but every request's adapter set to null.
Null routing is 86.19% of base; current is 95.38% of null routing. This points
to fixed execution costs beyond useful LoRA arithmetic. It is not an arithmetic
upper bound: active/null routing and graph/kernel choices can differ. Native
`skip_inactive_lora_batches` is false on this path, so null requests still
exercise the enabled-LoRA machinery.

Null versus base keeps 64/64 short and 32/32 long token sequences, with maximum
saved logprob differences 0.045371 and 0.049433. The long check uses a batched
32-request call with 128 input and 64 forced output tokens, in addition to the
short checks. CPU trace interval unions avoid counting nested runtime events
twice; the remaining host time is unattributed, not proven removable Python
cost or an end-to-end savings estimate.

A joint-projection candidate appends the four pinned adapters' A rows to base
weights and computes all resident A products with the base GEMM, then uses
B-at-consumer kernels. The complete GDN projection operator improves 1.19–1.23x
at batch 32. Whole-model short checks pass 64/64, but only **28/32 long token
sequences match**. This candidate is rejected before service timing; it is not
part of the maintained runtime or a valid full-model speedup. The change in
GEMM/reduction geometry warrants investigation; this result does not establish
that the drift is harmless.

The same job's stock full-prefill-graph probe fails during initialization:
`TritonAttnBackend._apply_cuda_graph_metadata` rejects `ForwardMode.EXTEND`.
The GDN base graph metadata method also lacks plain EXTEND handling. This is
a framework path gap, not an out-of-memory result. Exact-shape prefill graph
metadata and larger prefill batches are subsequent experiments, not measured
improvements in this diagnostic. See the [curated evidence](qwen35-fixed-cost-mi250.json).


## Cold-prefill graph prototype

The [process-local installer](../benchmarks/qwen35_cold_prefill_graph.py) supplies
plain EXTEND metadata missing from the tested full-graph Triton/GDN backends.
It captures **eight cold requests of exactly 128 tokens** in one 1024-token bucket.
A capture fixture of 1024 + seven zero-length requests is insufficient: GDN's
convolution grids and cached chunk tables must describe the same 8x128 layout.
The full attention prefix KV table stays empty; native output-cache locations
still receive K/V. GDN uses native eager physical state-slot translation, copied
into independent fixed-address buffers before replay. Decode metadata never
owns those buffers. Prefix cache and recurrent-state tracking are disabled.

The admission guard rejects other batch counts, input lengths, cached prefixes,
speculative/encoder and multimodal inputs. Those batches use eager prefill.
Scope remains BF16 Qwen3.5-0.8B, TP1, SGLang 0.5.22, one MI250 GCD and capacity 4.
SWA, MLA, DCP, DLLM and deterministic attention are excluded. No new LoRA
approximation or state-update reordering is introduced by this graph adapter.

Job **23481421**, nid005033, completes a pilot in 6m41s. C32 medians are
2608.00 tokens/s for the preceding row policy, 2970.65 with the prefill graph,
and 3173.62 for eager-prefill base. The graph candidate gains 13.91% over its
same-card preceding policy. All 64 short and 32 long sequences and saved logprobs
are **exactly equal to that preceding policy**. Audit records 117 prefill graph
replays, 2787 decode graph calls, fixed-address loads and 456 positive rank 4
entries. The graph is real; out-of-scope validation also exercises eager fallback.
This pilot does not give native/base the graph and is not the final fair ratio.

The standalone graph helper does not install the earlier private row/prefill
serving hooks. In the measured combined experiment those existing hooks admit
static prefill graph metadata while keeping their preceding kernels and BF16
rounding. The installed Fleet runtime remains unchanged.

For the public native/Fleet/base benchmark, add `--cold-prefill-graph` to
`benchmarks/qwen35_hybrid.py run` or `serve`. This installs the graph adapter in
each spawned worker and supplies the same full-prefill configuration to every
mode. The flag also enables extended validation: four serial 128 samples, one
8-way 128-input sample, and 32-way 64-output samples. Use
`--extended-validation` for the corresponding eager reference. It checks 96
sequences instead of the default 52; optional case sets must match the reference.
Use a separate output directory for each configuration with identical manifests;
`--validation-only` checks output/state repeatability without measuring throughput.
A graph flag without observed `EXTEND/graph` and GDN replay counters is not proof
of an optimized run.

CPU checks cover ten admission failures, stable metadata storage, real request
slot refresh after intervening decode, native fallback, idempotence and version
rejection. The public computation matches the GPU-tested prototype; its only
additional behavior is an explicit SGLang version guard and a worker CLI entry.


### Same-graph, bracketed complete-model result

Job **23481601**, nid005033, completes all eight arms in 16m36s. Three repeats
of 384 C32 requests per arm retain 128 input and 64 forced output tokens.
The native and Fleet graph server arguments are exactly equal. Medians:

| Arm, execution order | Output tokens/s |
| --- | ---: |
| Preceding row policy, eager prefill | 2620.46 |
| Row policy + cold-prefill graph, before | **2966.08** |
| Native compact rank 4, eager prefill | 2265.47 |
| Native compact rank 4 + same cold-prefill graph | 2529.99 |
| Base, eager prefill | 3162.26 |
| Base + same cold-prefill graph | 3445.90 |
| Row policy + cold-prefill graph, after | **2978.45** |
| Preceding row policy, eager prefill, after | 2618.20 |

The two graph medians improve the two preceding-policy controls by
**13.19–13.76%**; control drift is -0.086% and graph before/after drift is +0.417%.
The graph candidate is **1.1724–1.1773x same-graph native** and
**86.08–86.43% of same-graph base**, leaving 13.57–13.92% throughput gap.
Comparing it only with eager-prefill native gives 1.31x, but that is a different
execution configuration, not the same-graph algorithm ratio. The graph also
helps native/base themselves. These are closed-loop, complete-service timings,
not fixed arrival rates, kernel ratios, trained-quality results or confidence
intervals across allocations.

Graph versus its own eager arithmetic is **96/96 exact token/logprob sequences**
for Fleet, native and base. The final current-after control is also exact versus
current-before. Actual prefill/decode graphs, GDN state-index refresh,
456 positive retained rank 4 entries and fixed-address loads are audited.

**Expanded native parity fails:** the preceding fast research policy matches
64/64 short native sequences but only 28/32 long sequences. Therefore the graph
candidate matches 92/96 native sequences overall, with maximum common-prefix
logprob difference 0.120192. This is an existing fast-policy difference: graph
versus that policy adds zero drift. It is no longer accurate to infer 64-token
native parity from the earlier 8-token checks. The prior token-matched speed
opt-in does not turn this expanded failure into a pass. Throughput here is a
diagnostic of unchanged experimental arithmetic; retain the draft status and
do not enable these research fusions as an accepted native-equivalent runtime.
The standalone graph adapter itself preserves native arithmetic in this suite.

See [the full curated comparison](qwen35-cold-prefill-mi250.json) for repeated
samples, all numerical flags, graph counters and pinned framework SHA256 hashes.
Larger 2K/4K prefill batches have only been prepared as a follow-up; no speed or
numerical claim is made for them.


Job **23481853**, nid005037, then runs the public CLI and graph modules without
private fusion hooks. Native and base each pass the expanded 96-case suite,
including repeated long requests, and each is exactly equal to the earlier
same-mode graph result. Prefill/GDN graph counters are positive in both modes.
This validates the public worker/CLI wiring and version guard on a second MI250
allocation; it is a validation-only run, not another throughput measurement.
