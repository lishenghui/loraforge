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
