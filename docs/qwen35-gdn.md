# GDN fusion experiments on MI250

These are isolated Qwen3.5-0.8B BF16, TP=1 experiments on top of the opt-in
rank-4 tiny profile. The installed SGLang model file matches the preceding
study's SHA-256. Its input-GEMM packing requires CUDA or AITER, and separately
rejects LoRA. With AITER disabled in these LUMI runs, packing was disabled for
base-only serving too. The experiment enables a packed BF16 projection directly
on MI250; it does not attribute the previous gap solely to LoRA disabling it.

Two decode-only prototypes were compared with the existing fast profile:

- **Packed base projection:** concatenate the 8192-row qkvz and 32-row b/a base
  weights, execute one GEMM, then apply the existing LoRA paths to its strided
  output slices. Prefill keeps its previous path.
- **Expand at unpack:** keep the two base projections and shrink paths, then
  compute the tiny B corrections while writing the contiguous QKV, Z, b and a
  outputs. Those corrected tensors enter the original convolution, cache update,
  gating and recurrence. The prototype reads the native BF16 B pools with the
  retained-rank metadata. It does not use the compact RowExpand kernel internally.

The independent consumer kernel is in
[the benchmark prototype](../benchmarks/qwen35_gdn_consumer.py). The serving hooks
remain isolated experiment code; this report does not enable them in the library.

The packed prototype retains approximately **289 MiB** of extra BF16 base-weight
cache across 18 GDN layers. Packing precedes graph capture and timed requests.

## Operator checks

Job **23470608** completed 40 cases at rank 1/2/4/8, token count 1/8/32/128,
and 1/4/up-to-32 active adapters. Cases with multiple token rows include a null
route. Consumer outputs had zero maximum absolute difference from separate RowExpand
+ native b/a expand + existing unpack in every case. Graph replay also matched eager execution
after changing request routes. These synthetic cases do not prove equality for
all possible floating-point inputs.

The base projection segment took 34.65/35.52/37.60 microseconds at 1/8/32 rows
with two GEMMs, versus 24.29/25.14/26.92 with a packed GEMM. Padding to 8256 rows
gave no consistent improvement over 8224 rows. These are warmed operator timings,
not whole-service speedups.

Rank-4 decode consumer speedups range from 1.63x to 3.37x. At 128 rows with
one/four adapters, they fall to 0.83x/0.84x. Rank 8 at 32 rows/four adapters
reaches only 0.67x. Rank, token grouping and B reuse therefore matter; a universal
replacement is not supported by these measurements. Full-model consumer testing
below is restricted to rank-4 decode batches up to 32.

## Separate full-model effects

Job **23470800** completed on nid005027 in **11m31s**. All arms used the same
four compact rank-4 adapters, server settings and allocation, in fixed
control/packed/consumer order. Ratios are over the already-fast LoRAForge profile.

| Concurrency | Existing fast tokens/s | Packed base | Consumer | Packed/control | Consumer/control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 139.49 | 146.34 | 144.85 | 1.0491x | 1.0384x |
| 8 | 912.06 | 955.96 | 943.35 | 1.0481x | 1.0343x |
| 32 | 2400.99 | 2451.76 | 2433.89 | 1.0211x | 1.0137x |

Both prototypes match the control's 52/52 token sequences. Consumer's saved
logprobs are exact; packed-base maximum logprob difference is 0.0006001.
This reference is the existing fast profile, which already differs from native
SGLang. Repeated serial/mixed requests, actual graph execution and fixed-address
loads passed the serving checks. Hook counters verify all 18 GDN layers took the
intended path.

## Combined full-model control

Job **23471324** completed in **14m15s** on **nid005054**, with one MI250X GCD,
7 requested CPUs and 40GB host memory. It ran combined/control/native/base in
that order on one allocation. All LoRA arms share exact server options, weights
and physical/pool rank 4. Every point has three repeats, 128 input/64 forced output
tokens, and 32/192/384 requests at C1/8/32. Throughput includes the base model and
serving overhead. Base-only uses the existing unfused, no-AITER configuration.

| Concurrency | Base | Native | Existing fast | Combined | Combined/native | Combined/fast | Combined/base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 183.49 | 122.28 | 139.78 | 149.37 | 1.2215x | 1.0686x | 81.4% |
| 8 | 1208.94 | 833.92 | 907.49 | 965.91 | 1.1583x | 1.0644x | 79.9% |
| 32 | 3115.08 | 2262.13 | 2373.95 | 2458.33 | 1.0867x | 1.0355x | 78.9% |

The combined prototype adds **6.86% / 6.44% / 3.55%** over the existing tiny
fast profile. Gains do not add linearly. It reaches **1.222x / 1.158x / 1.087x**
native throughput, not 1.3x. These are single-run, fixed-order pilot results,
not across-run confidence intervals or a reproduction of the paper's workload.

Both existing-fast and combined match all 52 native token sequences. Their
maximum native logprob difference is 0.1187558 in each case, exceeding the strict
0.05 gate. The combined-versus-fast difference is 0.0006001. Logprob drift was
recorded without blocking this speed study. Numerical cases generate eight
output tokens; throughput cases generate 64. Synthetic factors do not measure
trained-adapter quality or long-generation stability.

Execution audits confirm decode graphs, fixed-address loads, all 18 GDN layers
prepared, and both modified paths captured. Actual runtime flags are
`is_cuda=false` and `use_aiter=false`, confirming the platform distinction above.
The first combination attempt (23471041) omitted its consumer callback and
failed a pending-data assertion during capture, before timing. Its logs and
completed baselines were preserved. A CPU four-mode hook-installation check and
the corrected complete run are recorded separately.

## Shared A finding

The current HF fixture has shared QKV A and independent Z A. The compatibility
bridge copies the QKV factor into three slices; the backend invokes one HIP
shrink kernel with four slices, whose slice-indexed work computes all four
products. Thus two unique contractions could replace four here. This is not four
kernel launches, and no shared-A speedup is claimed by this experiment. Fully
shared qkvz and fully independent formats need their own structural treatment.

## Reproduction boundary and evidence

The benchmark kernel's function AST is identical to the GPU-tested prototype;
only its module documentation, formatting and an output-pointer lint annotation
differ. Library defaults and the existing tiny-profile flag do not install these
GDN serving hooks. The original serving hook/runner scripts, immutable attempts
and full correctness records are retained in the workspace's
`loraforge-validation/qwen35-gdn-*` experiment directories.

[Complete timings, numerical results, source hashes and receipts](qwen35-gdn-fusion-mi250.json)
include both successful studies and the retained failed composition. The
[inspected SGLang model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5.py)
is the upstream interface reference; the installed file SHA accompanies the data.
