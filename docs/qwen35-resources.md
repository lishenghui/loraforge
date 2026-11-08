# Qwen3.5 BF16/NF4 memory and phase audit

The subsequent [storage/dequantization follow-up](qwen35-storage-nf4.md)
eliminates the measured duplicate GDN storage and tests exact NF4 byte-pair
reconstruction. The historical measurements below remain unchanged.

This study measures allocation stages, unprofiled serving peaks and throughput,
and separately samples GPU traces. Four synthetic rank32 sources and their
actual FlashTSQR-compressed rank4 files are reused unchanged. This is fixed-work
inference, not trained-quality or native-equivalent fast-policy certification.
The preceding 48/96 original/compressed and 88/96 compressed-native/fast token
agreements remain numerical failures, as recorded in the rank-comparison guide.

## Matched primary measurement

Job **23487369**, **nid005035**, completes all six arms in 13m50s on one MI250
GCD, small-g, 7 CPUs, 40GB, with a 20-minute cap. BF16 and NF4 use the same
canonical packed source checkpoint; 114 actual projection tensors are checked
against each fixture. NF4 is block64/double-quant with the exact cached-scale
linear path. Every LoRA buffer stays BF16. Both precisions use the same prompt
IDs, 128 input/64 forced output tokens, 8/32/96 requests at C1/8/32, three
repeats, pinned adapters and cold-prefill/decode graphs. Actual capacities are
verified in every arm: **16384 KV tokens, 32 request slots, 64 GDN state slots**.
This fixed GDN limit differs from earlier runs, so paired results here are the
primary memory comparison. Performance runs precede all profiling.

| Configuration | Loaded active GB | After LoRA GB | After caches GB | After graphs GB | C32 peak active GB | C32 throughput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16-base | 1.782 | 1.782 | 3.254 | 3.455 | 3.628 | 3404.77 |
| nf4-base | 1.045 | 1.045 | 2.518 | 2.750 | 2.920 | 2654.88 |
| bf16-native32 | 1.782 | 2.131 | 3.604 | 3.805 | 3.978 | 2419.77 |
| nf4-native32 | 1.045 | 1.374 | 2.847 | 3.079 | 3.249 | 2028.32 |
| bf16-fleet4 | 1.782 | 1.836 | 3.309 | 3.813 | 3.985 | 2995.11 |
| nf4-fleet4 | 1.045 | 1.099 | 2.572 | 2.804 | 2.974 | 2319.25 |

GB means decimal 1e9 bytes. These are PyTorch active/peak allocated bytes, not
checkpoint size or a continuously measured device-wide peak. Reserved bytes
and device-wide used bytes at each snapshot are retained separately. Runtime
peaks reset before three unprofiled repeats and are read by the next untimed
9-token sentinel; graph pools and warmed allocations remain. Profiler memory
is excluded. The observer does not empty allocator caches or change model math.

| Policy | NF4 C32 peak reduction versus BF16 |
| --- | ---: |
| base | 19.51% |
| native32 | 18.31% |
| fleet4 | 25.36% |

Quantization reduces weight-resident allocation, but fixed KV/GDN state and
activation/workspace allocations dilute the total peak reduction. It does not
provide fourfold total VRAM reduction in this fixture. The BF16 rank4 fast arm
has almost the same peak as native rank32; its selected GDN base-projection
packing creates an additional BF16 weight tensor, which can offset part of the
LoRA-pool reduction. The corrected component audit checks the actual sizes.

## Unprofiled throughput

| Concurrency | BF16 base | NF4 base | BF16 native32 | NF4 native32 | BF16 fast4 | NF4 fast4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 183.37 | 229.87 | 123.79 | 146.63 | 169.38 | 194.11 |
| 8 | 1210.72 | 878.77 | 819.35 | 648.59 | 1072.64 | 769.83 |
| 32 | 3404.77 | 2654.88 | 2419.77 | 2028.32 | 2995.11 | 2319.25 |

At C32, NF4 fast4 reaches 87.36% of this NF4 base
control, while BF16 fast4 reaches 87.97% of its BF16 base control.
Those are configuration-specific comparisons, not optimized hardware ceilings.
NF4 fast4 is 22.57% slower than BF16 fast4 at C32 but
14.60% faster at C1. Smaller stored weights do not establish higher
batched throughput for this dequantization/GEMM path.

## Trace accounting and corrected request scope

Each sample covers 12 forwards at batch 1, 8 or 32. Prefill and decode costs are
reported separately; a 12-forward sample is not a complete 64-output request
mix. Kernel shares use summed GPU kernel durations. Activity union removes
overlap; gaps are unattributed, not proven scheduler idle time. CPU ranges are
inclusive and must not be added together. Fused LoRA consumers include
activation/unpack work. Dense GEMM includes the output head. Generic attention
kernel names require a source-verified allowlist; unknowns remain explicit.

Initial traces in job23487369 requested saved logprobs and are retained as
**diagnostic**, not exact throughput-workload attribution. A corrected request
disables returned logprobs. The first corrected-profile attempt23487818 failed
before capture because a NumPy integer byte count could not serialize to JSON;
it is retained, not counted as completed profiling. Explicit integer conversion
and a regression test fix the observer. The corrected retry is a separate
single-card allocation; its phase ratios are paired within that allocation.

Corrected profile job **23488079** completes all 18 windows with logprob returns
disabled. Component storage is counted once per underlying storage within each
category; categories can share backing memory and are not automatically additive.


## Confirmed components and phase costs

Corrected job23488079 completes in9m55s on nid005035, one MI250 GCD,7CPUs,
40GB and a15-minute cap. This is a separate allocation from the primary
measurement, on the same node. Each phase's precision/policy comparisons stay
within its own allocation. The corrected profile-only run records zero
throughput repeats; primary throughput/peak figures above are unchanged.
All final public worker-source hashes match the completed corrected payload.

After graph capture, every arm has **1,269,964,800 bytes** of GDN state storage
and **201,338,880 bytes** of KV tensors. Parameter storage is1,705,972,416 bytes
in BF16 and959,550,144 bytes in NF4, excluding non-parameter metadata. NF4
adds31,100,928 bytes of cached FP32 scales. LoRA pool storage drops from
326,615,040 bytes at rank32 to40,826,880 bytes at rank4: exactly8x smaller.
BF16 fast4 also holds **303,169,536 bytes** of additional GDN packed base
weights, created with `torch.cat` while original projection parameters remain
live. This offsets the **285,788,160 bytes** saved in the LoRA pool. NF4 fast4
has no such BF16 packed-weight tensor. These owner-storage counts explain why
BF16 fast4's observed peak stays close to native32. Cross-component categories
can share backing storage in general and must not be summed without checking.

The following C32 decode shares use only kernels attributed to the sampled
DECODE forward ranges, excluding unattributed work. They are **GPU kernel-time
shares**, not request wall-time percentages. Combined LoRA consumers also do
activation/unpack work; dense GEMM includes the output head.

| Configuration | Dequant | Dense GEMM | Separate LoRA A+B | LoRA fused consumers | GDN state/conv | Sampled kernel sum per decode step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16-base | 0.00% | 59.37% | 0.00% | 0.00% | 23.64% | 6.411 ms |
| nf4-base | 24.28% | 44.84% | 0.00% | 0.00% | 17.37% | 8.685 ms |
| bf16-native32 | 0.00% | 43.87% | 26.69% | 0.00% | 16.42% | 9.061 ms |
| nf4-native32 | 18.60% | 34.81% | 21.91% | 0.00% | 13.78% | 11.220 ms |
| bf16-fleet4 | 0.00% | 51.88% | 1.42% | 10.89% | 20.89% | 6.959 ms |
| nf4-fleet4 | 21.93% | 41.84% | 1.25% | 7.75% | 15.41% | 9.599 ms |

For NF4 fast4, dequantization plus dense GEMM is about64% of sampled decode
kernel time, while separately named LoRA and fused-consumer kernels are about
9% (including non-LoRA consumer work). A/B rank reduction does not remove the
roughly2.1ms of dequantization per sampled decode step. BF16 fast4 avoids that
cost and uses the packed GDN projection; the memory/performance tradeoff is now
measured rather than inferred from rank or checkpoint size.

Prefill is sampled separately at1024 input tokens per forward:

| Configuration | Sampled kernel sum per1024-token prefill | Separate LoRA A+B share | Fused consumer share | Dequant share |
| --- | ---: | ---: | ---: | ---: |
| bf16-native32 | 39.549 ms | 29.17% | 0.00% | 0.00% |
| nf4-native32 | 41.716 ms | 27.72% | 0.00% | 5.21% |
| bf16-fleet4 | 32.923 ms | 6.35% | 11.38% | 0.00% |
| nf4-fleet4 | 35.113 ms | 6.01% | 10.66% | 6.19% |

These are12-forward samples (C32 contains4 prefill and8 decode forwards), not
complete request mixes. Unknowns and unattached sampling/metadata work remain
in the evidence, and profiler timing is not used as serving throughput.
Next candidates are GDN packed-weight ownership that avoids a duplicate BF16
copy, and a faster large-batch NF4 dequantization/GEMM path. Neither candidate
is implemented or claimed faster by this accounting change.

## Tools and checks

`run_qwen35_resources.py run ROOT` reads `ROOT/study.json`: named arms specify
model/manifest paths, mode, precision kind and explicit worker environment.
The public runner does not install the separately selected research fusion
hooks. `qwen35_resource_audit.py` wraps SGLang0.5.22 lifecycle checkpoints in
process only. `analyze_qwen35_resources.py` reads a Chrome trace on CPU.

CPU validation: **187 passed**, 3 skipped, 49 GPU deselected at the current
source snapshot. Trace tests cover overlap, runtime/GPU annotation attribution,
unknowns, fused consumers, shared-storage identity, request logprob flags and
NumPy byte-count serialization. All samples, stages, source hashes, graph and
weight checks are retained in [the curated evidence](qwen35-resource-mi250.json).
