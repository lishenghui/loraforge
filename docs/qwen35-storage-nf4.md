# GDN shared storage and exact NF4 byte-pair reconstruction

This follow-up optimizes the already compressed rank4 research policy. Ratios
below compare the same rank4 files and the same policy before/after one change;
they are **not** new compression-versus-rank32 speedups. Earlier original32 /
compressed4 and fast / compressed-native numerical failures remain unresolved.
The fixture is synthetic, with 128 input / 64 forced output tokens, four adapters,
C1/8/32, 8/32/96 requests per repeat, three repeats, identical decode and cold
8x128 prefill graphs, KV 16384 / request 32 / GDN 64 slots. No profiler runs during
these allocations. LoRA buffers remain BF16 and base checkpoints are verified.

## Exact NF4 dequantization

The old prepared path interpreted each reconstructed weight separately. The
new path loads one quantized byte, uses its high/low nibbles to access the
canonical FP32 codebook, applies the same FP32 block scale, rounds each value
to BF16 and interleaves the output. The following native linear call, LoRA
kernels and scheduling remain unchanged. It adds no persistent BF16 weight
cache. M<=4 retains the original BNB path; the small 32-output projection
retains the previous dequantizer. The selected kernel uses 1024 packed bytes
per block and 4 warps. Other tile/lookup choices are screening candidates only.

Operator job **23489102**, nid005059, completes in 46s, one MI250 GCD, 7 CPUs,
40GB, small-g, 20min cap. All six unique shapes reconstruct exactly versus
bitsandbytes across six tile/lookup choices. All selected dequant-plus-linear
outputs are exact against BF16 reconstruction at rows 8/32/1024. At 32 rows,
8192x1024 improves 64.96→55.33us,7168x1024 improves 56.40→48.13us,
5120x1024 improves 45.29→39.70us, and1024x3584 improves 47.88→44.20us.
These operator times are not full-service acceleration ratios. Split-K fused
MFMA candidates remain slower and are not enabled. Their changed FP32
reduction order also produces some nonidentical BF16 outputs.

Full-model job **23489176**, nid005042, completes in 12m06s on one MI250 GCD,
7CPU, 40GB, 20min cap. The accepted NF4 sequence brackets the candidate with the
old implementation. Each arm verifies 114 actual loaded projection tensors,
336 BF16 LoRA buffers and eager/graph execution. The candidate separately
checks exact reconstruction for all 96 selected large projections during eager
warmup; all 114 scales are prepared once. All 44 validation sequences and saved
logprobs exactly match the old path, including 32-way 64-output requests.

| Concurrency | Old before tok/s | Byte-pair tok/s | Old after tok/s | Gain versus bracketing controls |
| --- | ---: | ---: | ---: | ---: |
| 1 | 193.07 | 193.66 | 194.71 | -0.54% to +0.31% |
| 8 | 776.86 | 810.49 | 776.82 | +4.33% to +4.33% |
| 32 | 2340.27 | 2415.72 | 2312.83 | +3.22% to +4.45% |

C1 differences are too small for a speed claim. C32 baseline drift is
-1.17%; the reported gain interval uses both controls.
NF4 C32 peak allocated memory stays **2.974449GB**.
GB means decimal1e9bytes and PyTorch peak allocated, not device-wide peak.
The sampled prior decode dequantization fraction was about 22%; reducing only
that phase yields a smaller complete-service improvement. Native BF16 GEMM,
the output head, GDN state/conv and serving overhead remain. No new native32
arm or base-only arm is measured in this follow-up.

## GDN packed-weight ownership

The packed BF16 8192+32 by1024 projection previously coexisted with independent
original parameter storages. The new benchmark hook keeps the parameter
objects and metadata, validates exact values, and rebinds their data to
contiguous slices of the packed owner before graph capture. Packed computation
and native fallback linears now read the same values from one storage. There
is no post-capture rebinding, live weight reload or training support.

The first class-method-only wrapper in job 23489176 was a **no-op** for this
purpose: lazy initialization called a closed-over function directly. Duplicate
bytes remained 303,169,536; that arm is not evidence of shared storage. The
corrected hook prepares the packed owner at the projection entrypoint, covering
both explicit and closure-based initialization. A CPU regression reproduces
this bypass. The final runner can require zero extra storage before timing.

Retry job **23489614**, nid005026, completes in 7m18s using one MI250 GCD,
7 CPUs, 40GB, small-g and a 10min cap. Controls bracket the shared version
within this separate allocation.
At graph capture, packed GDN storage is 303,169,536bytes in every arm, but its
**additional storage beyond model parameters falls 303,169,536→0bytes**.
Union accounting deduplicates underlying storages instead of summing aliases.
All 44 validation sequences and saved logprobs match exactly in this allocation;
both original-weight/packed-cache value checks and graph execution pass.

| Configuration | C32 peak allocated GB | C32 output tok/s |
| --- | ---: | ---: |
| bf16-control | 3.985314 | 2934.76 |
| bf16-shared | 3.682145 | 2937.76 |
| bf16-control-after | 3.985314 | 2973.58 |

The measured C32 peak saving is **303.17MB**.
This change targets memory ownership. Throughput samples and baseline drift
are recorded, without treating storage reduction as a compute acceleration.
The GDN and NF4 allocation controls must not be mixed into one speed ratio.

## Reproduce and scope

Use the pinned Torch 2.11 ROCm 7.2, Triton 3.6, SGLang 0.5.22 and bitsandbytes 0.50.2
MI250/gfx90a environment from the preceding NF4 study. Quantization is
NF4/block64/doublequant with BF16 compute and LoRA. Run the portable screen:

```bash
python benchmarks/check_qwen35_nf4_candidates.py FIXTURE --output screen.json
```

The public NF4 runner accepts the explicit experimental opt-in:

```bash
python benchmarks/qwen35_hybrid.py run STUDY --nf4-experiment \
  --nf4-cached-scales --nf4-byte-pairs --cold-prefill-graph
```

These flags use the maintained policy; they do not install the separate fast
fusion research wiring used in the timing study. GDN storage sharing is a
benchmark hook enabled by `LF_GDN_STORAGE=1` with the existing packed BF16 GDN
prototype (`LF_GDN_ARM=packed/both`); it is not an installed runtime default.
`run_qwen35_resources.py` accepts `profile_concurrencies: []` for unprofiled
studies, `expect_gdn_shared_storage: true` for pre-timing ownership checks and
`expect_nf4_byte_pairs: true` for all 96 selected reconstruction checks. The last
gate was added after the completed NF4 allocation; its identical saved audit
was checked on CPU. Final numerical kernels match the measured NF4 payload;
the final GDN/observer/runner sources match the retry payload. Per-payload and
public-source hashes are retained separately.

Validation: 192 CPU tests passed, 3 skipped, 49 GPU deselected; Ruff checks and
wheel/sdist build passed. Shared SGLang, probe and paper sources remain unchanged.
Full samples, stages, windows, exact comparisons and rejected/no-op evidence:
[qwen35-storage-nf4-mi250.json](qwen35-storage-nf4-mi250.json).
