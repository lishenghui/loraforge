# NF4 base with BF16 LoRA on MI250

This experiment serves Qwen3.5-0.8B with an NF4 base and separate BF16 LoRA
factors. Compute activations, LoRA A/B buffers, KV and recurrent state keep the
preceding BF16 configuration. Four deterministic synthetic compact rank-4
adapters exercise batching; no training or trained-adapter quality is measured.

## Fixture and loading contract

The CPU [packing tool](../benchmarks/prepare_qwen35_packed.py) concatenates the
exact source BF16 rows under runtime projection names: q/k/v, gate/up, GDN
qkv/z and b/a. It compares every concatenated source slice before writing the
new checkpoint. All other tensors are retained, and source/output hashes are
recorded. The paired BF16 control uses this same fused checkpoint. Original
model and adapter files are untouched.

On an allocated GPU, [NF4 preparation](../benchmarks/prepare_qwen35_nf4.py)
quantizes all **114 decoder projections**, including GDN b/a, using bitsandbytes
NF4, block size 64 and nested/double statistics. Embeddings, norms, convolution
and non-decoder weights retain their source values. The checkpoint is not
uniformly four-bit. Target weight data shrinks from 995,229,696 to 248,807,424 bytes,
before quantization metadata. The complete NF4 checkpoint is 1,008,615,366 bytes.
Each fused projection has one NF4 quantization state; native and Fleet use
identical packed weights and scales. LoRA factors remain separate BF16 files.

The installed SGLang 0.5.22 has three gaps for this combination:

- Qwen3.5 does not advertise the bitsandbytes fused-parameter mapping. The fixture
  already uses runtime names, so its map is empty; no quant-state shard joining
  or unsupported GDN tuple loading is needed.
- Its ROCm admission list omits bitsandbytes, although the installed 0.50.2 backend
  executes the tested NF4 operators on gfx90a. The explicit
  [benchmark adapter](../benchmarks/qwen35_nf4_compat.py) adds only this method to
  that version's admission list. It retains all other verification, checks
  Qwen3.5/BF16/NF4/double-quant/gfx90a scope and requires the generated fixture's
  receipt. This is an out-of-tree experiment, not upstream supported ROCm serving.
- LoRAAdapter inherits the base `LoadFormat.BITSANDBYTES`, then passes it to a
  regular file loader that rejects it. The adapter copies that configuration
  with `safetensors` format. The original base configuration and all other
  fields stay intact. No adapter quantization or dtype change is introduced.

Only explicit `--nf4-experiment` on `benchmarks/qwen35_hybrid.py run/serve`
installs this adapter in each worker. The manifest model path must refer to
this prepared NF4 fixture. The flag implies the expanded 96-sequence checks.
Combine with `--cold-prefill-graph` to use the same cold 8x128 graph in native,
Fleet and base. Library entrypoints/defaults and the shared SGLang environment
are unchanged. Base and fast LoRAForge GDN fusion differs by supported dtype:
the BF16 research hook packs base qkvz+ba; NF4 uses its two quantized base
projections and retains the LoRA consumer fusion.

## Initial operator check and loading verification

Job 23482416, nid005053, prepares the NF4 checkpoint and passes 24 numerical
operator checks against the same dequantized weights. Timings include the
actual NF4 matmul/dequantization path versus original BF16 projection, with
three shuffled graph samples. It then hits the stock ROCm admission error;
that failed service attempt is retained separately from successful operators.

| Projection | Rows | BF16 microseconds | NF4 microseconds | BF16 time/NF4 time |
| --- | ---: | ---: | ---: | ---: |
| GDN QKVZ 8192x1024 | 1 | 25.25 | 13.05 | 1.93x |
| Gate/up 7168x1024 | 1 | 22.55 | 11.92 | 1.89x |
| Down 1024x3584 | 1 | 28.14 | 12.30 | 2.29x |
| GDN QKVZ 8192x1024 | 32 | 26.95 | 118.40 | 0.23x |
| Gate/up 7168x1024 | 32 | 22.96 | 105.25 | 0.22x |
| Down 1024x3584 | 32 | 28.48 | 75.06 | 0.38x |

NF4 is faster for the tested single-token projections but slower at rows 8/32
and 1024. Weight storage reduction alone does not establish higher throughput.
These operator ratios are not LoRAForge/native or full-service speedups.

Earlier full-model runs used canonical fused names with the stock substring
shard loader. `qkv_proj` matches `v_proj`, and `gate_up_proj` matches `up_proj`;
those parameters were silently skipped. Dtype/quant-state presence did not
prove their actual data was loaded. All full-model numbers from jobs 23482434
and 23482520 are **invalid and excluded**. The direct operator screen above reads
actual checkpoint weights and remains independent of that loader failure.

The [checked loader](../benchmarks/qwen35_packed_loader.py) sends all 114 canonical
projection tensors straight through the native full-weight loader, then requires
the model's actual storage to be exactly equal to the source tensor. Missing,
duplicate, changed or incomplete projections fail before graph capture/timing.
Non-projection weights still use native loading. The public runner activates
this only for a packed/NF4 fixture with a generated receipt, and propagates it
to spawned workers. Validation rejects missing/null and nonfinite logprobs
before any throughput measurements.

The matched study measures native BF16 LoRA, conservative Fleet, the preceding
research fast policy and no-adapter base under each base precision. Each
policy/native pair runs within one allocation and identical base weights; NF4/BF16 outputs are allowed to
differ because the base itself was quantized. Any research-policy native token
or logprob mismatch remains a failed numerical result, even when forced-length
diagnostic throughput is measured. It is never an equal-quality speed claim.

## Reproduce

Use the pinned SGLang 0.5.22, bitsandbytes 0.50.2, Torch 2.11 ROCm 7.2 environment.
On CPU prepare the control fixture; on one allocated gfx90a GPU quantize it:

```bash
python benchmarks/prepare_qwen35_packed.py LOCAL_QWEN35_MODEL FIXTURE/packed-bf16
python benchmarks/prepare_qwen35_nf4.py FIXTURE
# In FIXTURE, on the allocated GPU:
python /absolute/path/to/benchmarks/check_qwen35_nf4.py
```

Create a rank 4 adapter study whose model path is `FIXTURE/nf4`, then run the
public native/Fleet/base benchmark with `--nf4-experiment --cold-prefill-graph`.
Use a separate study with `FIXTURE/packed-bf16` for the BF16 control. Reuse the
same adapter factors, prompts, rank capacity, concurrency, output lengths and
request counts. The public CLI installs the maintained Fleet policy; earlier
private fusion hooks are separately identified research candidates and are
not installed by this flag.

References: [bitsandbytes AMD installation](https://huggingface.co/docs/bitsandbytes/installation)
and [SGLang quantization matrix](https://docs.sglang.io/docs/advanced_features/quantization).
The tested installed source and runtime audits take precedence over a broad
online compatibility description.


The installed bitsandbytes 0.50.2 ROCm dispatch selects its fused SIMT kernel
through M=4 for BF16 on gfx9. Larger batches use `_dequant_linear_fallback`:
it reconstructs nested absmax, materializes BF16 weights, and calls `F.linear`.
This explains the direct projection screen's large difference between rows 1
and 8/32. It is a source-backed execution-path finding, not a universal statement
about four-bit inference or other GPUs/backends. A fused batched NF4 GEMM is a
separate optimization target; its speed is not measured here.


## Checked-weight NF4 result

Job **23482836**, nid005035, completes native, conservative Fleet, fast research
Fleet and base under NF4. All four verify 114/114 actual projection tensors
against the checkpoint; all 336 adapter buffers in LoRA modes are BF16. Each
point has three repeats, 128 input/64 forced output tokens, and 8/32/96 requests
at C1/8/32. All modes use the same cold-prefill and decode graphs.

| Concurrency | NF4 base | Native LoRA | Conservative Fleet | Fast research Fleet |
| --- | ---: | ---: | ---: | ---: |
| 1 | 221.71 | 146.05 | 147.57 | 188.98 |
| 8 | 578.12 | 473.57 | 478.01 | 528.39 |
| 32 | 1887.01 | 1562.85 | 1563.27 | 1701.98 |

Values are output tokens/s. Conservative Fleet matches 96/96 native sequences
and saved logprobs exactly; its throughput is approximately equal to native.
The research fast policy matches 96/96 token sequences but reaches 0.0999453
maximum logprob difference, failing the strict 0.05 gate. Its fixed-work ratios
are 1.2939x/1.1158x/1.0890x; strict numerical success is not inferred from timing.
The three C1 fast repeats are 138.07, 188.98, 189.16 tokens/s; the reported result is their median. These short three-repeat samples are exploratory.

The NF4 phase was saved before intentionally cancelling the unused remainder
of its allocation at 12m42s. BF16 native, conservative and base completed in job
23482975 on nid005035. Its fast candidate was stopped before timing by a cache
audit placed before lazy cache creation. Job 23483167 on nid005032 retests BF16
native and fast together, checking all 18 GDN packed caches after the first
forward, before timing. Every cache is exactly equal to current base weights.

## BF16 controls and precision comparison

All values below are median complete-service output tokens/s, with the same
128/64 input/output lengths, adapters and graph configuration. The fast BF16
column uses the new same-allocation native control; the other BF16 columns use
job 23482975. Do not mix those two native controls when computing policy ratios.

| Concurrency | BF16 base | BF16 native | BF16 conservative | BF16 fast control | BF16 fast | NF4 fast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 185.25 | 122.00 | 124.40 | 122.16 | 165.83 | 188.98 |
| 8 | 1224.34 | 831.53 | 835.19 | 846.22 | 1060.17 | 528.39 |
| 32 | 3457.01 | 2484.42 | 2469.14 | 2533.46 | 2924.75 | 1701.98 |

| Concurrency | BF16 fast/native | NF4 fast/native | NF4 fast/BF16 fast |
| --- | ---: | ---: | ---: |
| 1 | 1.3574x | 1.2939x | 1.1397x |
| 8 | 1.2528x | 1.1158x | 0.4984x |
| 32 | 1.1544x | 1.0890x | 0.5819x |

Quantization does **not** increase the tested LoRAForge/native relative benefit.
Single-request absolute throughput improves, but NF4 is substantially slower
at C8/C32 on this installed ROCm backend. Absolute precision comparisons cross
allocations/nodes and are exploratory; policy/native ratios are paired within
one allocation. Short three-repeat medians are not confidence intervals.

Numerical results against each precision's own native control:

| Policy | Identical token sequences | Maximum saved logprob difference | Strict 0.05 gate |
| --- | ---: | ---: | --- |
| bf16-conservative | 96/96 | 0.081472695 | FAIL |
| bf16-retest-fast | 96/96 | 0.120192289 | FAIL |
| nf4-conservative | 96/96 | 0.000000000 | pass |
| nf4-fast | 96/96 | 0.099945307 | FAIL |

All these comparisons use the exact checked checkpoint within precision. The
conservative NF4 path passes strictly and provides approximately native
throughput. Research fast rates remain fixed-work diagnostics wherever their
numerical gate fails; they do not establish equal-quality acceleration.

An additional control against the earlier original/unfused BF16 native run
(job 23481601) matches **91/96** token sequences, with maximum common-prefix
logprob difference **0.0914304852**, failing the strict gate. The packed source
rows and all 114 loaded projection tensors are exact, but this older execution
has not been reproduced numerically; the cause is not localized. Do not use
this study as proof of equivalence to that older checkpoint execution, or
ascribe the difference to harmless rounding. Comparisons in this report use
the newly checked, precision-local native controls.

The direct operator screen points to a fused batched NF4 GEMM as the next
quantized-backend target. Merely reducing LoRA rank or weight storage cannot
remove the current batch>4 dequantization fallback. No speed for a replacement
quantized kernel is claimed here. This is NF4-base/BF16-LoRA **inference**, not
Unsloth training, a trained QLoRA evaluation, or general performance across
quantization formats and devices.

All samples, numerical flags, memory receipts and actual graph/weight audits
are retained in [the evidence summary](qwen35-nf4-mi250.json). The experiment
remains an explicit benchmark path with pinned compatibility scope.


## Large-batch NF4 optimization

`--nf4-experiment --nf4-cached-scales` explicitly enables a benchmark-only
large-batch path. The base keeps its original NF4 bytes and LoRA stays BF16.
The new path reconstructs immutable FP32 block scales once per parameter/state
version, dequantizes with one new Triton kernel, and calls the native BF16
linear operation directly. It avoids repeated double-scale recovery and the
SGLang shard-output copy. M<=4 and unsupported shapes/layouts keep the original
bitsandbytes implementation. This is not a fused NF4 GEMM: full BF16 temporary
weights are still written and read each invocation. There is no cache of their
values, but captured graphs retain their temporary-buffer allocation lifetimes.
The persistent FP32 scale cache across 114 projections is **31,100,928 bytes**.
Captured execution assumes an immutable base; eager versioned weight/scale
updates invalidate the scale cache. Shared installations and runtime defaults
are unchanged.

Jobs 23484723/23484738 each check six actual projection shapes at rows
1/8/32/1024. All 48 shape/batch cases use exactly reconstructed BF16 weights
and produce exact dequant-plus-linear outputs against the original NF4
dequantized weights. At rows 8/32 the selected operator is about 1.5–2.4x the stock path; these are operator ratios. The first
MFMA prototypes, including a register-codebook variant and forced native SIMT,
were also screened. None provided a generally better selected serving path;
slow and shared-memory-exhausted variants are retained as rejected candidates.
The resident-BF16 column is only a diagnostic ceiling, not the quantized path.

Job **23484772**, **nid005030**, runs old-native, old-fast, new-native and new-fast
in that order on one MI250 GCD, 7 CPUs, 40GB, with a 20-minute cap. Same NF4
checkpoint, four compact rank-4 adapters, 128 input/64 forced output tokens,
8/32/96 requests at C1/8/32, three repeats and cold-prefill/decode graphs.
Every arm verifies all 114 actual projection tensors and 336 BF16 LoRA buffers.
Both new arms record 114 scale preparations and actual eager/captured execution.

| Concurrency | Old native | New native | Old fast | New fast | New fast/old fast | New fast/new native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 147.15 | 150.65 | 187.96 | 194.61 | 1.0354x | 1.2918x |
| 8 | 473.07 | 662.99 | 523.87 | 775.53 | 1.4804x | 1.1697x |
| 32 | 1561.36 | 2096.27 | 1684.37 | 2343.23 | 1.3912x | 1.1178x |

Values are complete-service output tokens/s. Backend improvement also benefits
native LoRA; only the final column measures the research Fleet benefit against
native using the **same new quantized backend**. Comparing new-fast to old-native
would conflate two optimizations.

Both new-native/old-native and new-fast/old-fast retain **96/96 identical token
sequences and zero saved logprob difference**. The fast policy's pre-existing
native logprob failure remains; this change does not fix or relabel it:

| Comparison | Identical token sequences | Maximum saved logprob difference | Strict 0.05 gate |
| --- | ---: | ---: | --- |
| old-fast/corresponding native | 96/96 | 0.099945307 | FAIL |
| new-fast/corresponding native | 96/96 | 0.099945307 | FAIL |

This is a three-repeat synthetic inference study on one allocation, not trained
quality, training speed, a confidence interval or a claim of fourfold VRAM
reduction. The earlier original/unfused BF16 control limitation remains in the
preceding section. CPU validation now passes 169 tests; Ruff, wheel and sdist
builds pass. [All samples, rejected operator candidates, source hashes and graph,
weight and scale audits](qwen35-nf4-optimization-mi250.json) are retained.
