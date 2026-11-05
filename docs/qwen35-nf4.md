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

The matched study measures native BF16 LoRA, the preceding research fast policy,
and no-adapter base under each base precision on one allocation. Native/Fleet
comparisons stay within identical base weights; NF4/BF16 outputs are allowed to
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

The NF4 phase is saved before intentionally cancelling the unused remainder
of its allocation at 12m42s. BF16 comparisons run in a separate single-card
allocation, job 23482975. Each precision's policy/native comparison is paired
within that allocation. The four BF16 controls and final cross-precision table
are pending. See [the receipts and raw sample summary](qwen35-nf4-mi250.json).
