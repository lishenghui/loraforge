# Qwen3.5-4B native-graph LoRA training check

The [Qwen3.5-4B model](https://huggingface.co/Qwen/Qwen3.5-4B) retains a
short-sequence benefit, but does **not** reproduce the 0.8B model's 1024-token
speedup. All saved numerical comparisons are exact against original PEFT.
This measures the complete optimizer step, with both paths at rank four.

## Verified complete-step result

Job `23509083` completes in 5m53s on LUMI `nid005026`: small-g, one MI250 GCD,
seven CPUs, 40GB host memory, a 10-minute cap. Four required workers complete
all three lengths: default PEFT before/after, ordinary PEFT with the fixed
causal policy, and native-autograd graph with that policy. There are 12 cases,
120 actual optimizer updates and 12 additional changed-input backwards.

BF16 base/compute, FP32 LoRA masters, all-linear rank four, alpha four, dropout
zero, batch one, checkpointing off, no padding/packing or KV cache. The full
24-GDN/eight-attention text model, LM head, causal loss, backward, zero_grad and
foreach AdamW are included. Loading, auditing, JIT and capture setup are outside
step timing. Identical token IDs are used across models and paths.

Median synchronized wall time in milliseconds:

| Tokens | PEFT before / after | Fixed-policy ordinary PEFT | Native graph | PEFT / graph range |
| --- | ---: | ---: | ---: | ---: |
| 128 | 313.706 / 295.193 | 299.802 | 119.705 | **2.466–2.621x** |
| 1024 | 409.556 / 408.579 | 408.923 | 409.468 | **0.998–1.000x** |
| 4096 | 1387.608 / 1387.198 | 1386.518 | 1390.940 | **0.9973–0.9976x** |

The receipt retains all unrounded values/samples. Ratios bracket both
same-card default PEFT controls. Their median-time drift is **5.90% / 0.24% / 0.03%**. In particular,
2.62x alone overstates the certainty of the short-sequence measurement. There
are five timed samples after two warmup updates; ranges are control brackets,
not statistical confidence intervals. The 4096-token graph path is slightly
slower (about 0.25%) and has no practical speedup.

Normalized comparison with the [earlier 0.8B result](qwen35-training-autograd.md):

| Tokens | Qwen3.5-0.8B speed range | Qwen3.5-4B speed range |
| --- | ---: | ---: |
| 128 | 3.429–3.431x | 2.466–2.621x |
| 1024 | 1.454–1.540x | 0.998–1.000x |
| 4096 | 1.000–1.001x | 0.9973–0.9976x |

Each ratio uses its own job's bracketing controls; the two model sizes were
tested on different GCDs/nodes/dates of the same hardware family. This is not a
same-physical-device causal scaling study. The pattern is consistent with less
removable dispatch idle time as model/sequence compute grows. No per-stage
profile was performed here, so this does not identify GEMM, GDN or LM head as a
measured dominant kernel. It does not establish an Unsloth or QLoRA comparison.

## Numerical and lifecycle evidence

All 12 CPU comparisons are **exact**, not merely within the unchanged gates:

- Three correctness-update losses and sampled logits.
- All **496** trainable factor gradients and updated parameter tensors.
- Loss, logits and every factor gradient after changing the input sequence.
- All five timed-update losses after the separate parameter/optimizer reset.

The candidate retains 8,116,224 original PEFT trainable parameters. The
benchmark now derives their complete expected names from the original linear
modules and verifies exact ownership, rather than assuming the 0.8B count.
Every worker audits all **427 loaded state entries**, including the tied output
head, against the unchanged prepared weights. It requires actual isolated FLA
GDN kernels and causal convolution; no GPU Torch fallback is accepted.

Each graph shape records two captures and 22 replays across all checks and
updates. Exactly ten occur during five measured steps: one forward and one
backward per real AdamW update. There are no empty-graph warnings. Recorded
AccumulateGrad stream-mismatch warnings remain unsuppressed and did not alter
observed exactness. Graph execution is first-order and fixed-shape.

## Memory

All three lengths fit one GCD without checkpointing or quantization. Allocator
values are not total-device VRAM; graph private/reserved pools make allocated
peaks alone misleading:

| Tokens | Native / graph allocated GiB | Native / graph reserved GiB |
| --- | ---: | ---: |
| 128 | 9.210 / 8.282 | 9.328 / 9.480 |
| 1024 | 16.854 / 8.430 | 17.773 / 19.094 |
| 4096 | 43.003 / 8.579 | 46.125 / 47.625 |

The graph path reserves more memory. No total-VRAM saving is claimed.

## Model identity and preparation

Official repository `Qwen/Qwen3.5-4B`, pinned revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Its two original public shards
(total 9,319,828,096 bytes) are checked against the Hub's LFS SHA256 values.
The text-only extraction checks every one of 426 serialized tensors against
the source, excluding vision/MTP without changing any text values. The tied
output head adds the 427th loaded-state check. No serving-fused weights are used.

The actual text model has **4,205,751,296 base parameters**, 32 layers (24 GDN,
eight full attention), hidden size 2560, intermediate size 9216 and vocabulary
248320. The original-token fixture hash matches the 0.8B preparation.

`prepare_qwen35_training.py` accepts indexed multi-shard input, rejects duplicate,
missing or misindexed weights, and preserves the existing single-file path.
Layer counts, attention counts and adapter names are configuration-derived;
the same strict shape/load/dtype checks remain. This does not validate every
other model size or the older packed/custom-MLP compiler paths on 4B.

[The curated receipt](qwen35-training-4b-mi250.json) includes original checkpoint
hashes/revision, preparation/worker/analysis source hashes, every sample, all
exact-tensor counts, input-refresh evidence, capture counters and memory totals.
Raw checkpoints/gradients stay outside the package. No research or installed
framework source is modified. CPU: **230 passed**, three skipped, 49 GPU unit
cases deselected; Ruff and wheel/sdist checks accompany the change.

## Reproduction

With the pinned original HF snapshot available, prepare on CPU:

```bash
python benchmarks/prepare_qwen35_training.py SOURCE_HF_4B BF16_TEXT_4B
```

On an allocated gfx90a GPU with the same isolated FLA dependency setup as the
0.8B experiment:

```bash
PYTHONPATH=/absolute/path/to/deps:/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/run_qwen35_training.py BF16_TEXT_4B --output RESULTS \
  --native-graph-only --check-input-change --recompile-limit 64 --max-seconds 560
```

This selection requires all four workers and uses the original default-input
PEFT before/after controls. On CPU after the GPU job:

```bash
python benchmarks/analyze_qwen35_training.py RESULTS --output comparison.json
```

All GPU work and public model downloads for this check have finished. The
standalone experiment changes no public training default or serving integration.
