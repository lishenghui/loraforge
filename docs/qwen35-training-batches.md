# Qwen3.5-4B real-microbatch training

The batch-one short-sequence graph benefit does not establish a training
throughput benefit at larger batches. This study increases the number of
sequences processed together in one forward/backward and reports input
tokens/second, complete-step wall time and allocator memory separately.
Gradient accumulation remains **one** throughout.

## Measured result

LUMI job `23509496` completes in **14m31s** on `nid005034`: small-g, one
MI250X GCD (63.98 GiB device capacity), seven CPUs, 40GB host memory and a
20-minute cap. Seven shapes complete all three workers: **21 workers, 210
optimizer updates and 21 additional changed-input backwards**. The last
shape encounters explicit GPU OOM in both methods. No GPU allocation remains.

Median synchronized complete-step wall time:

| Batch | Sequence tokens | PEFT before / after ms | Native graph ms | Graph input tokens/s | Same-batch PEFT / graph |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 306.751 / 308.442 | 119.922 | 1,067 | **2.558–2.572x** |
| 8 | 128 | 382.421 / 382.077 | 383.237 | 2,672 | 0.9970–0.9979x |
| 32 | 128 | 1235.185 / 1234.263 | 1238.559 | 3,307 | 0.9965–0.9973x |
| 1 | 1024 | 400.346 / 400.264 | 402.083 | 2,547 | 0.9955–0.9957x |
| 2 | 1024 | 717.515 / 716.883 | 718.843 | 2,849 | 0.9973–0.9982x |
| 4 | 1024 | 1277.622 / 1281.191 | 1280.765 | 3,198 | 0.9975–1.0003x |
| 8 | 512 | 1252.190 / 1252.924 | 1256.585 | 3,260 | 0.9965–0.9971x |
| 8 | 1024 | GPU OOM | GPU OOM | — | No valid comparison |

At a fixed length of 128, increasing the graph path's batch from one to 32
raises throughput **3.10x**. At a fixed length of 1024, batch one to four raises
it **25.6%**. Original PEFT gains from batching too: once the batch is at least
eight at length 128, its throughput matches or slightly exceeds the graph
path. Across the tested larger batches, graph ratios span **0.9965–1.0003x**;
there is no meaningful graph speedup. All same-shape PEFT control medians
are within 0.56% of each other.

The earlier batch-one short-sequence gain must not be presented as normal
training throughput improvement. The pattern is consistent with amortizing
dispatch overhead as useful work per step grows; this is an inference, not a
per-kernel profile. Both paths remain rank four. This is neither a
rank-32-to-rank-four compression speedup nor an Unsloth/QLoRA comparison.
Sequence lengths are different workloads; the largest tokens/s across lengths
does not determine the appropriate training context length.

All **14** saved CPU comparisons pass and are **exact**, including the
per-sequence sampled logits, all 496 parameter gradients/updates and the
changed-input audits. Each successful graph shape records two captures, 22
replays overall and ten replays during the five timed updates. The
[curated receipt](qwen35-training-batches-mi250.json) retains all timings,
gates, original checkpoint identity, worker/analysis hashes and OOM traceback
tails. Raw audit tensors and full logs stay outside the package.

## Memory and the next capacity limit

Peak reserved allocator memory, in GiB (PEFT uses the larger of the two
controls):

| Batch × length | PEFT | Graph |
| --- | ---: | ---: |
| 1 × 128 | 9.328 | 9.480 |
| 8 × 128 | 17.727 | 18.752 |
| 32 × 128 | 46.104 | 46.293 |
| 1 × 1024 | 17.727 | 18.750 |
| 2 × 1024 | 27.061 | 29.004 |
| 4 × 1024 | 46.104 | 46.270 |
| 8 × 512 | 46.104 | 46.293 |

Batch eight at length 1024 fails in the original Transformers
`ForCausalLMLoss` at `logits = logits.float()` in both methods. The requested
allocation is **7.58 GiB**; native has 2.86 GiB free and graph 2.95 GiB, with
58.88 GiB already allocated in each. The graph failure occurs during graph
setup, before graph execution is available. This is not a graph-only capacity
regression or a Slurm host-memory failure. The second PEFT control is skipped
because the first did not produce a valid reference.

Within this measured no-checkpointing grid, **batch four × 1024** fits and
batch eight × 1024 does not; intermediate batches were not tested. A logical
next capacity experiment is a chunked or fused LM-head/loss path that avoids
materializing full-vocabulary FP32 logits. Its speed, numerical behavior and
memory benefit are untested here. Gradient checkpointing is another separate
capacity experiment; the current graph helper explicitly rejects it. Gradient
accumulation can increase effective batch without increasing microbatch, but
is not counted as a microbatch throughput gain.

## Protocol

The unchanged prepared [4B text checkpoint](qwen35-training-4b.md), BF16
base/compute, FP32 LoRA masters, all-linear rank four, alpha four and dropout
zero are retained. Both methods use the same actual batch, initial adapters
and input matrix. Sequences are deterministic rotations of the token fixture;
they occupy independent rows, rather than duplicating one sequence through
gradient accumulation. These are synthetic throughput inputs, not a dataset
or convergence experiment.

Each shape runs original PEFT before, native-autograd graph, then original
PEFT after, sequentially in fresh processes on the same GCD. There are three
correctness optimizer updates, one additional changed-input backward, a reset,
two warmup updates and five timed actual foreach-AdamW updates per successful
worker. Timings include zero_grad, the complete model, native causal loss,
backward and AdamW. Data is already on device; loading, full weight auditing,
JIT, graph capture and numerical audit copies are outside step timing.
No activation checkpointing, quantization, padding, packing or KV cache is used.
The optimizer remains outside graph capture.

Input throughput is `batch * sequence_length / step_seconds`; supervised
throughput is `batch * (sequence_length - 1) / step_seconds`. Both and
sequences/second are retained in each worker receipt. The reported speed
interval brackets the two same-batch PEFT controls; it is not a confidence
interval. Five timed samples do not establish long-running training speed.

The fixed-causal policy now accepts a positive rectangular batch while keeping
the original native SDPA callable and implicit causal/GQA behavior. Its three
RoPE axes equal the original default axes after `[1:]` for every row. This
extends the known unpacked benchmark contract; it does not add support for
arbitrary padding, user positions, packing, dropout or checkpointed graphs.
No installed framework source is modified.

## Numerical and capacity checks

Every worker verifies all 427 loaded state entries and the identities of all
496 trainable factor tensors (8,116,224 scalar parameters). Cross-method CPU
checks cover three update losses, all factor gradients/updates, sampled logits
from **every sequence**, five timed losses and changed-input loss/logits/all
gradients. Existing gates remain loss absolute error 0.01, sampled-logit
relative L2 0.02, each parameter's gradient relative L2 0.05, and update
absolute error 0.0001. An additional per-sequence logit relative-L2 gate of
0.02 prevents one wrong row from being diluted by a large-batch average.
Original/changed input hashes and batch dimensions must also agree.

Each shape/method is an independent process. A recognized GPU out-of-memory
exception is recorded explicitly, with its log preserved, and cannot poison
subsequent shapes. Other errors, timeouts and budget skips remain failures.
If the first PEFT control cannot fit, the graph capacity is still tested but
the second control is skipped: there is no valid paired speed comparison.
An OOM shape is never counted as a successful numerical test.

Allocator allocated/reserved peaks are not total-device VRAM. In particular,
graph private pools can produce low allocated peaks with larger reserved
peaks; these measurements do not establish total-memory savings.

## Reproduction

Use the environment and original checkpoint preparation recorded in the
[4B study](qwen35-training-4b.md). On one allocated gfx90a GCD:

```bash
PYTHONPATH=/absolute/path/to/deps:/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/run_qwen35_training_batches.py run BF16_TEXT_4B \
  --output RESULTS --max-seconds 1100 \
  --shapes 1x128 8x128 32x128 1x1024 2x1024 4x1024 8x512 8x1024
```

The run phase only trains and saves audits. After the GPU job, compare on CPU:

```bash
python benchmarks/run_qwen35_training_batches.py compare RESULTS \
  --output comparison.json
```

Keep failed attempts and raw audit tensors outside the package. The runner
creates a new result directory and refuses to reuse an existing one.

CPU checks: **242 passed**, three skipped, 49 GPU unit cases deselected;
Ruff and wheel/source-archive checks pass. This remains a standalone,
first-order fixed-shape training experiment, not a public Trainer integration.
