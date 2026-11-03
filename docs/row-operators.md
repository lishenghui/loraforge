# Retained-rank row operators

`loraforge.ops.row_shrink` and `row_expand` extract the project-owned direct-row
shrink and RowExpand from probe's `fast_decode.cu`. They operate on BF16 tensors
without importing a serving framework. They are independent of the existing
padded-slot vLLM GEMV integration and do not change its dispatch. This is the
first operator migration toward the Figure 12 serving path, not a reproduction
of that figure or a throughput result.

The source already has CUDA and HIP compilation branches. Hardware validation
is recorded below; compilation branches alone do not establish device support.
The initial interface accepts BF16 only, effective ranks 0..64 and one to three
fused projection slices. Rank zero denotes an inactive slot. The stored rank
capacity may exceed 64; only effective ranks up to 64 run these operators.

## Layout and numerics

For `row_shrink(x, a, output, slots, ranks, num_slices=S, max_rank=K)`:

- `x`: `[tokens, input_width]`; `a`: `[slots, S * capacity, input_width]`;
  `output`: `[tokens, S * capacity]`, all BF16 on one GPU.
- Each slot's A slices are packed at `s * ranks[slot]`, not `s * capacity`.
  Live output prefixes use that same layout. Other columns remain unchanged.
- `slots[t]` is an int32 slot index or -1. Rank-zero/base rows remain unchanged.
- Each live rank is at most `max_rank`, which is in 1..64 and no greater than
  capacity. Input width and row stride are multiples of eight BF16 elements;
  the X/A starting addresses must be 16-byte aligned. A/output are contiguous.
- Each warp/wave reduces an A row in FP32 and rounds the result to BF16.

`row_expand(z, b, output, **metadata)` consumes that packed intermediate:

- `z`: `[tokens, S * capacity]`; `output`: `[tokens, output_width]`.
  Z is contiguous. Output may be a column slice of a larger base projection,
  with contiguous columns and nonoverlapping rows.
- `b` is contiguous flat BF16 storage. Slot `g` starts at `b_offsets[g]`
  **elements** and has `output_width` rows of width `b_strides[g]`. The stride
  is a power of two in 1..64, at least the live rank; padded columns are zero.
  B's starting address is 16-byte aligned and slot offsets are multiples of
  eight elements. The containing buffer must cover each live slot's full region.
- A segment names one slot and at most 16 token rows. `segment_indptr` indexes
  `token_indices`; `segment_slots` contains the slot IDs. Empty segments and
  slot -1/rank-zero segments are skipped. `segment_count` is a one-element
  int32 device vector, so a captured graph can vary its number of active segments.
- `slice_offsets`, an int32 vector of length S+1, partitions output columns
  into strictly increasing slices, beginning at zero and ending at output width.
- `scales[g]` is a finite FP32 multiplier. RowExpand reduces B @ Z in FP32,
  scales the delta, rounds it to BF16, adds the base output and rounds to BF16
  again. This preserves the source convention and differs from the existing
  vLLM GEMV's FP32 intermediate convention. Scaling must be applied exactly once;
  a loader that folds scaling into B must supply unit scales.

The kernel loads each B row into registers once per token segment and stages
the segment's Z values in shared memory. `threads` is 128 or 256; `blocks`
defaults to twice the device SM/CU count. This explicit launch policy has not
been tuned for an end-to-end serving workload.

## Metadata and lifetime contract

These are low-level, inference-only tensor operators. Structural checks reject
wrong devices, dtypes, dimensions, layouts and shared input/output storage.
Tensor dimensions and the segment/column work count must fit signed int32.
They do not read device metadata values back to the CPU. Callers must establish:

- Slots are -1 or valid indices; ranks and buckets satisfy the bounds above.
- The active count is between zero and segment capacity. Indptr entries are
  nonnegative, nondecreasing, within token-index storage, and each active span
  has at most 16 tokens. Each indexed token is in range and appears at most once
  across active segments. Duplicate tokens would create conflicting output writes.
- Slice boundaries, B offsets, zero padding and scales satisfy the layout contract.
  Shrink/expand routes must agree when composing the two operators, and all live
  intermediate elements must have been written before expansion.

Warm each operator before graph capture; compilation during first-use capture
is rejected. Native launches use the input device's current PyTorch stream and
restore the caller's current device. Capture/replay reads ranks, offsets, scales,
routes, segment counts and weights from device memory. Callers retain all buffers
through completion, keep their addresses stable for captured graphs, and order
producer writes against consumers using streams/events. No global scratch or
adapter registry is used. Autograd is rejected. Concurrent writes to one output
or concurrent metadata mutation are unsupported.

The existing `GPUAdapterSlots` replaces allocations on load and cannot by itself
provide this fixed-address graph contract. A compact storage/metadata owner and
vLLM loading/eviction bridge remain separate migrations. Figure 12 additionally
requires its scheduling policy, model/adapters, fixed input workload and memory
accounting. The research results and launchers are not copied into this library.

## Validation

On an allocated GPU with a matching compiler and PyTorch build:

```bash
python -m pytest tests/test_row_operators_gpu.py -v
python examples/row_operators.py
```

Tests use independent FP64 dense references with BF16 rounding, mixed effective
ranks 0/1/2/3/7/16/17/33/64 in a rank-128 pool, one/two/three slices, segments
up to 16 rows, nonidentity routing, empty segments, skipped/base rows, zero and
negative scales, column views, and widths 4096/14336. Shrink live values and
isolated expand deltas use a relative L2 gate of 0.005; untouched output regions
are compared exactly. A dedicated case checks the two-stage BF16 rounding.
Graph tests mutate weights, ranks, compact offsets, scales, token routes and the
active segment count, then compare replay with eager execution exactly.

The 2026-11-03 CPU suite passed 86 tests; Ruff lint/format and wheel/sdist builds
passed. Fresh dependency-free wheel and wheel-from-sdist environments each
passed 31 tests, with six dependency-requiring modules skipped. These checks
do not execute the GPU operators.

CPU-side native compilation and extension loading succeeded with GCC 13.3.0 and
CUDA 12.6.20 for both `sm_89` and `sm_90` (`TORCH_CUDA_ARCH_LIST="8.9;9.0"`,
`MAX_JOBS=2`). No GPU kernel was executed by this build check.

Pelle H100 job `7545404` is queued for the command above and the example, using
one H100, four CPU cores, 16 GiB host RAM and a 15-minute limit. The environment
is Python 3.12.3, PyTorch 2.7.1+cu126 and CUDA toolkit 12.6.20. Immediate H100
and L40S allocation attempts found no free GPUs; neither ran numerical tests.
GPU validation remains pending. No serving or token/logprob parity claim follows
from these operator tests, even after they pass.
