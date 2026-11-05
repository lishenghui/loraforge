# Source provenance

## Qwen3.5 GDN adaptation — 2026-11-04

`integrations/_sglang_qwen35.py` is independently implemented against the
inspected SGLang 0.5.22 model/loader interfaces and the official Qwen3.5 config.
It copies no framework source. References and the exact compatibility boundary
are in [qwen35.md](qwen35.md); validation artifacts record runtime file hashes.
The four-slice RowExpand instantiation extends the existing project-owned
kernel below, retaining its author/source notices and original three-slice
shared-memory footprint. The synthetic adapters in the pilot are generated
test data, not research model outputs or imported trained adapters.

The B-only `row_tile=2/4/8` extension is new LoRAForge scheduling code around
the existing attributed RowExpand reduction. It splits each existing segment
inside B without changing A metadata, rank accumulation or BF16 rounding. The
original `row_tile=16` kernel remains the default low-level implementation.
`benchmarks/rowexpand_tiles.py` and its graph/rank/route regression cases are
new project code; no additional probe or framework implementation is copied.
MI250 validation and cache-regime timings are recorded in
[qwen35-rowtiles-mi250.json](qwen35-rowtiles-mi250.json).

The opt-in Qwen3.5 tiny-rank dispatch is new LoRAForge policy code derived
from the project's rank-4/rank-8 operator and full-model experiments. It selects
the already attributed HIP shrink, RowExpand and native implementations;
no new probe or framework kernel is imported. Its separate timing opt-in
preserves the original strict logprob result while allowing token-matched
performance measurements. See [qwen35.md](qwen35.md) for scope and results.

`benchmarks/qwen35_gdn_consumer.py` is an independently implemented research
kernel for the fixed Qwen3.5-0.8B contiguous projection boundary. It computes
BF16-rounded LoRA B corrections while producing QKV/Z/b/a for the original
GDN consumer. It copies no upstream kernel. Its functions match the measured
prototype AST; [the experiment report](qwen35-gdn.md) records GPU checks,
framework hashes, whole-model controls and remaining scope limits.

The new benchmark-only `qwen35_tiny_fused.cu` and `qwen35_wide_fused.cu` retain
the attributed `ops/csrc/rowexpand.cu` HIP shrink reduction, BF16 conversion
helpers and original probe source notice. Their shared-A, fused A+B and direct
GDN unpack scheduling are new LoRAForge experiment code; no additional upstream
kernel is imported. `qwen35_fused_lora.py` supplies explicit lazy ROCm loaders,
and `check_qwen35_wide_fusion.py` is a portable operator check against the
existing HIP/RowExpand/consumer implementations. Both native source files are
byte-identical to the GPU-study sources. Benchmark `.cu` files accompany the
source archive, alongside their Python callers; they do not become installed
runtime operators. [The follow-up report](qwen35-fused-lora.md) records numerical
checks, shape exclusions and whole-model controls.

`benchmarks/qwen35_rank4_fused.cu` specializes the same attributed HIP reduction
and BF16 conversion helpers to capacity 4 and the two measured narrow input
widths, with unrolled chunks and vector B loads. Its lazy loader and portable
30-case operator check are new experiment code. Live retained ranks and routes
remain GPU metadata, with mixed-rank and changed-metadata graph equality checks.
This source is also a benchmark prototype, not an installed library kernel.

## Foundation

Created on 2026-10-31 as new packaging, documentation, and CPU checks under
Apache-2.0. This commit imports no numerical kernels, research results,
third-party framework code, or paper content.

The migration reference is
[`lishenghui/probe` at `5fd6ae060d106066b306959f9a34b726a281ef4c`](https://github.com/lishenghui/probe/tree/5fd6ae060d106066b306959f9a34b726a281ef4c).
This is a baseline for inspection, not an assertion that its entire tree is
licensed for redistribution or has been migrated. Local uncommitted probe work
is outside this baseline and must be recorded explicitly if later selected.

## Code import ledger

### FlashTSQR tensor extraction

Source: `lishenghui/probe` commit
`5fd6ae060d106066b306959f9a34b726a281ef4c`, authored by Shenghui Li
(including the `lishenghui` Git identity), as recorded in source history.
The project-owned implementation is migrated under this repository's Apache-2.0
license, with author and source notices in the adapted files. No external
framework implementation or vendor source is copied.

| Source path under probe | Destination |
| --- | --- |
| `FlashTSQR/flashtsqr.py` | `src/loraforge/ops/flashtsqr.py`, `ops/_extension.py`, `compression/spectral.py` |
| `FlashTSQR/kernels/tsqr_full.cu` | `src/loraforge/ops/csrc/tsqr_full.cu` |
| `FlashTSQR/kernels/tsqr_wide.cu` | `src/loraforge/ops/csrc/tsqr_wide.cu` |
| `FlashTSQR/kernels/jacobi_eigh.cu` | `src/loraforge/ops/csrc/jacobi_eigh.cu` |

Original file hashes are in [flashtsqr-source-hashes.json](flashtsqr-source-hashes.json).
New bindings, result types, tests, and example code were written for this extraction.
Intentional changes:

- Replace static `g_h` with per-call reflector tensors owned by a Python handle.
  Build WY factors eagerly to avoid mutable state during concurrent application.
- Guard the input GPU and launch on its current PyTorch stream; record completion
  and consumer-stream lifetimes. Preserve input-device semantics.
- Return execution metadata instead of global `last_path` / `last_eigh` values.
- Disable ambient autocast, validate shapes/dtypes, and document inference-only
  behavior. Explicitly reject graph capture.
- Return physically compact cloned factors by default; legacy padding is opt-in.
  Support 2D and batched 3D factors through one interface.
- Zero numerical null eigenvalues at a documented FP64 relative threshold and
  check Jacobi eigen-residuals instead of silently accepting nonconvergence.
- Package one copy of native sources for lazy compilation; remove cluster-specific
  compiler mutations. Keep the existing Householder method and backend selection.

The opt-in experimental cuSOLVER batched EVD wrapper is not migrated. CUDA uses
the original default `torch.linalg.eigh` core. No production generic QR path or
unreported fallback is introduced. Source repositories and paper content are unchanged.
Validation scope and commands are in [flashtsqr.md](flashtsqr.md).

### PEFT file workflow

The PEFT file layer (`src/loraforge/adapters/`), its example and reload tests
were written for this repository on 2026-11-01. They implement the documented
PEFT 0.18.0 file/scaling contract without copying PEFT implementation code or
importing experiment scripts. References and the supported surface are recorded
in [peft.md](peft.md). Actual compression reuses the maintained FlashTSQR tensor
API at baseline `15734fdd7e983b2e2a97bda1861115fb600f9423`; no native kernels
or numerical algorithms change in this file-workflow step.

The rsLoRA extension at baseline `3082b452dd59d9749ac659ee4b76dcd8b6c47b42`
adds PEFT's `alpha/sqrt(r)` convention to the file layer, preserves the flag on
export, and records the scaling type in each manifest module. It is new project
code based on the same referenced PEFT contract; no kernels or upstream source
files are copied or changed.

### Future entries

Each import must add:

- Source repository URL, immutable commit, and source paths.
- Destination paths and retained authorship/copyright notices.
- Applicable license and any required third-party notices.
- Intentional API, algorithm, layout, and numerical changes.
- Verification commands, hardware/software environment, and results.

A new package license does not replace the terms of imported source files.

### Spectral budget planner (2026-11-03)

New dependency-free implementation in `compression/allocation.py`, following
Shenghui Li's global singular-value ordering in probe commit
`b217b17fac6cd6105cd3f04865962fb51d8a8ba3`,
`vla_fleet/compress_oft_gpu.py` lines 50–59. No source code is copied.
The source file was inspected locally; its uncommitted experiments are not imported.
The API accepts an integer direction budget and sorted spectra, validates inputs,
and resolves ties by module name and index. It does not perform compression.
CPU tests exhaustively compare all budgets in a small instance with every feasible
prefix-rank allocation. No GPU algorithm or hardware support changes.

The budget-to-PEFT extension adds a minimum retained rank (default zero in the
standalone planner, one in the file workflow), an explicit `budget` keyword,
and a two-pass FlashTSQR spectrum/compression workflow. This is new library
orchestration; no additional source files or numerical kernels are imported.
The objective deliberately uses unscaled spectra as in the recorded research
source. The export manifest records this convention and the selected ranks.

### Quality-curve budget planning (2026-11-03)

New implementation: `compression/quality.py`, with CPU exhaustive-search tests.
Conceptual reference: Shenghui Li's `tools/llm_fleet/allocate_qwen_tasks.py`,
function `dp`, at local probe commit
`b217b17fac6cd6105cd3f04865962fb51d8a8ba3`. No source code is copied.
The library uses exact integer costs and sparse dominated-state pruning rather
than the experiment's rounded-up cost grid. Calibration, isotonic fitting,
minimax objectives, experiment results and framework dependencies are not
imported. Losses and costs are explicit caller data; tie-breaking is documented.

### Compact CPU slot ownership (2026-11-03)

`runtime/slots.py` and its tests are new project code. No research loader or
framework implementation is copied. Design inspection included local probe
`tools/llm_fleet/sglang_plugin/probe_sglang_gemv/fast.py` and the untracked
`tools/llm_fleet/test_paging_loader.py`; these remain research evidence, not
imported source. The inspected checkout HEAD is
`b217b17fac6cd6105cd3f04865962fb51d8a8ba3`, but `fast.py` has local edits and the
test file is untracked, so that commit does not identify their working content.

The new component deliberately implements synchronous CPU snapshots and explicit
pool/generation ownership. It reuses LoRAForge's maintained PEFT validator and
preserves scaling. It does not claim equivalence to the research GPU loader,
persistent shadows, fused projections or graph replay. No numerical kernel changes.

### GPU residency and scoped leases (2026-11-03)

`runtime/gpu.py` and `tests/test_runtime_gpu.py` are new library code, building
on the CPU slot contract from commit `4f15389`. No research or third-party code
is copied and no numerical kernel changes. The implementation uses pinned host
staging, per-load completion events, explicit read leases and PyTorch stream
recording. Reload publishes freshly allocated compact factors rather than
mutating old addresses. GPU graph replay and SGLang/vLLM registration remain
outside the supported scope. Validation results are recorded in runtime docs.

### Decode GEMV and SGLang integration (2026-11-03)

Imported mathematical kernels: `tools/llm_fleet/sglang_plugin/probe_sglang_gemv/kernels.py`
from probe commit `b217b17fac6cd6105cd3f04865962fb51d8a8ba3`, authored by Shenghui Li
(file history commit `5ab76d171374d313ec7594dc7c24acfd5ccce796`). The working file is
unchanged from this pinned revision. Source checksum is recorded in
[gemv-source-hashes.json](gemv-source-hashes.json). The project-owned implementation
is migrated under Apache-2.0 with retained authorship/source notices.

Destination: `ops/_gemv_kernels.py` contains the three extracted shrink,
reduce and expand kernels. `ops/gemv.py` supplies adapted lazy public wrappers.
`integrations/sglang.py` is new orchestration implementing the source backend's
decode/native-prefill boundary through SGLang's installed public method contract.
No upstream SGLang implementation or source experiment/result trees are copied.

Intentional differences from the source: remove import-time Torch/Triton loading
from public modules, remove environment-driven tuning/global rank buckets, use
one explicit flat split-K schedule, initialize shrink output to zero, validate
shape/dtype/device contracts, and reject autograd. Kernel math, actual-rank A
slice packing, max-rank B stride, FP32 accumulation and intermediate/delta rounding
are retained. Compile specialization decorators and symbol names are simplified.
Native SGLang buffers and loader/scheduler own fixed addresses and lifetimes;
LoRAForge's independent runtime pools are not substituted into the framework.

The inspected LUMI SGLang checkout reports version 0.5.22 and source commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`. The integration is version-gated and
conflict-aware, opt-in per worker, and retains native eager prefill/pruned paths.
Validation scope and commands live in [sglang.md](sglang.md).

### BF16 decode numerical correction

The original project-owned split-K/tree GEMV reduction failed full-model BF16
parity despite passing operator tolerances. LoRAForge now implements separate
BF16 routed-token matrix-dot kernels: shrink uses sequential K=256 blocks and
16 output columns; expand uses sequential rank blocks of 16 and 64 output
columns. Each uses FP32 accumulation and BF16 intermediate/delta rounding.
These new kernels were implemented here against the observed SGLang 0.5.22
default reduction contract, retaining native slot layout but no framework
dependency. FP32/FP16 retain the extracted GEMV kernels. This is a correctness
compatibility path; performance equivalence and tuned SGLang configurations
are not claimed. Hardware/full-model results are tracked in sglang.md.

### Probe rank-aware policy extraction (2026-11-03)

`integrations/_sglang_rank_aware.py` extracts the non-split segmented shrink
kernel from public probe commit `5fd6ae060d106066b306959f9a34b726a281ef4c`,
`tools/llm_fleet/sglang_plugin/probe_sglang_gemv/rank_aware_shrink.py`.
The source identifies itself as derived from SGLang 0.5.22's
`sglang/kernels/ops/lora/dense/chunked_sgmv_shrink.py` (Apache-2.0), with Shenghui
Li's actual-rank column-block early exit. This extraction retains that
attribution and algorithm; it renames the kernel, omits the custom cache decorator
and split-K/prefill tuning, and adds a narrow wrapper using installed native
launch configuration. Native expand remains native. Explicit native split-K
settings delegate back to native shrink. No Qwen-specific table, global config
patch or Fleet adapter shadow pool is installed. The previous statement that
no upstream implementation was copied describes the original GEMV extraction;
this separately attributed extraction does include a derived SGLang kernel.

Pinned probe source SHA-256: `0af480cc8ead0a7c40292dee1d20551232a6228c3a189758051f9c92bd56a92a`.

### Isolated Fleet gate/up expansion experiment

The benchmark-only `sglang_gate_expand_experiment.py` follows the compact B
layout and device metadata design in public probe commit
`5fd6ae060d106066b306959f9a34b726a281ef4c`,
`tools/llm_fleet/sglang_plugin/probe_sglang_gemv/fast.py` and `fast_decode.cu`.
Its Python wrapper was implemented here with fixed per-slot regions, only
for gate/up, refreshed from native B buffers after native loading. It does
not import the per-checkpoint manifest cache, Qwen tuning tables or prefill
replacement. The explicitly supplied compiled HIP library remains external;
its hash is recorded by the service pilot. No native source is copied into
the installed package by this experiment. GPU/serving validation is recorded
separately in the SGLang guide.

### Complete Fleet / RowExpand extraction

The opt-in `fleet` SGLang policy extracts the measured retained-rank path from
probe commit `5fd6ae060d106066b306959f9a34b726a281ef4c`, under
`tools/llm_fleet/sglang_plugin/probe_sglang_gemv/`. Attribution: Shenghui Li,
Apache-2.0. The source ledger is:

| Source | SHA-256 | Destination |
| --- | --- | --- |
| `fast_decode.cu` | `e26ca56e16945362cb9c96a2be6f01de78920e42eee63908161deeb226a87260` | `ops/csrc/rowexpand.cu` |
| `fast.py` | `bd6a953ac82782dcdfafb05e4c4351ccc7f32109e0bf4cc19599b78312dfb1bc` | `runtime/retained.py`, `integrations/_sglang_fleet.py` |
| `mi250x_qwen50_fleet_dispatch.json` | `245318e64963720c2ba13034598a1aa815c7320aee52438519082b67bb5b63da` | `integrations/data/mi250x_fleet_dispatch.json` (byte-identical) |
| `stable_prefill.py` | `5cb43562ab9e53648ad024b31ed7bc9f8d68c2ce6cbb40b3b866f0b507e8d6a0` | `integrations/_sglang_stable_prefill.py` |

Native arithmetic and the frozen profile are retained: table-selected decode A,
all supported decode B ranks, prefill B only through rank 16, rank-aware native
fallback elsewhere. The B fields in the historical tuning table do not override
this measured profile. The native padded pool remains resident; compact B
regions belong to known checkpoints, with per-slot device ranks and offsets.
Fused A slices are repacked after native loading using effective module ranks.

Intentional maintenance changes: package and lazily compile the extension;
separate runtime layout from framework integration; use weak pointer registries
and pool-owned entries; make the adapter catalog explicit; validate padded
headers and nonzero tails before execution; reject unknown reloads before native
mutation; add same-device/shape/dtype/stride guards and autograd rejection;
separate launch statements for PyTorch HIPify parsing; replace probe environment
names and remove implicit stable-prefill enablement. Qwen stable prefill remains
an explicit, narrowly guarded hook; it is not enabled for Llama or by default.
The frozen dispatch is currently gated to gfx90a. This does not claim CUDA Fleet
validation, mixed-attention extensions, speculative/MoE/TP support or performance
portability. Other exploratory probe backends are not part of this measured path.

`adapters/sglang.py` is new integration glue: it explicitly pads compact PEFT
exports, folds per-module scaling into BF16 B and publishes true module ranks.
This is a recorded format/scaling conversion, not another compression algorithm.
The conversion can introduce BF16 rounding and requires corresponding validation.

The Qwen3.5 bounded-grid experiment used SGLang 0.5.22's sorted adapter groups
and fixed-size chunk contract. Its capacity bound and tests were new LoRAForge
code; no new upstream kernel was copied. Although numerically unchanged, it
regressed full-model service throughput and was reverted. It is recorded as
a rejected experiment in the Qwen3.5 guide, not an enabled optimization.

### Qwen3.5 prefill and metadata reuse prototypes

`benchmarks/qwen35_prefill_fused.py`, `qwen35_decode_silu.py` and
`qwen35_metadata_cache.py` are new
LoRAForge research implementations. They use SGLang 0.5.22's native sorted
16-token segment/pool contracts and the attributed Fleet retained-rank layout;
no additional upstream kernel is copied. The two-dot prefill implementation
keeps BF16 intermediates; B-at-SiLU preserves the tested native BF16 activation
rounding. The cache reuses only consecutive static decode metadata, scoped to
Fleet TP1/no-overlap workers and non-DP execution. These benchmark helpers do
not replace the installed backend. Their check programs and source comparison
receipts are documented in `docs/qwen35-high-concurrency.md`.


### Qwen3.5 cold-prefill graph experiment

`benchmarks/qwen35_cold_prefill_metadata.py` and
`qwen35_cold_prefill_graph.py` are new process-local benchmark glue against
SGLang 0.5.22's `ForwardMetadata`, prefill graph runner and GDN eager metadata
contracts. They add plain EXTEND preparation without copying upstream GPU
kernels. Cold 8x128 queries use fixed independent metadata buffers; physical
GDN state-slot translation remains native. Native convolution, recurrent,
attention, KV writes and LoRA computations retain their order. The capture
fixture must also be 8x128, because the default 1024+empty-requests fixture
captures different convolution/chunk grids. These helpers are explicit
benchmark opt-ins, not an installed-library runtime change.


### NF4 base with BF16 LoRA benchmark

`prepare_qwen35_packed.py` and `prepare_qwen35_nf4.py` generate a canonical
Qwen3.5-0.8B fixture using native BF16 row concatenation and bitsandbytes0.50.2
NF4 block64/double-quantization operations. No quantization kernel is copied. The direct `check_qwen35_nf4.py` program
uses those same library operations with the measured projection geometries.
`qwen35_nf4_compat.py` is version-scoped process-local compatibility glue: it
extends one SGLang0.5.22 ROCm method list while preserving the remaining
verification, supplies the already-fused name map and separates the BF16
adapter file loader configuration from the quantized base loader. It changes
no installed library default or shared SGLang file.

`qwen35_packed_loader.py` handles canonical fused checkpoint names directly
through native full-weight loading and verifies114 actual projection tensors
against their checkpoint values. This avoids upstream substring matching of
`qkv_proj` as `v_proj` and `gate_up_proj` as `up_proj`, which silently skipped
those parameters in rejected attempts. Other weights retain upstream loading.
Numerical validation rejects null/nonfinite logprobs before any timing.
See `docs/qwen35-nf4.md` for scope, rejected evidence and measured results.
