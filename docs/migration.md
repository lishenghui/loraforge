# Migration from probe

Move capabilities in small, reviewable PRs from a pinned source revision. Keep
research runs and paper reproduction in probe. Once a module is maintained here,
future fixes originate here and probe consumes a pinned LoRAForge revision.
Avoid maintaining two independently edited copies.

| Step | Deliverable | Acceptance |
| --- | --- | --- |
| 1 | Package foundation, contribution guide, provenance, CPU CI | Wheel and source archive install; import works without GPU dependencies |
| 2 | FlashTSQR and tensor compression API | Independent handles; reconstruction/truncation correctness; interleaved calls, device and stream checks on target GPUs |
| 3 | PEFT I/O, compact export, manifest | Save/reload preserves scaling and results; actual stored sizes are recorded |
| 4 | Budget allocation | Separate spectral and quality objectives; deterministic plans stay within budget |
| 5 | Serving operators, runtime, SGLang integration | Route changes, slot reuse, graph replay, and end-to-end numerical validation |
| 6 | vLLM integration and broader hardware support | Independently tested framework/hardware combinations |

Step 3's linear LoRA/rsLoRA file workflow is described in [peft.md](peft.md).
It exports positive ranks, preserves per-module scaling, and records a versioned
manifest; broader PEFT variants remain outside the initial supported surface.

## First code extraction

Start with `FlashTSQR/flashtsqr.py` and the kernels it actually invokes, rather
than copying the FlashTSQR experiment tree. Replace the global factorization
state with owned handles, preserve input-device semantics, and return execution
metadata with results. Keep the mathematical method and performance changes
separately reviewable. The initial extraction provides owned handles,
current-stream execution, tensor refactoring and compact truncation; see
[the API contracts](flashtsqr.md) and [the import ledger](provenance.md).

## Import procedure

For every extraction:

1. Pin the source repository, commit, paths, authorship and applicable licenses.
2. Record the target paths and intentional behavior changes in `provenance.md`.
3. Include the original-to-new comparison or reference in the PR; preserve
   attribution in source files and add required notices.
4. Validate the capability in isolation before attaching a framework or CLI.
5. Document the supported environment and update probe to consume the library
   only after the migrated implementation is verified.

Date-named experiment directories, private datasets, model weights, absolute
cluster paths, job launchers, paper submodules, and generated result trees stay
in their research repositories. Benchmarks use portable inputs and publish
hardware, versions, warmup, timing scope, tolerances, and limitations.

## Local audit and progress — 2026-11-03

Stages 1–3 have implementations and recorded validation. Stage 4 now includes
`loraforge.compression.allocation.allocate_spectral_budget(spectra, budget)`:
a CPU-only, deterministic equal-cost direction planner maximizing retained
spectral energy. For example, `{"a": [6, 2], "b": [5, 1]}` with budget 2 yields
`{"a": 1, "b": 1}`. The planner supports a minimum rank constraint. The PEFT workflow now accepts
`budget=` and reserves one direction per module, measures spectra through
FlashTSQR, and records the chosen ranks in the export manifest.

Stage 4 also provides `allocate_quality_budget` with explicit candidate ranks,
integer costs and calibration losses. It minimizes additive loss using exact
sparse multiple-choice knapsack, with deterministic ties and infeasible-budget
checks. See [allocation contracts](allocation.md). Automatic calibration,
minimax objectives and large-frontier approximations remain outside this API.
Stages 5–6 remain pending:
serving kernels, compact runtime/loading and slot lifecycle, SGLang, vLLM,
graph replay and framework/hardware validation. Experimental cuSOLVER batched
EVD is also intentionally absent. Research results, launchers and paper assets
are excluded from migration rather than counted as missing library features.

Validation after file-workflow integration: 49 CPU tests passed in the existing
PEFT environment, including real PEFT reload regressions and infeasible-budget
rejection. Ruff lint/format checks passed. Wheel and sdist built, and the wheel
was installed in a separate environment with a successful public planner check.
LUMI GPU job `23423930` completed successfully: all 6 tests passed, including
ordinary-LoRA/rsLoRA budget roundtrips and four existing file-compression
regressions. Hardware: one MI250X GCD (`gfx90a`), Python 3.12.11,
PyTorch 2.7.1+rocm6.2.4. Command: `python -m pytest
tests/test_peft_adapters.py -m gpu -v`. Slurm elapsed time was 2:01; pytest
time was 67.54 seconds. NVIDIA CUDA validation remains pending.

The local probe revision is `b217b17fac6cd6105cd3f04865962fb51d8a8ba3`;
LUMI reports `419256022d76c68d482b14323b350678feee0969`. These are distinct
snapshots and no remote code was imported. GitHub SSH authentication failed, but HTTPS read access succeeded:
LoRAForge HEAD matches the local baseline
`5fa6242dd40116b7218bda130eb2f2150ea42a12`; probe GitHub HEAD remains the original
migration baseline `5fd6ae060d106066b306959f9a34b726a281ef4c`.
The planner's later local source revision is recorded explicitly in provenance.

Quality-planner validation: 59 CPU tests passed after this extension, including
exhaustive enumeration of feasible choices, nonmonotonic losses, zero costs,
infeasible budgets, deterministic ordering and existing adapter regressions.
Wheel/sdist builds and an isolated installed-wheel example passed. No GPU
computation or numerical backend changed in this extension.

## Runtime ownership extraction — 2026-11-03

Stage 5 now has a working synchronous CPU component: `runtime.AdapterSlots`
loads validated PEFT factors into compact owned storage, enforces an optional
resident tensor-byte limit, and uses pool/generation handles to reject stale
references after slot reuse. Failed loads preserve the old occupant. Resolve
returns independent copies and eviction releases residency. See [runtime](runtime.md).
GPU layouts/transfers, async lifetime, serving operators, SGLang/vLLM integration
and graph replay remain pending; this component makes no GPU execution claim.

Validation: 75 CPU tests passed, including 16 slot tests covering compact PEFT
export loading, LoRA/rsLoRA scaling, stale/foreign handles, reload integrity
failures, byte-limit failures, detached ownership, invalid indices and concurrent
independent slots. Ruff checks, wheel/sdist builds, and a separate installed-wheel
runtime import without PyTorch passed. This step uses no GPU allocation.

## Compact GPU residency and leases — 2026-11-03

`runtime.GPUAdapterSlots` adds pinned staging, nonblocking compact H2D loading,
ready-event waits across streams and read leases that block active-slot reuse.
Consumer stream recording protects queued operations after eviction. Replacement
publishes new allocations and generations rather than overwriting old buffers.
See [runtime contracts](runtime.md) for residency accounting and caller duties.

Local validation: 78 CPU tests passed; lint/format checks, wheel and source
archive builds, wheel-from-source rebuild and an isolated installed-wheel import
without PyTorch passed. LUMI single-GCD job `23424086` completed successfully:
7 GPU cases passed across FP32/FP16/BF16 and LoRA/rsLoRA on MI250X (`gfx90a`),
Python 3.12.11, PyTorch 2.7.1+rocm6.2.4. Pytest took 16.35 seconds; Slurm
elapsed time was 57 seconds. Command: `python -m pytest tests/test_runtime_gpu.py
-m gpu -v`. Checks cover cross-stream values, lease protection, slot replacement,
eviction/lifetime, byte-limit rejection and exception release.
Fixed-address pools, graph replay, serving kernels, SGLang/vLLM and NVIDIA CUDA
validation remain pending.

## Decode operators and SGLang bridge — 2026-11-03

Stage 5 now has extracted split-K GEMV shrink/reduce/expand operators and an
opt-in SGLang 0.5.22 decode backend. It consumes native fixed-address padded
buffers and device rank/scale/route metadata; it does not replace SGLang's
scheduler or connect a competing pool. Eager prefill/pruned paths remain native.
Source checksums, authorship and intentional changes are in provenance.

Validation includes CPU registration/version/conflict contracts and isolated
installed-wheel plugin discovery without framework imports. LUMI job `23425475`
completed successfully: all 36 GPU cases passed, including raw numerics,
fixed-address graph replay with updated occupants/ranks/routes, real SGLang
batch preparation, dense projection methods and inherited native eager prefill.
The environment was MI250X (gfx90a), Python 3.12.3, SGLang 0.5.22 at the recorded
source revision, PyTorch 2.11.0+rocm7.2, Triton 3.6.0. Pytest took 49.11 seconds;
Slurm elapsed time was 61 seconds. Local CPU validation passed 85 tests.
Wheel/sdist and wheel-from-sdist builds, isolated-wheel lazy imports/plugin
metadata and fresh-process native SGLang plugin discovery passed.
The later full-model validation below covers native paging and loading in a
limited configuration. Compact serving storage, prefill graphs, TP/MoE and
performance benchmarking remain separate work.

### Full-model protocol correction

Job `23426358` reached real Llama-3.1-8B inference and passed two cycles of ten
single-request cases. It stopped in native eager before custom execution because
the original single-versus-two-request-batch assertion observed a 0.0572664
logprob difference, exceeding 0.05. This is retained as a failed control run.
Protocol version 2 reports cross-shape drift diagnostically, checks repeatability
within single and mixed shapes, and compares backends at matching shapes with
the unchanged 0.05 threshold. Regression checks retain rejection of above-limit
matched-backend differences and reject stale protocols/nonfinite values. The
updated CPU suite passed 91 checks. Rerun `23430057` uses a separate directory
and the same cached inputs on one small-g GCD (40GB host memory, 20 minutes).
Rerun `23430057` completed all four modes with real rank-changing slot reuse,
stable addresses and eager/graph execution. Final matched-shape comparison
failed: token IDs all matched, but native/custom logprob drift reached
0.08525431156158447 > 0.05. Native eager/graph, custom eager/graph and base-only
comparisons were exact. Isolated diagnostic job `23430621` completed: native-A/custom-B and
custom-A/native-B still differed by 0.0617663 and 0.0992774 respectively, with
identical generated tokens. A saved real shrink input demonstrates an FP32
reduction-order difference crossing a BF16 rounding midpoint; native matches
the rounded FP64 reference for that element. Replacing either operation alone
was insufficient. BF16 now uses sequential matrix-dot reductions and explicit
matrix-shaped store pointers. LUMI job `23431819` passed all 39 GPU checks and
all four full-model service modes. The final CPU comparison passed with exact
token IDs and maximum logprob difference **0.0** at matching batch shapes.
Real rank-changing slot reuse, fixed addresses and eager/graph custom execution
were observed. Scope and limitations are recorded in [sglang.md](sglang.md).

## NVIDIA CUDA validation — 2026-11-03

Fixed the wide FlashTSQR kernels to check the actual per-block device shared
memory limit and explicitly opt in to larger CUDA dynamic scratch. Full-path
opt-in errors are now checked as well. The new regression reproduced an H100
invalid-argument launch before the fix and passes afterward; excessive scratch
is rejected without invalidating the factorization handle.

Pelle Slurm job `7542015` on one H100 NVL completed with 123 passed, 1 skipped
(two visible GPUs required), followed by the tensor compression example.
This validates the rank-160 full/WY path, compact tensor/file compression and
GPU residency on this NVIDIA environment. See [the NVIDIA guide](nvidia.md)
for versions, commands, precision settings and remaining validation limits.
Lint/format, CPU tests and isolated wheel/source-archive checks passed.
Serving integration, graph replay and other hardware combinations remain pending.

Pelle job `7542157` subsequently validated two L40S GPUs: 126 tests passed and
one rank-160 full/WY case skipped because L40S shared memory does not fit its
policy. Full/wide two-device QR guards, second-GPU LoRA reconstruction and
per-device GPU residency tests all executed and passed; the compression example
also succeeded. See [the NVIDIA guide](nvidia.md) for commands and scope.
Dual-H100 and distributed execution remain unverified.

## Isolated SGLang speed candidates

The HIP-A/native-B service pilot (23438767) measured 1.1101x–1.1185x
end-to-end speedup on one MI250X GCD. Selective gate/up HIP-B follow-up
23439944 completed in 15m27s and measured 1.1169x–1.1233x. Cross-job
incremental latency savings were below 1%, while gate shadows added 224 MiB.
With a uniformly increased 256-token quality allowance, the 32-case medical
pilot scored 15/32 native versus 14/32 candidate; one candidate answer looped
until truncation. The small sample and changed allowance do not isolate the
effect of the B optimization. These remain explicit benchmark experiments,
not enabled library policies or quality certifications. Full measurements,
numerical differences and limitations are in [sglang.md](sglang.md).

## Complete retained-rank path extraction

The optional `fleet` SGLang policy now packages RowExpand and direct shrink,
per-module rank/A packing, checkpoint-compact B regions, the frozen MI250X
decode dispatch, low-rank prefill, unsupported-rank fallback, loader updates,
and graph metadata. The optional Qwen stable-prefill hook and a compact-PEFT to
SGLang export bridge are also extracted. See [RowExpand](rowexpand.md) for the
component checklist and limits. Job 23440720 passed 33 GPU checks, including
21 exact old/new kernel comparisons. The earlier partial-path speedups do not
validate this packaged path; its separate full-model result is recorded below.

Full-model job 23440815 exposed a fused-projection keyword interface mismatch
after native completed. QKV and gate/up now preserve SGLang's parameter names;
integration tests exercise public keyword calls rather than private helpers.
Retry 23441471 passed all 33 GPU regressions and completed both full-model modes
in 15m58s. CPU analysis verified trace IDs and custom prefill/decode execution.
The original-adapter batch-1/2 pilot measured 1.1126x–1.1251x versus native;
medical quality was 15/32 native versus 14/32 Fleet, with one repeated-output
truncation loss. Functional execution is verified in this scope, not native
parity or quality equivalence. The failed attempt is retained separately.

The compressed Qwen3 workload is now exercised in jobs 23451046 and 23452454:
50 real retained-rank adapters, 144 layouts, 150 exact single-request eager/graph
checks and three mixed batches. The repeated single-request records match
exactly; 23451046 also matches the archived probe records. Fixed-work serving
measured 1.1685x at offered rate 16 and 1.2012x–1.2207x at rate 32 against the
same compressed native reference. That native job ended in cleanup `TIMEOUT`
after its complete checks and timing points were saved; it is not reported as a
clean completed allocation. These single-seed points do not certify task quality.

An experimental eager-prefill empty-rank dispatch passed 45 GPU checks and exact
full-model comparisons in job 23452964, but measured 0.9922x the frozen control.
It remains disabled by default. The matched base-only job 23453345 and three
sampled GPU windows now quantify the remaining headroom; 1.3x has not been
demonstrated. See [the measurements and limitations](rowexpand.md).

## Qwen3.5 hybrid extension — 2026-11-04

The opt-in bridge now packs HF qkv+z and b+a adapter factors for dense Qwen3.5
text models, and Fleet supports four-slice retained layouts and RowExpand.
This is a new model adaptation; no probe research source or shared SGLang
installation was modified. The conservative hybrid dispatch retains native
shrink and prefill reductions after the first fast-policy pilot failed its
declared numerical gate.

Job 23457660 completed with 85 GPU checks and a full-model synthetic-adapter
pilot: 20/20 native/Fleet sequences matched, max logprob difference 0.0159646.
Matched median speedups were 1.032x/1.034x/1.009x at concurrency 1/8/32.
These are modest pilot gains, not a trained-adapter quality claim or general
hybrid certification. [Architecture, reproduction and limits](qwen35.md) and
[both accepted and rejected measurements](qwen35-mi250.json) are retained.

The follow-up numerical protocol includes a repeated mixed batch of 32 (52
cases total). The conservative implementation passes this stronger gate in job
23461243 with the same 0.0159646 maximum logprob difference. Profiling identifies
native A and the smaller RowExpand advantage at large batches as the main
remaining costs. Empty-grid bounds regressed service throughput; four selective
HIP-A profiles failed the unchanged model gate, including MLP down alone.
The B-only change was near the control and was not adopted. Source hashes,
same-card timings and rejected attempts are recorded in
[the optimization evidence](qwen35-optimization-mi250.json). Those investigations
did not establish an additional speedup.

The B-only row-subdivision follow-up passed 93 GPU checks and 32 operator cases
with bitwise agreement across every custom variant (23463508). It changes work
ownership, preserving each token's rank reduction and native A batching.
Model attempt 23464050 failed in the experiment's reference-path reader after
passing the model comparisons; the corrected same-card repeat **23464674**
completed in 13m07s. Both Fleet policies match native token IDs in 52/52 cases
(maximum logprob difference 0.0159646), and new/old Fleet token-logprob records
are exact. The shape/batch-bounded B profile improves old-Fleet throughput by
0.61%/0.63%/1.53% at C1/8/32. Against native the ratios are
1.0173x/1.0291x/0.9954x: high concurrency remains near parity. These are small,
single-run synthetic-pilot gains, not 1.3x or trained-adapter quality equivalence.
The implementation, limits and raw evidence are in [qwen35.md](qwen35.md) and
[qwen35-rowtiles-mi250.json](qwen35-rowtiles-mi250.json). Current CPU checks total
129; the original dense dispatch and hybrid native-A/prefill reductions remain.

The Qwen3.5 harness also prepares compact fixed rank-4/rank-8 synthetic fixtures.
Both engines now read the prepared pool capacity instead of using a hardcoded
rank-64 pool, and the CPU comparison checks matching server options. These
fixtures isolate small-rank execution; they do not validate trained compression
quality. See [the tiny-rank protocol](qwen35.md).

The compact full-model jobs **23465767** (rank 4) and **23466166** (rank 8)
completed with 52/52 matching native/Fleet token sequences each. Rank 4's saved
logprobs are exact; rank 8's maximum difference is 0.0351560, below 0.05.
Matching pool capacity, all 456 positive fused entries, fixed addresses and
decode graphs were audited. Fleet/native ratios at C1/8/32 are
**1.0191x/0.9992x/0.9855x** for rank 4 and **1.0001x/0.9831x/0.9710x** for rank 8.
Tiny ranks alone do not recover the paper's speedup under the conservative
hybrid policy. Runtime code is unchanged; native A/prefill remain enabled.
See [the evidence and limits](qwen35-tiny-rank-mi250.json).

The subsequent tiny-rank profile is a separate opt-in,
`LORAFORGE_SGLANG_QWEN35_TINY=1`, bounded to the measured Qwen3.5-0.8B shapes,
pool capacities 4/8 and batch ranges. It uses HIP A in decode, native narrow B,
and shape-selected prefill B; existing kernels are unchanged. Default hybrid
reductions remain conservative. The benchmark's explicit
`--allow-logprob-drift` permits token-matched speed studies without changing
the meaning of the strict numerical result.

Job **23469847** completed rank-4/rank-8 native/fast pairs on one MI250X GCD.
Fast/native ratios at C1/8/32 are **1.1428x/1.0848x/1.0586x** and
**1.1528x/1.0824x/1.0625x**, including base-model computation. Each rank keeps
52/52 token sequences, but maximum logprob differences **0.1187558/0.0909806**
exceed 0.05; these are reported separately from the authorized timing checks.
All 456 positive fused entries, graphs, fixed addresses, HIP A and prefill/decode
RowExpand were audited. CPU checks total **133**. This is a synthetic pilot,
not a trained-quality or 1.3x claim; see [results and limits](qwen35.md) and
[the recorded evidence](qwen35-tiny-speed-mi250.json).

The later [tiny-LoRA fusion experiments](qwen35-fused-lora.md) add standalone
benchmark kernels and curated evidence, without changing the installed tiny
profile. Shared A has little rank-4 latency benefit. Narrow A+B, wide A+B and
direct GDN A+B+unpack retain intermediate BF16 rounding and are checked on
MI250X, including null routes and graph metadata updates. Selected rank-4
full-model fusion reaches 1.2993x/1.2093x/1.1175x matched native throughput at
C1/8/32; before/after controls drift by 2.61%/1.30%/0.79%. All 52 native token
sequences match, with zero new saved logprob difference versus the preceding
combined prototype; the existing strict native logprob failure remains.
These research hooks are not installed by library flags. Benchmark native
sources now accompany the sdist so its Python operator checks can compile them.

The capacity-4 narrow specialization then fixes K/tile/layout constants,
unrolls the original input accumulation and vectorizes B reads. Job **23474915**
passes 30 exact operator/metadata-changing graph cases and completes matched
control-before/specialized/native/control-after serving. Ratios are
**1.2975x/1.2503x/1.1345x** native at C1/8/32, with **1.17–3.21%** incremental
gain over the two fused controls, 52/52 token sequences and zero new saved
logprob difference. These remain benchmark prototypes; the native strict gate
and broader quality/hardware validation remain outside the measured speed claim.

The [high-concurrency follow-up](qwen35-high-concurrency.md) adds standalone
capacity-four prefill A+B and B-at-SiLU prototypes plus a consecutive-decode
metadata cache candidate. The first MI250 screen rejects grouped decode
gate/up fusion and narrow prefill fusion, retaining only measured wide
128/1024-token candidates. CPU checks total134, including14 cache invalidation
transitions. Cache-only passes64 exact model sequences. The initial prefill model attempt
keeps64 token sequences but stops at its incremental logprob gate0.086396;
its timings are not reported. A per-row decode consumer subsequently passes
32 exact eager/changed-metadata graph cases and improves the operator segment
1.39–1.73x at batch32. Matched C32 whole-model timing is pending; none of those
operator ratios is an installed service speedup claim.
