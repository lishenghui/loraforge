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

Still pending in stage 4: quality-curve objectives and heterogeneous costs. Stages 5–6 remain pending:
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
