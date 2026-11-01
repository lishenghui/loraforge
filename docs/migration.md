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
