# Source provenance

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
