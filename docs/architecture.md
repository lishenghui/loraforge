# Architecture

Public alpha interfaces use the dependency-free `ExperimentalAPI` annotation.
Selected NF4 operators and Qwen3.5 integrations ship in the installed package;
their explicit opt-ins and compatibility boundaries are documented in
[experimental APIs](experimental.md). Benchmarks consume those implementations
and retain fixture preparation, measurements and rejected candidates.

LoRAForge provides LoRA compression, adapter formats, and efficient execution
components. The current API exposes `loraforge.ops.qr_factor` and
`loraforge.compression.refactor_lora` / `truncate_lora`.
`loraforge.adapters` provides local PEFT I/O, compact export and the file
compression workflow; its configuration rules live in an internal module.

The optional `integrations.vllm` module activates a pinned vLLM 0.26.0 linear
decode replacement. Framework-independent padded-slot kernels live in
`ops/vllm_gemv.py` and `ops/_vllm_gemv_kernels.py`; serving imports remain lazy.
See [the vLLM contract](vllm.md) for dispatch, ownership and validation limits.

`ops.row_shrink` and `ops.row_expand` separately provide BF16 retained-rank
tensor operators with a compact-B layout. They have no serving-framework
dependency and are not yet wired into vLLM. See [row operators](row-operators.md)
for the packed layout, device metadata and graph-lifetime contract.

## Package boundaries

| Location | Responsibility |
| --- | --- |
| `src/loraforge/compression/` | Spectral refactoring, compact truncation, budget allocation |
| `src/loraforge/adapters/` | Adapter structures, scaling, PEFT I/O, versioned manifests |
| `src/loraforge/ops/` | Tensor-level operator contracts and dispatch; Triton implementations |
| `src/loraforge/runtime/` | Compact layouts, slot ownership, loading and execution metadata |
| `src/loraforge/integrations/` | Thin, optional SGLang and vLLM adapters |
| `src/loraforge/ops/csrc/` | Native CUDA/HIP implementations and bindings, shipped for lazy compilation |
| `tests/` | Public behavior, numerics, lifecycle, and compatibility checks |
| `benchmarks/` | Reproducible performance programs with documented protocols |
| `examples/` | Small, complete user workflows |

Entrypoints and integrations depend on the library; tensor operations never
import a serving framework. Compression and runtime share operator and adapter
contracts rather than reaching into each other's internals.

## Execution contracts

FlashTSQR remains the default spectral compression backend. Reference
implementations serve correctness comparisons; they must not silently replace
the accelerated path. Return the backend actually used with operation results.

Factorization handles must own their reflectors and remain valid across later
factorizations. Device, stream, workspace lifetime, and concurrent use must have
explicit contracts before the first numerical API is published. A module-global
last-result field is insufficient for execution metadata.

Compact export stores the retained tensor dimensions. Padding is an explicit
compatibility option and cannot be counted as a storage reduction. Preserve
adapter scaling independently of the factorization and document its convention.

Core configuration belongs in typed arguments or configuration objects.
Framework registration must be repeatable within each worker process and must
validate the supported framework version. Plugin discovery does not by itself
provide a stable hook for arbitrary operator replacement.
