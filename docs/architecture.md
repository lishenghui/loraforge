# Architecture

LoRAForge will provide LoRA compression, adapter formats, and efficient execution
components. The foundation currently exposes only `loraforge.__version__`.
The following boundaries guide future implementation; empty placeholder modules
are deliberately omitted until they contain a working capability.

## Package boundaries

| Location | Responsibility |
| --- | --- |
| `src/loraforge/compression/` | Spectral refactoring, compact truncation, budget allocation |
| `src/loraforge/adapters/` | Adapter structures, scaling, PEFT I/O, versioned manifests |
| `src/loraforge/ops/` | Tensor-level operator contracts and dispatch; Triton implementations |
| `src/loraforge/runtime/` | Compact layouts, slot ownership, loading and execution metadata |
| `src/loraforge/integrations/` | Thin, optional SGLang and vLLM adapters |
| `src/loraforge/cli/` | User commands built on public library interfaces |
| `csrc/` | Native CUDA/HIP implementations and bindings |
| `tests/` | Public behavior, numerics, lifecycle, and compatibility checks |
| `benchmarks/` | Reproducible performance programs with documented protocols |
| `examples/` | Small, complete user workflows |

Entrypoints and integrations depend on the library; tensor operations never
import a serving framework. Compression and runtime share operator and adapter
contracts rather than reaching into each other's internals. Start with one
Python distribution. Split native packages only when build and release needs
justify the additional maintenance.

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

## Project organization references

The [vLLM repository](https://github.com/vllm-project/vllm) separates the Python
package, native code, tests, benchmarks, examples, and documentation.
[SGLang](https://github.com/sgl-project/sglang) similarly separates its Python
runtime from examples, documentation, and benchmarks. These boundaries inform
this smaller layout; their complete build systems are not required here.
See also [vLLM's plugin contracts](https://docs.vllm.ai/en/latest/design/plugin_system/).
