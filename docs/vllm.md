# vLLM linear decode integration

This experimental opt-in backend targets **vLLM 0.26.0** and its padded
`PunicaWrapperGPU` slot layout. It extracts the project-owned GEMV path from
`probe` at `5fd6ae060d106066b306959f9a34b726a281ef4c`. It is independent of the
SGLang integration and does not import its kernels or pending BF16 corrections.

## Activation and dispatch

Install into a separate vLLM environment, since vLLM pins its PyTorch and Triton
dependencies. The compression environment need not be upgraded:

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -e ".[vllm,dev]"
LORAFORGE_VLLM=1 VLLM_PLUGINS=loraforge vllm serve /path/to/model \
  --enable-lora --specialize-active-lora \
  --lora-modules adapter=/path/to/adapter
```

Alternatively call `loraforge.integrations.vllm.install_vllm_backend()` in
each worker before constructing model layers. The installed general plugin
is inert without `LORAFORGE_VLLM=1`. If other general plugins are needed, list
them explicitly in `VLLM_PLUGINS`. Registration rejects other vLLM versions,
foreign replacements and changed thresholds after activation; repeating the
same registration is safe. Public imports do not load torch, Triton or vLLM.

vLLM 0.26.0 has no dedicated registration hook for this linear method. This
integration replaces only `PunicaWrapperGPU.add_lora_linear` per worker,
retaining the original method for prefill, unspecialized execution and extra
keyword contracts. Other methods, including embedding, logits and MoE, remain
native. Only ranks 1..256 use the custom path.

Inside a mutating opaque PyTorch custom op, batches of at most four tokens per
native active-LoRA specialization use GEMV; larger batches call native vLLM
shrink/expand. `install_vllm_backend(tokens_per_lora=...)` can explicitly set
that threshold before activation. Native metadata's rounded active count is
used, which may include the base-model route; it is not an exact adapter count.
This follows the reference heuristic, not a new measured optimal schedule.

## Layout, numerical and graph contracts

The kernels use padded A `[slots, rank, width]`, B `[slots, output, rank]`,
or their vLLM `[slots, 1, ..., ...]` forms. Routes are GPU int32 slot indices,
with -1 denoting no adapter. Fused slices are separate tensors. The intermediate
is FP32 `[slices, tokens, rank]`. Scaling is applied in shrink; vLLM retains its
normal loaded-weight scaling convention. Unused rank columns must be zeroed by
the native adapter manager. This migration does not provide compact serving
allocations or replace the native scheduler/loading/eviction implementation.

The source split-K shrink uses FP32 atomic additions, so summation order is not
deterministic. Expand retains the FP32 intermediate, multiplies and sums in
FP32, then casts the final result to the output dtype. Native vLLM matrix-dot
casts intermediates to the B-weight dtype; bitwise agreement is not promised.
Operator tolerances cannot establish full-model token/logprob equivalence.
There is no throughput claim for this extraction.

Warmed calls support CUDA graph capture. Dispatch is fixed within each captured
active-count specialization; routes and weights are read from device buffers on
every replay. vLLM must manage graph selection, fixed addresses and lifetimes.
Changing specialization requires selecting/recapturing the correct graph.
Raw callers must validate route ranges (-1 or a valid slot), order input writes,
and keep buffers alive. Autograd is rejected. Base-route rows remain unchanged
even with `add_inputs=False`, matching the reference/native skipped-token contract.

## Validation

On an allocated GPU in the pinned vLLM environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS= \
  python -m pytest tests/test_vllm_gpu.py -v
python -m pytest tests/test_vllm_registration.py -v
```

Tests cover FP32/FP16/BF16 raw GEMV, ranks 1/7/33/128/256, fused slices, masked
tails, row strides, padded/compact source layouts, base routes and offset writes.
Real vLLM tests cover GEMV/native dispatch and graph replay with replaced slot
contents and changed routes. These tests use synthetic tensors and do not
download model weights. Full-model service acceptance, tensor parallelism,
distributed execution, native loader/eviction integration and benchmarks are
separate validation tasks.

### Recorded run — 2026-11-03

An allocated NVIDIA H100 on Pelle passed all **27 GPU cases** in 21.01 seconds.
Environment: Python 3.12.3, vLLM 0.26.0, PyTorch 2.11.0+cu130 and Triton 3.6.0.
The command above used four CPU cores, 16 GiB host memory and one GPU, with
`TORCHINDUCTOR_COMPILE_THREADS=1`. The FP32 relative-norm gate is 2e-5;
FP16/BF16 comparisons use 0.01, including comparisons against native vLLM.
Real kernel-launch observation confirms GEMV is selected for the small decode
cases and is absent from native fallback cases. CUDA graph replay and full-graph
`torch.compile` both passed. The graph tests replace slot values in place;
they do not exercise vLLM's actual adapter manager.

The existing PyTorch 2.7.1+cu126 / PEFT environment passed **86 CPU tests**,
including eight new lazy-import/registration checks. Ruff, wheel and sdist
builds passed. Fresh dependency-free wheel and wheel-from-sdist environments
each passed **31 tests**, with five dependency-requiring modules skipped.
After reinstalling the editable distribution to refresh entrypoint metadata,
a fresh interpreter confirmed native `vllm.plugins.load_general_plugins()`
discovers and activates this plugin. No model weights were downloaded or
full-model token/logprob acceptance results produced in this migration.


### Current-main integration — 2026-11-08

The installer, plugin entrypoint and raw GEMV operators now expose alpha API
metadata. Only explicit installation emits a once-per-process warning;
metadata-only numerical annotations preserve the original function identity.
The optional plugin is registered alongside SGLang, with neither framework
imported by discovery of a disabled plugin. Numerical function bodies and Triton
kernel sources match the H100-validated extraction; only alpha annotation
imports/decorators were added to the implementation modules.

A recovered L40S CUDA 13.0 receipt also contains 27 passing vLLM GPU cases
alongside 13 row tests. Its source-hash properties cover only the row files, so
it is not an independent hash-attested vLLM implementation record. See
[row-validation-l40s.json](row-validation-l40s.json) for the receipt and source
comparison details. Full-model acceptance and throughput remain unmeasured.
