# SGLang decode integration

This opt-in integration targets SGLang 0.5.22's dense linear LoRA backend.
`loraforge.ops.lora_shrink` and `lora_expand` provide GPU decode GEMV with packed
fused A slices, padded B row strides, device-resident ranks/scales/routes and
FP32 accumulation. The intermediate and scaled delta follow the activation
dtype rounding convention. Rank capacities 1..256 and FP32/FP16/BF16 are supported
by the interface; tested hardware and cases are recorded below.

## Activation

Install LoRAForge in the SGLang worker environment. Its PyTorch and Triton builds
must match the host CUDA/ROCm platform. No serving framework is imported by
ordinary `import loraforge` or `import loraforge.ops`.

For every worker process, enable the installed general plugin explicitly:

```bash
LORAFORGE_SGLANG=1 SGLANG_PLUGINS=loraforge python -m sglang.launch_server \
  --model-path /path/to/model --enable-lora --lora-backend csgmv \
  --lora-paths adapter=/path/to/adapter
```

`SGLANG_PLUGINS` restricts the active plugin set. If your deployment requires
other plugins, include them deliberately and avoid another csgmv replacement.
Alternatively, call `install_sglang_backend()` from
`loraforge.integrations.sglang` before constructing a backend in each worker.
Registration is repeatable and rejects unsupported SGLang versions and existing
foreign csgmv replacements. The general entrypoint is inert without
`LORAFORGE_SGLANG=1`. SGLang's loader logs plugin exceptions, so direct installer
calls provide a fail-fast version/conflict check for embedded deployments.

Decode uses LoRAForge GEMV for ordinary linear, fused QKV and fused gate/up
projections. Eager prefill and pruned batches retain native CSGMV execution.
Embedding, lm_head, MoE, tensor-parallel and full-model service behavior are not
certified by the initial tests. Prefill graph support is explicitly disabled in
the subclass until separately validated. No throughput improvement is claimed.

## Fixed addresses and ownership

The adapter consumes SGLang's existing fixed-address weight and routing buffers.
It does not create a second GPU pool or connect `GPUAdapterSlots` by replacing
SGLang's scheduler. Loading synchronization, eviction decisions and graph buffer
lifetimes remain with SGLang. Native max-rank padded allocations are still used;
compact file ranks do not imply compact serving memory in this integration.

Warmed decode calls can be captured. Captured kernels read updated rank, scale,
route and slice-offset values from the same device addresses. Replacing a slot
in place therefore does not freeze the old rank into the graph. Raw operator
callers must guarantee valid metadata ranges, slice boundaries and buffer
lifetimes/stream ordering. These low-level inference operators do not synchronize
to inspect device metadata and do not implement autograd.

The initial schedule is the source implementation's flat split-K path with an
explicit deterministic launch configuration. Rank-bucket schedules, environment
tuning tables, persistent B shadows, paging caches, HIP specializations and
attention/model monkey patches are not imported. Benchmarking and further
optimization are separate changes.
