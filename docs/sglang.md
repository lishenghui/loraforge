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

Decode uses LoRAForge operators for ordinary linear, fused QKV and fused gate/up
projections. Eager prefill and pruned batches retain native CSGMV execution.
Embedding, lm_head, MoE and tensor-parallel custom operators remain outside
this integration. Full-model coverage is limited to the configuration below. Prefill graph support is explicitly disabled in
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

## Recorded validation

On 2026-11-03, LUMI Slurm job `23425475` passed all 36 cases in:

```bash
python -m pytest tests/test_gemv_gpu.py tests/test_sglang_gpu.py -m gpu -v
```

Hardware/software: one MI250X GCD (gfx90a), Python 3.12.3, PyTorch
2.11.0+rocm7.2 LUMI build, Triton 3.6.0, SGLang 0.5.22, inspected SGLang source
commit `baa9befa381511cd4d56f83980951e5a45fffc6d`. Pytest time was 49.11 seconds;
Slurm elapsed time was 61 seconds. The existing AMD compatibility environment
was reused rather than claiming a fresh generic pip installation is certified.

Raw tests cover FP32/FP16/BF16, slice counts 1/2/3, rank capacities 8/33 plus
BF16 capacities 128/256, split-K with masked input tails, row-strided inputs,
base-model routes and in-place output views. Six raw graph cases update weights,
ranks, scales and routes after capture while retaining buffer addresses.

Ten real-framework cases cover repeatable registry activation, actual native
batch preparation and linear/QKV/gate-up methods in eager prefill/decode, plus
three graph cases using native static batch metadata refreshed before replay.
Slot replacement is simulated with in-place tensor writes; these tests do not
exercise the native adapter manager, full paging loader or scheduler.
A separate login-node check confirmed that SGLang's general plugin loader in a
fresh process discovers the installed entrypoint and selects the backend.
Local CPU suites passed 85 cases, including disabled activation, unsupported
versions, conflicts, lazy imports and existing compression/runtime regressions.
No model weights were downloaded. NVIDIA, full-model serving and throughput
claims require separate validation.

## Full-model service protocol

`benchmarks/prepare_serving_manifest.py` prepares existing local model and four
adapter paths on a CPU/login node. It hashes adapter configs/weights and the
model config and tokenizes two fixed chat prompts without downloading models.
On an allocated GPU, `benchmarks/validate_sglang_serving.py run` starts four
separate servers: native CSGMV eager/graph and LoRAForge eager/graph. It uses
fewer native GPU slots than adapters and observes the actual native loader and
model runner. Instrumentation is validation-only and affects timing.

Each server runs two cycles of base/adapter requests, two repeated mixed-adapter batches,
and explicit HTTP unload/reload. Acceptance requires repeated generated token
IDs and logprobs to agree, real slot reuse with rank changes, stable native
buffer addresses, expected eager/graph execution, and actual custom decode calls
in LoRAForge modes. The `compare` action runs on a CPU/login node after GPU work
and compares single requests against native-eager single requests and mixed
batches against native-eager mixed batches. Both require matching token IDs
and max absolute output logprob difference at most 0.05. This threshold is a protocol criterion, not
bitwise equality or an accuracy guarantee. Partial reports and server logs are
preserved on failure. Completed mode reports can be resumed for the same input
manifest; use a new output directory when changing protocol options.

```bash
# CPU/login node; all inputs must already exist locally.
python benchmarks/prepare_serving_manifest.py --model /path/to/model \
  --adapters /path/to/a /path/to/b /path/to/c /path/to/d --output manifest.json
# Allocated GPU; include LoRAForge in the SGLang worker environment.
python benchmarks/validate_sglang_serving.py run --manifest manifest.json --output validation
# CPU/login node, after the GPU job finishes.
python benchmarks/validate_sglang_serving.py compare --manifest manifest.json --output validation
```

The first LUMI full-model run is job `23426358`, using the cached
Llama-3.1-8B-Instruct snapshot `0e9e39f249a16976918f6564b8830bc894c89659`
and four existing ordinary-LoRA adapters with ranks 32/16/64/32. Input hashes
and paths are in the job directory's manifest. This attempt reached inference
and completed two cycles of all ten single-request cases with matching outputs.
It then failed the original cross-batch-size assertion: native eager single
versus a two-request batch differed by 0.0572664 in output logprob (criterion
0.05). No LoRAForge mode had run, so this is not evidence of a custom-backend
mismatch. The failed attempt and logs are retained.

Protocol version 2 records single-versus-mixed drift as a diagnostic, requires
repeatability within each batch shape, and compares native/custom eager/graph
results at matching batch shapes. The backend-comparison threshold stays 0.05;
within-shape repeat checks stay 1e-5. Version and input identity are checked when
resuming or comparing reports, and mixed outputs are saved before assertions.
A regression test demonstrates that 0.0573 cross-shape drift is recorded while
0.05-exceeding drift between matched native/custom batches still fails. The
updated CPU suite passed 91 checks.

Rerun `23430057` completed all four service modes in 6 minutes 31 seconds in
`loraforge-serving-20261103-v2`. Each mode recorded 12 adapter loads, 9 slot
reuses and 7 rank changes with stable addresses. Native and custom graph modes
each recorded 367 decode graph forwards; the custom observer also recorded
actual GEMV capture/execution. These per-mode checks passed, but the final CPU
comparison **failed**: generated token IDs matched for every single/mixed case,
while the largest native/custom output logprob difference was
0.08525431156158447, above the unchanged 0.05 criterion. Native eager/graph
outputs matched exactly, as did custom eager/graph outputs. Base-only outputs
also matched exactly. This narrows investigation to custom adapter decode
numerics; it does not establish the precise source of the discrepancy.
This earlier attempt did not satisfy full-model acceptance. An isolated
single-GCD diagnostic (`23430621`) completed three eager diagnostic variants
with identical inputs. These are diagnostic outputs, not acceptance results:

| Decode A / B implementation | Maximum logprob difference vs native |
| --- | --- |
| custom / custom (original validation) | 0.08525431156158447 |
| native / custom | 0.06176626682281494 |
| custom / native | 0.09927737712860107 |

All generated token IDs matched. Replacing either operation alone does not
restore the 0.05 criterion; both custom operator paths contribute to end-to-end
numerical differences. On identical real activations, the first 128 sampled
shrink calls differed in one of 7,168 live elements by 2.9802322387695312e-08;
the paired 128 expand calls matched in all 1,376,256 elements. This bounded
sample does not establish expand equality for later calls.

The saved shrink input (rank 32, two slices, input width 4096) has a live element
with FP64 reference -7.2882194217527285e-06. Native rounds to
-7.3015689849853516e-06, custom to -7.271766662597656e-06; their BF16 midpoint
is -7.286667823791504e-06. Native agrees with the FP64 result rounded to BF16
in this sample. The native matrix-dot and custom split/tree FP32 reductions
use different accumulation orders, producing a real BF16 rounding-boundary
disagreement. This is evidence for decode numerical parity work, not a paging
or graph-replay fix. The correction and successful validation follow below.

### BF16 correction and completed validation

The BF16 tensor operators now use sequential matrix-dot blocks in FP32,
retaining BF16 intermediate and scaled-delta rounding. Shrink uses K blocks of
256 with 16 output columns; expand uses rank blocks of 16 with 64 output
columns, matching the pinned native default configuration. Routed-token tiles
keep metadata on-device and support dynamic ranks under graph replay.
FP32/FP16 retain the original split-K/tree kernels. This compatibility path
has no speedup claim and does not certify alternate tuned native geometries.

Three new GPU regression cases require exact native/custom BF16 equality at
model input widths 4096/14336 and ranks 16/32/64, covering packed fused slices,
mixed ranks, base routes, scaling and addition to a nonzero base projection.
Local validation passed 91 CPU checks, Ruff checks and wheel/sdist builds.
Job `23431392` stopped at the GPU regression gate: 20 passed and 19 failed
because Triton requires the store pointer itself to have the full matrix tile
shape. No full-model mode ran. Both BF16 stores now explicitly broadcast their
pointer tiles to the value/mask shape.

Rerun `23431819` completed successfully in 15 minutes 17 seconds on one MI250X
GCD, using the same pinned environment and cached inputs. All **39 GPU checks
passed** in 37.64 seconds, including the three exact BF16 native comparisons.
All four full-model modes passed. Each mode observed 12 loads, 9 slot reuses
and 7 rank changes with fixed buffer addresses. Eager modes recorded 367 eager
decode forwards; graph modes recorded 367 graph decode forwards. Custom
execution/capture was observed in both LoRAForge modes.

The separate CPU `compare` step passed protocol version 2: all generated token
IDs and output logprobs matched native eager exactly in the ten single-request
cases and both mixed-batch outputs. Maximum absolute logprob differences were
**0.0** for native graph, LoRAForge eager and LoRAForge graph, within the
unchanged 0.05 criterion. The within-shape repeatability gate remained 1e-5.
Raw reports and prior failed attempts are retained in their separate directories.

This validates BF16 Llama-3.1-8B-Instruct with the four rank-32/16/64/32 adapters,
two 49-token prompts, 16 generated tokens, TP=1 and two native padded adapter
slots. It establishes correctness for this finite test set, not universal
bitwise equality, task accuracy or throughput. NVIDIA serving, alternate tuned
kernel geometries, compact serving storage, prefill graphs and TP/MoE still
require separate validation.

## BF16 operator latency — 2026-11-03

LUMI job `23434324` completed in 58 seconds on one MI250X GCD using the same
SGLang 0.5.22 / PyTorch 2.11 ROCm 7.2 / Triton 3.6 environment. This is an
operator microbenchmark, not full-model throughput. Compare native CSGMV,
pre-correction GEMV (`00b4ed9`) and the validated BF16 compatibility kernels.

`benchmarks/benchmark_sglang_decode.py` uses synthetic BF16 tensors at Llama 8B
QKV, O, gate/up and down-projection dimensions, ranks 16/32/64 and batches
1/2/8. All tokens use the same adapter, with weights and routing already
resident. A+B computation includes cloning a nonzero base output. Compilation
and warmup are excluded. Five trials of 50 iterations interleave backends in
seeded randomized order; inputs/weights are repeatedly reused (warm cache).
CUDA-event intervals in eager mode include host submission gaps. Graph mode
replays a captured A+B pair. No paging or base-model computation is timed.

For each batch, ratios below divide sums of per-case median event latencies
across its 12 projection/rank combinations. They are not model-weighted
throughput estimates. Larger ratios mean higher latency:

| Batch | Graph compatible/native | Graph compatible/legacy | Eager compatible/native | Eager compatible/legacy |
| --- | --- | --- | --- | --- |
| 1 | 1.014x | 1.766x | 1.071x | 0.771x |
| 2 | 1.091x | 1.627x | 1.069x | 0.770x |
| 8 | 1.500x | 1.263x | 1.066x | 0.763x |

At rank 32, batch 8, gate/up graph replay measured 33.26 us native,
66.55 us legacy and 72.82 us compatible. Across all graph cases, compatible
latency ranged from 0.987x to 2.729x native. Across case/backend combinations,
the median five-trial relative range was 0.26% for graph and 4.51% for eager;
maximum ranges were 2.42% and 9.51%, respectively. These describe within-run
variation, not uncertainty across independent allocations.

The compatibility path has lost the legacy small-batch graph advantage and
is slower than native for larger homogeneous batches. Eager is faster than
legacy but still slower than native. These results support no serving-speedup
claim. The legacy implementation remains a failed full-model numerical control,
even though this synthetic output sample happened to agree with native.

Reproduce with a trusted historical module from this repository:

```bash
git show 00b4ed9:src/loraforge/ops/gemv.py > /tmp/legacy_gemv.py
python benchmarks/benchmark_sglang_decode.py --legacy-module /tmp/legacy_gemv.py --output speed.json
```

Run on an allocated GPU with the validated framework environment. The JSON
records versions, source hashes, individual trial timings and numerical drift.

## Probe strategy migration in progress

Job `23435268` validated grouping same-slot tokens within each 16-token tile
and pipelining shrink loads while retaining the accumulation order: 42 GPU
checks passed and the four-mode full-model comparison again had identical token
IDs and maximum logprob delta 0.0. In the same-allocation operator benchmark,
graph latency versus native was 1.034x/1.119x/1.161x for batch 1/2/8; batch 8
was 22.5% faster than the prior compatible path, but small batches did not
recover the old GEMV advantage. These remain warm-cache operator measurements.

The public probe Fleet backend does not use one custom path universally. Its
rank-aware shrink skips column blocks outside actual adapter rank; its tuned
and Fleet variants select native or specialized kernels by shape and workload.
The Qwen3-4B/MI250X table is not a validated Llama-3.1-8B tuning table.

An explicit `LORAFORGE_SGLANG_POLICY=rank-aware` candidate now extracts the
non-split probe shrink with native launch configuration and uses native expand.
The default remains `compatible`. Native explicit split-K settings retain the
native path, and unsupported noncontiguous shrink inputs retain the tensor
operator. Prefill and framework globals are unchanged. See provenance for the
pinned upstream-derived source and license.

Validation job `23436482` is pending for the rank-aware candidate: GPU checks,
capacity-64/actual-rank-16/32/64 operator timings, and four-mode full-model
verification. The offline benchmark also loads a separately CPU-built public
Fleet V3 HIP library, including A-only/B-only hybrid comparisons with native.
Its compact B layout is prepared outside timing, as a resident shadow layout;
that is a layout-plus-kernel comparison. This does not migrate Fleet's loader,
per-layer compact pools or global hooks into the library. The isolated source
only adds line breaks before kernel launches to avoid the current HIP
translator's same-line parsing error. HIP paths are diagnostic candidates,
not validated serving defaults. Local CPU checks passed 94 cases.
