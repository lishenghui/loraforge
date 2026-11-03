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

Validation job `23436482` completed: 42 GPU checks passed, and the four-mode
full-model comparison passed with identical token IDs and maximum logprob
delta 0.0. Each mode exercised 12 loads, 9 slot reuses and 7 rank changes.
With capacity 64 and actual ranks 16/32/64, graph operator latency relative
to native was 1.0023x/1.0037x/1.0014x for batch 1/2/8 (sum of 12 case
medians per batch). This recovers native-level operator performance; it is
not an end-to-end throughput measurement.

The offline benchmark also loads a separately CPU-built public
Fleet V3 HIP library, including A-only/B-only hybrid comparisons with native.
Its compact B layout is prepared outside timing, as a resident shadow layout;
that is a layout-plus-kernel comparison. This does not migrate Fleet's loader,
per-layer compact pools or global hooks into the library. The isolated source
only adds line breaks before kernel launches to avoid the current HIP
translator's same-line parsing error. HIP paths are diagnostic candidates,
not validated serving defaults. The HIP-A/native-B hybrid's graph operator
speedup over native was 1.785x/1.772x/1.679x for batch 1/2/8. Synthetic
output drift was zero, which does not establish full-model numerical parity.
Isolated full-model validation job `23436657` completed in 17m06s, but the
final cross-mode comparison failed. Native eager and graph outputs matched
exactly. Both HIP-A hybrid modes changed generated token IDs for
`adapter_1/0`; among cases with matching token IDs, maximum logprob delta
was 0.0777238011, exceeding 0.05. Base-only outputs remained exact. The
per-mode service checks passing does not imply cross-mode numerical parity.
The hybrid therefore must not replace the validated rank-aware path or be
advertised as a validated serving speedup. Its observer patch remains outside
the production library. Local CPU checks passed 94 cases.

The isolated eager shadow diagnostic (`23437670`, completed in 10m17s) ran
HIP A beside native A on identical real activations, while returning native A
to the model. The resulting full-model outputs matched the prior native
baseline exactly. Across 2,048 sampled calls and 143,360 live elements, 53
elements differed; maximum absolute difference was 0.001953125. Sampling
covered rank 16/32/64 and the mixed rank-32/16 batch, with a limit of 128
calls per width/slice/rank-pattern group; this is not exhaustive coverage.

CPU replay of the HIP FP32 accumulation and shuffle order reproduced all
12 saved differing elements. A high-precision sum rounded to BF16 matched
HIP in 10 of those samples and native in 2. Thus these discrepancies are
accumulation/rounding differences, not evidence that HIP is uniformly less
accurate; increasing accumulator precision alone cannot guarantee native
compatibility. The FP64-accumulator diagnostic library was compiled in an
isolated directory, but no additional full-model GPU run was submitted for
it after this finding. Preserving the native accumulation behavior remains
necessary for the current output-parity contract. These observations do not
establish a model-quality improvement or an end-to-end serving speedup.

### Experimental quality and service-performance pilot

Strict native-output compatibility and task quality are separate contracts.
The Fleet candidate's failed strict comparison remains a failure of that
contract; it does not by itself establish degraded task quality. Likewise,
rounding differences do not establish that quality is unaffected. vLLM's
[reproducibility guidance](https://docs.vllm.ai/en/stable/usage/reproducibility/)
and [FAQ](https://docs.vllm.ai/en/stable/usage/faq/) describe analogous limitations
of default inference reproducibility; those statements do not validate this
LoRAForge candidate.

`benchmarks/benchmark_sglang_service.py` adds an **experimental, isolated**
HIP-A/native-B pilot. It requires an explicitly supplied compiled library
implementing the public probe Fleet shrink ABI; the production backend and
`validate_sglang_serving.py` acceptance rules are unchanged. Use the pinned
probe source described above. The protocol records library, benchmark-source,
manifest and quality-input SHA-256 hashes.

The GPU run performs three separate measurements, on the same allocation:

- **Service timing:** single-client HTTP wall time including base-model
  computation, prefill and decode. Workloads cover base-only, each adapter at
  batch 1/2, and mixed adapters. Each adapter set is warmed before three timed
  fixed-length requests. Model startup and adapter paging are excluded;
  logits capture and forced-token processing are disabled for these requests.
  Lightweight execution observation records each phase once, during warmup.
- **Labeled quality:** normal free generation, with no forced tokens. The first
  pilot uses 32 deterministically sampled MedMCQA validation questions with
  the rank-32 medical adapter. A brief explanation precedes an explicit final
  answer so the task exercises decode. Invalid answer formats count as wrong;
  raw outputs, completion lengths and finish reasons are retained. This does
  not assess SQL, translation or intent-adapter quality, and overlap with the
  adapter's original training data is unknown.
- **Same-prefix numerics:** capture full-vocabulary logits before any token
  intervention, then force the native continuation in both modes. Compare
  decode-step KL divergence, total variation, native-token logprob changes,
  top-1 changes and native top-two margins. Prefill positions are labeled
  separately. This tests actual decode computations, not prompt-only scoring.

Data preparation and trace analysis run on CPU. Each GPU attempt uses a fresh
output directory; old attempts are preserved. Example commands:

```bash
python benchmarks/benchmark_sglang_service.py run \
  --manifest manifest.json --quality quality.json \
  --fleet-library /absolute/path/loraforge_fleet_probe.so --output pilot-new
# After the GPU allocation ends, on CPU:
python benchmarks/benchmark_sglang_service.py analyze --output pilot-new
```

`quality.json` contains `split: "validation"`, `source`, `adapter` (a manifest
adapter name), and `cases` with unique `id`, tokenized `input_ids`, and an
`answer` digit from 0 to 3. Prompts must request the final `Answer: X` format.
The concrete dataset and model artifacts are not published in this repository.

This is a pilot, not a production qualification. Quality reports paired gains
and losses plus an approximate conservative 95% interval built from Bonferroni
Wilson intervals; even zero disagreements in 32 cases cannot certify a small
quality-regression bound. Performance uses a fixed native-then-candidate mode
order and few repetitions. A useful pilot should be followed by larger
adapter-relevant held-out evaluations and repeated/reversed-order speed runs.
There is no automatic promotion or merging based on these pilot results.

#### First pilot results — job 23438767

Completed on one MI250X GCD in 14m27s; separate CPU analysis verified actual
forced token IDs, complete trace positions and native/candidate graph-execution
evidence. In nine LoRA workloads (four adapters at batch 1/2 plus a mixed
batch), median HTTP speedup was **1.1101x–1.1185x**, corresponding to
**9.9%–10.6% lower request latency**. Base-only speedup was 1.0008x. For example,
rank-32 adapter_0 at batch 1 took 1.6064s native versus 1.4371s Fleet for 64
new tokens. Inputs were the two 49-token prompts; adapters were resident and
warmed. This includes base-model computation but excludes startup and paging.
Fixed mode order and three trials still require reversed-order replication.

On the 32 MedMCQA validation questions, both modes scored **14/32 (43.75%)**,
with the same set of correct answers (paired losses/gains both zero). Invalid
final-answer formats numbered 5 native and 4 Fleet; all reached the 96-token
output limit. These count as incorrect under the declared protocol. The
approximate conservative 95% degradation interval is **[-13.57, +13.57]
percentage points**, so the pilot does not certify quality non-inferiority.
Larger evaluation should increase the output allowance uniformly, retain
truncation counts, and cover the other adapters' relevant tasks.

Of 120 adapter decode positions under identical forced prefixes, one top-1
choice changed. Mean KL(native || Fleet) was 0.00038537, maximum KL 0.00546859,
and maximum total variation 0.0441468. The top-1 change occurred at
adapter_1/prompt_0, generated-token index 8, with native top-two logprob margin
0.125. All 30 base-only decode positions and 10 prefill positions matched
exactly (KL and total variation zero). These are measured distribution
changes, not a task-quality score. The strict compatibility failure remains
recorded; the candidate now has a positive service-performance pilot and an
inconclusive, rather than certified-equivalent, quality pilot.

#### Gate/up expansion candidate

The batch-1 pilot averaged 1.6092s native and 1.4420s HIP-A/native-B, versus
1.3046s for the base-only control. This is an approximate overhead comparison,
not an isolated kernel profile or a strict speedup ceiling: base computation,
prefill, scheduling and HTTP costs remain in every measurement.

The earlier graph operator sweep identifies gate/up B as a candidate for
further work. Relative to HIP-A/native-B, full HIP A+B improved the gate/up
projection by 1.383x at batch 1 and 1.337x at batch 2 (sums of medians across
ranks 16/32/64). Other projections showed little gain, and several regressed
at batch 8. The next isolated experiment therefore adds only **gate/up** HIP
expansion via `--fleet-expand gate`; the default remains `native` expansion.

`benchmarks/sglang_gate_expand_experiment.py` maintains fixed-address compact
B regions per native slot for this projection, refreshed after native loading.
It preserves the original buffers for native prefill and other projections.
This duplicates the gate/up B allocation, so it is not a memory-saving claim.
GPU-resident bucket/offset metadata and a stable active-segment-count tensor
allow route and rank changes during graph replay. The experiment is limited
to BF16, power-of-two slot capacities up to 64 and decode batches up to 16;
unsupported expand calls retain native execution. It is not a production policy.

`benchmarks/check_sglang_gate_expand.py` checks mixed routes, rank-changing
slot reuse, clearing and fixed addresses under graph replay, then compares
native B with Fleet launch grids 64/110/220/440/880 at batch 1/2/8 and ranks
16/32/64. These are synthetic operator checks; actual service integration,
quality and end-to-end benefit require a new full-model run. Local CPU checks
passed 99 cases, including compact-slot ownership and padding regressions.

The allocated one-GCD check (job **23439843**, 35 seconds) passed all four
mixed-route/slot-reuse graph states, with zero maximum absolute difference in
these synthetic outputs. At the retained 880-block launch, native/Fleet B-only
speedups were 1.264/1.450/1.741x for ranks 16/32/64 at batch 1,
1.236/1.402/1.652x at batch 2, and 1.128/1.223/1.236x at batch 8.
Grids 220/440/880 were close; this sweep does not justify a per-shape tuning
table. The full-model follow-up, job **23439944**, uses `--fleet-expand gate`
and uniformly raises the quality output allowance to 256 tokens. Its actual
service benefit and output-quality results are pending; the B-only ratios
must not be reported as model speedups.
