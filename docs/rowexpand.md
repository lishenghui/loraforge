# Retained-rank Fleet and RowExpand

The `fleet` policy now contains the complete measured LoRA execution path,
separately from the existing `compatible` and `rank-aware` policies. It is an
explicit opt-in on SGLang 0.5.22, BF16, TP=1, MI250X/gfx90a, dense qkv/o/gate-up/down
projections, 16-token LoRA chunks and non-overlapping native adapter loading.
It does not claim native-identical model outputs or task-quality equivalence.
The experimental [Qwen3.5 GDN bridge](qwen35.md) adds four-slice input projections,
the linear-attention output projection, and native retained-rank b/a gates.
Its model validation and performance scope are recorded separately.

## Migrated components

| Component | Implementation |
| --- | --- |
| Direct shrink and staged RowExpand | Packaged CUDA/HIP source, lazy build; no external probe `.so` |
| Rank buckets 1/2/4/8/16/32/64 | Compile-time RowExpand reduction widths, staged Z reuse and persistent output blocks |
| All four projection types | Fused QKV and gate/up plus o/down, including nonzero base accumulation |
| Effective module ranks | CPU catalog from tensor headers / checked compression metadata; fused slices use maximum rank |
| Compact B and repacked A | Fixed checkpoint regions and per-slot device metadata; native A repacked after loading |
| Decode dispatch | Original MI250X table selects A and grid; RowExpand handles ranks <=64 |
| Prefill dispatch | Rank-aware segmented shrink; RowExpand <=16, native expand above 16 |
| Unsupported ranks | Device masks split custom and native work, including rank 128 in mixed batches |
| Loader / graph lifecycle | Native slot ownership, fixed addresses and active-segment count; reject unknown sources |
| Qwen stable prefill | Extracted optional canonical-row/norm and graph-bucket hook; separate validation scope |
| Compact export bridge | Explicit SGLang padding and scaling conversion with true-rank metadata |

The compact B shadow is additional storage, not a replacement for the native
pool. Repacking A means all consumers of those buffers must use the effective
rank metadata. The native loader restores checkpoint layout before each reload;
no asynchronous load may race a consumer. Catalogs are frozen before graph
capture, and registering a new checkpoint requires a new worker/pool.

## Usage

Prepare a JSON file with `{"adapters": [{"path": "/local/padded-adapter"}]}`.
A full validation manifest with the same `adapters[].path` fields also works.
Sources must be local single-file PEFT checkpoints with a common padded rank
and scale. A `compression.json` may declare `module_ranks` keyed by full A tensor
names; the padding must be zero. Unknown paths, rank patterns and unsupported
layouts fail explicitly. For compact LoRAForge exports, first run:

```python
from loraforge.adapters.sglang import export_sglang_adapter

export_sglang_adapter("compact-peft", "new-sglang-padded")
```

This creates a new destination, leaves the source untouched, folds module scaling
into BF16 B, sets loader scaling to one, and records physical storage and ranks.
Compare against these same converted weights when isolating backend behavior.

Set before worker startup:

```bash
export LORAFORGE_SGLANG=1
export LORAFORGE_SGLANG_POLICY=fleet
export LORAFORGE_SGLANG_MANIFEST=/absolute/path/catalog.json
```

Compile before capture with `loraforge.ops.rowexpand.rowexpand_extension()`;
normal pool initialization does this automatically. ROCm builds on a login host
without a GPU need `PYTORCH_ROCM_ARCH=gfx90a`. No global environment installation
or changes to SGLang kernel-config getters are required.

The low-level `row_expand` operation accepts `row_tile=2/4/8/16` and
`threads=128/256`. The default `row_tile=16` invokes the original kernel.
Smaller tiles subdivide each existing segment inside B into disjoint output
rows, retaining native A batching and each token's rank-reduction order.
Metadata stays on device during graph replay. This trades B reuse for more
independent work; it is not uniformly faster across shapes and batch sizes.
The [Qwen3.5 study](qwen35.md#b-only-row-subdivision) records the measured choices.
Job 23463508 passed 93 GPU checks, including eight new combinations covering
changing routes, segment counts, ranks, null slots, uneven slices and strided
outputs during graph replay. All 32 operator-study cases were bitwise equal to
the original kernel for every custom tile/thread variant.

For the original Qwen3 small-prefill profile, additionally set
`LORAFORGE_STABLE_PREFILL=vllm` and `LORAFORGE_STABLE_PREFILL_MAX_TOKENS=256`.
The mode name refers to the inherited normalization path, not a vLLM serving
backend. This hook has explicit Qwen3/BF16/TP=1/hardware and normalization guards;
leave it disabled for Llama. Compressed Qwen3 eager/prefill-graph validation is recorded below; other
models and configurations require separate validation.

## Experimental active-batch prefill dispatch

`LORAFORGE_SGLANG_ACTIVE_PREFILL=1` opts the `fleet` policy into skipping an
expand branch when no request in the current eager prefill batch uses that rank
class. The default is `0`. Low ranks (1–16) still use RowExpand; higher ranks
still use native expand. This changes dispatch only, not their arithmetic.

The decision uses CPU slot records owned by the existing synchronized loader,
without reading GPU ranks. Unknown routes keep both branches. Graph preparation
and capture keep both branches because subsequent replay may change the routes
or reuse a slot at another rank. Snapshot counters report skipped native-high
and RowExpand-low prefill calls; capture/eager counters are not GPU replay counts.

LUMI job **23452964** passed **45 GPU checks** on one MI250X GCD, including
zero-tolerance comparisons with the unchanged path for low/high/mixed/inactive
batches, slot rank replacement, and prefill graph replay from low-only capture
to high/mixed/inactive routes. The job completed in 7m28s; its 150 single-request
and three mixed eager/graph records also matched the frozen Fleet control
exactly. At offered rate 32, all 1,000 requests completed at **551.48 tokens/s**,
versus **555.79** for control 23452454 (0.9922x). This single-run screen shows no
gain; the switch remains disabled by default. Do not promote it as a speed fix.

## Validation status

- 104 CPU checks passed, including rank packing/clearing, frozen profile selection,
  export scaling and lazy package imports. Ruff and wheel/source builds passed.
- LUMI job **23440720** completed in 37 seconds on one MI250X GCD: **33 GPU checks
  passed**. Twenty-one rank/slice cases compared migrated shrink and expand to
  the pinned compiled probe kernels with **zero tolerance**, and all matched.
  Twelve real SGLang cases covered four projections, decode batches 4/33 and
  prefill 67, mixed ranks with native rank-128 fallback, and decode graph replay
  while changing slot ranks through 16/0/64/3 and reversing routes.
- Full-model job **23440815** failed after 12m32s. Native completed; Fleet
  failed during startup because fused projection methods renamed the native
  `qkv_lora_a/b` and `gate_up_lora_a/b` keyword parameters to `a/b`. The previous
  GPU tests called private helpers and therefore missed this framework boundary.
  The parameter names are corrected, and all twelve integration cases now use
  the public methods with the same keywords as actual SGLang layers.
- Retry **23441471 completed in 15m58s**. All 33 GPU checks passed again,
  including the corrected public keyword calls. Both native and packaged Fleet
  then completed ten timing workloads, 32 quality cases and ten forced-prefix
  traces. CPU analysis verified actual continuation IDs, complete trace sets,
  RowExpand prefill/decode capture evidence and real decode graph execution.
  The model owned 128 retained layouts and **378 MiB** of compact B shadows.

### Full-model pilot after the interface fix

Nine LoRA workloads measured **1.1126x–1.1251x** speedup versus native, with
49-token inputs, 64 generated tokens, resident adapters, batch 1/2 and three
timing trials. Base-only measured 0.9859x. The four original adapters have ranks
32/16/64/32; this is not the paper's compressed-fleet concurrent workload.
Fixed mode order and the base-control difference limit interpretation of small
cross-job gains. These results do not show a clear speedup over the earlier
partial-path pilots or reproduce the paper's fleet-level throughput claims.

At the 256-token output allowance, medical quality was **15/32 native versus
14/32 Fleet**, one paired loss and no gain. Invalid answers were 3/5, all
truncated. Case 994 again yielded a native correct answer after 123 tokens,
while Fleet repeated its explanation until the cap. The conservative approximate
95% degradation interval remains **[-13.11, +18.52] percentage points**; this
small sample establishes neither general degradation nor non-inferiority.

Identical-prefix adapter decode had **1/120 top-1 flips**, mean KL 0.00047485,
maximum KL 0.00721362 and maximum total variation 0.0585396. All 30 base-only
decode positions remained exact. Unlike the partial-path pilots, prefill also
had nonzero distribution differences: no top-1 flips across ten positions,
maximum KL 0.00236344 and maximum total variation 0.0326424. Functional completion
and migration fidelity therefore do not establish native numerical or task-quality
equivalence. Qwen stable-prefill graph reproduction and matched compressed-fleet
serving validation remain outstanding; no automatic backend promotion follows.

The previous 1.11x–1.12x pilot used partial experimental paths. Those measurements
must not be labeled as results of this complete migrated runtime. vLLM integration
remains separate work. Source checksums and behavior changes are in provenance.md.


## Compressed Qwen3 serving and headroom diagnostic

The compressed Fleet workload uses Qwen3-4B-Instruct-2507 and 50 retained-rank
adapters from the paper export, with TP=1/BF16 on one MI250X GCD. Each point
replays the same 1,000 prompts, arrivals and per-request completion lengths
(55,965 output tokens), after a full 1,000-request conditioning run. KV capacity
is fixed at 86,000 tokens, prefill chunks at 8,192, prefill/decode interval at 4,
and maximum running requests at 256. Startup and profiling are excluded.

| Execution | Offered requests/s | Output tokens/s | Relative to compressed native |
| --- | ---: | ---: | ---: |
| Native, 23451543 | 16 | 437.65 | 1.0000x |
| Packaged Fleet, 23451046 | 16 | 511.39 | 1.1685x |
| Native, 23451543 | 32 | 455.33 | 1.0000x |
| Packaged Fleet, 23451046 | 32 | 546.93 | 1.2012x |
| Packaged Fleet control repeat, 23452454 | 32 | 555.79 | 1.2207x |
| Empty-rank prefill candidate, 23452964 | 32 | 551.48 | 1.2112x |
| Base only, 23453345 | 16 | 569.96 | 1.3023x |
| Base only, 23453345 | 32 | 648.25 | 1.4237x |

These are single-seed measurements on separate allocations, not paired timing
confidence intervals or the paper's three-seed aggregate. Native job 23451543
ended in Slurm `TIMEOUT` during cleanup: both complete timing points, numerical
checks and route audit were already written. Its allocation is not relabeled as
completed. Fresh probe comparison job 23451544 failed the startup timeout before
measurement, so it supplies no speed ratio.

Both packaged Fleet jobs passed 150 matched single-request eager/graph checks
(50 adapters, prompt lengths 131/255/257, eight output tokens) and three mixed
batches. All 150 single-request records matched exactly across those two jobs;
job 23451046 also matched the archived probe records exactly. The model had
144 retained layouts, custom RowExpand prefill/decode calls and actual graph
execution. These finite synthetic checks do not establish natural-EOS quality
or overturn the Llama quality limits above.

Job **23452454 completed in 19m45s** and captured three 32-step GPU windows in a
separate replay. CPU analysis found LoRA kernels at **14.82% / 16.30% / 15.26%**
of summed kernel time. Native high-rank prefill expand accounted for **2.88% /
4.21% / 0%**; tail decode RowExpand accounted for **9.69%**, with actual tail
batches of 25–28. Of 1,584 sampled eager-prefill projection calls, 1,148 (72.47%)
had no active high rank. Call fractions are not time savings. GPU overlap is
removed for busy-time analysis; CPU/GPU inclusive spans are not added together.
The profiled replay took 106.87s versus the unprofiled control's 100.69s, so its
throughput is not used. There was no after-profile control.

The 1.3x target at offered rate 32 is 591.92 output tokens/s against this native
reference: 6.50% above the latest control, or 6.10% less elapsed time. Skipping
empty prefill branches is a bounded first candidate; these sampled fractions do
not establish that it can achieve the whole target. Historical token tiling,
rank-cap specialization and split-rank launches did not deliver a material
serving gain and should not be promoted from microbenchmarks alone.


Base-only retry **23453345 completed in 6m16s**, with three single-request and
three mixed eager/graph checks, and both 1,000-request timing points completed
without failures. CPU comparison verified the same model/runtime, timed server
options, prompts, arrivals, prompt-token counts and per-request output lengths.
Only adapter execution was removed. The initial base job **23452455** failed
before timing because the batch harness sent a list of null adapter names to a
LoRA-disabled server. The isolated retry omits the adapter field altogether;
a CPU payload regression covers both base and adapter requests. Old artifacts
remain preserved. The retry and candidate use the runtime's existing checkpoint
prefetch option for untimed startup, with full conditioning still required.

At offered rate 32, the latest frozen Fleet control reaches **85.74% of base
throughput**. Reaching 1.3x native would require removing about **42.80% of the
measured control-to-base elapsed-time gap**, assuming the base timing remains a
useful same-configuration reference. At rate 16, the 1.3x target (568.95 tokens/s)
is already nearly the measured base rate (569.96). These observations constrain
the target for this workload; they are not universal hardware speed limits.


An isolated second screen, **23453892**, kept the frozen decode policy and used
RowExpand for all supported prefill ranks instead of native expand above 16.
It completed in **9m39s**, with 33 GPU checks and its own 150 single/three mixed
eager–graph checks passing. Fixed-work throughput was **559.06 tokens/s**, only
**1.0059x** the frozen control. Cross-policy CPU comparison found **135/150 exact
single-request records**, **148/150 identical token sequences**, and maximum
common-prefix logprob delta **0.0707774**. Token differences occurred for a02 at
257 prompt tokens and a17 at 131. All 24 requests in the three mixed checks
matched exactly. Its own eager–graph equality therefore does not establish
parity with the frozen policy. This tiny timing increment and numerical change
fail the screen; the process-local hook is not installed in the package, and
no further repeat/grid or default promotion follows from it.
