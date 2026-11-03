# Retained-rank Fleet and RowExpand

The `fleet` policy now contains the complete measured LoRA execution path,
separately from the existing `compatible` and `rank-aware` policies. It is an
explicit opt-in on SGLang 0.5.22, BF16, TP=1, MI250X/gfx90a, dense qkv/o/gate-up/down
projections, 16-token LoRA chunks and non-overlapping native adapter loading.
It does not claim native-identical model outputs or task-quality equivalence.

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

For the original Qwen3 small-prefill profile, additionally set
`LORAFORGE_STABLE_PREFILL=vllm` and `LORAFORGE_STABLE_PREFILL_MAX_TOKENS=256`.
The mode name refers to the inherited normalization path, not a vLLM serving
backend. This hook has explicit Qwen3/BF16/TP=1/hardware and normalization guards;
leave it disabled for Llama. Its extraction alone does not establish full-model
prefill-graph correctness in this package.

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
