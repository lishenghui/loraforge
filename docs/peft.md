# PEFT adapter files

LoRAForge reads local linear LoRA and rsLoRA adapters, compresses them using our
FlashTSQR tensor API, and writes a new PEFT adapter with a versioned manifest.
Reading and exporting need PyTorch and safetensors, but do not import PEFT,
Transformers, initialize a GPU, or execute model code. Real PEFT loading is
tested independently with synthetic models and no network access.

## Install and compress

For file operations, install `python -m pip install ".[adapters]"`.
For compression, first install PyTorch and a matching compiler for your GPU,
then install `python -m pip install ".[adapters,gpu]"`.

```python
from loraforge.adapters import compress_peft_adapter

result = compress_peft_adapter(
    "original_adapter",
    "compressed_adapter",  # must not exist; its parent must exist
    keep=8,
    device="cuda:0",  # explicit CUDA or ROCm GPU
)
print(result.directory)
print(result.manifest["output"]["files"]["adapter_model.safetensors"])
```

`keep` may instead map every canonical module name to its retained rank.
Names exclude `base_model.model.` and `.lora_A/B.weight`; use
`load_peft_adapter(path).modules` to inspect them. Partial maps, zero ranks,
and ranks above the source rank are rejected before GPU work. Zero-rank tensor
results remain valid in the tensor API but are not exported as PEFT modules.

Same-shape/dtype modules with the same retained rank are grouped into GPU batches
of at most `batch_size` (default 16). Lower this value to limit GPU memory.
Compact outputs are moved to CPU between batches. Device kernels and EVD
selection follow [the tensor API contracts](flashtsqr.md); no CPU QR or generic
GPU QR fallback exists.

There is also a runnable example:

```bash
python examples/compress_adapter.py original_adapter compressed_adapter \
  --keep 8 --device cuda:0
```

## Read and export without compression

```python
from loraforge.adapters import load_peft_adapter, save_peft_adapter

adapter = load_peft_adapter("original_adapter")
for module in adapter.modules:
    print(module.name, module.rank, module.alpha, module.scaling)
exported = save_peft_adapter(adapter, "copied_adapter")
```

The loaded object is a CPU snapshot; treat its configuration and tensors as
read-only. `save_peft_adapter` accepts an optional complete mapping of module
names to replacement `(a, b)` tensors with unchanged ambient dimensions and
positive, non-increasing ranks. It records an **export**, with no inferred
compression method or execution backend. Only `compress_peft_adapter` records
the FlashTSQR backend that actually ran.

## Scaling and storage

The reader detects the source `use_rslora` flag automatically. Both variants
compress the raw `B @ A` factors using the same numerical implementation; export
preserves the source flag and its effective scale within floating-point precision:

| Variant | Effective scale | Exported alpha for retained rank k |
| --- | --- | --- |
| LoRA | `alpha / r` | `source_alpha * k / source_rank` |
| rsLoRA | `alpha / sqrt(r)` | `source_alpha * sqrt(k / source_rank)` |

For example, rsLoRA rank 64 to 16 halves alpha, keeping the multiplier on the
compressed factors equal to the original multiplier. Alpha can be fractional. Exact
anchored module patterns avoid suffix collisions when ranks/alphas differ by
layer. The output target regex selects exactly the modules present in the file.
The original base-model identifier, revision, and applicable metadata are retained.

By default, saved tensors use each source module's dtype. Set `output_dtype` to
`float32`, `float16`, or `bfloat16` to override it. Casting can introduce rounding;
nonfinite factors, including cast overflow, are rejected. Outputs have physical
retained-rank shapes, with no nominal-rank padding. Manifest tensor bytes count
actual tensor payloads, while file bytes include safetensors headers. An FP32
override or a small rank reduction need not make the overall file smaller.

The input directory is read-only. Output must be a new directory outside it;
existing directories, files, and symlinks are rejected. Complete files are staged
beside the destination, then linked into an exclusively created directory without
replacing existing files. The manifest is published last. Failure cleanup removes
only files this export linked and preserves unrelated files. This requires a
local filesystem supporting hard links within the destination parent.

## Manifest version 1

Output contains `adapter_config.json`, `adapter_model.safetensors`, and
`loraforge_manifest.json`. The manifest's `format` is `loraforge.peft` and
`schema_version` is the integer 1.

| Field | Meaning |
| --- | --- |
| `loraforge_version` | Producing library version |
| `operation` | `export` or `compression` |
| `compression` | Requested method, EVD selection and batch size; null for plain export |
| `source` | Source directory name and file hashes/byte counts, including any previous manifest |
| `modules` | Canonical name, source/retained rank, alpha/scale, scaling type, dtypes, tensor bytes and actual execution metadata |
| `output.files` | Config/weight filenames, SHA-256 hashes and measured file bytes |
| `output.tensor_bytes` | Total output tensor payload bytes |
| `output.source_tensor_bytes` | Total original tensor payload bytes for comparison |

`load_peft_adapter` verifies hashes and file sizes when a LoRAForge manifest is
present, rejects unknown versions, and rechecks the source files for changes
during loading. Export checks that the source snapshot still matches its files.
Hashes provide consistency checking, not signature-based authentication.

Each new module record has `use_rslora` and `scaling_type` (`alpha_over_rank`
or `alpha_over_sqrt_rank`). These fields are additive in manifest version 1;
earlier ordinary-LoRA exports omit them. The loader obtains the operative mode
from the hash-checked PEFT configuration. Export rejects a snapshot whose module
scaling mode disagrees with its configuration.

## Supported surface

Input is one `adapter_model.safetensors` plus `adapter_config.json`, using saved
keys `base_model.model.<module>.lora_A.weight` and `lora_B.weight`. Factors must
be matching 2D float32/float16/bfloat16 pairs with ranks consistent with the
configuration, including ordered rank/alpha patterns.

This file API rejects DoRA, QALoRA, aLoRA, convolution/embedding
layouts, named in-memory adapter keys, bias weights, modules-to-save,
parameter-targeted adapters, layer replication, special initialization schemes,
unknown config fields, and extra tensors. Sharded checkpoints and pickle `.bin`
files are not loaded. This avoids silently dropping unsupported weights or
interpreting a different update rule as ordinary LoRA. It does not inspect or
certify a caller's quantized base model.

The format and scaling contracts were checked against PEFT 0.18.0's
[checkpoint documentation](https://huggingface.co/docs/peft/developer_guides/checkpoint),
[LoRA configuration](https://github.com/huggingface/peft/blob/v0.18.0/src/peft/tuners/lora/config.py),
and [pattern semantics](https://github.com/huggingface/peft/blob/v0.18.0/src/peft/utils/other.py).
The two scaling conventions follow PEFT's
[LoRA layer implementation](https://github.com/huggingface/peft/blob/v0.18.0/src/peft/tuners/lora/layer.py).
CPU CI uses PEFT 0.18.0 and Transformers 4.x with PyTorch 2.7.1. Other PEFT
versions are not broadly certified by those checks.

## Recorded validation

On 2026-11-01, the CPU file/reload suite passed locally with PyTorch 2.7.1,
PEFT 0.18.0, Transformers 4.57.6 and safetensors 0.8.0. It covers float32,
float16 and bfloat16 export; heterogeneous ranks and fractional alpha; suffix
patterns; unsupported formats; integrity checks; source changes; output
collisions; and failure cleanup. A separate serializer fixture checks that
returned execution metadata matches the JSON representation exactly.

LUMI Slurm job `23375283` used one MI250X GCD (`gfx90a`) on `small-g`, Python
3.12.11, PyTorch 2.7.1+rocm6.2.4, PEFT 0.18.0, Transformers 4.55.3 and
safetensors 0.6.2. Both actual GPU file-compression tests passed: one with
heterogeneous per-layer ranks and one grouping same-shape modules into a batch.
The checks compare truncation error to a dense SVD reference and PEFT forward
outputs to the saved factors with their effective scales. No model checkpoints
were downloaded. Dependencies and native compilation were prepared on the login
node before GPU allocation. The final return-value normalization is covered by
the CPU serializer test and does not alter the files or GPU computation.

These tests establish format and numerical correctness for synthetic linear
models, not task-level accuracy retention or end-to-end performance. This step
does not extend the tensor API's NVIDIA CUDA validation scope.

The rsLoRA extension passed 35 CPU file/reload checks on the same local CPU
environment and on LUMI's login node, including ordinary-LoRA regressions,
heterogeneous rank/alpha patterns, and FP32/FP16/BF16 export. LUMI job `23377615`
then passed both rsLoRA GPU file-compression cases (heterogeneous and grouped
modules) on one MI250X GCD, with the same PyTorch/PEFT/Transformers versions as
job `23375283`. Tests confirm the retained flag, square-root alpha adjustment,
PEFT's effective scales, and forward outputs after reload. Kernels are unchanged.

## Global direction budget

Instead of `keep`, pass an integer `budget` to `compress_peft_adapter`:

```python
result = compress_peft_adapter("original", "compressed", budget=128, device="cuda:0")
```

The budget is the sum of retained module ranks. Every module reserves one
rank; the remaining directions are selected by descending singular value,
with module name and direction index breaking ties. The objective uses the
unscaled factors B @ A, matching the research spectral allocation convention;
it does not optimize task quality, scaled update energy, bytes or latency.
Infeasible budgets and simultaneous `keep`/`budget` arguments are rejected.

Spectrum measurement uses GPU FlashTSQR per module, followed by the existing
batched compression/export pass. This computes factorization twice while bounding
GPU memory. The manifest records the objective, exact direction budget, minimum
rank and chosen ranks under `compression.allocation`. Plain `keep` calls retain
the existing workflow. GPU validation of the new two-pass workflow is separate
from earlier file-compression validation.

The included command-line example also accepts `--budget` in place of `--keep`:

```bash
python examples/compress_adapter.py original compressed --budget 128 --device cuda:0
```

On 2026-11-03, LUMI Slurm job `23423930` passed all six file-compression
GPU tests on one MI250X GCD (gfx90a), Python 3.12.11 and PyTorch
2.7.1+rocm6.2.4. This includes both new budget roundtrips (LoRA and rsLoRA),
checking exact total ranks, positive per-module ranks, manifest allocation,
preserved scaling and dense-SVD reconstruction error. The other four cases
cover the existing heterogeneous/grouped file workflow. No NVIDIA validation
is claimed by this run.
