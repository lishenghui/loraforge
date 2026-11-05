# Experimental APIs

Experimental functionality is shipped in the installed package, with explicit
alpha metadata and bounded runtime contracts. It is available to callers who
accept API changes between releases. The annotation describes interface
stability; it does not certify output quality, speed, or hardware support.

## Annotation and warnings

```python
from loraforge.annotations import ExperimentalAPI


@ExperimentalAPI(reason="Validated on one hardware family.", warn=True)
def enable_feature():
    """Enable a process-local integration."""
    ...
```

The callable exposes immutable `__experimental_api__` metadata containing
`stability="alpha"`, `since` and `reason`. Its docstring includes an experimental
notice. With `warn=True`, synchronous entrypoints emit `ExperimentalAPIWarning`
once per function per process, preserve signatures and propagate original
values/exceptions. The metadata-only default returns the original function or
class unchanged; numerical operations gain no call wrapper. Warning wrappers
are deliberately limited to synchronous non-generator functions. Imports do
not warn, import a tensor/framework dependency, initialize a device or compile.

To hide the notice deliberately, use Python's ordinary warning filter:

```python
import warnings
from loraforge.annotations import ExperimentalAPIWarning

warnings.filterwarnings("ignore", category=ExperimentalAPIWarning)
```

## Installed SGLang entrypoints

```python
from loraforge.integrations.sglang_experimental import (
    install_qwen35_nf4,
    install_qwen35_gdn,
    share_qwen35_gdn_storage,
)
```

Call installation entrypoints **before model construction, in each worker**.
Calling only in the parent process does not configure spawned workers. Prefer
the existing plugin entrypoint for propagation:

```bash
export SGLANG_PLUGINS=loraforge
export LORAFORGE_SGLANG_QWEN35=1
export LORAFORGE_EXPERIMENTAL_QWEN35_NF4=byte-pairs
```

NF4 accepts `stock`, `cached-scales` and `byte-pairs`, matching
`install_qwen35_nf4(mode="byte-pairs")`. It installs the prepared checkpoint
admission checks, exact 114-projection loader and optional linear optimization.
The prepared fixture must contain its complete quantization receipt. Original
adapter files and buffers remain BF16. The new kernel reconstructs canonical
NF4/block64 weights exactly and keeps native BF16 linear accumulation. Only
FP32 scales are cached persistently; BF16 weight values remain temporary.
M<=4 and unsupported shapes retain native BNB dispatch. Eager warmup prepares
scales before graph capture. Mode changes in a live process are rejected.

For the separate BF16 GDN experiment, unset the NF4 option and use:

```bash
unset LORAFORGE_EXPERIMENTAL_QWEN35_NF4
export LORAFORGE_EXPERIMENTAL_QWEN35_GDN=packed
```

`packed` corresponds to `install_qwen35_gdn(packed_checkpoint=True)` and enables
the checked loader for the canonical 114-projection BF16 fixture. Use `1`, or
`install_qwen35_gdn()` without that argument, for original checkpoint names.
LoRA decode with csgmv uses one fused base QKVZ/BA projection, followed by each
original Linear's adapter update. Prefill retains its original execution. The
original parameter objects become contiguous views of one packed owner before
graphs. No consumer, tiny-rank or private research hooks are installed by this
entrypoint. It is a standalone implementation inside the package. Existing
conflicting projection patches, quantized bases and mode changes are rejected.

`share_qwen35_gdn_storage(layer, packed_weight=owner)` also exposes the ownership
operation independently. The owner must already equal the loaded BF16
[8192+32,1024] projection weights. The operation preserves parameter identity
and metadata and returns whether a rebind occurred. Invoke it before capturing
graphs or publishing parameter views; it cannot discover previously captured
graph references. The original parameters keep storage alive after the caller
releases its owner reference. No live reload or training is supported.

Both installation paths are limited to Qwen3.5-0.8B text-only, TP1, BF16 LoRA,
ROCm gfx90a and SGLang0.5.22. NF4 additionally requires bitsandbytes0.50.2 and
the prepared checkpoint. The current validation environment is Torch2.11 /
ROCm7.2 / Triton3.6, with source hashes recorded in the validation report.
NF4 and BF16 GDN installation are mutually exclusive in one worker. These
options do not select LoRAForge's backend: native csgmv remains native unless
`LORAFORGE_SGLANG=1` and its documented policy are explicitly selected as well.

## Tensor primitive and package boundaries

`loraforge.ops.nf4.dequantize_nf4` is a metadata-only alpha API. It accepts
contiguous NF4 bytes, FP32 block64 scales, a canonical FP32 16-entry codebook
and a positive [N,K] shape with dimensions divisible by64. It returns BF16
weights on the current stream, or writes a caller-owned nonaliasing `out`.
All tensors share a gfx90a device. Codebook and scale values remain the
caller's responsibility during graph replay. The primitive imports no SGLang.

The selected kernels live in `ops/_nf4_kernels.py`; prepared dispatch,
admission, loading and GDN integration live in `integrations/`. Legacy benchmark
imports delegate to these installed implementations. Fixture preparation,
profiling, rejected MFMA/split-K candidates and measurements stay in benchmarks.
Other fast fusion research policies have not been promoted by this change.

## Validation and limitations

CPU checks cover metadata, class/function identity, per-process warnings,
signature/exception preservation, worker option propagation, invalid/conflicting
configuration, storage identity and stale-owner rejection. Fresh installed-wheel
imports reject GPU/framework dependencies to verify lazy loading.

GPU validation uses the actual installed wheel in an isolated directory. It
checks exact NF4 tensor reconstruction, caller-owned output, graph replay,
cross-stream execution and alias rejection, then compares legacy/package NF4
and legacy packed/package GDN serving paths at matching inputs. The numerical
kernel ASTs match the measured `6b2e36a` source. Validation status and exact
results are recorded separately after the allocated run completes.

Earlier compression/native numerical failures remain unchanged. Passing an
incremental migration comparison does not establish trained-adapter quality,
equal-quality compression or general upstream ROCm support. Historical
performance reports remain separate from this API migration validation.
