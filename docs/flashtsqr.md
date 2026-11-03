# FlashTSQR tensor API

This is an inference-only, early-development API. Install the CUDA/ROCm build of
PyTorch appropriate to your machine and its matching compiler, then install
`.[gpu]`. The base package remains importable without PyTorch. Native sources
are included in wheels and source archives under `loraforge/ops/csrc` and compile
on the first operation through PyTorch's extension loader. The cache key includes
source content and the PyTorch version. Standard PyTorch compiler/cache/architecture
environment overrides apply; the library does not change them.

## QR and implicit Q application

```python
from loraforge.ops import qr_factor

factor = qr_factor(x)  # x: GPU float32 [batch, m, n], m >= n > 0
reconstructed = factor.apply_q(factor.r)
other = qr_factor(y)  # does not invalidate factor
again = factor.apply_q(z)  # z: same-device float32 [batch, n, k], k > 0
print(factor.info)
```

`factor.r` is [batch, n, n]. Input tensors may be noncontiguous and are not
modified. Each handle owns its leaf/merge reflectors; its tensors must remain
unmodified by callers. There is no process-global last-factorization state.
An event orders applications after factorization on other streams, and
`record_stream` protects owned storage when the handle is released after enqueue.
Callers must order their own input writes before calling the operator and wait
for output production before consuming it on a different stream.

`path="auto"` selects shared-memory `tsqr_full` when the leaf and merge tiles
fit, otherwise `tsqr_wide`, the same Householder method with global workspaces.
Explicit `full`/`wide` selectors are available for validation. Neither calls
generic QR. The full path builds its large-rank WY panel factors before returning
the handle, so applications do not mutate shared caches. The imported
`apply_mode="cooperative"` remains an experimental full-path option.

All native launches use the input device's current PyTorch stream. Computation
does not move inputs to the current default GPU. Ambient autocast is disabled
inside operators. Autograd, graph capture, and torch.compile integration are
not supported. Wide application is limited by its per-block shared scratch
space; an oversized right-hand side raises an error rather than changing QR
backends. Wide scratch is checked against the actual device's per-block limit;
CUDA launches explicitly opt in to shared memory above the default 48 KiB.
Float32 ATen GEMMs follow the caller's TF32 policy on CUDA; validate
your precision settings for that environment before relying on tolerances.

## LoRA refactoring and truncation

```python
from loraforge.compression import refactor_lora, truncate_lora

full = refactor_lora(a, b)
compact = truncate_lora(a, b, keep=8)
padded = truncate_lora(a, b, keep=8, pad_to_original_rank=True)
```

Inputs are matching GPU float16, bfloat16 or float32 factors, either
`a[r, in], b[out, r]` or `a[batch, r, in], b[batch, out, r]`. Rank must be
positive and no greater than either ambient dimension. Outputs preserve the
batching convention and input device, and use float32. Input factors are unchanged.

The method factors the smaller ambient side once using FlashTSQR, forms a small
Gram core in float64, and splits each singular value evenly between the returned
factors. Numerical null directions below `rank * eps64 * max_eigenvalue` in the
Gram core are zeroed to avoid amplifying roundoff in deficient inputs. This
threshold is part of the new API's numerical contract, not a performance change.

Results contain `a`, `b`, the full descending `singular_values` spectrum, and
`info`. `keep=0` produces zero-size rank dimensions. Compact results have physical
rank `keep`; explicit padding preserves nominal shapes but saves no tensor bytes.
These operations represent `b @ a`. They do not change or absorb external
adapter scaling; callers retain that scaling. Use
[the adapter file API](peft.md) for configuration-preserving LoRA/rsLoRA export.

`eigh_backend="auto"` preserves the original selection: CUDA uses
`torch.linalg.eigh`; ROCm uses our FP64 Jacobi kernel when its core fits the
64 KiB policy, otherwise CPU FP64 EVD. The actual EVD backend is returned in
`info.eigh_backend`. Explicit `torch`, `jacobi`, or `cpu` is available. Jacobi
checks its eigen-residual and raises on nonconvergence. No EVD failure triggers
an unreported fallback. These operations can synchronize with the host.

## Verification

CPU CI verifies installation, native-source packaging and dependency-free imports.
It does not validate numerical kernels. On an allocated GPU with the compiler
available, run:

```bash
python -m pytest tests/test_flashtsqr_gpu.py -v
python examples/compress_tensors.py
```

The suite checks QR reconstruction/orthogonality, interleaved handles,
cross-stream lifetime, two-device guarding, spectral correctness, optimal
truncation error against a dense SVD reference, compact/padded/zero-rank outputs,
odd-rank Jacobi, deficient inputs, noncontiguous tensors, and autocast.
The FP32 relative reconstruction/orthogonality gates are 2e-5; dense SVD is a test
reference only. Performance has not been benchmarked for this extraction.

### Validated environment

On 2026-10-31, LUMI Slurm job `23374333` used two allocated MI250X GCDs
(`gfx90a`), Python 3.12.11, PyTorch 2.7.1+rocm6.2.4, ROCm 6.2.41134,
pytest 8.3.5, and GCC 12 as the host C++ compiler. The wheel was installed into
an isolated virtual environment with the container's PyTorch available.
Native compilation was completed on a login node before GPU allocation.

`python -m pytest tests -v --junitxml=gpu-results.xml` completed with
**32 passed, 1 skipped**, followed by a successful run of
`python examples/compress_tensors.py`. The full path was checked at column
counts 1, 16 and 48, and the wide path through 256, including non-power-of-two
leaf counts. The two-device test ran rather than being skipped. This is
correctness validation, not a performance benchmark.

The one skip is the 160-column full/WY test: its shared-memory policy does not
fit MI250X, where the public API uses the wide kernel instead. NVIDIA CUDA and
the large-rank full/WY path were not covered by that MI250X validation.
Subsequent single-H100 validation, including the full/WY path and CUDA shared
memory fix, and dual-L40S validation of full/wide device guards are recorded in
[the NVIDIA guide](nvidia.md). L40S skips the full/WY case because its shared
memory does not fit that policy. Other GPUs, framework versions, and precision
policies remain unverified.
