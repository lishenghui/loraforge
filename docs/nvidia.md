# NVIDIA CUDA

Use Python 3.10 or newer, a CUDA-enabled PyTorch build (2.7 or newer),
the matching CUDA toolkit including `nvcc`, and a compatible host C++ compiler.
The NVIDIA driver alone is insufficient for the lazy native extension build.
On a cluster, allocate a GPU before running the numerical tests.

## Installation

For the CUDA 12.6 / PyTorch 2.7.1 combination:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[gpu,dev,adapters,test-adapters]"
nvcc --version
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

Keep the virtual environment activated so the extension loader can find `ninja`.
Load your site's CUDA/compiler modules before activating it. If the node's
temporary filesystem is full, set `TMPDIR` to a writable location with space
before installation and compilation. These are caller environment settings;
LoRAForge does not change global paths, compilers, or precision settings.

The first operation compiles the packaged FlashTSQR sources. `MAX_JOBS=2`
can limit compilation memory usage. PyTorch normally targets the visible GPU;
to share a cache between L40S and H100 nodes, explicitly set
`TORCH_CUDA_ARCH_LIST="8.9;9.0"` before the first operation.

## Validation

Run on allocated hardware from the checkout, with the environment activated:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -v
python examples/compress_tensors.py
```

Two visible GPUs are required to exercise the device-guard test; a single GPU
skips it. Tests create small models locally and download no model checkpoints.
FlashTSQR remains the QR backend; CUDA's automatic EVD uses
`torch.linalg.eigh`. Explicit CPU EVD is reported separately in result metadata.
FP32 matmul follows the caller's TF32 policy. The recorded numerical tolerances
do not establish correctness under other precision settings.

## Validated environment — 2026-11-03

Slurm job `7542015` ran on one NVIDIA H100 NVL (compute capability 9.0,
95,830 MiB device memory) on UPPMAX Pelle node `p206`. The environment was
Python 3.12.3, PyTorch 2.7.1+cu126, CUDA toolkit 12.6.20, NVIDIA driver
615.71.09, GCC 13.3.0, pytest 9.2.0, PEFT 0.18.0 and Transformers 4.57.6.
The site modules were `Python/3.12.3-GCCcore-13.3.0` and `CUDA/12.6.0`.
Compilation used `MAX_JOBS=2` and `TORCH_CUDA_ARCH_LIST="8.9;9.0"`.
`torch.backends.cuda.matmul.allow_tf32` was false; cuDNN's TF32 setting was true.

The validation commands above completed with **123 passed, 1 skipped** in
124.18 seconds for pytest, followed by a successful compression example using
FlashTSQR's full path and torch EVD. The only skip required two visible GPUs.
Coverage includes full/wide QR, the rank-160 full/WY path, FP32/FP16/BF16 LoRA
compression, all explicit EVD backends, PEFT/rsLoRA and budget roundtrips,
and cross-stream GPU residency and slot reuse. QR reconstruction/orthogonality
use relative gates of 2e-5; tensor test shapes and dtype cases are recorded in
`tests/test_flashtsqr_gpu.py`.

The shared-memory regression reconstructs a `[1, 12500, 1]` FP32 tensor through
the wide kernels and applies Q to `[1, 1, 12500]` right-hand sides. These require
scratch above CUDA's default 48 KiB. Before the fix, this reproduced
`CUDA error: invalid argument` on the H100. The repaired kernels pass and reject
scratch beyond the device limit without invalidating subsequent handle use.

CPU validation passed 78 tests. Lint/format, wheel and sdist builds passed;
fresh wheel and wheel-from-sdist environments without PyTorch each passed
23 tests and skipped the four dependency-requiring test modules.
L40S and dual-device validation is recorded below. Other NVIDIA cards and
precision settings, performance, and a new ROCm regression run remain unverified.

## Dual L40S validation — 2026-11-03

Pelle Slurm job `7542157` ran on two allocated NVIDIA L40S GPUs on node `p204`.
Both devices reported compute capability 8.9 and 46,068 MiB via `nvidia-smi`,
with driver 615.71.09. Python/PyTorch/toolkit/compiler/framework versions,
compilation flags and FP32 matmul TF32 policy matched the H100 environment above.

With both devices visible, the commands were:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -v \
  --basetemp="$TMPDIR/pytest-dual-$SLURM_JOB_ID" \
  --junitxml="dual-l40s-$SLURM_JOB_ID.xml"
python examples/compress_tensors.py
```

The full suite completed with **126 passed, 1 skipped** in 14.06 seconds,
followed by a successful compression example. The only skip was rank-160 full/WY,
whose shared-memory policy does not fit L40S; the rank-256 automatic/wide test
passed. Both shared-memory regression tests passed on L40S as well.

The dual-device tests ran without skips: full and wide QR factorizations on
`cuda:1` while `cuda:0` was current preserved the input device and numerical
results, and rejected a right-hand side on the wrong device. LoRA refactoring
on the second GPU preserved its device and reconstructed the dense update.
Two new residency cases loaded, borrowed, evicted and collected slots on each
GPU while the other GPU was current; device placement, values and restoration
of the caller's current device passed. These are same-process device-guard
and residency checks, not distributed training or collective communication tests.

This run establishes the tested H100-single and L40S-dual combinations only;
dual-H100, mixed GPU pairs, distributed execution and performance remain unverified.
