![LoRAForge](docs/_static/loraforge-banner.png)

# LoRAForge

[![arXiv](https://img.shields.io/badge/arXiv-2611.01256-b31b1b.svg)](https://arxiv.org/abs/2611.01256)

**LoRA compression and efficient multi-LoRA execution.**

LoRAForge compresses trained LoRA adapters into smaller factors and provides
GPU operators and serving integrations for executing them. It combines
FlashTSQR factorization, spectral rank allocation, and compact adapter storage
to reduce adapter memory and preparation overhead. Compression operates on
existing adapter weights without retraining.

- **Compress tensors and adapter files.** Truncate LoRA factors on the GPU and
  export compact PEFT adapters while preserving LoRA and rsLoRA scaling.
- **Allocate ranks across modules.** Choose a fixed rank, per-module ranks, or a
  total rank budget. CPU planners also support measured quality curves and
  heterogeneous resource costs.
- **Manage adapter memory.** Load compact factors into CPU or GPU slots with
  byte limits, generation-checked handles, and stream-aware GPU lifetimes.
- **Integrate with serving engines.** Opt-in SGLang and vLLM backends provide
  specialized LoRA execution. Experimental Qwen3.5 APIs cover hybrid adapters,
  NF4 bases, and GDN storage sharing.

[Documentation](https://lishenghui.github.io/loraforge/) ·
[Paper](https://arxiv.org/abs/2611.01256) ·
[Examples](examples/) ·
[Issues](https://github.com/lishenghui/loraforge/issues)

## Installation

Use **Python 3.10 or newer** and install from this repository:

```bash
git clone https://github.com/lishenghui/loraforge.git
cd loraforge
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The base package has no runtime dependencies. CPU rank allocation works without
PyTorch; importing `loraforge` does not initialize a GPU or compile kernels.

For GPU compression and PEFT adapter files, first install a PyTorch build and
CUDA or ROCm compiler matching your GPU environment, then install:

```bash
python -m pip install ".[adapters,gpu]"
```

Native sources are included in the package and compile on the first GPU
operation. FlashTSQR is the default accelerated compression backend. See the
[GPU requirements](docs/flashtsqr.md) and [NVIDIA setup guide](docs/nvidia.md)
for tested hardware and toolchains.

## Quickstart

### Allocate a rank budget on CPU

Given each module's descending singular values, distribute a total number of
retained directions to maximize retained spectral energy:

```python
from loraforge.compression import allocate_spectral_budget

ranks = allocate_spectral_budget(
    {"query": [5.0, 2.0], "value": [3.0, 1.0]},
    budget=2,
)
print(ranks)  # {"query": 1, "value": 1}
```

Here the budget is the sum of retained ranks. For allocation using measured
quality losses or resource costs, see [budget allocation](docs/allocation.md).

### Compress LoRA tensors on a GPU

This example creates four rank-16 adapters and compresses each to rank 8.
It requires the GPU dependencies and a supported GPU with its compiler available:

```python
import torch

from loraforge.compression import truncate_lora

# PyTorch uses the "cuda" device name for both CUDA and ROCm GPUs.
a = torch.randn(4, 16, 128, device="cuda:0")  # [batch, rank, input]
b = torch.randn(4, 96, 16, device="cuda:0")  # [batch, output, rank]

result = truncate_lora(a, b, keep=8)
print(result.a.shape)  # torch.Size([4, 8, 128])
print(result.b.shape)  # torch.Size([4, 96, 8])
print(result.info.qr.path, result.info.eigh_backend)
```

The returned factors approximate `b @ a`; inputs remain unchanged. Outputs are
compact float32 tensors. External adapter scaling stays with the caller; the
file workflow below handles scaling in the exported adapter configuration.
See [tensor compression](docs/flashtsqr.md) for shapes, precision, and execution
contracts, or run `python examples/compress_tensors.py`.

### Compress a PEFT adapter

Prepare a local adapter directory containing `adapter_config.json` and
`adapter_model.safetensors` with supported linear LoRA or rsLoRA weights.
With the GPU and adapter dependencies installed:

```python
from loraforge.adapters import compress_peft_adapter

result = compress_peft_adapter(
    "original_adapter",
    "compressed_adapter",
    keep=8,
    device="cuda:0",
)
print(result.directory)
```

Replace `original_adapter` with your input directory. Each source module must
have rank at least 8. The output directory must not exist, and its parent must
exist. The input remains unchanged; output contains compact adapter weights,
a PEFT configuration preserving effective scaling, and `loraforge_manifest.json`
with ranks, file hashes, and compression metadata.

Use `budget=128` instead of `keep=8` to distribute 128 retained directions
across modules, reserving at least one per module. See the
[PEFT guide](docs/peft.md) for supported checkpoint formats, per-module ranks,
and export options.

## Serving and runtime

Install serving frameworks in their own compatible environments. The integrations
are opt-in and require the versions below; installing LoRAForge alone does not
activate them.

| Component | Available functionality | Guide |
| --- | --- | --- |
| SGLang 0.5.22 | Linear LoRA decode and retained-rank execution paths | [SGLang](docs/sglang.md) |
| vLLM 0.26.0 | Experimental specialized linear LoRA decode backend | [vLLM](docs/vllm.md) |
| Qwen3.5 on SGLang 0.5.22 | Experimental hybrid adapter bridge and NF4/GDN APIs | [Qwen3.5](docs/qwen35.md), [experimental APIs](docs/experimental.md) |
| Adapter residency | `AdapterSlots` and `GPUAdapterSlots` for compact factor ownership and loading | [Runtime](docs/runtime.md) |
| Retained-rank operators | BF16 `row_shrink` and `row_expand` with compact B factors | [Row operators](docs/row-operators.md) |

The guides provide activation commands, supported model and hardware
configurations, and validation results. Alpha entrypoints are marked with
`ExperimentalAPI`. GPU residency slots allocate new storage on replacement;
they are not fixed-address pools for graph replay. The retained-rank row
operators are separate from the vLLM integration.

## Documentation and development

The [documentation website](https://lishenghui.github.io/loraforge/) contains
installation guides, API references, examples, and validation details.
[CONTRIBUTING.md](CONTRIBUTING.md) covers development setup and checks;
[architecture](docs/architecture.md) describes the package layout.

To build the documentation locally, use Python 3.11 or newer:

```bash
python -m pip install ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## License

LoRAForge is licensed under [Apache-2.0](LICENSE). Source attribution and
third-party notices are documented in [provenance](docs/provenance.md) and the
corresponding source files.

## Citation

If you use LoRAForge in your research, please cite our paper:

Shenghui Li and Thiemo Voigt. **LoRAForge: Co-Designing Post-Hoc Compression and
Execution for Efficient Multi-LoRA Serving.** arXiv preprint, 2026.
[Paper](https://arxiv.org/abs/2611.01256)

```bibtex
@misc{li2026loraforge,
  title = {{LoRAForge}: Co-Designing Post-Hoc Compression and Execution for Efficient Multi-{LoRA} Serving},
  author = {Shenghui Li and Thiemo Voigt},
  year = {2026},
  eprint = {2611.01256},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2611.01256}
}
```
