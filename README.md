# LoRAForge

Building blocks for LoRA compression and efficient execution.

LoRAForge is being developed as a reusable library from the
[probe research repository](https://github.com/lishenghui/probe).
The library provides GPU FlashTSQR and tensor-level spectral LoRA compression.
It includes PEFT adapter I/O and compact export for linear LoRA and rsLoRA,
and is in early development. Serving integrations remain planned; see
[the migration plan](docs/migration.md).

## Install from source

Python 3.10 or newer is required. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -c "import loraforge; print(loraforge.__version__)"
```

The foundation has no runtime dependencies. Importing `loraforge` does not load
PyTorch, compile kernels, or initialize a GPU. Install from this repository;
no PyPI release is advertised at this stage.

## Tensor compression

Install a PyTorch build and compiler matching your CUDA/ROCm environment first,
then install the optional dependencies with `python -m pip install ".[gpu]"`.
Native sources ship in the wheel and compile lazily on the first GPU operation.
No GPU is needed to build or install the Python wheel.

```python
from loraforge.compression import truncate_lora

# a: [batch, rank, in_features], b: [batch, out_features, rank], on the same GPU
result = truncate_lora(a, b, keep=8)
# result.a / result.b have compact rank-8 shapes; inputs remain unmodified.
print(result.info.qr.path, result.info.eigh_backend)
```

For a complete synthetic example, see
[`examples/compress_tensors.py`](examples/compress_tensors.py).
[FlashTSQR contracts and validation](docs/flashtsqr.md) describe devices, dtypes,
stream ownership, numerical limits, and tested environments.

For NVIDIA CUDA installation and hardware validation, see
[the NVIDIA guide](docs/nvidia.md).

## Adapter files

For local adapter files, see [the PEFT workflow](docs/peft.md):

```python
from loraforge.adapters import compress_peft_adapter

result = compress_peft_adapter("original", "compressed", keep=8, device="cuda:0")
```

Output uses a new directory and preserves per-layer effective scaling. Install
`.[adapters,gpu]` in your configured GPU environment for this workflow.

## Adapter residency

`loraforge.runtime.AdapterSlots` loads compact PEFT factors into owned CPU slots
with byte limits and generation-checked handles. See [runtime contracts](docs/runtime.md).
`GPUAdapterSlots` adds compact GPU residency with event-ordered loading and
scoped stream leases. Serving integration and graph replay remain planned.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and checks,
[the architecture](docs/architecture.md) for module boundaries, and
[the migration plan](docs/migration.md) for the order of implementation.
[Provenance](docs/provenance.md) records the research baseline and imported code.

## License

The new foundation is licensed under [Apache-2.0](LICENSE).
Future code imports must retain their applicable licenses and attribution.
