# Installation

LoRAForge is in early development. Install from the repository; no PyPI release
is advertised. The library supports Python 3.10 or newer.

## Base package

```bash
git clone https://github.com/lishenghui/loraforge.git
cd loraforge
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The base package has no runtime dependencies. CPU allocation planners are
available immediately:

```python
from loraforge.compression import allocate_spectral_budget

ranks = allocate_spectral_budget({"query": [5.0, 2.0], "value": [3.0, 1.0]}, budget=2)
assert ranks == {"query": 1, "value": 1}
```

## Compression and PEFT files

First install a PyTorch build and compiler matching your CUDA or ROCm environment.
Then add the optional library dependencies:

```bash
python -m pip install ".[adapters,gpu]"
```

Native compression sources ship with the package and compile lazily on first
use. FlashTSQR needs an allocated GPU; there is no silent CPU QR fallback.
See [FlashTSQR contracts](../flashtsqr.md) and the
[NVIDIA guide](../nvidia.md) for tested combinations.

## Serving integrations

Install a compatible serving environment separately. The current SGLang bridge
requires exactly SGLang 0.5.22; optional integrations do not become base-package
dependencies. Start with the [SGLang guide](../sglang.md). Packaged Qwen3.5
NF4/GDN entrypoints are alpha APIs with narrower runtime contracts:
[experimental APIs](../experimental.md).

## Documentation tools

Documentation builds use Python 3.11 or newer; CI uses Python 3.12.

```bash
python -m pip install ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

This installs documentation tools and the lightweight package, without PyTorch,
Triton, SGLang or a GPU. See [documentation development](../development/documentation.md).
