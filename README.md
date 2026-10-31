# LoRAForge

Building blocks for LoRA compression and efficient execution.

LoRAForge is being developed as a reusable library from the
[probe research repository](https://github.com/lishenghui/probe).
The initial foundation provides packaging, contribution guidance, and CPU CI.
Compression operators and serving integrations are planned; this snapshot does
not yet implement them or claim GPU compatibility.

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

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and checks,
[the architecture](docs/architecture.md) for module boundaries, and
[the migration plan](docs/migration.md) for the order of implementation.
[Provenance](docs/provenance.md) records the research baseline and imported code.

## License

The new foundation is licensed under [Apache-2.0](LICENSE).
Future code imports must retain their applicable licenses and attribution.
