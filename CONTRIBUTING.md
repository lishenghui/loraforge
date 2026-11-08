# Contributing

Keep each change focused on one usable capability, with a small example and
tests for its public behavior. Start API or backend changes with an issue or PR
explaining the input/output contract and supported environment.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Check a change

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
```

CI also installs the wheel into a fresh environment and tests it there, then
repeats this with a wheel rebuilt from the source archive. The `src` layout
ensures repository-root imports cannot hide missing installation files.

To reproduce the wheel check locally after building:

```bash
python -m venv /tmp/loraforge-wheel-check
/tmp/loraforge-wheel-check/bin/python -m pip install pytest dist/*.whl
/tmp/loraforge-wheel-check/bin/python -m pytest
```

The CPU CI checks need no GPU. GPU functionality must include numerical and
lifecycle tests on its target hardware before being described as supported.
Report the hardware, Python/PyTorch/framework versions, dtype, shape, and exact
commands for hardware validation. Do not infer support for untested devices.
For the FlashTSQR numerical suite and current coverage, see
[docs/flashtsqr.md](docs/flashtsqr.md).

For CPU PEFT file/reload tests, install `.[adapters,test-adapters]` in an
environment with the appropriate PyTorch build, then run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest \
  tests/test_peft_adapters.py tests/test_package.py -m "not gpu"
```

The tests create small models locally and download no model checkpoints. On an
allocated GPU, `python -m pytest tests/test_peft_adapters.py -m gpu -v` checks
the actual FlashTSQR-to-file-to-PEFT path, including grouped modules.

## Code and API conventions

Public experimental APIs belong in the installed package and use
`loraforge.annotations.ExperimentalAPI` with an English reason and documented
support boundaries. Use warning wrappers only at explicit installation
entrypoints, keeping numerical operations metadata-only. See
[experimental APIs](docs/experimental.md) for lifecycle and validation rules.

Documentation uses Sphinx, MyST and the PyData theme. Build with
`python -m sphinx -W --keep-going -b html docs docs/_build/html` after installing
`.[docs]` with Python 3.11 or newer. Warnings fail CI. Edit the existing Markdown
guides and public docstrings directly; the website does not maintain duplicate
copies. See [documentation development](docs/development/documentation.md).

- Keep imports lightweight. Compile and initialize devices only when explicitly
  requested by an operation.
- Pass devices and configuration explicitly; avoid cluster paths and global
  environment changes. Optional integrations must not become core dependencies.
- Keep tensor operations independent of serving-framework internals. Put
  framework-specific conversion and registration in the integration layer.
- Use meaningful names and type annotations for new public interfaces. Document
  shapes, scaling, ownership, stream behavior, and numerical tolerances where
  relevant.
- Preserve applicable licenses and authorship when moving existing code. Add a
  record to [provenance](docs/provenance.md) in the same PR as an import.
- Benchmark against a named baseline with correctness checks and reproducible
  inputs. Separate operator timings from end-to-end serving measurements.

Use short-lived branches and ordinary PRs. Include the problem, resulting
behavior, validation, and any remaining hardware limitations in the PR body.
