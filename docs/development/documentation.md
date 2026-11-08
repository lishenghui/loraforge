# Documentation development

The site uses Sphinx, MyST Markdown and the PyData Sphinx Theme. Existing guide
and experiment Markdown files remain authoritative; do not maintain separate
copies for the website. API reference pages use installed package docstrings.

## Build and preview

Use Python 3.11 or newer; CI builds with Python 3.12:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8765 --directory docs/_build/html
```

Open `http://localhost:8765`. Generated HTML stays under the ignored `_build`
directory. No GPU, PyTorch, SGLang or Triton installation is needed to build.
Warnings fail the build. The documentation workflow retains a downloadable HTML
artifact for review before deployment.

## Add a guide or API

Add Markdown under `docs/` and link it from the appropriate section's toctree.
Use MyST directives for navigation and Sphinx Design components. Public Python
APIs are documented through `automodule` / `autofunction`; do not copy their
signatures by hand. `ExperimentalAPI` metadata renders as an alpha admonition
automatically, with the declared reason and version.

Keep support scope and quality evidence accurate. In particular, preserve failed
numerical controls and distinguish measured fixed-work throughput from quality.
Run the strict build after changing documentation or public docstrings.

## Publication

GitHub Pages serves the built static site. The documentation workflow builds on
pushes to `main`, pull requests and manual requests. Only the repository's `main` and `docs/site` branches
can deploy: `main` is the normal publication channel; `docs/site` is an explicit
development snapshot available through a manual workflow run before a feature
branch merges. Footer links identify
the exact published source revision. Neither publication path merges a code PR.

## Tooling references

Ray uses Sphinx with MyST and the PyData theme in its
[documentation configuration](https://github.com/ray-project/ray/blob/master/doc/source/conf.py).
PyTorch uses Sphinx with its own `pytorch_sphinx_theme2` in its
[configuration](https://github.com/pytorch/pytorch/blob/main/docs/source/conf.py).
Unsloth uses GitBook, as described in the
[GitBook customer story](https://www.gitbook.com/customers/unsloth-ai).
LoRAForge uses Sphinx so narrative Markdown and API docstrings can be built and
reviewed together in this repository.
