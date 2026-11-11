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

GitHub Pages publishes documentation from `main`. Pull requests build a
reviewable HTML artifact without deploying. Manual workflow runs can build any
branch; deployment is restricted to `main`. Footer links identify the exact
published source revision.

The historical migration log remains in `docs/migration.md` for repository
maintenance; it is excluded from the published site and search index. Feature
contracts and validation reports are the current user-facing documentation.
