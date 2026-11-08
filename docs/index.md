---
html_theme.sidebar_secondary.remove: true
---

<div class="lf-kicker">LoRA compression &amp; execution</div>

# Smaller adapters. Explicit control.

<div class="lf-intro">Compress LoRA factors, allocate a rank budget, and connect
optional serving kernels. Built as a Python library with clear ownership,
scaling, and hardware contracts.</div>

<div class="lf-actions">

:::{button-ref} getting-started/installation
:color: primary

Install LoRAForge
:::

:::{button-ref} getting-started/quickstart
:color: primary
:outline:

Quickstart
:::

:::{button-ref} reference/index
:color: secondary
:outline:

API reference
:::

</div>

<div class="lf-status"><strong>Development release.</strong> Experimental APIs
are explicitly marked alpha. Installation is from this repository; hardware
coverage and numerical results are documented per feature. The footer identifies
the documented source snapshot, including APIs not yet on the default branch.</div>

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Compress adapters
:link: peft
:link-type: doc

Apply spectral compression to PEFT LoRA factors and export physically compact
files while preserving adapter scaling.
:::

:::{grid-item-card} Plan a budget
:link: allocation
:link-type: doc

Allocate retained directions or caller-supplied quality curves with deterministic
CPU planners. No GPU needed.
:::

:::{grid-item-card} Serve with explicit opt-ins
:link: experimental
:link-type: doc

Connect SGLang and explore packaged NF4/GDN experiments with clear version,
initialization, and hardware boundaries.
:::
::::

## Start with one adapter

Install matching GPU dependencies, then compress a local adapter:

```python
from loraforge.adapters import compress_peft_adapter

result = compress_peft_adapter(
    "adapter-original",
    "adapter-r4",
    keep=4,
    device="cuda:0",
)
```

FlashTSQR is the default compression backend. The output directory is new;
input files stay unchanged. Compression quality depends on the adapter and
retained rank: measure it on your task before deployment.

## Find your path

| Goal | Start here |
| --- | --- |
| Install the library and choose dependencies | [Installation](getting-started/installation.md) |
| Run a CPU planner or compress PEFT files | [Quickstart](getting-started/quickstart.md) |
| Understand tensor ownership and scaling | [User guides](guides/index.md) |
| Inspect signatures and experimental status | [API reference](reference/index.md) |
| Evaluate the recorded measurements | [Experiments](experiments/index.md) |
| Contribute or build this site | [Development](development/index.md) |

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
guides/index
reference/index
experiments/index
development/index
```
