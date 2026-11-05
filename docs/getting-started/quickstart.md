# Quickstart

## Allocate ranks on CPU

Choose directions using spectral energy without importing a tensor library:

```python
from loraforge.compression import allocate_spectral_budget

ranks = allocate_spectral_budget(
    {"query": [5.0, 2.0], "value": [3.0, 1.0]},
    budget=2,
)
assert ranks == {"query": 1, "value": 1}
```

For heterogeneous resource costs and measured calibration losses, use
[quality-budget allocation](../allocation.md).

## Compress a local PEFT adapter

On a supported GPU, compress physical factors into a new directory:

```python
from loraforge.adapters import compress_peft_adapter

result = compress_peft_adapter(
    "adapter-original",
    "adapter-r4",
    keep=4,
    device="cuda:0",
)
```

The default accelerated backend is FlashTSQR. Adapter scaling, including rsLoRA
when present, is preserved independently of retained rank. Input files remain
unchanged; compact output stores the retained tensor dimensions. See the
[PEFT contract](../peft.md) for supported modules, manifests and reload checks.

## Enable an alpha serving path

In each SGLang worker, before model construction:

```python
from loraforge.integrations.sglang_experimental import install_qwen35_nf4

install_qwen35_nf4(mode="byte-pairs")
```

This path requires the prepared Qwen3.5-0.8B NF4 fixture, TP1, gfx90a,
SGLang 0.5.22 and bitsandbytes 0.50.2. For spawned processes, use the documented
plugin environment options in [experimental APIs](../experimental.md).
The marker describes interface stability; it does not establish model quality.

## Read the contracts

- [Tensor refactoring and truncation](../flashtsqr.md)
- [Compact residency and lifetime](../runtime.md)
- [Serving integration and validation](../sglang.md)
- [Generated API signatures](../reference/index.md)
