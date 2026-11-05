# Experimental API reference

Alpha APIs ship in the package and may change between releases. Their markers
are generated from runtime metadata, not manually assigned website labels.
Read [experimental usage and lifecycle](../experimental.md) before enabling them.

## SGLang alpha entrypoints

```{eval-rst}
.. automodule:: loraforge.integrations.sglang_experimental
   :members: install_qwen35_nf4, install_qwen35_gdn, share_qwen35_gdn_storage
```

## Exact NF4 tensor primitive

```{eval-rst}
.. autofunction:: loraforge.ops.nf4.dequantize_nf4
```

## Stability annotations

```{eval-rst}
.. automodule:: loraforge.annotations
   :members: ExperimentalAPI, ExperimentalInfo, ExperimentalAPIWarning
```
