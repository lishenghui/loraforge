# Serving integrations and row operators

These entrypoints are explicit process-local opt-ins. Read the
[SGLang contract](../sglang.md) before model construction.

```{eval-rst}
.. automodule:: loraforge.integrations.sglang
   :members: install, install_sglang_backend, get_backend
```

## Experimental vLLM linear decode

Read the [vLLM contract](../vllm.md). Registration is inert unless explicitly
enabled, pins vLLM 0.26.0 and retains native fallback execution.

```{eval-rst}
.. automodule:: loraforge.integrations.vllm
   :members: install_vllm_backend

.. automodule:: loraforge.ops.vllm_gemv
   :members: shrink, expand
```

## Experimental retained-rank BF16 row operators

Read the [metadata, rounding and lifetime contract](../row-operators.md).
These tensor operators do not install a serving backend or adapter manager.

```{eval-rst}
.. automodule:: loraforge.ops.lora_rows
   :members: row_shrink, row_expand
```
