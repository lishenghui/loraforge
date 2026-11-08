# Experiments and evidence

These reports preserve accepted measurements, failed attempts, hardware scope
and numerical limits. Historical throughput numbers are fixed-work diagnostics;
they do not establish trained-adapter quality or a universal speedup.

:::{important}
The existing rank32-to-rank4 synthetic comparison and fast-versus-native
comparison retain their documented quality/numerical failures. API alpha status
and successful incremental comparisons do not change those results.
:::

```{toctree}
:maxdepth: 1

../qwen35
../qwen35-gdn
../qwen35-fused-lora
../qwen35-high-concurrency
../qwen35-nf4
../qwen35-nf4-rank-comparison
../qwen35-resources
../qwen35-storage-nf4
```

## Standalone training experiments

These programs remain research harnesses rather than installed Trainer APIs.
Their reports distinguish original-reference comparisons, rejected numerical
paths, capacity-only results and untested convergence/task quality.

```{toctree}
:maxdepth: 1

../lora-training-transfer
../lora-swiglu-training
../qwen35-training
../qwen35-training-autograd
../qwen35-training-4b
../qwen35-training-batches
../qwen35-training-loss
../qwen35-training-head
```
