# Whole-model BF16 LoRA training experiment

This experiment attaches the [MLP training consumer fusion](lora-swiglu-training.md)
to the complete Qwen3.5-0.8B text model. It measures real optimizer steps, not
manual CUDA-graph replay. It is a standalone first-order training feasibility
study with fixed text batches, not a convergence or Unsloth comparison.

Full-model GPU validation remains pending. CPU checks pass (199 passed, three
skipped, 49 GPU cases deselected), including actual PEFT parameter ownership,
checkpointed updates and numerical-comparison failure gates. Job `23491457`
was cancelled before starting, with zero allocated GPU time, after waiting for
priority. Its replacement `23492718` requests a 10-minute main comparison;
the optional complete-model compiler worker is deferred. No MLP-only timing
is presented here as a measured full-model training result.

## Model and parameter contract

The original local Hugging Face checkpoint is extracted without numerical
transformation by `benchmarks/prepare_qwen35_training.py`. Only the known
`model.language_model.` prefix changes to `model.`. Vision/MTP tensors are
explicitly excluded for text-only training; all remaining names/shapes must
match a meta-instantiated `Qwen3_5ForCausalLM`. Every one of the 320 serialized
text tensors is checked against its original value. The original shard SHA256
is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`;
the text-only checkpoint is
`aa364f2a0e1d96b7ffb69e52bbdd4f1f07634ce27d10d3686ffff0c90824f7dd`.
These are original unmerged HF weights, not the serving checkpoint's fused
names. Each GPU worker additionally audits the actual loaded tensors, missing
and unexpected keys, and the tied embedding/output-head storage.

The complete decoder has 24 layers: 18 GDN and six full-attention layers. Its
native GDN chunk operation must resolve to the installed isolated FLA training
implementation, with an available causal convolution; the benchmark rejects a
silent Torch fallback. SDPA remains the configured full-attention interface.

PEFT wraps all text linear projections except the output head with ordinary
rank-4 LoRA, alpha=4, dropout=0, FP32 master factors and BF16 autocast compute.
The base stays frozen. `qwen35_training_mlp.py` replaces only the 24 MLPs,
preserving the actual six PEFT Parameter objects per MLP and their scaling.
GDN and attention adapters retain ordinary PEFT execution. Stable canonical
names and a complete initial adapter hash must agree across workers.
Concatenated gate/up base weights belong to the replacement; discarded modules
are not retained, and no inactive baseline model occupies the same process.

## Comparisons and timing

Each path runs in a separate process on the same allocated GCD:

| Worker | MLP path | Other model execution |
| --- | --- | --- |
| `peft-before` / `peft-after` | Unmodified PEFT MLP | Native HF/PEFT; bracketing controls |
| `packed` | Frozen gate/up concatenation plus ordinary Torch LoRA/autograd | Native HF/PEFT |
| `compiled-mlp` | Same packed MLP compiled with `fullgraph=True, dynamic=False` | Native HF/PEFT |
| `fused` | Consumer fusion, separate scaled/unscaled backward buffers | Native HF/PEFT |
| `compact` | Consumer fusion with fresh gradient-buffer reuse | Native HF/PEFT |
| `compiled-model` | Ordinary PEFT; complete model compiled with `fullgraph=False` | Record graph breaks and any compiler failure |

The `packed` control separates base-projection concatenation and PEFT casting
overheads from consumer-kernel gains. A `compiled-mlp` result is **not** a
fully compiled model result. The full-model compiler worker has a bounded
timeout and tests without checkpointing; a failed or timed-out compiler cannot
be presented as a successfully measured optimized baseline.

Batch one, sequence lengths 128/1024/4096, gradient accumulation one. Each
required worker tests both no checkpointing and non-reentrant decoder-layer
activation checkpointing. Cache reuse is disabled. Timing includes
`zero_grad(set_to_none=True)`, all decoder/GDN/attention operations, the tied
full-vocabulary LM head, native causal language-model loss, backward, and a
foreach FP32 AdamW update at learning rate 1e-4. Two warmups precede five real
timed updates. The primary time is synchronized wall-clock time; CUDA events
are recorded too. Tokenization, on-device batch preparation, model loading,
compilation, audits and correctness artifact copies are excluded from timing.

Peak allocated/reserved values are PyTorch allocator measurements, not total
device memory. Each process has only its active model, parameters and optimizer.
The same authored repeated text and labels are used everywhere; neither random
text repetition nor three optimizer updates establish SFT task quality.

## Numerical checks

Each shape/checkpoint pair first performs three updates from identical
zero-B adapter initialization with a fresh AdamW optimizer. It saves every
LoRA parameter gradient from the third backward, the updated parameters, the
three losses, and sampled logits (positions 0/middle/penultimate, vocabulary
stride 257). All trainable gradients and timed losses must be finite. Parameters
are reset before the separate warmup and timed-update trajectory.

`analyze_qwen35_training.py` compares on CPU using predeclared gates:

- Maximum absolute loss drift across the three steps: 0.01.
- Relative L2 drift of sampled logits: 0.02.
- Relative L2 drift of **each** LoRA parameter gradient: 0.05.
- Maximum absolute drift of updated parameters: 1e-4.

Observed error, exact tensor counts, failed tensor names and timing are
recorded separately. A speed result does not override a failed numerical gate.
Direct PEFT-versus-fused checks and packed-versus-fused checks are both needed;
the latter alone cannot establish equivalence to the ordinary PEFT model.

## Environment and reproduction

LUMI small-g, one MI250 GCD, seven CPUs, 40GB host memory and a 20-minute cap.
Torch 2.11.0 ROCm 7.2, Transformers 5.10.1, PEFT 0.21.1. FLA 0.6.0 pure-Python
wheels are extracted only into the test directory and prepended to PYTHONPATH;
the existing Torch/Triton/SGLang installation is unchanged. Dependency receipts
pin both wheels:

- `fla_core-0.6.0-py3-none-any.whl`:
  `b0708b70e0b119cd58a0a5146ea56cb175cef6ed969d8ac60267e3e077ce3ebd`
- `flash_linear_attention-0.6.0-py3-none-any.whl`:
  `6dec9735f8454523c13553622e98aff30152dde6586b4ac7d88010fbb7a06b5f`

CPU preparation, without exposing the FLA GPU-only constructor:

```bash
python benchmarks/prepare_qwen35_training.py ORIGINAL_HF_MODEL BF16_TEXT
```

Then on an allocated gfx90a GPU with the isolated dependencies available:

```bash
PYTHONPATH=/absolute/path/to/deps:/absolute/path/to/benchmarks:$PYTHONPATH \
  python benchmarks/run_qwen35_training.py BF16_TEXT --output RESULTS
```

When a shorter allocation is needed, `--skip-whole-compile --max-seconds 520`
keeps all six required workers, token lengths and checkpoint settings, and
omits the separate optional whole-model compilation worker. Its measurement
still includes the compiled-MLP baseline; it does not establish a comparison
against compiling the complete model. That comparison needs a separate run.
`--whole-compile-only` runs ordinary PEFT before/after controls around the
complete-model compiler, all without checkpointing, and requires the compiler
worker to complete for that run to count as complete. These workers still run
in separate processes and measure identical full optimizer steps.

CPU comparison after collection:

```bash
python benchmarks/analyze_qwen35_training.py RESULTS --output comparison.json
```

No training default or public serving integration is changed by these tools.
