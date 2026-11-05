# Actual rank-32 to rank-4 NF4 serving comparison

**Fixed-work throughput only:** compressed outputs match only 48/96 original
rank32 sequences; the research fast path matches 88/96 compressed-native
sequences. Both numerical gates fail. These timings are not equal-quality
compression or native-equivalent execution speedups.

This is a same-card Qwen3.5-0.8B NF4 inference comparison of native rank-32
adapters and the physically rank-4 files obtained by actually compressing them.
It is distinct from the preceding same-rank backend optimization: the earlier
rank-4 fixtures were independently generated, not compressed rank-32 files.
Four deterministic **synthetic random** source adapters exercise execution;
there is no trained-adapter or equal-quality compression claim.

## Compression and fixture provenance

The CPU preparation uses `benchmarks/qwen35_hybrid.py prepare STUDY --rank 32`.
All source A/B factors physically contain 32 BF16 channels, including GDN b/a;
there is no rank-4 data padded to look like rank 32. External alpha/r stays 1.
The [compression program](../benchmarks/compress_qwen35_fixture.py), on one
allocated GPU, calls the public `truncate_lora(..., keep=4)` implementation.
All 744 factor pairs report **FlashTSQR + Jacobi**; output A/B are stored at
physical BF16 rank 4, with alpha/r still 1. Original files remain unchanged.
The 16-output b/a gates have mathematical matrix rank at most 16 even with 32
LoRA channels; temporary zero output rows satisfy the rank-32 QR dimension
contract and are removed before export. This adds no stored serving rows.
Small-Gram Frobenius identities check the BF16 export against the discarded
spectrum, without materializing full decoder weight updates or substituting QR.

Source factor tensors occupy **173,162,496 bytes**, compressed factor tensors
**21,645,312 bytes**, an exact **8x tensor-storage reduction** before file headers.
The retained spectral-energy fraction has minimum/median/maximum
**0.1612/0.1698/0.5683**. Random spectra are comparatively flat;
these figures are fixture diagnostics, not trained compression quality.
Source and compressed files, dimensions, backend hashes and errors are retained
in the [evidence](qwen35-nf4-rank32-to4-mi250.json).

## Same-card serving protocol

Job **23485422**, **nid005035**, uses one MI250 GCD, small-g, 7 CPUs, 40GB,
a 20-minute cap. Compression occurs once before serving and is excluded from
throughput. The same immutable NF4 block64/double-quant base is used throughout;
all LoRA factors and buffers remain BF16. Prompt IDs, 128 input/64 forced output
tokens, four adapters, request counts 8/32/96 at C1/8/32, three repeats, capacity
five, cold-prefill and decode graphs are matched. The only server-option changes
are actual adapter paths and maximum LoRA rank 32 versus 4. Each arm checks
114/114 actual projection tensors, 336 BF16 buffers, and 168 B buffers with the
expected rank. Fixed-address loading and actual graph execution are audited.

The four sequential arms separate three contributions:

- Native rank32 + stock BNB linear, using the common NF4 loading compatibility.
- Native rank32 + optimized NF4 linear (cached scales/exact dequantization).
- Native compressed rank4 + the same optimized NF4 linear.
- Research Fleet compressed rank4 + the same optimized NF4 linear and prior
  selected tiny-rank fusion/metadata hooks.

| Concurrency | Native32 stock | Native32 optimized | Native4 compressed | Fleet4 compressed |
| --- | ---: | ---: | ---: | ---: |
| 1 | 144.04 | 146.10 | 150.84 | 191.98 |
| 8 | 479.98 | 645.15 | 661.51 | 771.00 |
| 32 | 1565.11 | 2015.74 | 2080.08 | 2321.83 |

Values are median complete-service output tokens/s, including base computation.

| Concurrency | Fleet4 / Native32, same optimized NF4 | Fleet4 / Native32 stock | Rank reduction alone, native4/native32 | Fleet4/native4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.3140x | 1.3329x | 1.0325x | 1.2727x |
| 8 | 1.1951x | 1.6063x | 1.0254x | 1.1655x |
| 32 | 1.1518x | 1.4835x | 1.0319x | 1.1162x |

The first ratio answers compression plus optimized execution with identical
quantized base execution on both sides. The second also includes the NF4
backend optimization; it must not be labeled a compression-only gain.
An eightfold reduction in LoRA tensor storage does not imply eightfold model
throughput. Here the native rank reduction alone changes complete throughput
only modestly; base operations, launches and other fixed work remain.

## Output comparisons and limitations

Each arm repeats its own serial and mixed-request checks and rejects null or
nonfinite saved logprobs before timing. Rank32 old/new backend comparisons are
checked independently from compression. Token and saved-logprob comparisons:

| Comparison | Identical token sequences | Maximum common-prefix logprob difference | Strict 0.05 gate |
| --- | ---: | ---: | --- |
| Rank32 new/stock | 96/96 | 0.000000000 | pass |
| Native4 compressed / Native32 | 48/96 | 1.041455626 | FAIL |
| Fleet4 compressed / Native32 | 48/96 | 1.041455626 | FAIL |
| Fleet4 / Native4, same compressed files | 88/96 | 0.083525196 | FAIL |

Timings use the same forced amount of work even when outputs differ; numerical
failures are preserved. Cross-rank output agreement on this small synthetic
fixture does not validate trained-adapter quality. The experiment has one GPU
allocation and three short repeats, not a statistical confidence interval or
universal speedup. Previous limitations of the original/unfused BF16 baseline
remain recorded in the NF4 guide and are not resolved by this rank study.

CPU checks: 172 passed, 3 skipped, 49 GPU tests deselected; Ruff and isolated
wheel/sdist builds pass. The public CLI admits compact rank32 alongside rank4
for this NF4/cold-graph benchmark; this does not broaden installed runtime
support or enable the research fast hooks by default.
