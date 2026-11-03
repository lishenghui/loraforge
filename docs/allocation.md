# Budget allocation

Both planners run on CPU without tensor or serving dependencies. They consume
caller-supplied spectra or calibration curves; they do not measure task quality.

## Spectral directions

`allocate_spectral_budget(spectra, budget, min_rank=0)` maximizes retained
unscaled spectral energy with equal direction costs. It returns prefix ranks
summing exactly to the integer budget. Spectra must be descending, finite and
nonnegative. Ties use item name and direction index. `min_rank` reserves a prefix
for every item. The PEFT compression API's `budget=` workflow uses this planner
with `min_rank=1`; see [PEFT](peft.md).

## Quality curves and heterogeneous costs

```python
from loraforge.compression import QualityOption, allocate_quality_budget

curves = {
    "adapter_a": [
        QualityOption(rank=2, cost=10, loss=0.4),
        QualityOption(rank=4, cost=20, loss=0.1),
    ],
    "adapter_b": [
        QualityOption(rank=2, cost=15, loss=0.3),
        QualityOption(rank=4, cost=30, loss=0.0),
    ],
}
ranks = allocate_quality_budget(curves, budget=35)
# {"adapter_a": 4, "adapter_b": 2}; total cost 35, summed loss 0.4
```

Exactly one candidate per item is chosen. Integer costs may represent measured
compact tensor bytes or another additive resource; units must be consistent.
The objective minimizes summed finite losses, then total cost, then ranks in
sorted item-name order. Candidate ranks must be unique per item. Empty curves
and infeasible budgets raise ValueError. Nonmonotonic losses and zero-cost
options are supported; no interpolation or monotonic fitting is applied.

Losses must be comparable across items. This additive objective does not imply
that independently calibrated per-layer losses predict a composed model's task
quality. Callers supply an appropriate calibration protocol and evaluation.

The sparse multiple-choice knapsack solver prunes dominated states and performs
no cost rounding. Large reachable-cost frontiers can consume substantial CPU
memory and time. It is intended for finite offline candidate sets. Quality
plans are independent of the PEFT automatic spectral workflow; a compatible
positive-rank module plan can be supplied through its existing `keep` mapping.
