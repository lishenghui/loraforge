# SPDX-License-Identifier: Apache-2.0
"""CPU allocation from caller-provided calibration losses and integer costs."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityOption:
    """One measured candidate; lower loss is better, cost units are caller-defined."""

    rank: int
    cost: int
    loss: float


def allocate_quality_budget(
    curves: Mapping[str, Sequence[QualityOption]], budget: int
) -> dict[str, int]:
    """Choose one rank per item minimizing summed loss within an integer budget.

    Costs may represent bytes or another additive integer resource. Losses must
    be comparable and additive across items; no monotonicity, interpolation or
    isotonic fitting is assumed. Ties prefer lower cost then lexicographic ranks
    in sorted item-name order. Returns ranks; inputs remain unchanged.

    Exact sparse multiple-choice knapsack with dominated-state pruning. Runtime
    and memory depend on the number of reachable costs, so large frontiers can
    be expensive. No cost rounding or silent approximation is performed.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if any(not isinstance(name, str) or not name for name in curves):
        raise ValueError("item names must be nonempty strings")
    names = sorted(curves)
    options = {}
    for name in names:
        row = list(curves[name])
        if not row:
            raise ValueError("each item needs at least one option")
        seen = set()
        for option in row:
            if not isinstance(option, QualityOption):
                raise ValueError("expected QualityOption")
            if type(option.rank) is not int or option.rank < 0:
                raise ValueError("rank must be a nonnegative integer")
            if type(option.cost) is not int or option.cost < 0:
                raise ValueError("cost must be a nonnegative integer")
            if not math.isfinite(option.loss):
                raise ValueError("loss must be finite")
            if option.rank in seen:
                raise ValueError("duplicate candidate rank")
            seen.add(option.rank)
        options[name] = sorted(row, key=lambda option: option.rank)
    states = {0: (0.0, ())}
    for name in names:
        candidates = {}
        for spent, (loss, ranks) in states.items():
            for option in options[name]:
                cost = spent + option.cost
                if cost > budget:
                    continue
                total = loss + option.loss
                if not math.isfinite(total):
                    raise ValueError("summed loss overflow")
                candidate = (total, ranks + (option.rank,))
                if cost not in candidates or candidate < candidates[cost]:
                    candidates[cost] = candidate
        if not candidates:
            raise ValueError("budget cannot fund one option per item")
        states = {}
        best_loss = math.inf
        for cost, candidate in sorted(candidates.items()):
            if candidate[0] < best_loss:
                states[cost] = candidate
                best_loss = candidate[0]
    _, (_, ranks) = min(
        states.items(), key=lambda item: (item[1][0], item[0], item[1][1])
    )
    return dict(zip(names, ranks, strict=True))
