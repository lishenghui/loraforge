# SPDX-License-Identifier: Apache-2.0
"""Equal-cost spectral planning without tensor or device dependencies."""

import math
from collections.abc import Mapping, Sequence


def allocate_spectral_budget(
    spectra: Mapping[str, Sequence[float]], budget: int, *, min_rank: int = 0
) -> dict[str, int]:
    """Maximize retained squared singular values under a direction budget.

    Inputs are descending nonnegative finite spectra. Ties use module name,
    then direction index. Returns prefix ranks totaling exactly budget without
    changing inputs. Zero ranks are allowed; PEFT export requires positive ranks.
    This objective does not model quality, parameter bytes, or serving latency.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if any(not isinstance(k, str) or not k for k in spectra):
        raise ValueError("module names must be nonempty strings")
    if isinstance(min_rank, bool) or not isinstance(min_rank, int) or min_rank < 0:
        raise ValueError("min_rank must be a nonnegative integer")
    ranks = dict.fromkeys(sorted(spectra), min_rank)
    pool = []
    for name in ranks:
        previous = math.inf
        for index, raw in enumerate(spectra[name]):
            value = float(raw)
            if not math.isfinite(value) or value < 0 or value > previous:
                raise ValueError("spectra must be finite, nonnegative and descending")
            if index >= min_rank:
                pool.append((-value, name, index))
            previous = value
        if len(spectra[name]) < min_rank:
            raise ValueError("min_rank exceeds a module rank")
    remaining = budget - min_rank * len(ranks)
    if remaining < 0:
        raise ValueError("budget cannot satisfy min_rank")
    if remaining > len(pool):
        raise ValueError("budget exceeds available directions")
    for _, name, _ in sorted(pool)[:remaining]:
        ranks[name] += 1
    return ranks
