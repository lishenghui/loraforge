from itertools import product

import pytest

from loraforge.compression.allocation import allocate_spectral_budget


def test_optimal_all_budgets():
    spectra = {"b": [5, 2], "a": [6, 2, 0]}
    for budget in range(6):
        plan = allocate_spectral_budget(spectra, budget)
        energy = sum(sum(x * x for x in spectra[k][:r]) for k, r in plan.items())
        optimum = max(
            sum(x * x for x in spectra["a"][:a] + spectra["b"][:b])
            for a, b in product(range(4), range(3))
            if a + b == budget
        )
        assert sum(plan.values()) == budget
        assert energy == optimum
    assert allocate_spectral_budget({"b": [2], "a": [2, 2]}, 2) == {"a": 2, "b": 0}
    assert allocate_spectral_budget({}, 0) == {}


@pytest.mark.parametrize("values", [[1, 2], [-1], [float("nan")], [float("inf")]])
def test_invalid_spectrum(values):
    with pytest.raises(ValueError):
        allocate_spectral_budget({"a": values}, 0)


@pytest.mark.parametrize("budget", [-1, True, 1.5, 3])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        allocate_spectral_budget({"a": [2]}, budget)


def test_minimum_rank():
    spectra = {"a": [10, 9, 8], "b": [1, 0]}
    assert allocate_spectral_budget(spectra, 3, min_rank=1) == {"a": 2, "b": 1}
    for budget in (0, 1, 6):
        with pytest.raises(ValueError):
            allocate_spectral_budget(spectra, budget, min_rank=1)
    with pytest.raises(ValueError):
        allocate_spectral_budget({"a": []}, 1, min_rank=1)


def test_constrained_optimum_and_mapping_order():
    spectra = {"z": [9, 8, 0], "a": [3, 3], "m": [1, 0]}
    for budget in range(3, 8):
        plan = allocate_spectral_budget(spectra, budget, min_rank=1)
        reversed_plan = allocate_spectral_budget(
            dict(reversed(list(spectra.items()))), budget, min_rank=1
        )
        assert plan == reversed_plan
        energy = sum(sum(s * s for s in spectra[k][:r]) for k, r in plan.items())
        optimum = max(
            sum(
                sum(s * s for s in spectra[k][:r])
                for k, r in zip(spectra, ranks, strict=True)
            )
            for ranks in product(range(1, 4), range(1, 3), range(1, 3))
            if sum(ranks) == budget
        )
        assert energy == optimum
        assert sum(plan.values()) == budget
