from itertools import product

import pytest

from loraforge.compression import QualityOption as O
from loraforge.compression import allocate_quality_budget


def test_exhaustive_optimum_and_ties():
    curves = {"b": [O(1, 2, 5), O(4, 7, 1)], "a": [O(0, 0, 8), O(2, 3, 3), O(3, 5, 4)]}
    for budget in range(2, 15):
        choices = list(product(curves["a"], curves["b"]))
        feasible = [x for x in choices if sum(o.cost for o in x) <= budget]
        best = min(
            feasible,
            key=lambda x: (
                sum(o.loss for o in x),
                sum(o.cost for o in x),
                tuple(o.rank for o in x),
            ),
        )
        expected = dict(zip(["a", "b"], [o.rank for o in best], strict=True))
        assert allocate_quality_budget(curves, budget) == expected
        assert (
            allocate_quality_budget(dict(reversed(list(curves.items()))), budget)
            == expected
        )
    assert allocate_quality_budget({}, 0) == {}
    assert allocate_quality_budget({"x": [O(2, 0, -1), O(1, 0, -1)]}, 0) == {"x": 1}


@pytest.mark.parametrize(
    "curves,budget",
    [
        ({"x": []}, 1),
        ({"x": [O(1, 2, 0)]}, 1),
        ({"x": [O(1, 0, float("nan"))]}, 0),
        ({"x": [O(1, -1, 0)]}, 0),
        ({"x": [O(-1, 0, 0)]}, 0),
        ({"x": [O(1, 0, 0), O(1, 1, 0)]}, 1),
        ({}, True),
        ({}, -1),
        ({}, 1.2),
    ],
)
def test_invalid(curves, budget):
    with pytest.raises(ValueError):
        allocate_quality_budget(curves, budget)
