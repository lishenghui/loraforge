# SPDX-License-Identifier: Apache-2.0
"""Experimental status is inspectable and warnings do not alter call contracts."""

import inspect
import warnings
from dataclasses import FrozenInstanceError

import pytest

from loraforge import annotations as a


def test_metadata_preserves_function_identity_and_adds_documentation():
    def target(value: int = 3) -> int:
        """Original contract."""
        return value

    result = a.ExperimentalAPI(reason="Measured on one hardware family.")(target)
    assert result is target and result(4) == 4
    assert "Original contract." in result.__doc__
    assert "Experimental API" in result.__doc__
    assert result.__experimental_api__.stability == "alpha"
    with pytest.raises(FrozenInstanceError):
        result.__experimental_api__.reason = "changed"


def test_class_annotation_preserves_type_identity():
    class Example:
        pass

    assert a.ExperimentalAPI(reason="Class contract may change.")(Example) is Example
    assert isinstance(Example(), Example)


def test_warning_once_per_process_preserves_signature_values_and_exceptions(
    monkeypatch,
):
    @a.ExperimentalAPI(reason="Limited runtime coverage.", warn=True)
    def target(value: int, *, offset: int = 2) -> int:
        if value < 0:
            raise ValueError("original failure")
        return value + offset

    assert inspect.signature(target) == inspect.signature(target.__wrapped__)
    monkeypatch.setattr(a.os, "getpid", lambda: 100)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        assert target(3, offset=4) == 7
        assert target(1) == 3
        with pytest.raises(ValueError, match="original failure"):
            target(-1)
        assert len(records) == 1
        assert records[0].category is a.ExperimentalAPIWarning
        monkeypatch.setattr(a.os, "getpid", lambda: 200)
        target(1)
        assert len(records) == 2


@pytest.mark.parametrize("options", [{"reason": ""}, {"reason": "x", "since": ""}])
def test_invalid_annotation_metadata_is_rejected(options):
    with pytest.raises(ValueError):
        a.ExperimentalAPI(**options)


def test_warning_wrapper_refuses_classes_and_coroutines():
    class Example:
        pass

    async def asynchronous():
        pass

    for target in (Example, asynchronous):
        with pytest.raises(TypeError, match="synchronous"):
            a.ExperimentalAPI(reason="Unsupported wrapper kind.", warn=True)(target)
