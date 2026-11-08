# SPDX-License-Identifier: Apache-2.0
"""Dependency-free stability annotations for public experimental APIs."""

import inspect
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class ExperimentalAPIWarning(UserWarning):
    """An explicitly enabled public API has no compatibility guarantee yet."""


@dataclass(frozen=True)
class ExperimentalInfo:
    """Stability metadata attached as ``__experimental_api__``."""

    reason: str
    since: str
    stability: str = "alpha"


def ExperimentalAPI(
    *, reason: str, since: str = "0.1.0.dev0", warn: bool = False
) -> Callable[[F], F]:
    """Document an alpha API, optionally warning once per function per process.

    The default returns the original function/class unchanged, so numerical
    operations gain no call wrapper. ``warn=True`` wraps synchronous functions
    such as installation entrypoints, preserving introspection and emitting
    one warning per process. It is intentionally unsupported for classes and
    asynchronous/generator functions; those can use metadata-only annotation.
    This annotation states API stability, not numerical or hardware certification.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a nonempty description")
    if not isinstance(since, str) or not since.strip():
        raise ValueError("since must be a nonempty version string")
    if type(warn) is not bool:
        raise TypeError("warn must be a bool")
    info = ExperimentalInfo(reason=reason, since=since)

    def decorate(obj: F) -> F:
        if not callable(obj):
            raise TypeError("ExperimentalAPI requires a callable")
        if warn and (
            not inspect.isfunction(obj)
            or inspect.iscoroutinefunction(obj)
            or inspect.isgeneratorfunction(obj)
            or inspect.isasyncgenfunction(obj)
        ):
            raise TypeError("warn=True requires a synchronous non-generator function")
        obj.__experimental_api__ = info
        note = (
            f"\n\nExperimental API (alpha, since {since}). "
            f"{reason} API compatibility is not guaranteed."
        )
        obj.__doc__ = (obj.__doc__ or "") + note
        if not warn:
            return obj
        warned = set()
        lock = Lock()

        @wraps(obj)
        def call(*args, **kwargs):
            pid = os.getpid()
            if pid not in warned:
                with lock:
                    if pid not in warned:
                        warnings.warn(
                            f"{obj.__module__}.{obj.__qualname__} is experimental. {reason}",
                            ExperimentalAPIWarning,
                            stacklevel=2,
                        )
                        warned.add(pid)
            return obj(*args, **kwargs)

        return cast(F, call)

    return decorate
