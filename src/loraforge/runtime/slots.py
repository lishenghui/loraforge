# SPDX-License-Identifier: Apache-2.0
"""Synchronous CPU residency with generation-checked slot reuse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from loraforge.adapters import AdapterModule, PeftAdapter, load_peft_adapter


@dataclass(frozen=True)
class SlotHandle:
    """Opaque ownership token; valid only in its originating pool generation."""

    pool: str
    slot: int
    generation: int


class AdapterSlots:
    """Own compact CPU snapshots in a bounded number of explicitly managed slots.

    Loading clones tensors into contiguous storage. A failed load leaves the old
    occupant and its handle intact. Successful replacement or eviction invalidates
    old handles. resolve() returns a fresh clone, so callers cannot mutate pool
    storage. No automatic eviction, GPU copies, graph replay or async lifetime is
    implied. Operations are serialized by a per-pool lock.
    """

    def __init__(self, capacity: int, *, max_tensor_bytes: int | None = None):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if max_tensor_bytes is not None and (
            type(max_tensor_bytes) is not int or max_tensor_bytes < 0
        ):
            raise ValueError("max_tensor_bytes must be a nonnegative integer")
        self._capacity = capacity
        self._limit = max_tensor_bytes
        self._id = uuid4().hex
        self._lock = RLock()
        self._generations = [0] * capacity
        self._entries: dict[int, tuple[AdapterModule, ...]] = {}

    def _slot(self, slot: int) -> None:
        if type(slot) is not int or not 0 <= slot < self._capacity:
            raise ValueError("slot outside pool capacity")

    def _check(self, handle: SlotHandle) -> None:
        if not isinstance(handle, SlotHandle) or handle.pool != self._id:
            raise ValueError("handle belongs to a different pool")
        self._slot(handle.slot)
        if (
            handle.generation != self._generations[handle.slot]
            or handle.slot not in self._entries
        ):
            raise ValueError("stale or evicted slot handle")

    @staticmethod
    def _clone(modules: tuple[AdapterModule, ...]) -> tuple[AdapterModule, ...]:
        return tuple(
            AdapterModule(
                m.name,
                m.a.detach().clone().contiguous(),
                m.b.detach().clone().contiguous(),
                m.alpha,
                m.use_rslora,
            )
            for m in modules
        )

    @staticmethod
    def _bytes(modules: tuple[AdapterModule, ...]) -> int:
        return sum(t.numel() * t.element_size() for m in modules for t in (m.a, m.b))

    @property
    def tensor_bytes(self) -> int:
        """Resident factor payload bytes; excludes Python and transient load copies."""
        with self._lock:
            return sum(self._bytes(modules) for modules in self._entries.values())

    def load(self, source: str | Path, slot: int) -> SlotHandle:
        """Validate a local PEFT file snapshot and atomically replace one CPU slot."""
        with self._lock:
            self._slot(slot)
            adapter: PeftAdapter = load_peft_adapter(source)
            modules = self._clone(adapter.modules)
            resident = self.tensor_bytes - self._bytes(self._entries.get(slot, ()))
            if (
                self._limit is not None
                and resident + self._bytes(modules) > self._limit
            ):
                raise ValueError("resident tensor byte limit exceeded")
            self._entries[slot] = modules
            self._generations[slot] += 1
            return SlotHandle(self._id, slot, self._generations[slot])

    def resolve(self, handle: SlotHandle) -> tuple[AdapterModule, ...]:
        """Return detached compact CPU copies, preserving rank and effective scaling."""
        with self._lock:
            self._check(handle)
            return self._clone(self._entries[handle.slot])

    def evict(self, handle: SlotHandle) -> None:
        """Release residency and invalidate the handle; external copies remain valid."""
        with self._lock:
            self._check(handle)
            del self._entries[handle.slot]
            self._generations[handle.slot] += 1
