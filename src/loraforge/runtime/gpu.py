# SPDX-License-Identifier: Apache-2.0
"""Compact GPU residency with event-ordered copies and scoped stream leases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from loraforge.adapters import AdapterModule, load_peft_adapter

from .slots import AdapterSlots, SlotHandle


@dataclass
class _Resident:
    modules: tuple[AdapterModule, ...]
    staging: tuple[AdapterModule, ...]
    ready: object
    readers: int = 0


class GPUAdapterSlots:
    """Explicit GPU slots; use factors only inside borrow() on its current stream.

    Each load allocates compact storage rather than overwriting old device
    addresses. Active leases prevent replacement/eviction. record_stream protects
    enqueued work after lease exit. Captured graphs and mutation are unsupported.
    """

    def __init__(
        self, capacity: int, *, device: str, max_tensor_bytes: int | None = None
    ):
        import torch

        # Share CPU argument validation without creating tensor/device state.
        AdapterSlots(capacity, max_tensor_bytes=max_tensor_bytes)
        selected = torch.device(device)
        if selected.type != "cuda" or selected.index is None:
            raise ValueError("device must explicitly select a CUDA/ROCm GPU index")
        with torch.cuda.device(selected):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("GPU slots do not support graph capture")
        self.device = selected
        self._capacity = capacity
        self._limit = max_tensor_bytes
        self._id = uuid4().hex
        self._lock = RLock()
        self._generations = [0] * capacity
        self._entries: dict[int, _Resident] = {}
        self._pending: list[_Resident] = []

    def _slot(self, slot: int) -> None:
        if type(slot) is not int or not 0 <= slot < self._capacity:
            raise ValueError("slot outside pool capacity")

    def _entry(self, handle: SlotHandle) -> _Resident:
        if not isinstance(handle, SlotHandle) or handle.pool != self._id:
            raise ValueError("handle belongs to a different pool")
        self._slot(handle.slot)
        if (
            handle.generation != self._generations[handle.slot]
            or handle.slot not in self._entries
        ):
            raise ValueError("stale or evicted slot handle")
        return self._entries[handle.slot]

    def _stream(self):
        import torch

        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("GPU slots do not support graph capture")
            return torch.cuda.current_stream(self.device)

    def _reap(self) -> None:
        self._pending = [entry for entry in self._pending if not entry.ready.query()]
        for entry in self._entries.values():
            if entry.staging and entry.ready.query():
                entry.staging = ()

    @property
    def tensor_bytes(self) -> int:
        """Committed factor payload bytes, excluding staging and retired allocations."""
        with self._lock:
            return sum(AdapterSlots._bytes(e.modules) for e in self._entries.values())

    def load(self, source: str | Path, slot: int) -> SlotHandle:
        """Validate on CPU, stage pinned copies, enqueue H2D on current pool stream.

        A returned handle can be borrowed immediately on another stream; borrow
        inserts an event wait. File parsing and pinning are synchronous host work.
        Failure preserves the previous occupant and generation.
        """
        import torch

        with self._lock:
            self._slot(slot)
            stream = self._stream()
            self._reap()
            old = self._entries.get(slot)
            if old is not None and old.readers:
                raise RuntimeError("slot has active leases")
            adapter = load_peft_adapter(source)
            size = AdapterSlots._bytes(adapter.modules)
            resident = self.tensor_bytes - (
                AdapterSlots._bytes(old.modules) if old else 0
            )
            if self._limit is not None and resident + size > self._limit:
                raise ValueError("resident tensor byte limit exceeded")
            staging, modules = [], []
            try:
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    for module in adapter.modules:
                        host = AdapterModule(
                            module.name,
                            module.a.pin_memory(),
                            module.b.pin_memory(),
                            module.alpha,
                            module.use_rslora,
                        )
                        staging.append(host)
                        modules.append(
                            AdapterModule(
                                host.name,
                                host.a.to(self.device, non_blocking=True),
                                host.b.to(self.device, non_blocking=True),
                                host.alpha,
                                host.use_rslora,
                            )
                        )
                    ready = torch.cuda.Event()
                    ready.record(stream)
            except BaseException:
                # Retain partial pinned buffers until outstanding copies finish.
                stream.synchronize()
                raise
            entry = _Resident(tuple(modules), tuple(staging), ready)
            if old is not None and old.staging:
                self._pending.append(old)
            self._entries[slot] = entry
            self._generations[slot] += 1
            return SlotHandle(self._id, slot, self._generations[slot])

    @contextmanager
    def borrow(self, handle: SlotHandle):
        """Lease read-only factors for work enqueued on the current pool-device stream.

        The lease must cover all enqueue calls. Do not keep tensor references or
        enqueue on another stream outside its scope. Scaling remains explicit.
        """
        with self._lock:
            stream = self._stream()
            self._reap()
            entry = self._entry(handle)
            stream.wait_event(entry.ready)
            for module in entry.modules:
                module.a.record_stream(stream)
                module.b.record_stream(stream)
            entry.readers += 1
        try:
            yield entry.modules
        finally:
            with self._lock:
                entry.readers -= 1

    def evict(self, handle: SlotHandle) -> None:
        """Invalidate residency; allocator stream tracking protects queued consumers."""
        with self._lock:
            self._stream()
            self._reap()
            entry = self._entry(handle)
            if entry.readers:
                raise RuntimeError("slot has active leases")
            if entry.staging:
                self._pending.append(entry)
            del self._entries[handle.slot]
            self._generations[handle.slot] += 1

    def collect(self) -> None:
        """Release completed pinned staging without waiting for GPU completion."""
        with self._lock:
            self._stream()
            self._reap()
