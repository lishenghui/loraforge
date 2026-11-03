# SPDX-License-Identifier: Apache-2.0
"""Compact CPU/GPU residency and explicit slot ownership."""

from .gpu import GPUAdapterSlots
from .slots import AdapterSlots, SlotHandle

__all__ = ["GPUAdapterSlots", "AdapterSlots", "SlotHandle"]
