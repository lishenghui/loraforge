# Compact CPU slots

`AdapterSlots` is a synchronous CPU residency component. It loads validated
local PEFT files using the existing file contract, owns contiguous compact
factor copies, and preserves each module's rank, alpha and LoRA/rsLoRA scaling.

```python
from loraforge.runtime import AdapterSlots

slots = AdapterSlots(2, max_tensor_bytes=512 * 1024 * 1024)
handle = slots.load("compressed_adapter", slot=0)
modules = slots.resolve(handle)
print(slots.tensor_bytes)
slots.evict(handle)
```

Slot indices are explicit; there is no automatic LRU policy. Successful reloads
increment a slot generation and invalidate previous handles. Handles from a
different pool, stale handles and double evictions raise ValueError. Failed
loads (including corrupt input or exceeded resident byte limits) preserve the
existing occupant and its handle. Operations are serialized within each pool.

`resolve` returns detached contiguous copies rather than pool-owned tensors.
Mutating those copies cannot affect residency; they remain usable after an
occupant is evicted. The returned module carries its effective `scaling`;
consumers must apply it when using B @ A.

`tensor_bytes` counts only resident A/B tensor payloads. The optional byte limit
applies to committed residency, not peak process memory: parsing, transactional
replacement and resolve copies require additional transient memory. Source
configuration and manifests are validated by the PEFT loader, not retained as
mutable runtime state. The original files are unchanged.

This step provides CPU ownership and compact file loading. It does not implement
GPU storage, asynchronous transfers, GPU stream lifetime, stable device pointers,
graph replay, fused layouts or framework registration. Those need separate
contracts and hardware validation before SGLang/vLLM use.
