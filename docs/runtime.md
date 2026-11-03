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

## Asynchronous compact GPU residency

```python
from loraforge.runtime import GPUAdapterSlots

slots = GPUAdapterSlots(2, device="cuda:0", max_tensor_bytes=512 * 1024 * 1024)
handle = slots.load("compressed_adapter", slot=0)
with slots.borrow(handle) as modules:
    # Enqueue consumer work on this device's current stream within this scope.
    module = modules[0]
    # update = (x @ module.a.T @ module.b.T) * module.scaling
slots.evict(handle)
slots.collect()
```

Construction requires an explicit indexed CUDA/ROCm device. Imports remain free
of tensor/device initialization. Loading validates files on CPU, pins compact
host factors, and enqueues nonblocking copies on the selected device's current
stream. Parsing and pinning are synchronous host work. A ready event orders
subsequent borrowing on any stream of the same device, without a global sync.

A lease returns the pool's actual GPU tensors. Treat them as read-only and keep
all enqueue calls inside the lease, on the stream current when borrowing starts.
Do not retain references for later work or switch consumer streams within a
lease. Multiple read leases are supported. Active leases block replacement and
eviction. Exception exit releases the lease. Device tensors are recorded on the
consumer stream so queued work remains protected after scope exit and eviction.
This is an execution contract, not protection against deliberate tensor mutation.

Reload allocates new compact storage, then atomically publishes its generation;
it does not overwrite old buffers. Failed loads preserve existing handles.
Pinned staging is retained through copy completion. `collect()` polls events to
release completed staging; load/borrow/evict also collect. Pending old transfers
can temporarily retain retired device factors. The residency byte limit counts
committed factor payloads, not pinned staging, retired allocations, allocator
caches or peak replacement memory. Partial transfer failures synchronize the
load stream before propagating the exception.

GPU operations reject graph capture. Addresses change on replacement; graph
replay and fixed-address device pools are not supported. There is no automatic
LRU, framework registration or serving operator in this component.

## Recorded GPU validation

LUMI job `23424086` completed on 2026-11-03 using one MI250X GCD (gfx90a),
Python 3.12.11 and PyTorch 2.7.1+rocm6.2.4. All seven cases in
`python -m pytest tests/test_runtime_gpu.py -m gpu -v` passed (16.35 seconds).
Tests cover FP32/FP16/BF16, LoRA/rsLoRA scaling and dense update values after
cross-stream loading, active-lease replacement/eviction rejection, stale and
foreign handles, allocator churn after reuse, byte limits and exception release.
The single allocated device does not establish multi-GPU or NVIDIA CUDA support.
These are correctness/lifecycle checks, not serving performance measurements.
