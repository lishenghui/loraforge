# SPDX-License-Identifier: Apache-2.0
"""Experimental gate/up-only compact B shadow for the public probe Fleet ABI.

Design reference: probe 5fd6ae060d106066b306959f9a34b726a281ef4c,
fast.py/fast_decode.cu. This wrapper uses fixed per-slot regions instead of
per-checkpoint regions and refreshes them after native loading. It neither
alters native B buffers nor enables HIP expansion for other projections.
"""

import json
import os


class GateShadow:
    def __init__(self, weights):
        import torch

        slots, width, capacity = weights.shape
        if (
            weights.dtype != torch.bfloat16
            or capacity not in (1, 2, 4, 8, 16, 32, 64)
            or width % 2
        ):
            raise ValueError("unsupported gate/up buffer")
        self.source = weights
        self.slots, self.width, self.capacity = slots, width, capacity
        self.store = torch.zeros_like(weights).flatten()
        self.boff = (
            torch.arange(slots, device=weights.device, dtype=torch.int64)
            * width
            * capacity
        )
        self.bucket = torch.zeros(slots, device=weights.device, dtype=torch.int32)
        self.offsets = torch.tensor(
            [0, width // 2, width], device=weights.device, dtype=torch.int32
        )
        self.ranks = [0] * slots

    def addresses(self):
        return tuple(
            t.data_ptr()
            for t in (self.source, self.store, self.boff, self.bucket, self.offsets)
        )

    def refresh(self, slot, rank):
        if not 0 <= slot < self.slots or not 0 <= rank <= self.capacity:
            raise ValueError("invalid slot or rank")
        begin = slot * self.width * self.capacity
        region = self.store[begin : begin + self.width * self.capacity]
        region.zero_()
        bucket = 1 << (rank - 1).bit_length() if rank else 0
        if rank:
            view = region[: self.width * bucket].view(self.width, bucket)
            view[:, :rank].copy_(self.source[slot, :, :rank])
        self.bucket[slot] = bucket
        self.ranks[slot] = rank


def install_gate_expand(backend, module, output, blocks=880):
    import torch
    from sglang.srt.lora.mem_pool import LoRAMemoryPool

    entries = {}
    loads = []
    old_init_pool = LoRAMemoryPool.init_buffers
    old_load = LoRAMemoryPool.load_lora_weight_to_buffer
    old_init_backend = backend.__init__
    old_prepare = backend.prepare_lora_batch
    old_expand = backend._expand
    observed = set()

    def init_pool(self, *args, **kwargs):
        result = old_init_pool(self, *args, **kwargs)
        self._fleet_gate_shadows = []
        for tensor in self.B_buffer.get("gate_up_proj", []):
            if tensor is not None:
                entry = GateShadow(tensor)
                entries[tensor.data_ptr()] = entry
                self._fleet_gate_shadows.append(entry)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"gate-pools-{os.getpid()}.json").write_text(
            json.dumps(
                dict(
                    modules=len(self._fleet_gate_shadows),
                    extra_bytes=sum(
                        e.store.numel() * e.store.element_size()
                        for e in self._fleet_gate_shadows
                    ),
                )
            )
        )
        return result

    def load(self, uid, buffer_id, lora_adapter, *args, **kwargs):
        result = old_load(self, uid, buffer_id, lora_adapter, *args, **kwargs)
        rank = 0 if uid is None else int(lora_adapter.config.r)
        for entry in self._fleet_gate_shadows:
            before = entry.addresses()
            entry.refresh(buffer_id, rank)
            if before != entry.addresses():
                raise AssertionError("shadow addresses changed during slot reuse")
        loads.append(
            dict(
                uid=str(uid),
                slot=buffer_id,
                rank=rank,
                modules=len(self._fleet_gate_shadows),
            )
        )
        (output / f"gate-loads-{os.getpid()}.json").write_text(json.dumps(loads))
        return result

    def init_backend(self, *args, **kwargs):
        old_init_backend(self, *args, **kwargs)
        self._fleet_gate_nseg = torch.zeros((), dtype=torch.int32, device=self.device)

    def prepare(self, *args, **kwargs):
        result = old_prepare(self, *args, **kwargs)
        self._fleet_gate_nseg.fill_(self.batch_info.num_segments)
        return result

    def expand(self, x, weights, offsets, max_slice, base, offsets_cpu=None):
        entry = entries.get(weights.data_ptr())
        info = self.batch_info
        supported = (
            entry is not None
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[0] <= 16
            and (info.max_len or 16) <= 16
            and info.scalings.dtype == torch.float32
            and offsets_cpu is not None
            and offsets_cpu.tolist() == [0, entry.width // 2, entry.width]
            and (base is None or (base.dtype == x.dtype and base.stride(1) == 1))
        )
        if not supported:
            return old_expand(self, x, weights, offsets, max_slice, base, offsets_cpu)
        capture = torch.cuda.is_current_stream_capturing()
        if capture not in observed:
            observed.add(capture)
            (output / f"gate-expand-{os.getpid()}-{int(capture)}.json").write_text(
                json.dumps(dict(capture=capture, blocks=blocks))
            )
        if base is None:
            base = torch.zeros(
                (x.shape[0], entry.width), dtype=x.dtype, device=x.device
            )
        module.expand(
            x,
            entry.store,
            entry.boff,
            entry.bucket,
            base,
            info.seg_indptr,
            info.weight_indices,
            info.permutation,
            info.lora_ranks,
            info.scalings,
            entry.offsets,
            self._fleet_gate_nseg,
            entry.width,
            entry.capacity,
            blocks,
            256,
        )
        return base

    LoRAMemoryPool.init_buffers = init_pool
    LoRAMemoryPool.load_lora_weight_to_buffer = load
    backend.__init__ = init_backend
    backend.prepare_lora_batch = prepare
    backend._expand = expand
