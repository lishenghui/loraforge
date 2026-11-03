# SPDX-License-Identifier: Apache-2.0
# Probe Fleet fast.py / backend.py design, Shenghui Li, commit 5fd6ae060d106066b306959f9a34b726a281ef4c.
"""Complete retained-rank SGLang bridge, opt-in separately from compatibility.

No global SGLang kernel configuration is replaced. Pool hooks are installed
only in the opted-in worker; native loading and synchronization remain owners.
"""

import copy
import json
import os
import weakref
from collections import Counter
from pathlib import Path

from loraforge.runtime.retained import RetainedLayout, read_rank_catalog


class FleetPolicy:
    """Frozen MI250X profile: table-selected decode A, all decode B, low prefill B."""

    def __init__(self, table=None):
        self.table = (
            json.loads(
                (
                    Path(__file__).with_name("data") / "mi250x_fleet_dispatch.json"
                ).read_text()
            )
            if table is None
            else table
        )

    def select(self, name, phase, tokens):
        points = self.table.get(phase, {}).get(name, {})
        if not points:
            return dict(a="native", b="native", nblocks=880, threads=256)
        sizes = sorted(map(int, points))
        size = next((n for n in sizes if n >= tokens), sizes[-1])
        cfg = points[str(size)]
        # Probe's measured profile overrides the table's experimental B choices.
        return dict(
            a=cfg.get("a", "native") if phase == "decode" else "native",
            b="all" if phase == "decode" else "low",
            nblocks=cfg["nblocks"],
            threads=cfg["threads"],
        )


def _masked(info, ranks):
    result = copy.copy(info)
    result.lora_ranks = ranks
    return result


def create_backend(catalog):
    import torch
    from sglang.kernels.ops.lora.dense.chunked_sgmv_expand import (
        chunked_sgmv_lora_expand_forward as native_expand,
    )
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
    from sglang.srt.lora.mem_pool import LoRAMemoryPool

    from loraforge.ops.rowexpand import direct_shrink, row_expand, rowexpand_extension

    from ._sglang_rank_aware import shrink as rank_shrink

    a_entries, b_entries = weakref.WeakValueDictionary(), weakref.WeakValueDictionary()
    policy = FleetPolicy()
    if getattr(LoRAMemoryPool, "_loraforge_fleet", False):
        raise RuntimeError(
            "Fleet pool hooks already belong to another backend instance"
        )
    old_init, old_load = (
        LoRAMemoryPool.init_buffers,
        LoRAMemoryPool.load_lora_weight_to_buffer,
    )

    def init_pool(self, *args, **kwargs):
        if self.tp_size != 1 or self.enable_lora_overlap_loading:
            raise ValueError("Fleet requires TP=1 and non-overlapping adapter loading")
        old_init(self, *args, **kwargs)
        self._loraforge_layouts = []
        for name, aa in self.A_buffer.items():
            if name not in ("qkv_proj", "gate_up_proj", "o_proj", "down_proj"):
                raise ValueError(f"unsupported Fleet projection: {name}")
            for layer, (a, b) in enumerate(zip(aa, self.B_buffer[name], strict=True)):
                if a is None or b is None:
                    continue
                entry = RetainedLayout(a, b, name, layer, catalog)
                self._loraforge_layouts.append(entry)
                a_entries[a.data_ptr()] = entry
                b_entries[b.data_ptr()] = entry
        rowexpand_extension()  # Before any graph capture, never from first replay.

    def load_pool(self, uid, buffer_id, adapter, *args, **kwargs):
        path = None if uid is None else str(Path(adapter.config.path).resolve())
        rank = 0 if uid is None else int(adapter.config.r)
        # Reject unknown checkpoints before native buffers can be overwritten.
        for e in self._loraforge_layouts:
            e.validate_load(path, rank)
        result = old_load(self, uid, buffer_id, adapter, *args, **kwargs)
        for e in self._loraforge_layouts:
            e.load(path, buffer_id, rank)
        return result

    LoRAMemoryPool.init_buffers = init_pool
    LoRAMemoryPool.load_lora_weight_to_buffer = load_pool
    LoRAMemoryPool._loraforge_fleet = True

    class FleetBackend(ChunkedSgmvLoRABackend):
        supports_prefill_cuda_graph = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            arch = getattr(
                torch.cuda.get_device_properties(self.device), "gcnArchName", ""
            )
            if not torch.version.hip or not arch.startswith("gfx90a"):
                raise ValueError(
                    "frozen Fleet dispatch currently requires MI250 gfx90a"
                )
            if self.max_chunk_size != 16:
                raise ValueError("Fleet requires max_lora_chunk_size=16")
            self._fleet_segments = torch.zeros(
                (), dtype=torch.int32, device=self.device
            )
            self._fleet_decode = False
            self._fleet_hits = Counter()

        def prepare_lora_batch(self, forward_batch, *args, **kwargs):
            super().prepare_lora_batch(forward_batch, *args, **kwargs)
            self._fleet_segments.fill_(self.batch_info.num_segments)
            self._fleet_decode = forward_batch.forward_mode.is_decode()

        def _a(self, x, weights, info, slices):
            x = x.contiguous()
            e = a_entries.get(weights.data_ptr())
            if e is None:
                return rank_shrink(x, weights, info, slices)
            cfg = policy.select(
                e.name, "decode" if self._fleet_decode else "prefill", x.shape[0]
            )
            eligible = (
                cfg["a"] == "hip"
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
                and x.shape[1] % 8 == 0
                and x.data_ptr() % 16 == 0
                and info.req_weight_indices is not None
            )
            if not eligible:
                self._fleet_hits["rank_aware_shrink"] += 1
                return rank_shrink(x, weights, _masked(info, e.rank), slices)
            self._fleet_hits["direct_shrink"] += 1
            z = (
                rank_shrink(x, weights, _masked(info, e.unsupported), slices)
                if e.has_unsupported
                else torch.empty(
                    (x.shape[0], weights.shape[1]), dtype=x.dtype, device=x.device
                )
            )
            return direct_shrink(
                x, weights, z, info.req_weight_indices, e.supported, slices, e.max_rank
            )

        def _b(self, z, weights, info, offsets, max_slice, base, cpu):
            e = b_entries.get(weights.data_ptr())
            if e is None:
                return native_expand(
                    z, weights, info, offsets, max_slice, base, slice_offsets_cpu=cpu
                )
            phase = "decode" if self._fleet_decode else "prefill"
            cfg = policy.select(e.name, phase, z.shape[0])
            eligible = (
                cfg["b"] != "native"
                and info is self.batch_info
                and (info.max_len or 16) <= 16
                and cpu is not None
                and 2 <= cpu.numel() <= 4
                and z.dtype == torch.bfloat16
                and z.is_contiguous()
                and (base is None or (base.dtype == z.dtype and base.stride(1) == 1))
            )
            if not eligible:
                return native_expand(
                    z,
                    weights,
                    _masked(info, e.rank),
                    offsets,
                    max_slice,
                    base,
                    slice_offsets_cpu=cpu,
                )
            if base is None:
                base = torch.zeros(
                    (z.shape[0], weights.shape[1]), dtype=z.dtype, device=z.device
                )
            if not hasattr(e, "slice_offsets"):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm each projection before capturing Fleet graphs"
                    )
                e.slice_offsets = torch.tensor(
                    cpu.tolist(), device=z.device, dtype=torch.int32
                )
            low = cfg["b"] == "low"
            if e.has_high if low else e.has_unsupported:
                base = native_expand(
                    z,
                    weights,
                    _masked(info, e.high if low else e.unsupported),
                    offsets,
                    max_slice,
                    base,
                    slice_offsets_cpu=cpu,
                )
            self._fleet_hits[f"rowexpand_{phase}"] += 1
            return row_expand(
                z,
                e.store,
                e.boff,
                e.bucket,
                base,
                info.seg_indptr,
                info.weight_indices,
                info.permutation,
                e.low if low else e.supported,
                info.scalings,
                e.slice_offsets,
                self._fleet_segments,
                e.capacity,
                blocks=cfg["nblocks"],
                threads=cfg["threads"],
            )

        def run_lora_a_sgemm(
            self, x, weights, pruned_batch_info=None, stack_num=1, *args, **kwargs
        ):
            return self._a(
                x,
                weights,
                self.batch_info if pruned_batch_info is None else pruned_batch_info,
                stack_num,
            )

        def run_lora_b_sgemm(
            self,
            x,
            weights,
            output_offset,
            base_output=None,
            pruned_batch_info=None,
            *args,
            **kwargs,
        ):
            return self._b(
                x,
                weights,
                self.batch_info if pruned_batch_info is None else pruned_batch_info,
                output_offset,
                weights.shape[-2],
                base_output,
                kwargs.get("output_offset_cpu"),
            )

        def run_qkv_lora(
            self,
            x,
            a,
            b,
            output_offset,
            max_qkv_out_dim,
            base_output=None,
            n_slices=3,
            *args,
            **kwargs,
        ):
            return self._b(
                self._a(x, a, self.batch_info, n_slices),
                b,
                self.batch_info,
                output_offset,
                max_qkv_out_dim,
                base_output,
                kwargs.get("output_offset_cpu"),
            )

        def run_gate_up_lora(
            self, x, a, b, output_offset, base_output=None, *args, **kwargs
        ):
            return self._b(
                self._a(x, a, self.batch_info, 2),
                b,
                self.batch_info,
                output_offset,
                b.shape[-2] // 2,
                base_output,
                kwargs.get("output_offset_cpu"),
            )

        def fleet_snapshot(self):
            entries = list(a_entries.values())
            return dict(
                policy="fleet",
                python_capture_eager_calls=dict(self._fleet_hits),
                modules=len(entries),
                shadow_bytes=sum(
                    e.store.numel() * e.store.element_size() for e in entries
                ),
                loaded={f"{e.layer}:{e.name}": e.loaded for e in entries},
            )

    # An explicit test hook also permits framework-free pool lifecycle checks.
    FleetBackend.bind_layout = staticmethod(
        lambda e: (
            a_entries.__setitem__(e.a.data_ptr(), e),
            b_entries.__setitem__(e.b.data_ptr(), e),
        )
    )
    return FleetBackend


def backend_from_environment():
    manifest = Path(os.environ["LORAFORGE_SGLANG_MANIFEST"])
    data = json.loads(manifest.read_text())
    paths = [a.get("path", a.get("source")) for a in data["adapters"]]
    if any(not p for p in paths):
        raise ValueError("every adapter must declare a local path")
    return create_backend(read_rank_catalog(paths))
