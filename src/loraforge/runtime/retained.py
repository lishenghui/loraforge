# SPDX-License-Identifier: Apache-2.0
# Layout/packing design: Shenghui Li, probe fast.py at 5fd6ae060d106066b306959f9a34b726a281ef4c.
"""Fixed-address per-checkpoint compact B storage with per-slot rank metadata.

Catalog construction and slot loading are outside graph capture. The serving
framework owns synchronization; loading must finish before any consuming graph
replay and may not race consumers. Known checkpoints are frozen at construction.
The native padded B pool remains resident. A is repacked in place after native
loading, so every consumer must use this layout's effective ranks thereafter.
"""

import json
import re
from pathlib import Path


def read_rank_catalog(paths):
    """Read local padded PEFT headers and optional probe compression rank metadata.

    Global checkpoint rank/scaling is required by the native SGLang loader.
    Compact heterogeneous LoRAForge exports must first be converted explicitly
    with export_sglang_adapter; rank metadata never changes adapter scaling.
    """
    catalog = {}
    for raw in paths:
        path = Path(raw).resolve()
        config = json.loads((path / "adapter_config.json").read_text())
        rank = config["r"]
        if type(rank) is not int or rank < 1:
            raise ValueError("invalid checkpoint rank")
        if (
            config.get("rank_pattern")
            or config.get("alpha_pattern")
            or config.get("use_rslora")
        ):
            raise ValueError("convert per-module rank/scaling to SGLang padding first")
        with (path / "adapter_model.safetensors").open("rb") as f:
            length = int.from_bytes(f.read(8), "little")
            if not 2 <= length <= 64 * 1024 * 1024:
                raise ValueError("invalid safetensors header length")
            header = json.loads(f.read(length))
        ranks = {}
        for key, value in header.items():
            if key.endswith(".lora_A.weight"):
                bkey = key.replace(".lora_A.weight", ".lora_B.weight")
                if (
                    len(value["shape"]) != 2
                    or value["shape"][0] != rank
                    or bkey not in header
                    or header[bkey]["shape"][-1] != rank
                ):
                    raise ValueError(
                        "native SGLang requires padded global-rank tensors"
                    )
                ranks[key] = rank
        if not ranks:
            raise ValueError("no linear LoRA A/B pairs found")
        metadata = path / "compression.json"
        if metadata.exists():
            declared = json.loads(metadata.read_text())["module_ranks"]
            if declared.keys() != ranks.keys():
                raise ValueError(
                    "retained rank metadata must cover exactly the A tensors"
                )
            for key, k in declared.items():
                if type(k) is not int or not 0 < k <= rank:
                    raise ValueError("retained rank exceeds checkpoint rank")
                ranks[key] = k
            # Padded columns are a representation contract, not a rank guess.
            # Check once on CPU before pools/graphs exist; never in timed execution.
            from safetensors import safe_open

            with safe_open(
                str(path / "adapter_model.safetensors"), framework="pt", device="cpu"
            ) as weights:
                for key, k in ranks.items():
                    if k < rank:
                        a = weights.get_tensor(key)
                        b = weights.get_tensor(
                            key.replace(".lora_A.weight", ".lora_B.weight")
                        )
                        if (
                            a[k:].count_nonzero().item()
                            or b[:, k:].count_nonzero().item()
                        ):
                            raise ValueError(
                                "retained-rank metadata truncates nonzero padded factors"
                            )
        catalog[str(path)] = ranks
    if not catalog:
        raise ValueError("a nonempty frozen adapter catalog is required")
    return catalog


def effective_rank(ranks, name, layer):
    targets = {
        "qkv_proj": {"qkv_proj", "q_proj", "k_proj", "v_proj"},
        "gate_up_proj": {"gate_up_proj", "gate_proj", "up_proj"},
        "in_proj_qkvz": {
            "in_proj_qkvz",
            "in_proj_qkv",
            "in_proj_q",
            "in_proj_k",
            "in_proj_v",
            "in_proj_z",
        },
        "in_proj_ba": {"in_proj_ba", "in_proj_b", "in_proj_a"},
    }.get(name, {name})
    return max(
        (
            r
            for key, r in ranks.items()
            if re.search(rf"\.layers\.{layer}\.", key) and key.split(".")[-3] in targets
        ),
        default=0,
    )


class RetainedLayout:
    """Owned compact B regions and fixed slot metadata for one fused projection."""

    def __init__(self, a, b, name, layer, catalog):
        import torch

        if (
            a.ndim != 3
            or b.ndim != 3
            or a.dtype != torch.bfloat16
            or b.dtype != a.dtype
            or a.device != b.device
        ):
            raise ValueError("retained layout requires matching BF16 3D buffers")
        slots, width, capacity = b.shape
        if (
            a.shape[0] != slots
            or capacity < 1
            or a.shape[1] % capacity
            or not 1 <= a.shape[1] // capacity <= 4
        ):
            raise ValueError("invalid fused A/B capacity")
        if not a.is_contiguous() or not b.is_contiguous():
            raise ValueError("native buffers must be contiguous")
        self.a, self.b, self.name, self.layer = a, b, name, layer
        self.capacity, self.slices = capacity, a.shape[1] // capacity
        self.regions, end = {}, 0
        for path, ranks in catalog.items():
            k = effective_rank(ranks, name, layer)
            if k > capacity:
                raise ValueError("effective rank exceeds native capacity")
            bucket = 1 << (max(1, k) - 1).bit_length()
            end = (end + 7) // 8 * 8
            self.regions[path] = (end, bucket, k)
            if 0 < k <= 64:
                end += width * bucket
        self.store = torch.zeros(max(1, end), dtype=b.dtype, device=b.device)
        for field in ("rank", "supported", "unsupported", "low", "high", "bucket"):
            setattr(self, field, torch.zeros(slots, dtype=torch.int32, device=b.device))
        self.boff = torch.zeros(slots, dtype=torch.int64, device=b.device)
        self.has_high = any(k > 16 for _, _, k in self.regions.values())
        self.has_unsupported = any(k > 64 for _, _, k in self.regions.values())
        self.max_rank = max(
            1, max(min(64, bucket) for _, bucket, _ in self.regions.values())
        )
        self.loaded = {}

    def validate_load(self, path, checkpoint_rank):
        if path is None:
            return
        if path not in self.regions:
            raise ValueError("checkpoint is absent from frozen rank catalog")
        if not self.regions[path][2] <= checkpoint_rank <= self.capacity:
            raise ValueError("checkpoint rank does not fit retained layout")

    def load(self, path, slot, checkpoint_rank=0):
        import torch

        self.validate_load(path, checkpoint_rank)
        if not 0 <= slot < self.b.shape[0]:
            raise ValueError("slot out of range")
        off, bucket, k = (0, 0, 0) if path is None else self.regions[path]
        if k and k != checkpoint_rank:
            packed = torch.cat(
                [
                    self.a[slot, s * checkpoint_rank : s * checkpoint_rank + k].clone()
                    for s in range(self.slices)
                ]
            )
            self.a[slot, : self.slices * k].copy_(packed)
        if 0 < k <= 64:
            view = self.store[off : off + self.b.shape[1] * bucket].view(
                self.b.shape[1], bucket
            )
            view.zero_()
            view[:, :k].copy_(self.b[slot, :, :k])
        for field, value in dict(
            rank=k,
            supported=k if k <= 64 else 0,
            unsupported=k if k > 64 else 0,
            low=k if k <= 16 else 0,
            high=k if k > 16 else 0,
            bucket=bucket,
            boff=off,
        ).items():
            getattr(self, field)[slot] = value
        self.loaded[slot] = dict(path=path, rank=k, checkpoint_rank=checkpoint_rank)
