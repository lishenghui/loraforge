# SPDX-License-Identifier: Apache-2.0
"""Explicit padding/scaling export for SGLang's global-rank loader."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .peft import _hash, load_peft_adapter


def export_sglang_adapter(source, destination):
    """Convert a local compact PEFT adapter to a new, padded serving directory.

    Preserve each module's scaling by folding it into BF16 B, and publish the
    true ranks separately for Fleet execution. BF16 rescaling may round; this is
    recorded and must be included in quality comparisons. Never overwrite files.
    Returned metadata reports physical bytes, not a fictitious compact saving.
    """
    import torch
    from safetensors.torch import save_file

    adapter = load_peft_adapter(source)
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".loraforge-serving-", dir=target.parent))
    try:
        rank = max(m.rank for m in adapter.modules)
        tensors, ranks = {}, {}
        for m in adapter.modules:
            key = f"base_model.model.{m.name}"
            a = torch.zeros((rank, m.a.shape[1]), dtype=torch.bfloat16)
            b = torch.zeros((m.b.shape[0], rank), dtype=torch.bfloat16)
            a[: m.rank].copy_(m.a)
            b[:, : m.rank].copy_((m.b.float() * m.scaling).to(torch.bfloat16))
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("serving conversion overflowed BF16")
            tensors[key + ".lora_A.weight"] = a
            tensors[key + ".lora_B.weight"] = b
            ranks[key + ".lora_A.weight"] = m.rank
        config = dict(adapter.config)
        config.update(
            r=rank,
            lora_alpha=rank,
            rank_pattern={},
            alpha_pattern={},
            use_rslora=False,
            inference_mode=True,
            target_modules=sorted({m.name.split(".")[-1] for m in adapter.modules}),
        )
        (stage / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(tensors, str(stage / "adapter_model.safetensors"))
        report = dict(
            format="loraforge.sglang-padded",
            schema_version=1,
            module_ranks=ranks,
            source_sha256=adapter.source_sha256,
            checkpoint_rank=rank,
            scaling="folded into BF16 B; loader alpha/r=1",
            physical_tensor_bytes=sum(
                t.numel() * t.element_size() for t in tensors.values()
            ),
            files={
                name: _hash(stage / name)
                for name in ("adapter_config.json", "adapter_model.safetensors")
            },
        )
        (stage / "compression.json").write_text(json.dumps(report, indent=2) + "\n")
        # Atomic publication after validation; a competing nonempty target fails.
        if target.exists():
            raise FileExistsError(target)
        os.rename(stage, target)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)
