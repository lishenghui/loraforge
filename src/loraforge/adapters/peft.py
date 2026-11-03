# SPDX-License-Identifier: Apache-2.0
"""Portable PEFT files and manifests for unmerged linear LoRA and rsLoRA.

This module implements the file contract independently; it does not import PEFT
or Transformers, execute model code, or download models. See docs/peft.md for
the intentionally limited configuration surface and reference specifications.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loraforge import __version__

from ._config import _config, _finite, _matches, _pattern_value, _positive_int

if TYPE_CHECKING:
    from torch import Tensor

    from loraforge.compression.spectral import CompressionInfo, EighBackend

OutputDtype = Literal["source", "float32", "float16", "bfloat16"]
CONFIG = "adapter_config.json"
WEIGHTS = "adapter_model.safetensors"
MANIFEST = "loraforge_manifest.json"
_PREFIX = "base_model.model."
_KEY = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")


@dataclass(frozen=True)
class AdapterModule:
    name: str
    a: Tensor
    b: Tensor
    alpha: float
    use_rslora: bool = False

    @property
    def rank(self) -> int:
        return self.a.shape[0]

    @property
    def scaling(self) -> float:
        denominator = math.sqrt(self.rank) if self.use_rslora else self.rank
        return self.alpha / denominator


@dataclass(frozen=True)
class PeftAdapter:
    """CPU snapshot of a local adapter. Treat config and tensors as read-only."""

    source: Path
    config: dict
    modules: tuple[AdapterModule, ...]
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class AdapterExport:
    directory: Path
    manifest: dict


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    data = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _check_manifest(directory: Path) -> None:
    path = directory / MANIFEST
    if not path.exists():
        return
    manifest = _json(path)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("format") != "loraforge.peft"
    ):
        raise ValueError("unsupported LoRAForge manifest version or format")
    output = manifest.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("files"), dict):
        raise ValueError("manifest output.files must be an object")
    files = output["files"]
    for name in (CONFIG, WEIGHTS):
        record = files.get(name, {})
        if not isinstance(record, dict):
            raise ValueError(f"manifest file record must be an object: {name}")
        if (
            record.get("sha256") != _hash(directory / name)
            or record.get("bytes") != (directory / name).stat().st_size
        ):
            raise ValueError(f"manifest integrity check failed for {name}")


def load_peft_adapter(directory: str | Path) -> PeftAdapter:
    """Read a local, single-file safetensors PEFT adapter into CPU tensors.

    Reject unsupported config/tensors rather than dropping weights. No model or
    PEFT installation is needed. Existing LoRAForge manifests are verified.
    """
    import torch
    from safetensors.torch import load_file

    source = Path(directory).resolve(strict=True)
    if not source.is_dir() or not all(
        (source / name).is_file() for name in (CONFIG, WEIGHTS)
    ):
        raise ValueError(
            "expected adapter_config.json and adapter_model.safetensors in a directory"
        )
    _check_manifest(source)
    names = [CONFIG, WEIGHTS] + ([MANIFEST] if (source / MANIFEST).exists() else [])
    hashes = {name: _hash(source / name) for name in names}
    config = _json(source / CONFIG)
    _config(config)
    weights = load_file(source / WEIGHTS, device="cpu")
    pairs = {}
    for key, tensor in weights.items():
        match = _KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"unsupported tensor key: {key}")
        name, side = match.groups()
        if tensor.ndim != 2 or tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(f"unsupported tensor layout or dtype: {key}")
        pairs.setdefault(name, {})[side] = tensor
    if not pairs:
        raise ValueError("adapter has no LoRA tensor pairs")
    modules = []
    for name, pair in sorted(pairs.items()):
        if pair.keys() != {"A", "B"}:
            raise ValueError(f"incomplete A/B pair: {name}")
        a, b = pair["A"], pair["B"]
        rank = _pattern_value(config.get("rank_pattern", {}), name, config["r"])
        alpha = _pattern_value(
            config.get("alpha_pattern", {}), name, config["lora_alpha"]
        )
        if (
            min(*a.shape, *b.shape) < 1
            or a.shape[0] != rank
            or b.shape[1] != rank
            or a.dtype != b.dtype
        ):
            raise ValueError(
                f"factor shape/dtype disagrees with configured rank: {name}"
            )
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise ValueError(f"nonfinite factors: {name}")
        if not _matches(config["target_modules"], name) or _matches(
            config.get("exclude_modules"), name
        ):
            raise ValueError(
                f"tensor module disagrees with target/exclude configuration: {name}"
            )
        modules.append(
            AdapterModule(
                name,
                a.clone(),
                b.clone(),
                float(alpha),
                use_rslora=config.get("use_rslora", False),
            )
        )
    if any(_hash(source / name) != digest for name, digest in hashes.items()):
        raise ValueError("source adapter changed while it was being read")
    return PeftAdapter(source, config, tuple(modules), hashes)


def _destination(adapter: PeftAdapter, output: str | Path) -> Path:
    raw = Path(output).absolute()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"output must be a new directory: {raw}")
    result = raw.resolve()
    if adapter.source == result or adapter.source in result.parents:
        raise ValueError("output must be outside the source adapter directory")
    if not result.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    return result


def _export(
    adapter: PeftAdapter,
    output: str | Path,
    factors: Mapping[str, tuple[Tensor, Tensor]] | None,
    output_dtype: OutputDtype,
    executions: Mapping[str, CompressionInfo] | None = None,
    parameters: dict | None = None,
) -> AdapterExport:
    import torch
    from safetensors.torch import save_file

    destination = _destination(adapter, output)
    if output_dtype not in ("source", "float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    if factors is not None and set(factors) != {
        module.name for module in adapter.modules
    }:
        raise ValueError("factors must contain exactly all adapter module names")
    if any(
        _hash(adapter.source / name) != digest
        for name, digest in adapter.source_sha256.items()
    ):
        raise ValueError("source adapter changed after loading")
    config = copy.deepcopy(adapter.config)
    _config(config)
    tensors, records = {}, []
    ranks, alphas = {}, {}
    for module in adapter.modules:
        if module.use_rslora != config.get("use_rslora", False):
            raise ValueError("module scaling mode disagrees with adapter configuration")
        a, b = factors[module.name] if factors is not None else (module.a, module.b)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("output factors must be torch.Tensor instances")
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[1] != module.a.shape[1]
            or b.shape[0] != module.b.shape[0]
            or a.shape[0] != b.shape[1]
        ):
            raise ValueError(f"output factor dimensions disagree: {module.name}")
        rank = _positive_int(a.shape[0], "exported rank")
        if rank > module.rank:
            raise ValueError("export cannot increase the original rank")
        if (
            a.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or b.dtype != a.dtype
        ):
            raise ValueError(
                "output factors must have matching float16/bfloat16/float32 dtype"
            )
        dtype = (
            module.a.dtype if output_dtype == "source" else getattr(torch, output_dtype)
        )
        a, b = (
            t.detach().to(device="cpu", dtype=dtype).clone().contiguous()
            for t in (a, b)
        )
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise ValueError(
                f"output factors are nonfinite after dtype conversion: {module.name}"
            )
        denominator = math.sqrt(rank) if module.use_rslora else rank
        alpha = module.scaling * denominator
        _finite(alpha, "exported alpha")
        pattern = "^" + re.escape(module.name)
        ranks[pattern], alphas[pattern] = rank, alpha
        tensors[f"{_PREFIX}{module.name}.lora_A.weight"] = a
        tensors[f"{_PREFIX}{module.name}.lora_B.weight"] = b
        records.append(
            {
                "name": module.name,
                "original_rank": module.rank,
                "retained_rank": rank,
                "source_alpha": module.alpha,
                "output_alpha": alpha,
                "scaling": module.scaling,
                "use_rslora": module.use_rslora,
                "scaling_type": "alpha_over_sqrt_rank"
                if module.use_rslora
                else "alpha_over_rank",
                "source_dtype": str(module.a.dtype),
                "output_dtype": str(dtype),
                "source_tensor_bytes": sum(
                    t.numel() * t.element_size() for t in (module.a, module.b)
                ),
                "output_tensor_bytes": sum(
                    t.numel() * t.element_size() for t in (a, b)
                ),
                "execution": asdict(executions[module.name])
                if executions is not None
                else None,
            }
        )
    config.update(
        r=records[0]["retained_rank"],
        lora_alpha=records[0]["output_alpha"],
        rank_pattern=ranks,
        alpha_pattern=alphas,
        inference_mode=True,
        target_modules="^(?:"
        + "|".join(re.escape(module.name) for module in adapter.modules)
        + ")$",
    )
    manifest = {
        "format": "loraforge.peft",
        "schema_version": 1,
        "loraforge_version": __version__,
        "operation": "compression" if executions is not None else "export",
        "compression": parameters,
        "source": {
            "directory_name": adapter.source.name,
            "files": {
                name: {
                    "sha256": digest,
                    "bytes": (adapter.source / name).stat().st_size,
                }
                for name, digest in adapter.source_sha256.items()
            },
        },
        "modules": records,
        "output": {
            "files": {},
            "source_tensor_bytes": sum(r["source_tensor_bytes"] for r in records),
            "tensor_bytes": sum(r["output_tensor_bytes"] for r in records),
        },
    }
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    created = False
    linked = []
    try:
        save_file(tensors, stage / WEIGHTS, metadata={"format": "pt"})
        (stage / CONFIG).write_text(
            json.dumps(config, indent=2, allow_nan=False) + "\n"
        )
        for name in (CONFIG, WEIGHTS):
            manifest["output"]["files"][name] = {
                "sha256": _hash(stage / name),
                "bytes": (stage / name).stat().st_size,
            }
        (stage / MANIFEST).write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        # Return the same JSON-native values that consumers read from disk.
        manifest = _json(stage / MANIFEST)
        # Reserve a new directory atomically; link complete files without overwrite.
        # The manifest is published last. Existing paths always fail unchanged.
        destination.mkdir()
        created = True
        for name in (CONFIG, WEIGHTS, MANIFEST):
            os.link(stage / name, destination / name)
            linked.append(name)
    except BaseException:
        for name in linked:
            path = destination / name
            if path.exists() and path.samefile(stage / name):
                path.unlink()
        if created:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(stage)
    return AdapterExport(destination, manifest)


def save_peft_adapter(
    adapter: PeftAdapter,
    output: str | Path,
    *,
    factors: Mapping[str, tuple[Tensor, Tensor]] | None = None,
    output_dtype: OutputDtype = "source",
) -> AdapterExport:
    """Export to a new directory, preserving each module's effective scale.

    Ordinary LoRA uses alpha/r; rsLoRA uses alpha/sqrt(r). Export preserves the
    source use_rslora flag and updates alpha for the retained rank accordingly.

    Optional replacement factors must cover exactly all modules and use positive
    ranks no larger than the originals. This is a file export, not a compression
    claim; its manifest has no inferred execution backend. No source file changes.
    """
    return _export(adapter, output, factors, output_dtype)


def compress_peft_adapter(
    source: str | Path,
    output: str | Path,
    keep: int | Mapping[str, int] | None = None,
    *,
    budget: int | None = None,
    device: str,
    output_dtype: OutputDtype = "source",
    eigh_backend: EighBackend = "auto",
    batch_size: int = 16,
) -> AdapterExport:
    """Load, compress through GPU FlashTSQR, and export a standard PEFT adapter.

    Provide keep (a positive rank or exact module-name mapping), or budget (an
    integer total direction count). Budget mode reserves one rank per module
    then maximizes unscaled spectral energy. It measures spectra with FlashTSQR
    before a second compression pass; it does not optimize quality or bytes.
    device must explicitly select a CUDA/ROCm GPU. Same-shape, same-dtype modules
    with equal retained rank are batched up to batch_size to bound GPU memory.
    """
    import torch

    from loraforge.compression import truncate_lora

    adapter = load_peft_adapter(source)
    _destination(adapter, output)
    _positive_int(batch_size, "batch_size")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("compression requires an explicit CUDA/ROCm GPU device")
    if output_dtype not in ("source", "float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported output dtype: {output_dtype}")
    if eigh_backend not in ("auto", "torch", "jacobi", "cpu"):
        raise ValueError(f"unknown EVD backend: {eigh_backend}")
    if (keep is None) == (budget is None):
        raise ValueError("provide exactly one of keep or budget")
    if budget is not None:
        from loraforge.compression.allocation import allocate_spectral_budget

        # Check feasibility before any GPU work. PEFT requires positive ranks.
        allocate_spectral_budget(
            {m.name: [0.0] * m.rank for m in adapter.modules}, budget, min_rank=1
        )
        selected = torch.device(device)
        if selected.type != "cuda":
            raise ValueError("compression requires an explicit CUDA/ROCm GPU device")
        spectra = {}
        for module in adapter.modules:
            result = truncate_lora(
                module.a.to(selected),
                module.b.to(selected),
                0,
                eigh_backend=eigh_backend,
            )
            spectra[module.name] = result.singular_values.cpu().tolist()
            del result
        keep = allocate_spectral_budget(spectra, budget, min_rank=1)
    if isinstance(keep, Mapping):
        if set(keep) != {module.name for module in adapter.modules}:
            raise ValueError(
                "keep mapping must contain exactly all adapter module names"
            )
        plan = dict(keep)
    else:
        plan = {module.name: keep for module in adapter.modules}
    for module in adapter.modules:
        if _positive_int(plan[module.name], f"keep[{module.name}]") > min(
            module.rank, module.a.shape[1], module.b.shape[0]
        ):
            raise ValueError(f"invalid retained rank for {module.name}")
        if module.rank > min(module.a.shape[1], module.b.shape[0]):
            raise ValueError(
                f"FlashTSQR requires rank <= ambient dimensions: {module.name}"
            )
    factors, executions = {}, {}
    groups = {}
    for module in adapter.modules:
        key = (
            tuple(module.a.shape),
            tuple(module.b.shape),
            module.a.dtype,
            plan[module.name],
        )
        groups.setdefault(key, []).append(module)
    for modules in groups.values():
        for start in range(0, len(modules), batch_size):
            batch = modules[start : start + batch_size]
            result = truncate_lora(
                torch.stack([m.a for m in batch]).to(selected),
                torch.stack([m.b for m in batch]).to(selected),
                plan[batch[0].name],
                eigh_backend=eigh_backend,
            )
            for index, module in enumerate(batch):
                factors[module.name] = result.a[index].cpu(), result.b[index].cpu()
                executions[module.name] = result.info
            del result
    parameters = {
        "method": "one-sided-spectral",
        "requested_eigh_backend": eigh_backend,
        "batch_size": batch_size,
    }
    if budget is not None:
        parameters["allocation"] = {
            "objective": "unscaled-spectral-energy",
            "direction_budget": budget,
            "min_rank": 1,
            "ranks": plan,
        }
    return _export(adapter, output, factors, output_dtype, executions, parameters)
