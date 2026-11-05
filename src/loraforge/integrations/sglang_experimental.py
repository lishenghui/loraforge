# SPDX-License-Identifier: Apache-2.0
"""Explicit alpha integrations; imports do not load SGLang or device libraries."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from loraforge.annotations import ExperimentalAPI

_NF4_MODE = None
_GDN_ENABLED = False
_GDN_PACKED = None


def _check_runtime():
    try:
        actual = version("sglang")
    except PackageNotFoundError as exc:
        raise RuntimeError("install the optional SGLang 0.5.22 runtime first") from exc
    if actual != "0.5.22":
        raise RuntimeError("experimental Qwen3.5 integration requires SGLang 0.5.22")
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("experimental Qwen3.5 integration requires ROCm gfx90a")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if arch.split(":")[0] != "gfx90a":
        raise RuntimeError("experimental Qwen3.5 integration requires ROCm gfx90a")


@ExperimentalAPI(
    reason="Prepared Qwen3.5-0.8B NF4/BF16 fixture, SGLang 0.5.22, TP1, gfx90a only.",
    warn=True,
)
def install_qwen35_nf4(*, mode: str = "byte-pairs") -> None:
    """Install NF4 admission, checked loading and optional prepared linear dispatch.

    Call before constructing SGLang models in each worker. ``mode`` is stock,
    cached-scales or byte-pairs. LoRA remains BF16. This does not select a LoRA
    backend or install other fusion prototypes. Base weights are immutable
    during graph replay. Repeated identical installation is harmless; changing
    the mode in a live process is rejected. Prepared checkpoint receipts and
    all114 actual projections are checked before serving.
    """
    global _NF4_MODE
    if mode not in ("stock", "cached-scales", "byte-pairs"):
        raise ValueError("mode must be stock, cached-scales, or byte-pairs")
    if _GDN_ENABLED:
        raise RuntimeError("BF16 GDN projection fusion cannot be combined with NF4")
    if _NF4_MODE is not None and _NF4_MODE != mode:
        raise RuntimeError(
            "NF4 integration is already installed with different options"
        )
    _check_runtime()
    if version("bitsandbytes") != "0.50.2":
        raise RuntimeError("experimental Qwen3.5 NF4 requires bitsandbytes 0.50.2")
    from ._sglang_nf4_compat import install as install_compat
    from ._sglang_qwen35_loader import install as install_loader

    install_compat()
    install_loader()
    if mode != "stock":
        from ._sglang_nf4_linear import install as install_linear

        install_linear(byte_pairs=mode == "byte-pairs")
    _NF4_MODE = mode


@ExperimentalAPI(
    reason="Immutable BF16 Qwen3.5-0.8B GDN parameters must be prepared before graphs.",
)
def share_qwen35_gdn_storage(layer: Any, *, packed_weight: Any = None) -> bool:
    """Make original parameters alias an existing packed [8224,1024] owner.

    ``layer`` exposes in_proj_qkvz and in_proj_ba, optionally LoRA-wrapped.
    Pass the already loaded packed BF16 weight explicitly, or use the owner's
    existing ``_lf_packed``. This operation validates equality and preserves
    parameter objects/metadata. Returns whether a rebind occurred. It cannot
    validate whether a previously captured graph still references old storage:
    callers must invoke it before capture or publishing parameter views.
    """
    from ._sglang_gdn_storage import consolidate

    result = consolidate(layer, packed=packed_weight)
    if result is None:
        raise ValueError("an existing packed BF16 GDN weight is required")
    return result


@ExperimentalAPI(
    reason="BF16 Qwen3.5-0.8B GDN decode fusion, SGLang 0.5.22, TP1, gfx90a only.",
    warn=True,
)
def install_qwen35_gdn(*, packed_checkpoint: bool = False) -> None:
    """Fuse GDN base projections, retain individual LoRA updates and share storage.

    Install before model construction in each worker. Decode with csgmv and
    BF16 [8192,1024]/[32,1024] base weights uses one packed projection, then
    each original Linear's LoRA update. Prefill retains the original execution.
    This installs no consumer, tiny-rank or private benchmark hooks. Projection
    parameters are immutable after loading; TP, quantized bases and live reload
    are unsupported. Existing conflicting projection patches are rejected.
    Set ``packed_checkpoint=True`` only for the canonical 114-projection BF16
    fixture; it installs the checked loader as well. Leave it false for the
    original checkpoint layout.
    """
    global _GDN_ENABLED, _GDN_PACKED
    if type(packed_checkpoint) is not bool:
        raise TypeError("packed_checkpoint must be a bool")
    if _GDN_ENABLED and _GDN_PACKED != packed_checkpoint:
        raise RuntimeError(
            "GDN integration is already installed with different options"
        )
    if _NF4_MODE is not None:
        raise RuntimeError("BF16 GDN projection fusion cannot be combined with NF4")
    _check_runtime()
    from ._sglang_gdn import install

    if packed_checkpoint:
        from ._sglang_qwen35_loader import install as install_loader

        install_loader()
    install()
    _GDN_ENABLED = True
    _GDN_PACKED = packed_checkpoint
