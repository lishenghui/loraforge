# SPDX-License-Identifier: Apache-2.0
"""Opt-in, immutable BF16 GDN base-storage consolidation before graph capture.

The packed projection owns the values; original Linear parameters keep their
identity and metadata, and become contiguous views of that same storage.
This is inference-only, TP1 and specific to the measured 8192+32 by 1024 pair.
Never call after capturing a graph or publishing parameter views to consumers.
"""


def consolidate(layer, *, packed=None):
    import torch

    packed = getattr(layer, "_lf_packed", None) if packed is None else packed
    if packed is None:
        return None
    q = getattr(layer.in_proj_qkvz, "base_layer", layer.in_proj_qkvz).weight
    b = getattr(layer.in_proj_ba, "base_layer", layer.in_proj_ba).weight
    if (
        packed.shape != (8224, 1024)
        or q.shape != (8192, 1024)
        or b.shape != (32, 1024)
        or not packed.is_contiguous()
        or any(t.dtype != torch.bfloat16 for t in (packed, q, b))
        or any(t.device != packed.device for t in (q, b))
        or any(t.requires_grad for t in (packed, q, b))
    ):
        raise ValueError("requires immutable, same-device BF16 TP1 GDN weights")
    if packed.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("consolidate GDN weights before graph capture")
    qview, bview = packed[:8192], packed[8192:]
    if q.data_ptr() == qview.data_ptr() and b.data_ptr() == bview.data_ptr():
        return False
    if not torch.equal(q, qview) or not torch.equal(b, bview):
        raise ValueError("packed GDN projection differs from current parameters")
    q.data = qview
    b.data = bview
    return True


def wrap_class(cls):
    """Cover both explicit finalization and the prototype's closure-based lazy path."""
    if getattr(cls, "_lf_consolidated_storage", False):
        return
    original = cls.finalize_fused_in_proj
    projection = cls._forward_input_proj

    def finalize(self):
        # The existing benchmark hook prepares the packed projection after
        # loading. Its warmup runs before graphs; do not replace a live graph.
        result = original(self)
        consolidate(self)
        self._lf_storage_ready = True
        return result

    def forward(self, *args, **kwargs):
        if not getattr(self, "_lf_storage_ready", False):
            # The prototype calls a closed-over finalize(), bypassing the class
            # method. Prepare first so that path observes an existing owner.
            if not hasattr(self, "_lf_packed"):
                finalize(self)
            else:
                consolidate(self)
                self._lf_storage_ready = True
        return projection(self, *args, **kwargs)

    cls.finalize_fused_in_proj = finalize
    cls._forward_input_proj = forward
    cls._lf_consolidated_storage = True


def install():
    import os

    from sglang import __version__
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    assert __version__ == "0.5.22"
    assert os.environ.get("LF_GDN_STORAGE") == "1"
    assert os.environ.get("LF_GDN_ARM") in ("packed", "both")
    wrap_class(Qwen3_5GatedDeltaNet)
