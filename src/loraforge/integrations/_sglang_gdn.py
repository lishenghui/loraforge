# SPDX-License-Identifier: Apache-2.0
"""Process-local BF16 GDN decode fusion with native adapter updates."""


def install():
    import torch
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
    from sglang.srt.models import qwen3_5 as model

    from ._sglang_gdn_storage import consolidate

    cls = model.Qwen3_5GatedDeltaNet
    if getattr(cls, "_loraforge_gdn_installed", False):
        return
    if cls._forward_input_proj.__module__ != "sglang.srt.models.qwen3_5":
        raise RuntimeError("GDN projection is already owned by another patch")
    original = cls._forward_input_proj
    stats = dict(prepared=0, fused_calls=0, fallback_calls=0)

    def prepare(self):
        if getattr(self, "_loraforge_gdn_ready", False):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepare GDN base storage in eager warmup before capture"
            )
        q = getattr(self.in_proj_qkvz, "base_layer", self.in_proj_qkvz)
        b = getattr(self.in_proj_ba, "base_layer", self.in_proj_ba)
        if (
            getattr(self, "attn_tp_size", 1) != 1
            or q.weight.shape != (8192, 1024)
            or b.weight.shape != (32, 1024)
            or q.weight.dtype != torch.bfloat16
            or b.weight.dtype != torch.bfloat16
            or q.bias is not None
            or b.bias is not None
            or q.weight.requires_grad
            or b.weight.requires_grad
            or q.weight.device != b.weight.device
        ):
            raise ValueError(
                "GDN fusion requires immutable unbiased BF16 TP1 Qwen3.5-0.8B projections"
            )
        packed = getattr(self, "_fused_in_proj_weight", None)
        if packed is None:
            packed = torch.cat((q.weight.detach(), b.weight.detach()), dim=0)
        self._lf_packed = packed
        consolidate(self)
        self._loraforge_gdn_ready = True
        stats["prepared"] += 1

    def forward(self, x):
        q, b = self.in_proj_qkvz, self.in_proj_ba
        if not hasattr(q, "lora_backend"):
            stats["fallback_calls"] += 1
            return original(self, x)
        if not isinstance(q.lora_backend, ChunkedSgmvLoRABackend):
            raise ValueError("experimental GDN fusion requires the csgmv LoRA backend")
        prepare(self)
        if not getattr(q.lora_backend, "_loraforge_gdn_decode", False):
            stats["fallback_calls"] += 1
            return original(self, x)
        base = model.bf16_gemm_dispatch(x, self._lf_packed, None)
        qbase, bbase = base[:, :8192], base[:, 8192:]
        stats["fused_calls"] += 1
        return (
            q.apply_lora(qbase, x) if q.lora_active else qbase,
            b.apply_lora(bbase, x) if b.lora_active else bbase,
        )

    previous_prepare = ChunkedSgmvLoRABackend.prepare_lora_batch

    def prepare_batch(self, batch, *args, **kwargs):
        result = previous_prepare(self, batch, *args, **kwargs)
        self._loraforge_gdn_decode = batch.forward_mode.is_decode()
        return result

    ChunkedSgmvLoRABackend.prepare_lora_batch = prepare_batch
    cls._forward_input_proj = forward
    cls._loraforge_gdn_installed = True
    cls._loraforge_gdn_stats = stats
