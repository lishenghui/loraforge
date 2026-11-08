# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in decode integration for SGLang 0.5.22."""

import os
from functools import lru_cache
from importlib.metadata import version


def _check_version():
    if version("sglang") != "0.5.22":
        raise RuntimeError(
            "LoRAForge SGLang integration requires exactly SGLang 0.5.22"
        )


def _decode_policy():
    policy = os.environ.get("LORAFORGE_SGLANG_POLICY", "compatible")
    if policy not in ("compatible", "rank-aware", "fleet"):
        raise ValueError(
            "LORAFORGE_SGLANG_POLICY must be compatible, rank-aware, or fleet"
        )
    return policy


@lru_cache(maxsize=1)
def get_backend():
    """Resolve the explicitly selected policy inside each worker."""
    _check_version()
    policy = _decode_policy()
    if policy == "fleet":
        if os.environ.get("LORAFORGE_STABLE_PREFILL", "0") not in ("", "0"):
            from ._sglang_stable_prefill import install as install_prefill

            install_prefill()
        from ._sglang_fleet import backend_from_environment

        return backend_from_environment()
    import torch
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend

    from loraforge.ops.gemv import lora_expand, lora_shrink

    class LoRAForgeDecodeBackend(ChunkedSgmvLoRABackend):
        supports_prefill_cuda_graph = False

        def prepare_lora_batch(self, forward_batch, *args, **kwargs):
            super().prepare_lora_batch(forward_batch, *args, **kwargs)
            self._loraforge_decode = forward_batch.forward_mode.is_decode()

        def _use_decode(self, x, pruned=None):
            return (
                getattr(self, "_loraforge_decode", False)
                and pruned is None
                and x.shape[0] > 0
                and self.batch_info is not None
                and self.batch_info.req_weight_indices is not None
            )

        def _shrink(self, x, weights, slices=1):
            if (
                policy == "rank-aware"
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
            ):
                from ._sglang_rank_aware import shrink

                return shrink(x, weights, self.batch_info, slices)
            return lora_shrink(
                x,
                weights,
                self.batch_info.lora_ranks,
                self.batch_info.req_weight_indices,
                slices,
            )

        def _expand(self, x, weights, offsets, max_slice, base, offsets_cpu=None):
            if policy == "rank-aware" and x.dtype == torch.bfloat16:
                from sglang.kernels.ops.lora.dense.chunked_sgmv_expand import (
                    chunked_sgmv_lora_expand_forward,
                )

                return chunked_sgmv_lora_expand_forward(
                    x=x,
                    weights=weights,
                    batch_info=self.batch_info,
                    slice_offsets=offsets,
                    slice_offsets_cpu=offsets_cpu,
                    max_slice_size=max_slice,
                    base_output=base,
                )
            return lora_expand(
                x,
                weights,
                self.batch_info.lora_ranks,
                self.batch_info.scalings,
                self.batch_info.req_weight_indices,
                offsets,
                max_slice,
                base,
            )

        def run_lora_a_sgemm(
            self, x, weights, pruned_batch_info=None, stack_num=1, *args, **kwargs
        ):
            if self._use_decode(x, pruned_batch_info):
                return self._shrink(x, weights, stack_num)
            return super().run_lora_a_sgemm(
                x, weights, pruned_batch_info, stack_num, *args, **kwargs
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
            if self._use_decode(x, pruned_batch_info):
                return self._expand(
                    x,
                    weights,
                    output_offset,
                    weights.shape[-2],
                    base_output,
                    kwargs.get("output_offset_cpu"),
                )
            return super().run_lora_b_sgemm(
                x,
                weights,
                output_offset,
                base_output,
                pruned_batch_info,
                *args,
                **kwargs,
            )

        def run_qkv_lora(
            self,
            x,
            qkv_lora_a,
            qkv_lora_b,
            output_offset,
            max_qkv_out_dim,
            base_output=None,
            n_slices=3,
            *args,
            **kwargs,
        ):
            if self._use_decode(x):
                return self._expand(
                    self._shrink(x, qkv_lora_a, n_slices),
                    qkv_lora_b,
                    output_offset,
                    max_qkv_out_dim,
                    base_output,
                    kwargs.get("output_offset_cpu"),
                )
            return super().run_qkv_lora(
                x,
                qkv_lora_a,
                qkv_lora_b,
                output_offset,
                max_qkv_out_dim,
                base_output,
                n_slices,
                *args,
                **kwargs,
            )

        def run_gate_up_lora(
            self,
            x,
            gate_up_lora_a,
            gate_up_lora_b,
            output_offset,
            base_output=None,
            *args,
            **kwargs,
        ):
            if self._use_decode(x):
                return self._expand(
                    self._shrink(x, gate_up_lora_a, 2),
                    gate_up_lora_b,
                    output_offset,
                    gate_up_lora_b.shape[-2] // 2,
                    base_output,
                    kwargs.get("output_offset_cpu"),
                )
            return super().run_gate_up_lora(
                x,
                gate_up_lora_a,
                gate_up_lora_b,
                output_offset,
                base_output,
                *args,
                **kwargs,
            )

    return LoRAForgeDecodeBackend


def install_sglang_backend():
    """Opt into LoRAForge decode for csgmv in this process; repeatable per worker.

    Refuse to overwrite another plugin. Native SGLang owns fixed-address buffers,
    routing updates, loader synchronization and graph lifetimes.
    """
    _check_version()
    from sglang.srt.lora.backend.lora_registry import LORA_SUPPORTED_BACKENDS

    current = LORA_SUPPORTED_BACKENDS.get("csgmv")
    if current is get_backend:
        return
    if (
        getattr(current, "__module__", None) != "sglang.srt.lora.backend.lora_registry"
        or getattr(current, "__name__", None) != "create_triton_csgmv_backend"
    ):
        raise RuntimeError("csgmv is already owned by another plugin")
    LORA_SUPPORTED_BACKENDS["csgmv"] = get_backend


def install():
    """SGLang plugin entrypoint; inert unless explicitly enabled."""
    nf4 = os.environ.get("LORAFORGE_EXPERIMENTAL_QWEN35_NF4", "")
    gdn = os.environ.get("LORAFORGE_EXPERIMENTAL_QWEN35_GDN", "0")
    if nf4 and gdn not in ("", "0"):
        raise ValueError("experimental NF4 and BF16 GDN fusion are mutually exclusive")
    if gdn not in ("", "0", "1", "packed"):
        raise ValueError("LORAFORGE_EXPERIMENTAL_QWEN35_GDN must be 0, 1, or packed")
    if nf4:
        from .sglang_experimental import install_qwen35_nf4

        install_qwen35_nf4(mode=nf4)
    if gdn in ("1", "packed"):
        from .sglang_experimental import install_qwen35_gdn

        install_qwen35_gdn(packed_checkpoint=gdn == "packed")
    if os.environ.get("LORAFORGE_SGLANG_QWEN35") == "1":
        from ._sglang_qwen35 import install_qwen35_lora

        install_qwen35_lora()
    if os.environ.get("LORAFORGE_SGLANG") == "1":
        install_sglang_backend()
