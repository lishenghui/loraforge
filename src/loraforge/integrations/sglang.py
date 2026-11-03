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


@lru_cache(maxsize=1)
def get_backend():
    """Resolve inside each worker; preserve the native CSGMV prefill path."""
    _check_version()
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
            return lora_shrink(
                x,
                weights,
                self.batch_info.lora_ranks,
                self.batch_info.req_weight_indices,
                slices,
            )

        def _expand(self, x, weights, offsets, max_slice, base):
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
                    x, weights, output_offset, weights.shape[-2], base_output
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
    if os.environ.get("LORAFORGE_SGLANG") == "1":
        install_sglang_backend()
