# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Shenghui Li. Extracted from probe 5fd6ae060d106066b306959f9a34b726a281ef4c stable_prefill.py.
"""Opt-in Qwen3 BF16 prefill numerics shared by eager and bucketed graphs.

Fixed row-wise RMSNorm and canonical GEMM rows are used only up to the
configured token cap. Decode and larger eager prefills retain their old path.
"""

import functools
import logging
import os
from importlib.metadata import version

import torch
import triton
import triton.language as tl

PREFILL = [False]
TOKEN_ROWS = [0]


def canonical_tokens(n):
    if n <= 0:
        raise ValueError("prefill token count must be positive")
    return max(4, 1 << (n - 1).bit_length())


@triton.jit
def _norm(
    X,
    R,
    W,
    Y,
    RO,
    D: tl.constexpr,
    EPS: tl.constexpr,
    RESIDUAL: tl.constexpr,
    HF_CAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    if RESIDUAL:
        r = tl.load(R + row * D + col, col < D, other=0).to(tl.float32)
        x = x + r
        tl.store(RO + row * D + col, x, col < D)
    var = tl.sum(x * x, 0) / D
    z = x * tl.rsqrt(var + EPS)
    if HF_CAST:
        z = z.to(Y.dtype.element_ty).to(tl.float32)
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    tl.store(Y + row * D + col, z * w, col < D)


def fixed_norm(module, x, residual=None, post=None):
    if post is not None:
        # Preserve the existing HIP path's BF16 residual+post addition.
        residual = residual + post if residual is not None else residual
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    y = torch.empty_like(x)
    ro = torch.empty_like(residual) if residual is not None else y
    d = x.shape[-1]
    _norm[(x.numel() // d,)](
        x,
        residual if residual is not None else x,
        module.weight,
        y,
        ro,
        d,
        module.variance_epsilon,
        residual is not None,
        module.cast_x_before_out_mul,
        triton.next_power_of_2(d),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (y, ro) if residual is not None else y


def install():
    mode = os.environ.get("LORAFORGE_STABLE_PREFILL", "0")
    if mode in ("", "0"):
        return
    if mode not in ("triton", "vllm"):
        raise ValueError("LORAFORGE_STABLE_PREFILL must be 0, triton, or vllm")
    if version("sglang") != "0.5.22":
        raise RuntimeError("Stable prefill was validated for SGLang 0.5.22 only")
    if os.environ.get("LORAFORGE_PREFILL_GRAPH_EXACT") == "1":
        raise RuntimeError("Choose stable prefill or the legacy exact-token gate")
    cap = int(os.environ.get("LORAFORGE_STABLE_PREFILL_MAX_TOKENS", "256"))
    if cap < 4 or cap & (cap - 1):
        raise ValueError(
            "LORAFORGE_STABLE_PREFILL_MAX_TOKENS must be a power of two >= 4"
        )
    from sglang.srt.layers import layernorm as layernorm_module
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.qwen3 import Qwen3Model

    if layernorm_module._use_aiter:
        raise RuntimeError("Stable prefill requires SGLANG_USE_AITER=0")
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    if getattr(Qwen3Model, "_loraforge_stable_prefill_installed", False):
        return
    logger = logging.getLogger(__name__)
    reported = set()
    buckets = set()
    capturing = [0]

    old_init = ModelRunner.maybe_init_lora_manager

    def initialize(self, *args, **kwargs):
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_exec

        base = getattr(self.model, "model", None)
        arch = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()),
            "gcnArchName",
            "",
        )
        cuda_opt_in = (
            not torch.version.hip
            and os.environ.get("LORAFORGE_FLEET_CUDA") == "1"
            and torch.cuda.get_device_capability() == (9, 0)
        )
        if not arch.startswith("gfx90a") and not cuda_opt_in:
            raise RuntimeError(
                "Stable prefill requires gfx90a or explicit LORAFORGE_FLEET_CUDA=1 on SM90; validate eager/graph outputs before timing"
            )
        if (
            type(base) is not Qwen3Model
            or self.is_draft_worker
            or get_tensor_model_parallel_world_size() != 1
        ):
            raise RuntimeError("Stable prefill requires a single-TP Qwen3 target model")
        if next(self.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError("Stable prefill requires BF16 model weights")
        if (
            get_exec().deterministic.rl_on_policy_target is not None
            or envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        ):
            raise RuntimeError(
                "Stable prefill has not been validated with deterministic/RL modes"
            )
        qkv = base.layers[base.start_layer].self_attn.qkv_proj
        if type(qkv.quant_method).__name__ != "UnquantizedLinearMethod":
            raise RuntimeError("Stable prefill requires unquantized BF16 linears")
        for norm in self.model.modules():
            if isinstance(norm, RMSNorm) and (
                norm.variance_size_override is not None
                or norm.fp32_residual
                or norm.override_orig_dtype is not None
                or norm._quant_linear is not None
            ):
                raise RuntimeError(
                    "Stable prefill does not support specialized RMSNorm modes"
                )
        return old_init(self, *args, **kwargs)

    ModelRunner.maybe_init_lora_manager = initialize

    old_model = Qwen3Model.forward

    @functools.wraps(old_model)
    def model_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        previous, previous_rows = PREFILL[0], TOKEN_ROWS[0]
        TOKEN_ROWS[0] = len(forward_batch.input_ids)
        PREFILL[0] = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_cuda_graph()
            and 0 < TOKEN_ROWS[0] <= cap
        )
        try:
            return old_model(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            PREFILL[0], TOKEN_ROWS[0] = previous, previous_rows

    Qwen3Model.forward = model_forward

    norm_method = "forward_hip" if torch.version.hip else "forward_cuda"
    old_norm = getattr(RMSNorm, norm_method)

    def norm_forward(
        self, x, residual=None, post_residual_addition=None, quant_linear=None
    ):
        if (
            not PREFILL[0]
            or x.numel() == 0
            or self.variance_size_override is not None
            or self.fp32_residual
            or self.override_orig_dtype is not None
            or quant_linear is not None
            or self._quant_linear is not None
        ):
            return old_norm(self, x, residual, post_residual_addition, quant_linear)
        if "norm" not in reported:
            logger.info(
                "LORAFORGE_STABLE_PREFILL_NORM_USED mode=%s shape=%s cap=%s",
                mode,
                tuple(x.shape),
                cap,
            )
            reported.add("norm")
        if mode == "triton":
            return fixed_norm(self, x, residual, post_residual_addition)
        shape, d = x.shape, x.shape[-1]
        x = x.reshape(-1, d)
        n = x.shape[0]
        assert n % TOKEN_ROWS[0] == 0
        target = canonical_tokens(TOKEN_ROWS[0]) * (n // TOKEN_ROWS[0])

        def padded(t):
            return (
                None
                if t is None
                else torch.nn.functional.pad(t.reshape(-1, d), (0, 0, 0, target - n))
            )

        out = old_norm(
            self,
            padded(x),
            padded(residual),
            padded(post_residual_addition),
            quant_linear,
        )
        if isinstance(out, tuple):
            return tuple(v[:n].reshape(shape) for v in out)
        return out[:n].reshape(shape)

    setattr(RMSNorm, norm_method, norm_forward)

    old_linear = UnquantizedLinearMethod.apply

    def linear(self, layer, x, bias=None):
        if PREFILL[0] and x.ndim == 2 and x.shape[0] > 0:
            n, target = x.shape[0], canonical_tokens(x.shape[0])
            if target != n:
                if "gemm" not in reported:
                    logger.info(
                        "LORAFORGE_STABLE_PREFILL_GEMM_PAD rows=%s canonical=%s",
                        n,
                        target,
                    )
                    reported.add("gemm")
                return old_linear(
                    self, layer, torch.nn.functional.pad(x, (0, 0, 0, target - n)), bias
                )[:n]
        return old_linear(self, layer, x, bias)

    UnquantizedLinearMethod.apply = linear

    # Graph capture and eager reference must use the same canonical rows. A
    # sparse/custom capture list may otherwise round up beyond that shape.
    old_capture = PrefillCudaGraphRunner.capture_one_shape

    @functools.wraps(old_capture)
    def capture_one_shape(self, *args, **kwargs):
        capturing[0] += 1
        try:
            return old_capture(self, *args, **kwargs)
        finally:
            capturing[0] -= 1

    PrefillCudaGraphRunner.capture_one_shape = capture_one_shape
    old_prepare = PrefillCudaGraphRunner.capture_prepare

    def capture_prepare(self, *args, **kwargs):
        buckets.update(self.capture_num_tokens)
        return old_prepare(self, *args, **kwargs)

    PrefillCudaGraphRunner.capture_prepare = capture_prepare
    old_can = PrefillCudaGraphRunner.can_run_graph

    def can_run(self, fb, *args, **kwargs):
        n = len(fb.input_ids)
        if not 0 < n <= cap or canonical_tokens(n) not in self.capture_num_tokens:
            return False
        return old_can(self, fb, *args, **kwargs)

    PrefillCudaGraphRunner.can_run_graph = can_run
    old_lora = LoRAManager.can_use_prefill_cuda_graph

    def can_use_lora(self, fb, *args, **kwargs):
        if not old_lora(self, fb, *args, **kwargs):
            return False
        if capturing[0]:
            return (
                True  # All capture buckets must bind refreshable static LoRA metadata.
            )
        n = len(fb.input_ids)
        return 0 < n <= cap and canonical_tokens(n) in buckets

    LoRAManager.can_use_prefill_cuda_graph = can_use_lora
    Qwen3Model._loraforge_stable_prefill_installed = True
    logger.info("LORAFORGE_STABLE_PREFILL_INSTALLED mode=%s max_tokens=%s", mode, cap)
