# SPDX-License-Identifier: Apache-2.0
"""Experimental exact NF4 tensor operations; device dependencies load on use."""

from loraforge.annotations import ExperimentalAPI


def _dequantize(qweight, scales, code, shape, *, byte_pairs=True, out=None):
    """Internal prepared dispatch; callers own validation and immutable metadata."""
    import torch
    import triton

    from ._nf4_kernels import _dequant, _dequant_pairs

    if out is None:
        out = torch.empty(shape, device=qweight.device, dtype=torch.bfloat16)
    if byte_pairs:
        _dequant_pairs[(triton.cdiv(out.numel(), 2048),)](
            qweight,
            scales,
            code,
            out,
            out.numel(),
            1024,
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _dequant[(triton.cdiv(out.numel(), 1024),)](
            qweight,
            scales,
            code,
            out,
            out.numel(),
            1024,
            enable_fp_fusion=False,
        )
    return out


@ExperimentalAPI(
    reason="Validated on gfx90a with canonical NF4/block64 and BF16 output."
)
def dequantize_nf4(qweight, scales, code, shape: tuple[int, int], *, out=None):
    """Reconstruct exact BF16 weights from contiguous packed NF4 bytes.

    ``qweight`` contains N*K/2 uint8 bytes (high nibble first), ``scales``
    N*K/64 FP32 block scales, and ``code`` the canonical FP32 16-entry NF4
    codebook. All tensors share a ROCm gfx90a device and are contiguous.
    Shape dimensions are positive multiples of64. Optional ``out`` is a
    nonaliasing contiguous BF16 [N,K] allocation on that device. Immutable
    inputs and metadata values are the caller's responsibility during replay.
    The operation executes on the current PyTorch stream and has no BF16 cache.
    The marker is metadata-only and adds no per-call warning wrapper.
    """
    import torch

    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(v) is not int or v <= 0 or v % 64 for v in shape)
    ):
        raise ValueError("shape must contain two positive multiples of64")
    size = shape[0] * shape[1]
    if any(not isinstance(t, torch.Tensor) for t in (qweight, scales, code)):
        raise TypeError("packed weights, scales and code must be tensors")
    device = qweight.device
    if device.type != "cuda" or any(t.device != device for t in (scales, code)):
        raise ValueError("all tensors must share one CUDA/ROCm device")
    if (
        not torch.version.hip
        or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        != "gfx90a"
    ):
        raise RuntimeError("experimental NF4 reconstruction requires ROCm gfx90a")
    if (
        qweight.dtype != torch.uint8
        or qweight.numel() != size // 2
        or scales.dtype != torch.float32
        or scales.numel() != size // 64
        or code.dtype != torch.float32
        or code.numel() != 16
        or any(
            not t.is_contiguous() or t.requires_grad for t in (qweight, scales, code)
        )
    ):
        raise ValueError(
            "requires contiguous uint8 bytes, FP32 block64 scales and a 16-entry codebook"
        )
    if out is not None:
        if (
            not isinstance(out, torch.Tensor)
            or out.device != device
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != shape
            or not out.is_contiguous()
            or out.requires_grad
        ):
            raise ValueError("out must be contiguous BF16 [N,K] on the input device")
        if any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (qweight, scales, code)
        ):
            raise ValueError("out must not alias an input storage")
    with torch.cuda.device(device):
        return _dequantize(qweight, scales, code, shape, out=out)


def prepare_scales(state):
    """Own a lossless FP32 scale vector; run before capture, for immutable weights."""
    import torch
    from bitsandbytes.functional import dequantize_blockwise

    if (
        state.quant_type != "nf4"
        or state.blocksize != 64
        or state.dtype != torch.bfloat16
    ):
        raise ValueError("requires NF4 block64 BF16 source weights")
    expected = torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020117282867,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=torch.float32,
        device=state.code.device,
    )
    if not torch.equal(expected, state.code):
        raise ValueError("requires the canonical NF4 codebook")
    if state.nested:
        return (
            dequantize_blockwise(state.absmax, state.state2) + state.offset
        ).contiguous()
    return state.absmax.contiguous()
