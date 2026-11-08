# SPDX-License-Identifier: Apache-2.0
"""Exact project-owned NF4 reconstruction kernels, imported on first use."""

import triton
import triton.language as tl


@triton.jit
def _code_value(code):
    # Canonical NF4 codebook, expressed as register selections instead of
    # per-element indirect global loads. Exact FP32 constants, not an approximation.
    a = tl.where((code & 1) != 0, -0.6961928009986877, -1.0)
    b = tl.where((code & 1) != 0, -0.39491748809814453, -0.5250730514526367)
    c = tl.where((code & 1) != 0, -0.18477343022823334, -0.28444138169288635)
    d = tl.where((code & 1) != 0, 0.0, -0.09105003625154495)
    e = tl.where((code & 1) != 0, 0.16093020117282867, 0.07958029955625534)
    f = tl.where((code & 1) != 0, 0.33791524171829224, 0.24611230194568634)
    g = tl.where((code & 1) != 0, 0.5626170039176941, 0.44070982933044434)
    h = tl.where((code & 1) != 0, 1.0, 0.7229568362236023)
    ab = tl.where((code & 2) != 0, b, a)
    cd = tl.where((code & 2) != 0, d, c)
    ef = tl.where((code & 2) != 0, f, e)
    gh = tl.where((code & 2) != 0, h, g)
    low = tl.where((code & 4) != 0, cd, ab)
    high = tl.where((code & 4) != 0, gh, ef)
    return tl.where((code & 8) != 0, high, low)


@triton.jit
def _values(Q, S, C, index, valid):
    byte = tl.load(Q + index // 2, mask=valid, other=0).to(tl.int32)
    code = tl.where(index % 2 == 0, byte >> 4, byte & 15)
    value = _code_value(code)
    scale = tl.load(S + index // 64, mask=valid, other=0)
    return (value * scale).to(tl.bfloat16)


@triton.jit
def _dequant(Q, S, C, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, _values(Q, S, C, i, i < SIZE), mask=i < SIZE)


@triton.jit
def _dequant_pairs(
    Q, S, C, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr, LOOKUP: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(Q + i, mask=i < SIZE // 2, other=0).to(tl.int32)
    scale = tl.load(S + i // 32, mask=i < SIZE // 2, other=0)
    if LOOKUP:
        high_value = tl.load(C + (byte >> 4))
        low_value = tl.load(C + (byte & 15))
    else:
        high_value = _code_value(byte >> 4)
        low_value = _code_value(byte & 15)
    high = (high_value * scale).to(tl.bfloat16)
    low = (low_value * scale).to(tl.bfloat16)
    values = tl.interleave(high, low)
    j = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    tl.store(Y + j, values, mask=j < SIZE)
