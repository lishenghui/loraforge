# SPDX-License-Identifier: Apache-2.0
"""Invalid calls fail before importing Triton or substituting CPU computation."""

import pytest

torch = pytest.importorskip("torch")

from loraforge.ops import lora_expand, lora_shrink  # noqa: E402


def test_shrink_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA/ROCm"):
        lora_shrink(
            torch.zeros(1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )


def test_expand_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA/ROCm"):
        lora_expand(
            torch.zeros(1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([0, 1]),
            1,
        )


def test_non_tensor_input():
    with pytest.raises(TypeError, match="tensors"):
        lora_shrink(None, None, None, None)
