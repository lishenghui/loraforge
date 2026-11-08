# SPDX-License-Identifier: Apache-2.0
"""One compact BF16 adapter, two token rows and one base-only row."""

import torch

from loraforge.ops import row_expand, row_shrink

device = "cuda:0"
x = torch.ones(3, 8, device=device, dtype=torch.bfloat16)
a = torch.ones(1, 2, 8, device=device, dtype=torch.bfloat16)
z = torch.zeros(3, 2, device=device, dtype=torch.bfloat16)
b = torch.ones(6, device=device, dtype=torch.bfloat16)
y = torch.ones(3, 3, device=device, dtype=torch.bfloat16)


def metadata(values, dtype=torch.int32):
    return torch.tensor(values, device=device, dtype=dtype)


ranks = metadata([2])
row_shrink(x, a, z, metadata([0, -1, 0]), ranks, max_rank=2)
row_expand(
    z,
    b,
    y,
    b_offsets=metadata([0], torch.int64),
    b_strides=metadata([2]),
    ranks=ranks,
    scales=metadata([0.5], torch.float32),
    segment_indptr=metadata([0, 2]),
    segment_slots=metadata([0]),
    token_indices=metadata([2, 0]),
    slice_offsets=metadata([0, 3]),
    segment_count=metadata([1]),
)
torch.testing.assert_close(
    y.cpu(), torch.tensor([[9, 9, 9], [1, 1, 1], [9, 9, 9]], dtype=torch.bfloat16)
)
print(y.cpu())
