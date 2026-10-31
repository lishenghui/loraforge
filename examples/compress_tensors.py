# SPDX-License-Identifier: Apache-2.0
"""Run on an allocated GPU: python examples/compress_tensors.py."""

import torch

from loraforge.compression import truncate_lora


def main() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(123)
    a = torch.randn(4, 16, 128, device=device, generator=generator)
    b = torch.randn(4, 96, 16, device=device, generator=generator)
    result = truncate_lora(a, b, keep=8)
    print("Compact shapes:", result.a.shape, result.b.shape)
    print("Executed:", result.info)
    print(
        "Tensor bytes:", sum(t.numel() * t.element_size() for t in (result.a, result.b))
    )


if __name__ == "__main__":
    main()
