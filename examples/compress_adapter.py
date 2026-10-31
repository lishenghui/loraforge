# SPDX-License-Identifier: Apache-2.0
"""Compress a local PEFT adapter on an allocated GPU; no model download needed."""

import argparse

from loraforge.adapters import compress_peft_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--keep", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-dtype",
        choices=["source", "float32", "float16", "bfloat16"],
        default="source",
    )
    args = parser.parse_args()
    result = compress_peft_adapter(
        args.source,
        args.output,
        args.keep,
        device=args.device,
        batch_size=args.batch_size,
        output_dtype=args.output_dtype,
    )
    print("Saved:", result.directory)
    print("Tensor bytes:", result.manifest["output"]["tensor_bytes"])
    print(
        "Weight-file bytes:",
        result.manifest["output"]["files"]["adapter_model.safetensors"]["bytes"],
    )


if __name__ == "__main__":
    main()
