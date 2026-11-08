# SPDX-License-Identifier: Apache-2.0
"""GPU-only NF4 quantization, with BF16 embeddings/non-decoder tensors retained."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def main():
    import torch
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    from safetensors import safe_open
    from safetensors.torch import save_file

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    if (root / "nf4").exists() or (root / "nf4").is_symlink():
        raise FileExistsError("retain existing NF4 attempt; use a new output root")
    src = root / "packed-bf16"
    out = root / f"nf4-attempt-{os.environ.get('SLURM_JOB_ID', str(time.time_ns()))}"
    out.mkdir()
    receipt = json.loads((src / "packed-receipt.json").read_text())
    targets = receipt["targets"]
    weights = {}
    checks = []
    torch.set_num_threads(7)
    with safe_open(src / "model.safetensors", framework="pt", device="cpu") as f:
        for name in f.keys():
            value = f.get_tensor(name)
            if name in targets:
                assert value.dtype == torch.bfloat16 and tuple(value.shape) == tuple(
                    targets[name]
                )
                gpu = value.cuda()
                packed, state = quantize_4bit(
                    gpu, blocksize=64, compress_statistics=True, quant_type="nf4"
                )
                restored = dequantize_4bit(packed, quant_state=state)
                delta = gpu.float() - restored.float()
                checks.append(
                    dict(
                        name=name,
                        shape=list(value.shape),
                        source_bytes=value.numel() * 2,
                        packed_bytes=packed.numel(),
                        quant_type=state.quant_type,
                        source_dtype=str(state.dtype),
                        nested=state.nested,
                        quantization_rmse=delta.square().mean().sqrt().item(),
                        quantization_max_abs=delta.abs().max().item(),
                    )
                )
                weights[name] = packed.cpu()
                for key, v in state.as_dict(packed=True).items():
                    weights[name + "." + key] = v.cpu().contiguous()
                del gpu, packed, state, restored, delta
            else:
                weights[name] = value
    assert len(checks) == 114
    save_file(weights, str(out / "model.safetensors"))
    for f in src.iterdir():
        if (
            f.is_file()
            and f.suffix in (".json", ".jinja", ".txt")
            and f.name != "packed-receipt.json"
        ):
            shutil.copy2(f, out / f.name)
    config = json.loads((out / "config.json").read_text())
    config["quantization_config"] = dict(
        quant_method="bitsandbytes",
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_storage="uint8",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head", "conv1d"],
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = dict(
        state="complete",
        source_bf16_sha256=receipt["packed_sha256"],
        quantized_layers=checks,
        total_checkpoint_tensor_bytes=sum(
            t.numel() * t.element_size() for t in weights.values()
        ),
        quantized_projection_bytes=sum(x["packed_bytes"] for x in checks),
        bf16_projection_bytes=sum(x["source_bytes"] for x in checks),
        scope="NF4+double quantization for114 dense decoder projections; remaining weights unchanged; LoRA files untouched",
    )
    (out / "quantization-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "nf4").symlink_to(out.name, target_is_directory=True)
    print(
        "NF4_CHECKPOINT_READY",
        len(checks),
        report["total_checkpoint_tensor_bytes"],
        flush=True,
    )


if __name__ == "__main__":
    main()
