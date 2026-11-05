# SPDX-License-Identifier: Apache-2.0
"""CPU-only canonical fused BF16 checkpoint, preserving every source tensor."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def normalize(name):
    return name.replace("model.language_model.", "model.").replace(".self_attn.", ".")


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    config = json.loads((args.source / "config.json").read_text())["text_config"]
    assert (
        config["model_type"],
        config["hidden_size"],
        config["intermediate_size"],
        len(config["layer_types"]),
    ) == ("qwen3_5_text", 1024, 3584, 24)
    assert config["layer_types"].count("linear_attention") == 18
    files = list(args.source.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("pilot expects one source safetensors shard")
    out = args.output
    out.mkdir(exist_ok=False)
    source_file = files[0]
    torch.set_num_threads(2)
    with safe_open(source_file, framework="pt", device="cpu") as source:
        original = {normalize(k): source.get_tensor(k) for k in source.keys()}
    groups = {
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
        "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
        "in_proj_ba": ("in_proj_b", "in_proj_a"),
    }
    merged = {}
    receipts = []
    for name in list(original):
        for target, parts in groups.items():
            if not name.endswith("." + parts[0] + ".weight") or ".visual." in name:
                continue
            prefix = name[: -len(parts[0] + ".weight")]
            keys = [prefix + part + ".weight" for part in parts]
            assert all(k in original for k in keys)
            tensors = [original[k] for k in keys]
            value = torch.cat(tensors, dim=0).contiguous()
            key = prefix + target + ".weight"
            merged[key] = value
            offset = 0
            for k, t in zip(keys, tensors):
                assert torch.equal(value[offset : offset + t.shape[0]], t)
                offset += t.shape[0]
                del original[k]
            receipts.append(dict(name=key, shape=list(value.shape), sources=keys))
            break
    merged.update(original)
    save_file(merged, str(out / "model.safetensors"))
    for f in args.source.iterdir():
        if (
            f.is_file()
            and f.suffix in (".json", ".jinja", ".txt")
            and f.name not in ("model.safetensors.index.json",)
        ):
            shutil.copy2(f, out / f.name)
    targets = {
        n: list(t.shape)
        for n, t in merged.items()
        if n.startswith("model.layers.")
        and n.endswith(".weight")
        and n.split(".")[-2]
        in {
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
        }
    }
    assert len(targets) == 114, (len(targets), list(targets))
    (out / "packed-receipt.json").write_text(
        json.dumps(
            dict(
                targets=targets,
                merges=receipts,
                source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                packed_sha256=hashlib.sha256(
                    (out / "model.safetensors").read_bytes()
                ).hexdigest(),
                state="complete",
                scope="Exact BF16 row concatenation; no numerical transformation",
            ),
            indent=2,
        )
        + "\n"
    )
    assert not torch.cuda.is_initialized()
    print("CPU_PACKED_CHECKPOINT_READY", len(targets), flush=True)


if __name__ == "__main__":
    main()
