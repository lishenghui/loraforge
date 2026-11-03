# SPDX-License-Identifier: Apache-2.0
"""Prepare and hash existing offline model/adapter inputs on a CPU/login node."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve()
    config = json.loads((model / "config.json").read_text())
    index = model / "model.safetensors.index.json"
    if index.exists():
        files = set(json.loads(index.read_text())["weight_map"].values())
    else:
        files = {"model.safetensors"}
    for name in files:
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    adapters, targets = [], set()
    for i, directory in enumerate(args.adapters):
        directory = directory.resolve()
        path = directory / "adapter_config.json"
        data = json.loads(path.read_text())
        if (
            data.get("use_dora")
            or data.get("use_rslora")
            or data.get("modules_to_save")
        ):
            raise ValueError("initial service protocol requires ordinary linear LoRA")
        if not isinstance(data.get("target_modules"), list):
            raise ValueError("explicit target_modules are required")
        targets.update(data["target_modules"])
        weights = directory / "adapter_model.safetensors"
        adapters.append(
            {
                "name": f"adapter_{i}",
                "path": str(directory),
                "rank": data["r"],
                "base_model": data.get("base_model_name_or_path"),
                "files": {
                    p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                    for p in (path, weights)
                },
            }
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        for text in (
            "Write one short sentence about the moon.",
            "Translate 'Good morning' into French.",
        )
    ]
    manifest = {
        "model": str(model),
        "model_type": config["model_type"],
        "model_config_sha256": digest(model / "config.json"),
        "adapters": adapters,
        "target_modules": sorted(targets),
        "prompts": prompts,
        "scope": "existing real weights; correctness inputs, not task quality evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": str(model),
                "ranks": [a["rank"] for a in adapters],
                "prompt_lengths": [len(p) for p in prompts],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
