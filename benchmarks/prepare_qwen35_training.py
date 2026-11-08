# SPDX-License-Identifier: Apache-2.0
"""CPU-only extraction of an audited, unchanged BF16 Qwen3.5 text checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path


def text_key(name: str) -> str | None:
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name == "lm_head.weight":
        return name
    if name.startswith(("model.visual.", "mtp.", "model.mtp.")):
        return None
    raise ValueError(f"unrecognized source weight: {name}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_text_shards(files, expected, weight_map=None):
    """Read unmodified tensors; reject duplicate, missing or misindexed weights."""
    import torch
    from safetensors import safe_open

    weights, mapping, skipped, seen = {}, {}, [], set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for source in f.keys():
                assert source not in seen, source
                seen.add(source)
                if weight_map is not None:
                    assert weight_map.get(source) == file.name, source
                target = text_key(source)
                if target is None:
                    skipped.append(source)
                    continue
                assert target in expected and target not in weights, (source, target)
                t = f.get_tensor(source)
                assert t.dtype in (torch.bfloat16, torch.float32), (source, t.dtype)
                if t.ndim == 2:
                    assert t.dtype == torch.bfloat16, (source, t.dtype)
                assert t.shape == expected[target].shape, (source, t.shape)
                weights[target], mapping[source] = t, target
    if weight_map is not None:
        assert seen == set(weight_map), "incomplete shard index"
    return weights, mapping, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.set_num_threads(2)
    cfg = json.loads((args.source / "config.json").read_text())["text_config"]
    assert len(cfg["layer_types"]) == cfg["num_hidden_layers"]
    assert set(cfg["layer_types"]) == {"linear_attention", "full_attention"}
    config = Qwen3_5TextConfig(**cfg)
    config.use_cache = False
    config.architectures = ["Qwen3_5ForCausalLM"]
    with torch.device("meta"):
        expected = Qwen3_5ForCausalLM(config).state_dict()
    index = args.source / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
    files = (
        [args.source / n for n in sorted(set(weight_map.values()))]
        if weight_map is not None
        else sorted(args.source.glob("*.safetensors"))
    )
    assert files and all(f.is_file() and f.parent == args.source for f in files), files
    assert weight_map is not None or len(files) == 1, "multiple shards need an index"
    weights, mapping, skipped = read_text_shards(files, expected, weight_map)
    tied = set()
    if config.tie_word_embeddings and "lm_head.weight" not in weights:
        tied.add("lm_head.weight")
    assert set(weights) | tied == set(expected), set(expected) - set(weights) - tied
    args.output.mkdir(exist_ok=False)
    save_file(
        weights, str(args.output / "model.safetensors"), metadata={"format": "pt"}
    )
    config.save_pretrained(args.output)
    # Verify every serialized text tensor, not just names or a clean load report.
    with safe_open(
        args.output / "model.safetensors", framework="pt", device="cpu"
    ) as f:
        assert set(f.keys()) == set(weights)
        for key, t in weights.items():
            assert torch.equal(f.get_tensor(key), t), key
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    text = (
        "A training step computes a prediction, a loss and its gradients. "
        "The optimizer updates only the low rank adapter parameters. "
        "We compare identical inputs, model weights and numerical precision. "
        "长序列训练需要同时检查运行时间、显存和梯度，不能把局部算子的加速当成整模型的收益。\n"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    (args.output / "input-tokens.json").write_text(json.dumps(ids) + "\n")
    receipt = dict(
        state="complete",
        scope="unchanged BF16 text weights; no vision inputs",
        source_files={f.name: sha256(f) for f in files},
        model_config={
            k: cfg[k]
            for k in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "vocab_size",
            )
        },
        text_sha256=sha256(args.output / "model.safetensors"),
        source_to_text=mapping,
        skipped_vision_or_mtp=skipped,
        tied_keys=sorted(tied),
        checked_tensors=len(weights),
        layer_types=cfg["layer_types"],
        input_tokens_sha256=sha256(args.output / "input-tokens.json"),
    )
    if len(files) == 1:
        receipt.update(
            source_file=files[0].name,
            source_sha256=receipt["source_files"][files[0].name],
        )
    if (args.source / "download-receipt.json").exists():
        receipt["download"] = json.loads(
            (args.source / "download-receipt.json").read_text()
        )
    (args.output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print("TEXT_CHECKPOINT_READY", len(weights), flush=True)


if __name__ == "__main__":
    main()
