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
    assert (cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]) == (
        1024,
        3584,
        24,
    )
    config = Qwen3_5TextConfig(**cfg)
    config.use_cache = False
    config.architectures = ["Qwen3_5ForCausalLM"]
    with torch.device("meta"):
        expected = Qwen3_5ForCausalLM(config).state_dict()
    files = list(args.source.glob("*.safetensors"))
    assert len(files) == 1, files
    weights, mapping, skipped = {}, {}, []
    with safe_open(files[0], framework="pt", device="cpu") as f:
        for source in f.keys():
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
        source_file=files[0].name,
        source_sha256=sha256(files[0]),
        text_sha256=sha256(args.output / "model.safetensors"),
        source_to_text=mapping,
        skipped_vision_or_mtp=skipped,
        tied_keys=sorted(tied),
        checked_tensors=len(weights),
        layer_types=cfg["layer_types"],
        input_tokens_sha256=sha256(args.output / "input-tokens.json"),
    )
    (args.output / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print("TEXT_CHECKPOINT_READY", len(weights), flush=True)


if __name__ == "__main__":
    main()
