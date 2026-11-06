# SPDX-License-Identifier: Apache-2.0
"""Sharded text extraction must preserve data and reject inconsistent indexes."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safe = pytest.importorskip("safetensors.torch")
spec = importlib.util.spec_from_file_location(
    "prepare_qwen35_training",
    Path(__file__).resolve().parents[1] / "benchmarks/prepare_qwen35_training.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize(
    "failure", [None, "wrong-shard", "missing-tensor", "duplicate", "fp32-matrix"]
)
def test_shard_data_and_index_contract(tmp_path, failure):
    embedding = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    norm = torch.tensor([0.9, 1.0, 1.1])
    a = {"model.language_model.embed_tokens.weight": embedding}
    b = {"model.language_model.norm.weight": norm, "model.visual.weight": torch.ones(2)}
    if failure == "duplicate":
        b.update(a)
    if failure == "fp32-matrix":
        a = {k: v.float() for k, v in a.items()}
    files = [tmp_path / "a.safetensors", tmp_path / "b.safetensors"]
    for file, weights in zip(files, (a, b), strict=True):
        safe.save_file(weights, file)
    index = {
        key: file.name
        for file, weights in zip(files, (a, b), strict=True)
        for key in weights
    }
    if failure == "wrong-shard":
        index["model.language_model.norm.weight"] = "a.safetensors"
    if failure == "missing-tensor":
        index["model.language_model.missing.weight"] = "a.safetensors"
    expected = {"model.embed_tokens.weight": embedding, "model.norm.weight": norm}
    if failure:
        with pytest.raises(AssertionError):
            probe.read_text_shards(files, expected, index)
    else:
        weights, mapping, skipped = probe.read_text_shards(files, expected, index)
        assert set(weights) == set(expected) and len(mapping) == 2
        assert skipped == ["model.visual.weight"]
        for key, value in expected.items():
            assert (
                torch.equal(weights[key], value) and weights[key].dtype == value.dtype
            )
