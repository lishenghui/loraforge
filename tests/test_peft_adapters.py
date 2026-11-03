# SPDX-License-Identifier: Apache-2.0
"""CPU-only file-format and real PEFT reload tests; no downloaded models."""

import copy
import json
import math

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
safetensors = pytest.importorskip("safetensors.torch")

from loraforge.adapters import (  # noqa: E402
    compress_peft_adapter,
    load_peft_adapter,
    save_peft_adapter,
)
from loraforge.adapters.peft import CONFIG, MANIFEST, WEIGHTS  # noqa: E402


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(24, 20, bias=False)
        self.block = torch.nn.ModuleDict({"proj": torch.nn.Linear(24, 20, bias=False)})

    def forward(self, x):
        return self.proj(x) + self.block["proj"](x)


def make_source(tmp_path, *, dtype=torch.float32, patterns=None, use_rslora=False):
    torch.manual_seed(123)
    base = TinyModel().to(dtype)
    original_base = copy.deepcopy(base)
    config = peft.LoraConfig(
        r=8,
        lora_alpha=11,
        target_modules=["proj"],
        rank_pattern=patterns or {"block.proj": 6},
        alpha_pattern={"block.proj": 7},
        use_rslora=use_rslora,
    )
    model = peft.get_peft_model(base, config).to(dtype).eval()
    generator = torch.Generator().manual_seed(456)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator).to(dtype)
                )
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    return source, model, original_base


@pytest.fixture
def source(tmp_path):
    return make_source(tmp_path)


def update_config(source, **changes):
    config = json.loads((source / CONFIG).read_text())
    config.update(changes)
    (source / CONFIG).write_text(json.dumps(config))


def reference_factors(adapter, ranks):
    # Dense SVD is only a CPU test reference for export; production compression
    # always calls FlashTSQR and has its own allocated-GPU test.
    results = {}
    for module in adapter.modules:
        u, s, vh = torch.linalg.svd(
            module.b.double() @ module.a.double(), full_matrices=False
        )
        k = ranks[module.name]
        root = s[:k].sqrt()
        results[module.name] = (
            (root[:, None] * vh[:k]).float(),
            (u[:, :k] * root).float(),
        )
    return results


@pytest.mark.parametrize("use_rslora", [False, True], ids=["lora", "rslora"])
def test_load_and_identity_export_matches_peft(tmp_path, use_rslora):
    directory, model, base = make_source(tmp_path, use_rslora=use_rslora)
    adapter = load_peft_adapter(directory)
    assert {m.name: m.rank for m in adapter.modules} == {"proj": 8, "block.proj": 6}
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    exported = save_peft_adapter(adapter, tmp_path / "output")
    reloaded = peft.PeftModel.from_pretrained(
        base, exported.directory, autocast_adapter_dtype=False
    ).eval()
    x = torch.randn(4, 24)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x), rtol=1e-5, atol=1e-5)
    assert before == {
        p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()
    }
    manifest = exported.manifest
    assert manifest["schema_version"] == 1 and manifest["operation"] == "export"
    assert all(m["execution"] is None for m in manifest["modules"])
    assert (
        json.loads((exported.directory / CONFIG).read_text())["use_rslora"]
        is use_rslora
    )
    for record in manifest["modules"]:
        assert record["use_rslora"] is use_rslora
        assert record["scaling_type"] == (
            "alpha_over_sqrt_rank" if use_rslora else "alpha_over_rank"
        )
    loaded = load_peft_adapter(exported.directory)
    for module in loaded.modules:
        layer = model.base_model.model.get_submodule(module.name)
        assert module.use_rslora is use_rslora
        assert module.scaling == pytest.approx(layer.scaling["default"], rel=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_rslora", [False, True], ids=["lora", "rslora"])
def test_heterogeneous_compact_export_preserves_scaling(tmp_path, dtype, use_rslora):
    directory, original, base = make_source(
        tmp_path, dtype=dtype, use_rslora=use_rslora
    )
    adapter = load_peft_adapter(directory)
    for module in adapter.modules:
        expected = original.base_model.model.get_submodule(module.name).scaling[
            "default"
        ]
        assert module.scaling == pytest.approx(expected, rel=1e-14)
    factors = reference_factors(adapter, {"proj": 3, "block.proj": 2})
    exported = save_peft_adapter(adapter, tmp_path / "compact", factors=factors)
    base_weights = {
        m.name: base.get_submodule(m.name).weight.detach().clone()
        for m in adapter.modules
    }
    reloaded = peft.PeftModel.from_pretrained(
        base, exported.directory, autocast_adapter_dtype=False
    ).eval()
    saved = load_peft_adapter(exported.directory)
    assert {m.name: m.rank for m in saved.modules} == {"proj": 3, "block.proj": 2}
    x = torch.randn(4, 24).to(dtype)
    cast = {m.name: (m.a, m.b) for m in saved.modules}
    with torch.no_grad():
        parts = {}
        for module in adapter.modules:
            a, b = cast[module.name]
            parts[module.name] = torch.nn.functional.linear(
                x, base_weights[module.name]
            ) + module.scaling * torch.nn.functional.linear(
                torch.nn.functional.linear(x, a), b
            )
        expected = parts["proj"] + parts["block.proj"]
        tolerance = (
            0.04
            if dtype == torch.bfloat16
            else 0.004
            if dtype == torch.float16
            else 2e-5
        )
        torch.testing.assert_close(
            reloaded(x), expected, rtol=tolerance, atol=tolerance
        )
    assert (
        exported.manifest["output"]["tensor_bytes"]
        < exported.manifest["output"]["source_tensor_bytes"]
    )
    assert (exported.directory / WEIGHTS).stat().st_size < (
        directory / WEIGHTS
    ).stat().st_size
    for module in saved.modules:
        assert module.a.dtype == module.b.dtype == dtype
        assert module.scaling == pytest.approx(
            next(m.scaling for m in adapter.modules if m.name == module.name), rel=1e-14
        )
        assert module.use_rslora is use_rslora
    assert reloaded.base_model.model.proj.lora_alpha["default"] == pytest.approx(
        11 * (math.sqrt(3 / 8) if use_rslora else 3 / 8)
    )


def test_ordered_suffix_patterns_match_peft(tmp_path):
    # Saved PEFT configs sort dictionary keys; preserve that rule order in the
    # source model so its serialized config still describes its actual weights.
    directory, model, _ = make_source(tmp_path, patterns={"block.proj": 6, "proj": 5})
    adapter = load_peft_adapter(directory)
    for module in adapter.modules:
        layer = model.base_model.model.get_submodule(module.name)
        assert module.rank == layer.r["default"]
        assert module.scaling == layer.scaling["default"]


def test_execution_metadata_matches_serialized_manifest(source, tmp_path):
    from loraforge.adapters.peft import _export
    from loraforge.compression import CompressionInfo
    from loraforge.ops import ExecutionInfo

    adapter = load_peft_adapter(source[0])
    # A serializer fixture, explicitly labeled as such; no GPU execution claim.
    execution = ExecutionInfo(
        "fixture", "cpu", "fixture", (("fixture", "hash"),), backend="fixture"
    )
    metadata = {
        m.name: CompressionInfo(execution, "fixture", m.rank, m.rank)
        for m in adapter.modules
    }
    result = _export(adapter, tmp_path / "metadata", None, "source", metadata)
    assert result.manifest == json.loads((result.directory / MANIFEST).read_text())


@pytest.mark.parametrize(
    "changes",
    [
        {"use_dora": True},
        {"use_rslora": "true"},
        {"use_rslora": 1},
        {"use_rslora": None},
        {"fan_in_fan_out": True},
        {"bias": "all"},
        {"lora_bias": True},
        {"modules_to_save": ["head"]},
        {"use_qalora": True},
        {"target_parameters": ["weight"]},
        {"future_variant": True},
        {"init_lora_weights": "pissa"},
    ],
)
def test_rejects_unsupported_configuration(source, changes):
    directory, _, _ = source
    update_config(directory, **changes)
    with pytest.raises(ValueError):
        load_peft_adapter(directory)


@pytest.mark.parametrize(
    "kind", ["extra", "missing", "shape", "nonfinite", "conv", "named_adapter"]
)
def test_rejects_unrepresentable_weights(source, kind):
    directory, _, _ = source
    weights = safetensors.load_file(directory / WEIGHTS)
    key = "base_model.model.proj.lora_A.weight"
    if kind == "extra":
        weights["base_model.model.head.weight"] = torch.ones(3, 3)
    elif kind == "missing":
        del weights[key]
    elif kind == "shape":
        weights[key] = weights[key][:2].contiguous()
    elif kind == "nonfinite":
        weights[key][0, 0] = float("nan")
    elif kind == "conv":
        weights[key] = weights[key].unsqueeze(-1)
    else:
        weights[key.replace(".weight", ".default.weight")] = weights.pop(key)
    safetensors.save_file(weights, directory / WEIGHTS)
    with pytest.raises(ValueError):
        load_peft_adapter(directory)


def test_output_is_new_and_outside_source(source, tmp_path):
    directory, _, _ = source
    adapter = load_peft_adapter(directory)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        save_peft_adapter(adapter, output)
    assert (output / "keep").read_text() == "untouched"
    with pytest.raises(ValueError, match="outside"):
        save_peft_adapter(adapter, directory / "nested")
    assert not (directory / "nested").exists()


def test_manifest_detects_changed_output_and_unknown_version(source, tmp_path):
    directory, _, _ = source
    out = save_peft_adapter(load_peft_adapter(directory), tmp_path / "output")
    manifest_path = out.directory / MANIFEST
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        load_peft_adapter(out.directory)
    manifest_path.write_text(original)
    update_config(out.directory, lora_alpha=123)
    with pytest.raises(ValueError, match="integrity"):
        load_peft_adapter(out.directory)


def test_source_mutation_and_invalid_export_do_not_publish(source, tmp_path):
    directory, _, _ = source
    adapter = load_peft_adapter(directory)
    factors = {m.name: (m.a[:0], m.b[:, :0]) for m in adapter.modules}
    with pytest.raises(ValueError, match="positive"):
        save_peft_adapter(adapter, tmp_path / "zero", factors=factors)
    assert not (tmp_path / "zero").exists()
    with pytest.raises(ValueError, match="exactly"):
        save_peft_adapter(adapter, tmp_path / "partial", factors={})
    update_config(directory, lora_alpha=123)
    with pytest.raises(ValueError, match="changed"):
        save_peft_adapter(adapter, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_compression_preflight_rejects_cpu_and_bad_plans(source, tmp_path):
    directory, _, _ = source
    for keep in (0, True, -1, 9, {"proj": 2}):
        with pytest.raises(ValueError):
            compress_peft_adapter(directory, tmp_path / "bad", keep, device="cuda:0")
    with pytest.raises(ValueError, match="GPU"):
        compress_peft_adapter(directory, tmp_path / "cpu", 2, device="cpu")
    assert not (tmp_path / "cpu").exists()


def test_publication_failure_keeps_external_file(source, tmp_path, monkeypatch):
    import loraforge.adapters.peft as io

    directory, _, _ = source
    output = tmp_path / "output"
    link = io.os.link

    def fail_second(a, b):
        if b.name == WEIGHTS:
            (output / "external").write_text("preserve")
            raise OSError("simulated publication failure")
        return link(a, b)

    monkeypatch.setattr(io.os, "link", fail_second)
    with pytest.raises(OSError, match="simulated"):
        save_peft_adapter(load_peft_adapter(directory), output)
    assert sorted(p.name for p in output.iterdir()) == ["external"]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="allocated GPU required")
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("use_rslora", [False, True], ids=["lora", "rslora"])
def test_gpu_file_compression_roundtrip(tmp_path, batched, use_rslora):
    directory, _, base = make_source(
        tmp_path,
        patterns={"proj": 8} if batched else None,
        use_rslora=use_rslora,
    )
    original = load_peft_adapter(directory)
    plan = {"proj": 3, "block.proj": 3 if batched else 2}
    exported = compress_peft_adapter(
        directory,
        tmp_path / "compressed",
        plan,
        device="cuda:0",
        batch_size=2,
    )
    saved = load_peft_adapter(exported.directory)
    base_weights = {
        m.name: base.get_submodule(m.name).weight.detach().clone()
        for m in original.modules
    }
    loaded = peft.PeftModel.from_pretrained(
        base, exported.directory, autocast_adapter_dtype=False
    ).eval()
    x = torch.randn(4, 24)
    parts = {}
    with torch.no_grad():
        for module in saved.modules:
            parts[module.name] = torch.nn.functional.linear(
                x, base_weights[module.name]
            ) + module.scaling * torch.nn.functional.linear(
                torch.nn.functional.linear(x, module.a), module.b
            )
            source_module = next(m for m in original.modules if m.name == module.name)
            dense = source_module.b.double() @ source_module.a.double()
            error = (module.b.double() @ module.a.double() - dense).norm()
            optimal = torch.linalg.svdvals(dense)[plan[module.name] :].norm()
            torch.testing.assert_close(error, optimal, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            loaded(x), parts["proj"] + parts["block.proj"], rtol=2e-5, atol=2e-5
        )
    assert exported.manifest["operation"] == "compression"
    assert exported.manifest["compression"]["batch_size"] == 2
    for record in exported.manifest["modules"]:
        assert record["execution"]["qr"]["backend"] == "FlashTSQR"
        assert record["execution"]["retained_rank"] == plan[record["name"]]
        assert record["output_tensor_bytes"] < record["source_tensor_bytes"]
        assert record["use_rslora"] is use_rslora
        layer = loaded.base_model.model.get_submodule(record["name"])
        source_layer = next(m for m in original.modules if m.name == record["name"])
        expected_scale = source_layer.alpha / (
            math.sqrt(source_layer.rank) if use_rslora else source_layer.rank
        )
        assert layer.scaling["default"] == pytest.approx(expected_scale, rel=1e-14)


@pytest.mark.parametrize("kwargs", [{}, {"keep": 2, "budget": 4}, {"budget": 1}])
def test_budget_rejected_before_gpu_work(source, tmp_path, kwargs):
    output = tmp_path / "invalid-budget"
    with pytest.raises(ValueError):
        compress_peft_adapter(source[0], output, device="cuda:0", **kwargs)
    assert not output.exists()


@pytest.mark.gpu
@pytest.mark.parametrize("use_rslora", [False, True])
def test_gpu_budget_roundtrip(tmp_path, use_rslora):
    if not torch.cuda.is_available():
        pytest.skip("requires allocated GPU")
    source = make_source(tmp_path, use_rslora=use_rslora)
    adapter = load_peft_adapter(source[0])
    budget = len(adapter.modules) + 3
    result = compress_peft_adapter(
        source[0], tmp_path / "budget", budget=budget, device="cuda:0"
    )
    loaded = load_peft_adapter(result.directory)
    assert sum(m.rank for m in loaded.modules) == budget
    assert all(m.rank >= 1 for m in loaded.modules)
    plan = result.manifest["compression"]["allocation"]
    assert plan["direction_budget"] == budget
    for original, compact in zip(adapter.modules, loaded.modules, strict=True):
        assert compact.scaling == pytest.approx(original.scaling)
        singular = torch.linalg.svdvals(original.b.float() @ original.a.float())
        error = (
            original.b.float() @ original.a.float()
            - compact.b.float() @ compact.a.float()
        ).norm()
        assert error.item() == pytest.approx(
            singular[compact.rank :].norm().item(), rel=0.02, abs=0.02
        )
