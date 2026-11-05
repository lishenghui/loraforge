# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from loraforge.integrations._sglang_fleet import (  # noqa: E402
    FleetPolicy,
    _active_rank_classes,
)
from loraforge.runtime.retained import RetainedLayout, effective_rank  # noqa: E402


def ranks(k, other=None):
    return {
        f"base_model.model.model.layers.0.mlp.{name}.lora_A.weight": value
        for name, value in (
            ("gate_proj", k),
            ("up_proj", k if other is None else other),
        )
    }


def test_fixed_regions_repack_reload_and_masks():
    a = torch.arange(2 * 2 * 128 * 8).reshape(2, 256, 8).to(torch.bfloat16)
    b = torch.arange(2 * 32 * 128).reshape(2, 32, 128).to(torch.bfloat16)
    catalog = {"low": ranks(3, 2), "high": ranks(32), "fallback": ranks(128)}
    e = RetainedLayout(a, b, "gate_up_proj", 0, catalog)
    original_a, original_b = a.clone(), b.clone()
    pointers = [
        t.data_ptr()
        for t in (e.a, e.b, e.store, e.rank, e.low, e.high, e.boff, e.bucket)
    ]
    e.load("low", 1, 128)
    torch.testing.assert_close(
        a[1, :6], torch.cat((original_a[1, :3], original_a[1, 128:131]))
    )
    off, br, k = e.regions["low"]
    assert (br, k) == (4, 3)
    packed = e.store[off : off + 32 * br].view(32, br)
    torch.testing.assert_close(packed[:, :3], original_b[1, :, :3])
    assert not packed[:, 3].any()
    assert e.low.tolist() == [0, 3] and e.high.tolist() == [0, 0]
    # Simulate native reload before repacking; another checkpoint owns another region.
    a[1].copy_(original_a[1])
    e.load("fallback", 1, 128)
    assert e.unsupported.tolist() == [0, 128] and e.supported.tolist() == [0, 0]
    a[1].copy_(original_a[1])
    e.load("high", 1, 128)
    assert e.high.tolist() == [0, 32] and e.supported.tolist() == [0, 32]
    e.load(None, 1)
    assert not e.rank.any() and not e.bucket.any()
    assert pointers == [
        t.data_ptr()
        for t in (e.a, e.b, e.store, e.rank, e.low, e.high, e.boff, e.bucket)
    ]
    torch.testing.assert_close(b, original_b)
    before = a.clone()
    with pytest.raises(ValueError, match="catalog"):
        e.load("unknown", 1, 128)
    torch.testing.assert_close(a, before)


def test_effective_rank_fused_missing_and_layer():
    assert effective_rank(ranks(3, 7), "gate_up_proj", 0) == 7
    assert effective_rank(ranks(3), "gate_up_proj", 1) == 0
    assert effective_rank(ranks(3), "qkv_proj", 0) == 0


def test_frozen_policy_preserves_profile_overrides():
    p = FleetPolicy()
    # Paper profile uses all B / low B, not the table's experimental B modes.
    assert p.select("qkv_proj", "decode", 16)["b"] == "all"
    assert p.select("qkv_proj", "prefill", 16)["b"] == "low"
    assert p.select("qkv_proj", "decode", 17)["a"] == "native"
    assert p.select("gate_up_proj", "decode", 17)["a"] == "hip"
    assert p.select("unknown", "decode", 1)["b"] == "native"


def test_hybrid_keeps_native_shrink_and_prefill_reductions():
    policy = FleetPolicy(hybrid=True)
    for name in (
        "qkv_proj",
        "gate_up_proj",
        "down_proj",
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
    ):
        assert policy.select(name, "prefill", 128)["b"] == "native"
        assert policy.select(name, "decode", 1)["a"] == "native"
        assert policy.select(name, "decode", 32)["b"] == (
            "native" if name == "in_proj_ba" else "all"
        )


def test_hybrid_b_tiles_leave_a_and_prefill_native():
    policy = FleetPolicy(hybrid=True)
    for name in ("out_proj", "down_proj", "o_proj"):
        small = policy.select(name, "decode", 8, output_width=1024)
        large = policy.select(name, "decode", 32, output_width=1024)
        assert (small["row_tile"], small["threads"]) == (16, 128)
        assert (large["row_tile"], large["threads"]) == (2, 128)
        assert small["a"] == large["a"] == "native"
        prefill = policy.select(name, "prefill", 32, output_width=1024)
        assert prefill["a"] == prefill["b"] == "native"
        assert (prefill["row_tile"], prefill["threads"]) == (16, 256)
    qkv = policy.select("qkv_proj", "decode", 32, output_width=5120)
    assert (qkv["a"], qkv["row_tile"], qkv["threads"]) == ("native", 4, 256)


def test_hybrid_b_tiles_are_bounded_by_shape_and_batch():
    policy = FleetPolicy(hybrid=True)
    for name, width, tokens in (
        ("out_proj", None, 32),
        ("out_proj", 2048, 32),
        ("down_proj", 1024, 0),
        ("down_proj", 1024, 9),
        ("down_proj", 1024, 15),
        ("down_proj", 1024, 33),
        ("qkv_proj", 5120, 8),
        ("qkv_proj", 6144, 32),
        ("qkv_proj", 5120, 64),
        ("in_proj_qkvz", 8192, 32),
        ("gate_up_proj", 7168, 32),
        ("in_proj_ba", 32, 32),
    ):
        cfg = policy.select(name, "decode", tokens, output_width=width)
        assert (cfg["row_tile"], cfg["threads"]) == (16, 256)
    # Supplying a width cannot change the independently validated dense policy.
    dense = FleetPolicy()
    assert dense.select("qkv_proj", "decode", 32, output_width=5120) == dense.select(
        "qkv_proj", "decode", 32
    )


@pytest.mark.parametrize("capacity", [4, 8])
def test_tiny_hybrid_requires_opt_in_and_complete_geometry(capacity):
    old = FleetPolicy(hybrid=True)
    fast = FleetPolicy(hybrid=True, hybrid_tiny=True)
    shapes = {
        "in_proj_qkvz": (1024, 8192, 4),
        "in_proj_ba": (1024, 32, 2),
        "out_proj": (2048, 1024, 1),
        "o_proj": (2048, 1024, 1),
        "gate_up_proj": (1024, 7168, 2),
        "down_proj": (3584, 1024, 1),
        "qkv_proj": (1024, 5120, 3),
    }
    for name, (nin, nout, slices) in shapes.items():
        shape = dict(
            input_width=nin, output_width=nout, slices=slices, rank_capacity=capacity
        )
        assert old.select(name, "decode", 32, **shape)["a"] == "native"
        assert fast.select(name, "decode", 32, **shape)["a"] == "hip"
        assert fast.select(name, "decode", 32, **shape)["b"] == (
            "native" if nout <= 1024 else "all"
        )
        assert fast.select(name, "prefill", 1024, **shape)["a"] == "native"
        assert fast.select(name, "prefill", 1024, **shape)["b"] == (
            "native" if name == "in_proj_ba" else "all"
        )
        for override in (
            dict(rank_capacity=None),
            dict(rank_capacity=16),
            dict(rank_capacity=64),
            dict(input_width=nin + 8),
            dict(output_width=nout + 8),
            dict(slices=slices + 1),
        ):
            other = dict(shape, **override)
            for phase, tokens in (("decode", 32), ("prefill", 1024)):
                assert fast.select(name, phase, tokens, **other) == old.select(
                    name, phase, tokens, **other
                )
        for phase, tokens in (
            ("decode", 0),
            ("decode", 33),
            ("prefill", 127),
            ("prefill", 1025),
        ):
            assert fast.select(name, phase, tokens, **shape) == old.select(
                name, phase, tokens, **shape
            )
        if nout <= 1024:
            assert fast.select(name, "prefill", 128, **shape)["b"] == "native"
    with pytest.raises(ValueError, match="requires.*hybrid"):
        FleetPolicy(hybrid_tiny=True)


def test_active_prefill_uses_live_slots_and_keeps_unknown_routes():
    loaded = {0: {"rank": 0}, 1: {"rank": 3}, 2: {"rank": 128}}
    assert _active_rank_classes(loaded, [0]) == (False, False)
    assert _active_rank_classes(loaded, [0, 1]) == (True, False)
    assert _active_rank_classes(loaded, [2]) == (False, True)
    assert _active_rank_classes(loaded, [1, 2]) == (True, True)
    assert _active_rank_classes(loaded, [1, 3]) == (True, True)
    # Reusing a slot must change the decision without a stale per-slot cache.
    loaded[1] = {"rank": 32}
    assert _active_rank_classes(loaded, [1]) == (False, True)
    loaded[1] = {"rank": 16}
    assert _active_rank_classes(loaded, [1]) == (True, False)


@pytest.mark.parametrize("flag,enabled", [(None, False), ("0", False), ("1", True)])
def test_active_prefill_is_explicit_opt_in(tmp_path, monkeypatch, flag, enabled):
    import json

    from loraforge.integrations import _sglang_fleet as fleet

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"adapters": [{"path": "adapter"}]}))
    monkeypatch.setenv("LORAFORGE_SGLANG_MANIFEST", str(manifest))
    monkeypatch.delenv("LORAFORGE_SGLANG_ACTIVE_PREFILL", raising=False)
    monkeypatch.delenv("LORAFORGE_SGLANG_QWEN35", raising=False)
    monkeypatch.delenv("LORAFORGE_SGLANG_QWEN35_TINY", raising=False)
    if flag is not None:
        monkeypatch.setenv("LORAFORGE_SGLANG_ACTIVE_PREFILL", flag)
    monkeypatch.setattr(fleet, "read_rank_catalog", lambda paths: paths)
    monkeypatch.setattr(
        fleet, "create_backend", lambda catalog, **kwargs: (catalog, kwargs)
    )
    assert fleet.backend_from_environment() == (
        ["adapter"],
        {"active_prefill": enabled, "hybrid": False, "hybrid_tiny": False},
    )
    monkeypatch.setenv("LORAFORGE_SGLANG_ACTIVE_PREFILL", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        fleet.backend_from_environment()


def test_tiny_hybrid_environment_is_explicit(tmp_path, monkeypatch):
    import json

    from loraforge.integrations import _sglang_fleet as fleet

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"adapters": [{"path": "adapter"}]}))
    monkeypatch.setenv("LORAFORGE_SGLANG_MANIFEST", str(manifest))
    monkeypatch.setenv("LORAFORGE_SGLANG_QWEN35", "1")
    monkeypatch.setenv("LORAFORGE_SGLANG_QWEN35_TINY", "1")
    monkeypatch.setenv("LORAFORGE_SGLANG_ACTIVE_PREFILL", "0")
    monkeypatch.setattr(fleet, "read_rank_catalog", lambda paths: paths)
    monkeypatch.setattr(fleet, "create_backend", lambda catalog, **kwargs: kwargs)
    assert fleet.backend_from_environment() == dict(
        active_prefill=False, hybrid=True, hybrid_tiny=True
    )
    monkeypatch.setenv("LORAFORGE_SGLANG_QWEN35_TINY", "yes")
    with pytest.raises(ValueError, match="QWEN35_TINY must be 0 or 1"):
        fleet.backend_from_environment()


def test_serving_export_retained_ranks_and_scaling(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    from loraforge.adapters import sglang
    from loraforge.adapters.peft import AdapterModule
    from loraforge.runtime.retained import read_rank_catalog

    modules = tuple(
        AdapterModule(
            f"model.layers.0.mlp.{name}", torch.ones(k, 8), torch.ones(16, k), alpha
        )
        for name, k, alpha in [("gate_proj", 3, 6), ("up_proj", 7, 3.5)]
    )
    adapter = SimpleNamespace(
        modules=modules,
        config={"peft_type": "LORA", "r": 3, "lora_alpha": 6},
        source_sha256={},
    )
    monkeypatch.setattr(sglang, "load_peft_adapter", lambda _: adapter)
    target = tmp_path / "serving"
    report = sglang.export_sglang_adapter("source", target)
    assert report["checkpoint_rank"] == 7
    catalog = read_rank_catalog([target])
    assert effective_rank(catalog[str(target)], "gate_up_proj", 0) == 7
    w = load_file(str(target / "adapter_model.safetensors"))
    for m in modules:
        key = "base_model.model." + m.name
        a, b = w[key + ".lora_A.weight"], w[key + ".lora_B.weight"]
        torch.testing.assert_close(
            b.float() @ a.float(), (m.b.float() @ m.a.float()) * m.scaling
        )
    with pytest.raises(FileExistsError):
        sglang.export_sglang_adapter("source", target)
