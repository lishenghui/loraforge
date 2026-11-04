# SPDX-License-Identifier: Apache-2.0
from itertools import product
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from loraforge.integrations._sglang_fleet import (  # noqa: E402
    FleetPolicy,
    _active_rank_classes,
)
from loraforge.runtime.retained import (  # noqa: E402
    RetainedLayout,
    effective_rank,
    packed_segment_capacity,
)


def test_segment_bound_covers_all_adapter_distributions():
    for slots in (1, 2, 3, 4):
        for chunk in (1, 2, 4, 16):
            for counts in product(range(7), repeat=slots):
                actual = sum((n + chunk - 1) // chunk for n in counts)
                assert actual <= packed_segment_capacity(sum(counts), slots, chunk)
    # All slots active, one larger group just beyond a chunk boundary.
    assert packed_segment_capacity(33, 4, 16) == 5
    assert packed_segment_capacity(32, 5, 16) == 6
    assert packed_segment_capacity(1, 5, 16) == 1
    assert packed_segment_capacity(0, 5, 16) == 0


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
    if flag is not None:
        monkeypatch.setenv("LORAFORGE_SGLANG_ACTIVE_PREFILL", flag)
    monkeypatch.setattr(fleet, "read_rank_catalog", lambda paths: paths)
    monkeypatch.setattr(
        fleet, "create_backend", lambda catalog, **kwargs: (catalog, kwargs)
    )
    assert fleet.backend_from_environment() == (
        ["adapter"],
        {"active_prefill": enabled, "hybrid": False},
    )
    monkeypatch.setenv("LORAFORGE_SGLANG_ACTIVE_PREFILL", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
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
