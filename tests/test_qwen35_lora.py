# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from loraforge.integrations._sglang_qwen35 import normalize_gdn_weights  # noqa: E402
from loraforge.runtime.retained import RetainedLayout, effective_rank  # noqa: E402

CFG = SimpleNamespace(
    model_type="qwen3_5_text",
    hidden_size=8,
    linear_key_head_dim=3,
    linear_num_key_heads=2,
    linear_value_head_dim=4,
    linear_num_value_heads=2,
)
PREFIX = "base_model.model.model.language_model.layers.0.linear_attn."


@pytest.mark.parametrize("format", ["hf", "split", "fused", "partial"])
def test_gdn_packing_preserves_independent_products(format):
    torch.manual_seed(49)
    widths = {
        "in_proj_q": 6,
        "in_proj_k": 6,
        "in_proj_v": 8,
        "in_proj_z": 8,
        "in_proj_b": 2,
        "in_proj_a": 2,
    }
    if format == "hf":
        widths = {"in_proj_qkv": 20, "in_proj_z": 8, "in_proj_b": 2, "in_proj_a": 2}
    elif format == "fused":
        widths = {"in_proj_qkvz": 28, "in_proj_ba": 4}
    elif format == "partial":
        widths = {"in_proj_z": 8, "in_proj_a": 2}
    weights, products = {}, {}
    for name, width in widths.items():
        a = torch.randn(3, 8).to(torch.bfloat16)
        b = torch.randn(width, 3).to(torch.bfloat16)
        weights[PREFIX + name + ".lora_A.weight"] = a
        weights[PREFIX + name + ".lora_B.weight"] = b
        products[name] = b.float() @ a.float()
    normalize_gdn_weights(weights, CFG, 3)
    if format == "fused":
        qkvz, ba = products["in_proj_qkvz"], products["in_proj_ba"]
    else:
        qkv = products.get("in_proj_qkv")
        if qkv is None:
            qkv = torch.cat(
                [
                    products.get(n, torch.zeros(w, 8))
                    for n, w in [("in_proj_q", 6), ("in_proj_k", 6), ("in_proj_v", 8)]
                ]
            )
        qkvz = torch.cat([qkv, products["in_proj_z"]])
        ba = torch.cat(
            [products.get("in_proj_b", torch.zeros(2, 8)), products["in_proj_a"]]
        )
    assert len(weights) == 4
    for name, widths, expected in [
        ("in_proj_qkvz", [6, 6, 8, 8], qkvz),
        ("in_proj_ba", [2, 2], ba),
    ]:
        a, b = (weights[PREFIX + name + f".lora_{ab}.weight"] for ab in "AB")
        actual = torch.cat(
            [
                bb.float() @ aa.float()
                for aa, bb in zip(a.split(3), b.split(widths), strict=True)
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_format_is_atomic():
    weights = {
        PREFIX + "in_proj_qkvz.lora_A.weight": torch.ones(3, 8),
        PREFIX + "in_proj_qkvz.lora_B.weight": torch.ones(28, 3),
        PREFIX + "in_proj_z.lora_A.weight": torch.ones(3, 8),
        PREFIX + "in_proj_z.lora_B.weight": torch.ones(8, 3),
    }
    before = dict(weights)
    with pytest.raises(ValueError, match="overlapping"):
        normalize_gdn_weights(weights, CFG, 3)
    assert all(weights[k] is v for k, v in before.items())
    del weights[PREFIX + "in_proj_z.lora_B.weight"]
    with pytest.raises(ValueError, match="incomplete"):
        normalize_gdn_weights(weights, CFG, 3)


def test_four_slice_retained_layout_and_aliases():
    catalog = {
        "hf": {
            PREFIX + "in_proj_qkv.lora_A.weight": 3,
            PREFIX + "in_proj_z.lora_A.weight": 7,
            PREFIX + "in_proj_a.lora_A.weight": 2,
        }
    }
    assert effective_rank(catalog["hf"], "in_proj_qkvz", 0) == 7
    assert effective_rank(catalog["hf"], "in_proj_ba", 0) == 2
    assert effective_rank(catalog["hf"], "in_proj_qkvz", 1) == 0
    a = torch.arange(2 * 4 * 16 * 8).reshape(2, 64, 8).to(torch.bfloat16)
    b = torch.ones(2, 28, 16, dtype=torch.bfloat16)
    e = RetainedLayout(a, b, "in_proj_qkvz", 0, catalog)
    raw = a.clone()
    pointers = (a.data_ptr(), e.store.data_ptr(), e.rank.data_ptr())
    e.load("hf", 1, 16)
    torch.testing.assert_close(
        a[1, :28], torch.cat([raw[1, i * 16 : i * 16 + 7] for i in range(4)])
    )
    e.load(None, 1)
    assert (a.data_ptr(), e.store.data_ptr(), e.rank.data_ptr()) == pointers
    assert e.rank.tolist() == [0, 0]
