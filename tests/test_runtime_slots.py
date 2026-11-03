# SPDX-License-Identifier: Apache-2.0
import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from loraforge.runtime import AdapterSlots  # noqa: E402


def source(tmp_path, name, rank=3, rslora=False):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": rank,
                "lora_alpha": 6,
                "target_modules": ["proj"],
                "bias": "none",
                "use_rslora": rslora,
            }
        )
    )
    safetensors.save_file(
        {
            "base_model.model.proj.lora_A.weight": torch.arange(rank * 5)
            .float()
            .reshape(rank, 5),
            "base_model.model.proj.lora_B.weight": torch.ones(4, rank),
        },
        directory / "adapter_model.safetensors",
    )
    return directory


@pytest.mark.parametrize("rslora", [False, True])
def test_reuse_scaling_and_ownership(tmp_path, rslora):
    first = source(tmp_path, "first", rslora=rslora)
    second = source(tmp_path, "second", rank=1)
    pool = AdapterSlots(1)
    handle = pool.load(first, 0)
    (module,) = pool.resolve(handle)
    assert module.rank == 3
    assert module.scaling == pytest.approx(6 / (3**0.5 if rslora else 3))
    assert module.a.is_contiguous() and module.b.is_contiguous()
    assert pool.tensor_bytes == (3 * 5 + 4 * 3) * 4
    module.a.zero_()
    assert pool.resolve(handle)[0].a.sum() > 0
    replacement = pool.load(second, 0)
    with pytest.raises(ValueError, match="stale"):
        pool.resolve(handle)
    with pytest.raises(ValueError, match="stale"):
        pool.evict(handle)
    assert pool.resolve(replacement)[0].rank == 1
    assert module.rank == 3
    pool.evict(replacement)
    assert pool.tensor_bytes == 0
    with pytest.raises(ValueError, match="stale"):
        pool.resolve(replacement)


def test_failed_load_preserves_occupant_and_budget(tmp_path):
    small = source(tmp_path, "small", rank=1)
    large = source(tmp_path, "large", rank=3)
    pool = AdapterSlots(2, max_tensor_bytes=36)
    handle = pool.load(small, 0)
    for path, slot in [(large, 0), (small, 1), (tmp_path / "missing", 0)]:
        with pytest.raises((ValueError, FileNotFoundError)):
            pool.load(path, slot)
        assert pool.resolve(handle)[0].rank == 1
        assert pool.tensor_bytes == 36
    with pytest.raises(ValueError, match="different pool"):
        AdapterSlots(2).resolve(handle)
    pool.evict(handle)
    assert pool.load(small, 1).slot == 1


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity(capacity):
    with pytest.raises(ValueError):
        AdapterSlots(capacity)


def test_compact_export_and_corrupt_reload(tmp_path):
    from loraforge.adapters import load_peft_adapter, save_peft_adapter

    original = source(tmp_path, "original", rank=3, rslora=True)
    snapshot = load_peft_adapter(original)
    (module,) = snapshot.modules
    compact = tmp_path / "compact"
    save_peft_adapter(
        snapshot,
        compact,
        factors={module.name: (module.a[:1].clone(), module.b[:, :1].clone())},
    )
    pool = AdapterSlots(1)
    handle = pool.load(compact, 0)
    (resident,) = pool.resolve(handle)
    assert resident.rank == 1
    assert resident.scaling == pytest.approx(module.scaling)
    assert pool.tensor_bytes == 36
    (compact / "adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        pool.load(compact, 0)
    assert pool.resolve(handle)[0].rank == 1


def test_concurrent_independent_slots(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = source(tmp_path, "shared")
    pool = AdapterSlots(4)
    with ThreadPoolExecutor(max_workers=4) as executor:
        handles = list(executor.map(lambda index: pool.load(path, index), range(4)))
    assert pool.tensor_bytes == 4 * 108
    for handle in handles:
        assert pool.resolve(handle)[0].rank == 3
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(pool.evict, handles))
    assert pool.tensor_bytes == 0


@pytest.mark.parametrize("slot", [-1, 1, True, 0.5])
def test_invalid_slot(tmp_path, slot):
    pool = AdapterSlots(1)
    with pytest.raises(ValueError, match="slot"):
        pool.load(tmp_path / "absent", slot)


@pytest.mark.parametrize("limit", [-1, True, 0.5])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        AdapterSlots(1, max_tensor_bytes=limit)
