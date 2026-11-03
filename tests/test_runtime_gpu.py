# SPDX-License-Identifier: Apache-2.0
"""Real GPU H2D ordering, ownership and slot-reuse tests."""

import gc
import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]

from loraforge.runtime import GPUAdapterSlots  # noqa: E402


def source(tmp_path, name, rank=3, dtype=torch.float32, rslora=False):
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
    a = torch.arange(rank * 64).reshape(rank, 64).to(dtype) / 100
    b = torch.arange(32 * rank).reshape(32, rank).to(dtype) / 100
    safetensors.save_file(
        {
            "base_model.model.proj.lora_A.weight": a,
            "base_model.model.proj.lora_B.weight": b,
        },
        directory / "adapter_model.safetensors",
    )
    return directory, a, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rslora", [False, True])
def test_cross_stream_reload_and_lifetime(tmp_path, dtype, rslora):
    path, a, b = source(tmp_path, "first", dtype=dtype, rslora=rslora)
    other, _, _ = source(tmp_path, "other", rank=1, dtype=dtype)
    pool = GPUAdapterSlots(1, device="cuda:0")
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        handle = pool.load(path, 0)
    with torch.cuda.stream(consumer):
        with pool.borrow(handle) as modules:
            (m,) = modules
            assert m.a.device == torch.device("cuda:0")
            assert m.rank == 3 and m.a.is_contiguous() and m.b.is_contiguous()
            result = (m.b.float() @ m.a.float()) * m.scaling
            with pytest.raises(RuntimeError, match="active"):
                pool.load(other, 0)
            with pytest.raises(RuntimeError, match="active"):
                pool.evict(handle)
        del modules, m
    with torch.cuda.stream(producer):
        replacement = pool.load(other, 0)
    with pytest.raises(ValueError, match="stale"):
        with pool.borrow(handle):
            pass
    pool.evict(replacement)
    del pool
    gc.collect()
    # Churn allocations before waiting to exercise record_stream lifetime.
    churn = [torch.empty((3, 64), device="cuda:0", dtype=dtype) for _ in range(32)]
    consumer.synchronize()
    scale = 6 / (3**0.5 if rslora else 3)
    torch.testing.assert_close(result.cpu(), (b.float() @ a.float()) * scale)
    del churn


def test_failure_budget_and_exception_release(tmp_path):
    path, _, _ = source(tmp_path, "first", rank=1)
    large, _, _ = source(tmp_path, "large", rank=3)
    pool = GPUAdapterSlots(1, device="cuda:0", max_tensor_bytes=384)
    handle = pool.load(path, 0)
    assert pool.tensor_bytes == 384
    with pytest.raises(ValueError, match="limit"):
        pool.load(large, 0)
    with pytest.raises(ValueError, match="different"):
        with GPUAdapterSlots(1, device="cuda:0").borrow(handle):
            pass
    with pytest.raises(RuntimeError, match="consumer failure"):
        with pool.borrow(handle):
            raise RuntimeError("consumer failure")
    pool.collect()
    pool.evict(handle)
    assert pool.tensor_bytes == 0
    with pytest.raises(ValueError, match="stale"):
        pool.evict(handle)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two visible GPUs required")
@pytest.mark.parametrize("index", [0, 1])
def test_residency_device_guard(tmp_path, index):
    path, a, b = source(tmp_path, "adapter")
    previous = torch.cuda.current_device()
    other = 1 - index
    device = torch.device("cuda", index)
    try:
        torch.cuda.set_device(other)
        pool = GPUAdapterSlots(1, device=str(device))
        handle = pool.load(path, 0)
        assert torch.cuda.current_device() == other
        with pool.borrow(handle) as modules:
            (module,) = modules
            assert module.a.device == module.b.device == device
            result = (module.b @ module.a) * module.scaling
        assert torch.cuda.current_device() == other
        pool.evict(handle)
        pool.collect()
        assert torch.cuda.current_device() == other
        torch.testing.assert_close(result.cpu(), (b @ a) * 2)
    finally:
        torch.cuda.set_device(previous)
