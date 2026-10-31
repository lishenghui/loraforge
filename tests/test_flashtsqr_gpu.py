# SPDX-License-Identifier: Apache-2.0
"""Numerical and ownership contracts on an allocated CUDA/ROCm GPU."""

import gc

import pytest

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]

from loraforge.compression import refactor_lora, truncate_lora  # noqa: E402
from loraforge.ops import qr_factor  # noqa: E402


def random(shape, device="cuda", dtype=torch.float32, seed=123):
    return torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def relative(actual, expected):
    return (
        (actual.double() - expected.double()).norm()
        / expected.double().norm().clamp_min(1e-30)
    ).item()


@pytest.mark.parametrize(
    "path,n,m",
    [
        ("full", 1, 17),
        ("full", 16, 513),
        ("full", 48, 777),
        ("wide", 16, 1201),
        ("auto", 256, 777),
    ],
)
def test_qr_reconstruction_and_orthogonality(path, n, m):
    x = random((2, m, n))
    original = x.clone()
    factor = qr_factor(x, path=path)
    eye = torch.eye(n, device=x.device).expand(2, n, n)
    q = factor.apply_q(eye)
    assert relative(q @ factor.r, x) < 2e-5
    assert relative(q.transpose(1, 2).double() @ q.double(), eye) < 2e-5
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert factor.info.backend == "FlashTSQR"
    assert factor.info.source_sha256


@pytest.mark.parametrize("path", ["full", "wide"])
def test_interleaved_owned_factorizations(path):
    x = random((2, 777, 16))
    factor = qr_factor(x, path=path)
    expected = factor.apply_q(factor.r)
    other_x = random((3, 1025, 32), seed=456)
    other = qr_factor(other_x, path=path)
    assert relative(other.apply_q(other.r), other_x) < 2e-5
    assert relative(factor.apply_q(factor.r), expected) < 1e-7
    del other, other_x
    gc.collect()
    assert relative(factor.apply_q(factor.r), x) < 2e-5


@pytest.mark.parametrize("path", ["full", "wide"])
def test_cross_stream_use_and_lifetime(path):
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        # Inputs are created on the producer; default-stream kernels would race.
        x = random((2, 1025, 32))
        factor = qr_factor(x, path=path)
    with torch.cuda.stream(consumer):
        out = factor.apply_q(factor.r)
    del factor
    gc.collect()
    # Put allocator pressure on the stream that owned the reflectors.
    with torch.cuda.stream(producer):
        pressure = [torch.empty((2, 1025, 32), device="cuda") for _ in range(16)]
    consumer.synchronize()
    producer.synchronize()
    assert relative(out, x) < 2e-5
    del pressure


def test_same_handle_multiple_consumer_streams():
    x = random((2, 513, 32))
    factor = qr_factor(x)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            outputs.append(factor.apply_q(factor.r))
    for stream in streams:
        stream.synchronize()
    for out in outputs:
        assert relative(out, x) < 2e-5


def test_large_rank_wy_handle():
    x = random((2, 777, 160))
    try:
        factor = qr_factor(x, path="full")
    except ValueError as error:
        if "does not fit" in str(error):
            pytest.skip("WY full-path test requires more device shared memory")
        raise
    other = qr_factor(random((2, 513, 16)))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = factor.apply_q(factor.r)
    stream.synchronize()
    assert relative(out, x) < 2e-5
    del other


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two visible GPUs required")
def test_device_guard_and_input_device_preservation():
    old = torch.cuda.current_device()
    try:
        x = random((2, 513, 32), device="cuda:1")
        torch.cuda.set_device(0)
        first = qr_factor(x)
        second_x = random((2, 513, 16), device="cuda:0")
        second = qr_factor(second_x)
        assert relative(first.apply_q(first.r), x) < 2e-5
        assert relative(second.apply_q(second.r), second_x) < 2e-5
        assert first.r.device.index == 1
        a, b = (
            random((2, 16, 80), device="cuda:1"),
            random((2, 64, 16), device="cuda:1"),
        )
        out = refactor_lora(a, b)
        assert out.a.device.index == out.b.device.index == 1
        assert torch.cuda.current_device() == 0
        with pytest.raises(ValueError, match="device"):
            first.apply_q(second.r)
    finally:
        torch.cuda.set_device(old)


@pytest.mark.parametrize(
    "single,din,dout,dtype",
    [
        (False, 80, 64, torch.float32),
        (False, 64, 80, torch.bfloat16),
        (True, 80, 64, torch.float16),
    ],
)
def test_refactor_matches_dense_spectrum(single, din, dout, dtype):
    a, b = (
        random((2, 16, din), dtype=dtype),
        random((2, dout, 16), dtype=dtype, seed=456),
    )
    if single:
        a, b = a[0], b[0]
    a_before, b_before = a.clone(), b.clone()
    out = refactor_lora(a, b)
    dense = b.double() @ a.double()
    assert relative(out.b.double() @ out.a.double(), dense) < 2e-5
    reference = torch.linalg.svdvals(dense)[..., :16]
    torch.testing.assert_close(
        out.singular_values.double(), reference, rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(a, a_before, rtol=0, atol=0)
    torch.testing.assert_close(b, b_before, rtol=0, atol=0)
    assert out.a.dtype == out.b.dtype == torch.float32
    assert out.info.qr.backend == "FlashTSQR"


@pytest.mark.parametrize("backend", ["torch", "jacobi", "cpu"])
def test_evd_backend_selection_and_odd_rank(backend):
    a, b = random((2, 7, 40)), random((2, 48, 7), seed=456)
    out = refactor_lora(a, b, eigh_backend=backend)
    assert relative(out.b @ out.a, b @ a) < 2e-5
    assert out.info.eigh_backend == backend


@pytest.mark.parametrize("keep", [0, 1, 8, 16])
@pytest.mark.parametrize("single", [False, True])
def test_compact_and_padded_truncation(keep, single):
    a, b = random((2, 16, 64)), random((2, 80, 16), seed=456)
    if single:
        a, b = a[0], b[0]
    out = truncate_lora(a, b, keep)
    padded = truncate_lora(a, b, keep, pad_to_original_rank=True)
    assert out.a.shape[-2] == out.b.shape[-1] == keep
    for tensor in (out.a, out.b):
        assert (
            tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        )
    assert padded.a.shape == a.shape and padded.b.shape == b.shape
    dense = b.double() @ a.double()
    error = (out.b.double() @ out.a.double() - dense).norm(dim=(-2, -1))
    singular = torch.linalg.svdvals(dense)
    optimal = singular[..., keep:].norm(dim=-1)
    torch.testing.assert_close(error, optimal, rtol=2e-5, atol=2e-4)
    assert relative(padded.b @ padded.a, out.b @ out.a) < 2e-5
    assert out.info.retained_rank == keep
    assert not out.info.padded and padded.info.padded


def test_zero_and_rank_deficient_factors():
    a = random((2, 8, 40))
    b = random((2, 48, 8))
    a[:, 4:] = 0
    out = refactor_lora(a, b)
    assert relative(out.b @ out.a, b @ a) < 2e-5
    out = refactor_lora(torch.zeros_like(a), b)
    assert torch.isfinite(out.a).all() and torch.isfinite(out.b).all()
    assert (out.b @ out.a).count_nonzero() == 0


def test_noncontiguous_inputs_and_apply():
    x = random((2, 16, 513)).transpose(1, 2)
    factor = qr_factor(x)
    z = random((2, 7, 16)).transpose(1, 2)
    q = factor.apply_q(torch.eye(16, device="cuda").expand(2, 16, 16))
    torch.testing.assert_close(factor.apply_q(z), q @ z, rtol=2e-5, atol=2e-5)


def test_cooperative_apply():
    x = random((2, 513, 16))
    factor = qr_factor(x, apply_mode="cooperative")
    assert relative(factor.apply_q(factor.r), x) < 2e-5


def test_autocast_does_not_change_operator_precision():
    a, b = random((2, 16, 64)), random((2, 80, 16), seed=456)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = refactor_lora(a, b)
    assert out.a.dtype == out.b.dtype == torch.float32
    assert relative(out.b @ out.a, b @ a) < 2e-5


def test_invalid_gpu_inputs():
    x = random((2, 32, 8))
    with pytest.raises(ValueError, match="float32"):
        qr_factor(x.half())
    with pytest.raises(ValueError, match="rows"):
        qr_factor(x.transpose(1, 2))
    with pytest.raises(ValueError, match="unknown"):
        qr_factor(x, path="invalid")
    with pytest.raises(ValueError, match="full path"):
        qr_factor(x, path="wide", apply_mode="cooperative")
    a, b = random((8, 32)), random((40, 8))
    for keep in (-1, 9):
        with pytest.raises(ValueError, match="keep"):
            truncate_lora(a, b, keep)
    with pytest.raises(TypeError, match="integer"):
        truncate_lora(a, b, True)
    with pytest.raises(ValueError, match="EVD"):
        refactor_lora(a, b, eigh_backend="invalid")
