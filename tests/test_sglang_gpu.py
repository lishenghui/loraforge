# SPDX-License-Identifier: Apache-2.0
"""Actual SGLang 0.5.22 batch preparation and backend graph replay; no model downloads."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]

from loraforge.integrations.sglang import get_backend, install_sglang_backend  # noqa: E402


def make_backend(monkeypatch):
    import sglang.srt.lora.backend.chunked_backend as native

    # Explicit configuration for the real backend; no scheduler/model is constructed.
    monkeypatch.setattr(
        native, "get_lora", lambda: SimpleNamespace(max_lora_chunk_size=16)
    )
    return get_backend()(
        max_loras_per_batch=4, device=torch.device("cuda:0"), server_args=None
    )


def batch(decode=True):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        batch_size=7,
        extend_num_tokens=7,
        extend_seq_lens_cpu=[1] * 7,
        extend_seq_lens=torch.ones(7, dtype=torch.int32, device="cuda:0"),
        return_logprob=False,
    )


def inputs(slices):
    torch.manual_seed(21)
    dtype = torch.bfloat16
    x = torch.randn(7, 128, device="cuda:0", dtype=dtype) / 10
    a = torch.randn(4, slices * 8, 128, device="cuda:0", dtype=dtype) / 10
    b = torch.randn(4, slices * 32, 8, device="cuda:0", dtype=dtype) / 10
    offsets = torch.arange(slices + 1, device="cuda:0", dtype=torch.int64) * 32
    return x, a, b, offsets


def compute(backend, slices, x, a, b, offsets):
    if slices == 1:
        z = backend.run_lora_a_sgemm(x, a)
        return backend.run_lora_b_sgemm(z, b, offsets)
    if slices == 2:
        return backend.run_gate_up_lora(x, a, b, offsets)
    return backend.run_qkv_lora(x, a, b, offsets, 32)


def expected(x, a, b, ranks, routes, scales, slices):
    output = torch.zeros(7, slices * 32, device=x.device, dtype=x.dtype)
    for i, slot in enumerate(routes):
        rank = ranks[slot]
        for sl in range(slices):
            z = (x[i].float() @ a[slot, sl * rank : (sl + 1) * rank].float().T).to(
                x.dtype
            )
            output[i, sl * 32 : (sl + 1) * 32] = (
                z.float()
                @ b[slot, sl * 32 : (sl + 1) * 32, :rank].float().T
                * scales[slot]
            ).to(x.dtype)
    return output


def test_real_registration(monkeypatch):
    from sglang.srt.lora.backend.lora_registry import (
        LORA_SUPPORTED_BACKENDS,
        get_backend_from_name,
    )

    original = LORA_SUPPORTED_BACKENDS["csgmv"]
    monkeypatch.setitem(LORA_SUPPORTED_BACKENDS, "csgmv", original)
    install_sglang_backend()
    install_sglang_backend()
    assert get_backend_from_name("csgmv") is get_backend()


@pytest.mark.parametrize("slices", [1, 2, 3])
@pytest.mark.parametrize("decode", [False, True])
def test_real_batch_preparation_and_numerics(monkeypatch, slices, decode):
    backend = make_backend(monkeypatch)
    x, a, b, offsets = inputs(slices)
    ranks, routes, scales = [0, 1, 3, 8], [3, 1, 2, 0, 2, 3, 0], [0, 0.5, 1.5, 0.75]
    backend.prepare_lora_batch(batch(decode), routes, ranks, scales, False)
    assert backend._loraforge_decode is decode
    actual = compute(backend, slices, x, a, b, offsets)
    reference = expected(x, a, b, ranks, routes, scales, slices)
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.02)


@pytest.mark.parametrize("slices", [1, 2, 3])
def test_real_graph_metadata_and_fixed_slots(monkeypatch, slices):
    backend = make_backend(monkeypatch)
    x, a, b, offsets = inputs(slices)
    ranks, routes, scales = [0, 1, 3, 8], [3, 1, 2, 0, 2, 3, 0], [0, 0.5, 1.5, 0.75]
    forward = batch()
    backend.init_cuda_graph_batch_info(7, 1)
    backend.prepare_lora_batch(forward, routes, ranks, scales, True)
    info = backend.batch_info
    pointers = tuple(
        t.data_ptr()
        for t in (a, b, info.lora_ranks, info.req_weight_indices, info.scalings)
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compute(backend, slices, x, a, b, offsets)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = compute(backend, slices, x, a, b, offsets)
    for rank in (2, 0, 5):
        # Simulate a new occupant in native fixed-address slot 3.
        a[3].fill_(0.125)
        b[3].fill_(0.25)
        ranks[3] = rank
        scales[3] = 0.5 + rank
        routes = [0, 3, 3, 1, 2, 0, 3]
        backend.prepare_lora_batch(forward, routes, ranks, scales, True)
        assert backend.batch_info is info
        graph.replay()
        torch.cuda.synchronize()
        reference = expected(x, a, b, ranks, routes, scales, slices)
        torch.testing.assert_close(result, reference, rtol=0.03, atol=0.02)
    assert pointers == tuple(
        t.data_ptr()
        for t in (a, b, info.lora_ranks, info.req_weight_indices, info.scalings)
    )


@pytest.mark.parametrize(
    "slices,width,rank", [(1, 14336, 64), (2, 4096, 32), (3, 4096, 16)]
)
def test_bf16_decode_matches_native_accumulation(monkeypatch, slices, width, rank):
    """Model-width reductions must preserve native BF16 rounding boundaries."""
    from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend

    backend = make_backend(monkeypatch)
    torch.manual_seed(741)
    x = torch.randn(7, width, device="cuda:0", dtype=torch.bfloat16)
    a = torch.randn(4, slices * rank, width, device=x.device, dtype=x.dtype) / 100
    b = torch.randn(4, slices * 128, rank, device=x.device, dtype=x.dtype) / 100
    offsets = torch.arange(slices + 1, device=x.device, dtype=torch.int64) * 128
    backend.prepare_lora_batch(
        batch(),
        [1, 2, 3, 0, 1, 2, 3],
        [0, rank, rank // 2, rank],
        [0, 0.5, 1.5, 2.0],
        False,
    )
    z = backend._shrink(x, a, slices)
    native_z = ChunkedSgmvLoRABackend.run_lora_a_sgemm(backend, x, a, stack_num=slices)
    for i, g in enumerate([1, 2, 3, 0, 1, 2, 3]):
        live = [0, rank, rank // 2, rank][g] * slices
        torch.testing.assert_close(z[i, :live], native_z[i, :live], rtol=0, atol=0)
    base = torch.randn(7, slices * 128, device=x.device, dtype=x.dtype)
    expected = ChunkedSgmvLoRABackend.run_lora_b_sgemm(
        backend, z, b, offsets, base.clone()
    )
    actual = backend._expand(z, b, offsets, 128, base.clone())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
