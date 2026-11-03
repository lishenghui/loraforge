# SPDX-License-Identifier: Apache-2.0
"""Decode numerics and fixed-address replay using synthetic packed slot buffers."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]

from loraforge.ops import lora_expand, lora_shrink  # noqa: E402


def fixture(dtype, slices, capacity):
    generator = torch.Generator(device="cuda:0").manual_seed(23)
    sizes = [5, 3, 7][:slices]
    x = torch.randn(14, 777, generator=generator, device="cuda:0", dtype=dtype) / 10
    x = x[::2]
    a = (
        torch.randn(
            4, slices * capacity, 777, generator=generator, device="cuda:0", dtype=dtype
        )
        / 10
    )
    b = (
        torch.randn(
            4, sum(sizes), capacity, generator=generator, device="cuda:0", dtype=dtype
        )
        / 10
    )
    ranks = torch.tensor(
        [0, 1, min(3, capacity), capacity], device="cuda:0", dtype=torch.int64
    )
    routes = torch.tensor([0, 3, 2, 1, 3, 0, 2], device="cuda:0", dtype=torch.int64)
    scales = torch.tensor([0, 0.25, 1.5, 0.75], device="cuda:0")
    offsets = torch.tensor(
        [0] + [sum(sizes[: i + 1]) for i in range(slices)],
        device="cuda:0",
        dtype=torch.int64,
    )
    base = (
        torch.randn(
            7, sum(sizes) + 4, generator=generator, device="cuda:0", dtype=dtype
        )
        / 10
    )
    return x, a, b, ranks, routes, scales, offsets, sizes, base


def reference(x, a, b, ranks, routes, scales, offsets, base):
    expected = base.clone()
    for row, slot in enumerate(routes.cpu().tolist()):
        rank = int(ranks[slot])
        if not rank:
            continue
        for i in range(offsets.numel() - 1):
            start, end = int(offsets[i]), int(offsets[i + 1])
            z = (x[row].float() @ a[slot, i * rank : (i + 1) * rank].float().T).to(
                x.dtype
            )
            delta = (z.float() @ b[slot, start:end, :rank].float().T * scales[slot]).to(
                x.dtype
            )
            expected[row, start:end] += delta
    return expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("slices", [1, 2, 3])
@pytest.mark.parametrize("capacity", [8, 33])
def test_dense_decode(dtype, slices, capacity):
    x, a, b, ranks, routes, scales, offsets, sizes, base = fixture(
        dtype, slices, capacity
    )
    expected = reference(x, a, b, ranks, routes, scales, offsets, base)
    z = lora_shrink(x, a, ranks, routes, slices)
    result = lora_expand(z, b, ranks, scales, routes, offsets, max(sizes), base)
    assert result.data_ptr() == base.data_ptr()
    tol = 2e-5 if dtype == torch.float32 else 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(result, expected, rtol=tol, atol=tol)
    assert z.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("slices", [1, 3])
def test_graph_routing_and_slot_reload(dtype, slices):
    x, a, b, ranks, routes, scales, offsets, sizes, initial = fixture(dtype, slices, 8)
    pointers = tuple(t.data_ptr() for t in (a, b, ranks, routes, scales))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            z = lora_shrink(x, a, ranks, routes, slices)
            lora_expand(z, b, ranks, scales, routes, offsets, max(sizes))
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        z = lora_shrink(x, a, ranks, routes, slices)
        result = lora_expand(z, b, ranks, scales, routes, offsets, max(sizes))
    for rank in (2, 0, 5):
        # Same address, new occupant/rank, changed routing and scale after capture.
        a[3].fill_(0.125)
        b[3].fill_(0.25)
        ranks[3] = rank
        scales[3] = 0.5 + rank
        routes.copy_(torch.tensor([3, 0, 1, 2, 3, 3, 0], device="cuda:0"))
        expected = reference(
            x,
            a,
            b,
            ranks,
            routes,
            scales,
            offsets,
            torch.zeros_like(initial[:, : sum(sizes)]),
        )
        graph.replay()
        torch.cuda.synchronize()
        tol = (
            2e-5 if dtype == torch.float32 else 3e-3 if dtype == torch.float16 else 2e-2
        )
        torch.testing.assert_close(result, expected, rtol=tol, atol=tol)
    assert pointers == tuple(t.data_ptr() for t in (a, b, ranks, routes, scales))


@pytest.mark.parametrize("capacity", [128, 256])
def test_large_rank_capacity(capacity):
    x, a, b, ranks, routes, scales, offsets, sizes, base = fixture(
        torch.bfloat16, 1, capacity
    )
    expected = reference(x, a, b, ranks, routes, scales, offsets, base)
    z = lora_shrink(x, a, ranks, routes)
    actual = lora_expand(z, b, ranks, scales, routes, offsets, max(sizes), base)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
