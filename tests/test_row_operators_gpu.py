# SPDX-License-Identifier: Apache-2.0
"""Retained-rank BF16 operators against dense references and graph slot updates."""

import hashlib
import platform
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="allocated GPU required"),
]

from loraforge.ops import row_expand, row_shrink  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def record_environment(record_testsuite_property):
    import loraforge.ops

    properties = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "capability": str(torch.cuda.get_device_capability(0)),
    }
    root = Path(loraforge.ops.__file__).parent
    for name in ("lora_rows.py", "_row_extension.py", "csrc/lora_rows.cu"):
        properties[name + ".sha256"] = hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()
    for name, value in properties.items():
        record_testsuite_property(name, str(value))


def case(slices=1, width=520, columns=None, device="cuda:0"):
    torch.manual_seed(31)
    ranks = [1, 2, 3, 7, 16, 17, 33, 64, 0]
    sizes = [1, 2, 3, 4, 8, 16, 15, 16, 3, 2]
    m, capacity = sum(sizes), 128
    widths = [19, 13, 11][:slices] if columns is None else [columns] * slices
    n = sum(widths)
    x = (torch.randn(m, width + 8, device=device) * 0.1).bfloat16()[:, :width]
    a = (
        torch.randn(len(ranks), slices * capacity, width, device=device) * 0.1
    ).bfloat16()
    z = torch.full((m, slices * capacity), 3.0, dtype=torch.bfloat16, device=device)
    storage = (torch.randn(m, n + 6, device=device) * 0.1).bfloat16()
    output = storage[:, 3 : 3 + n]
    permutation = torch.randperm(m).tolist()
    starts, slots = [0], []
    routes = [-1] * m
    for slot, size in enumerate(sizes):
        actual = slot if slot < len(ranks) else -1
        slots.append(actual)
        for t in permutation[starts[-1] : starts[-1] + size]:
            routes[t] = actual
        starts.append(starts[-1] + size)
    # Include an empty active segment and unused graph-capacity entries.
    slots += [0, -1]
    starts += [m, m]
    pieces, offsets, buckets, cursor = [], [], [], 0
    for rank in ranks:
        padding = (-cursor) % 8
        pieces.append(torch.zeros(padding, dtype=torch.bfloat16, device=device))
        cursor += padding
        bucket = 1 << (max(1, rank) - 1).bit_length()
        b = (torch.randn(n, bucket, device=device) * 0.1).bfloat16()
        b[:, rank:] = 0
        offsets.append(cursor)
        buckets.append(bucket)
        pieces.append(b.flatten())
        cursor += b.numel()
    boundaries = [0]
    for value in widths:
        boundaries.append(boundaries[-1] + value)

    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=device)

    meta = dict(
        b_offsets=tensor(offsets, torch.int64),
        b_strides=tensor(buckets),
        ranks=tensor(ranks),
        scales=tensor([1, -0.7, 0.5, 1.25, 2, 0, 0.3, -1, 1], torch.float32),
        segment_indptr=tensor(starts),
        segment_slots=tensor(slots),
        token_indices=tensor(permutation),
        slice_offsets=tensor(boundaries),
        segment_count=tensor([len(slots) - 1]),
    )
    return x, a, z, torch.cat(pieces), storage, output, tensor(routes), meta


def shrink_reference(x, a, z, routes, ranks, slices):
    expected = z.clone()
    for row, slot in enumerate(routes.tolist()):
        rank = int(ranks[slot]) if slot >= 0 else 0
        if rank:
            value = a[slot, : slices * rank].double() @ x[row].double()
            expected[row, : slices * rank] = value.bfloat16()
    return expected


def expand_reference(z, b, output, meta):
    expected = output.clone()
    ranks = meta["ranks"].tolist()
    indptr = meta["segment_indptr"].tolist()
    slots = meta["segment_slots"].tolist()
    indices = meta["token_indices"].tolist()
    boundaries = meta["slice_offsets"].tolist()
    for segment in range(meta["segment_count"].item()):
        slot = slots[segment]
        if slot < 0 or not ranks[slot]:
            continue
        rank = ranks[slot]
        offset = meta["b_offsets"][slot].item()
        bucket = meta["b_strides"][slot].item()
        weights = b[offset : offset + output.shape[1] * bucket].view(-1, bucket)
        for row in indices[indptr[segment] : indptr[segment + 1]]:
            for s, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                value = (
                    weights[start:end, :rank].double()
                    @ z[row, s * rank : (s + 1) * rank].double()
                )
                delta = (value * meta["scales"][slot].item()).bfloat16()
                expected[row, start:end] = (
                    output[row, start:end].float() + delta.float()
                ).bfloat16()
    return expected


def close(actual, expected):
    relative = (
        actual.double() - expected.double()
    ).norm() / expected.double().norm().clamp_min(1e-30)
    assert relative.item() < 0.005


@pytest.mark.parametrize("slices", [1, 2, 3])
@pytest.mark.parametrize("threads", [128, 256])
def test_mixed_ranks_slices_segments_and_base(slices, threads):
    x, a, z, b, storage, output, routes, meta = case(slices)
    before, all_before = output.clone(), storage.clone()
    expected_z = shrink_reference(x, a, z, routes, meta["ranks"], slices)
    row_shrink(x, a, z, routes, meta["ranks"], num_slices=slices)
    # Compare live values separately: untouched sentinel columns cannot dominate.
    for row, slot in enumerate(routes.tolist()):
        rank = meta["ranks"][slot].item() if slot >= 0 else 0
        if rank:
            close(z[row, : slices * rank], expected_z[row, : slices * rank])
        torch.testing.assert_close(
            z[row, slices * rank :], expected_z[row, slices * rank :], rtol=0, atol=0
        )
    expected = expand_reference(z, b, before, meta)
    # One block forces several differently ranked segments through the same CTA.
    row_expand(z, b, output, **meta, threads=threads, blocks=1)
    close(output, expected)
    delta = torch.zeros_like(output)
    row_expand(z, b, delta, **meta, threads=threads, blocks=2)
    close(delta, expand_reference(z, b, torch.zeros_like(output), meta))
    torch.testing.assert_close(storage[:, :3], all_before[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(storage[:, -3:], all_before[:, -3:], rtol=0, atol=0)
    skipped = (routes == -1) | (routes == 8)
    torch.testing.assert_close(output[skipped], before[skipped], rtol=0, atol=0)


@pytest.mark.parametrize("width,columns", [(4096, 14336), (14336, 4096)])
def test_model_projection_sizes(width, columns):
    x, a, z, b, _, output, routes, meta = case(width=width, columns=columns)
    expected_z = shrink_reference(x, a, z, routes, meta["ranks"], 1)
    row_shrink(x, a, z, routes, meta["ranks"])
    live = (
        torch.arange(128, device=x.device)[None, :]
        < torch.where(routes >= 0, meta["ranks"][routes], 0)[:, None]
    )
    close(z[live], expected_z[live])
    expected = expand_reference(z, b, output, meta)
    row_expand(z, b, output, **meta)
    close(output, expected)


def test_graph_replay_changed_routes_ranks_offsets_and_weights():
    x, a, z, b, _, output, routes, meta = case(slices=3)
    base = output.clone()

    def run():
        row_shrink(x, a, z, routes, meta["ranks"], num_slices=3)
        row_expand(z, b, output, **meta)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Slot 2 is reloaded with rank 7 and points at slot 3's compact region.
    # Both shrink and expand see the new slot metadata at the same addresses.
    meta["ranks"][2] = 7
    meta["b_strides"][2] = 8
    meta["b_offsets"][2] = meta["b_offsets"][3]
    a[2].mul_(0.5)
    b.mul_(0.75)
    x.mul_(0.75)
    meta["scales"].mul_(0.5)
    # Reverse the physical token permutation and update per-token shrink routes.
    meta["token_indices"].copy_(meta["token_indices"].flip(0))
    for seg, slot in enumerate(meta["segment_slots"].tolist()):
        start, end = meta["segment_indptr"][seg : seg + 2].tolist()
        routes[meta["token_indices"][start:end].long()] = slot
    meta["segment_count"].sub_(2)  # Omit the base and empty trailing segments.
    output.copy_(base)
    z.fill_(3)
    expected_z = shrink_reference(x, a, z, routes, meta["ranks"], 3)
    expected = expand_reference(expected_z, b, base, meta)
    graph.replay()
    torch.cuda.synchronize()
    close(output, expected)
    # Explicit eager comparison isolates graph replay from FP32 reduction order.
    replay = output.clone()
    output.copy_(base)
    run()
    torch.testing.assert_close(output, replay, rtol=0, atol=0)


def test_scaled_delta_rounding_contract():
    # Round a midpoint delta first, then add the base. One final rounding alone
    # would give a different result, even though B @ Z is exactly representable.
    device = "cuda:0"
    z = torch.tensor([[1.0]], device=device, dtype=torch.bfloat16)
    b = torch.tensor([1.0], device=device, dtype=torch.bfloat16)
    y = torch.tensor([[0.0078125]], device=device, dtype=torch.bfloat16)

    def i(values, dtype=torch.int32):
        return torch.tensor(values, device=device, dtype=dtype)

    row_expand(
        z,
        b,
        y,
        b_offsets=i([0], torch.int64),
        b_strides=i([1]),
        ranks=i([1]),
        scales=i([1.00390625], torch.float32),
        segment_indptr=i([0, 1]),
        segment_slots=i([0]),
        token_indices=i([0]),
        slice_offsets=i([0, 1]),
        segment_count=i([1]),
    )
    assert y.item() == 1.0078125
    assert torch.tensor(1.00390625 + 0.0078125).bfloat16().item() == 1.015625


def test_reject_invalid_structures_and_aliases():
    x, a, z, b, _, output, routes, meta = case()
    with pytest.raises(ValueError, match="same CUDA"):
        row_shrink(x, a.cpu(), z, routes, meta["ranks"])
    with pytest.raises(ValueError, match="aligned"):
        row_shrink(x[:, 1:513], a[:, :, :512].contiguous(), z, routes, meta["ranks"])
    with pytest.raises(ValueError, match="inference"):
        row_expand(z.requires_grad_(), b, output, **meta)
    z.requires_grad_(False)
    with pytest.raises(ValueError, match="length"):
        row_expand(z, b, output, **(meta | {"b_offsets": meta["b_offsets"][:2]}))
    with pytest.raises(ValueError, match="threads"):
        row_expand(z, b, output, **meta, threads=64)
    with pytest.raises(ValueError, match="share storage"):
        row_expand(z, b, z[:, : output.shape[1]], **meta)


def test_empty_batch_and_zero_active_segments():
    x, a, z, b, _, output, routes, meta = case()
    before = output.clone()
    meta["segment_count"].zero_()
    row_expand(z, b, output, **meta)
    torch.testing.assert_close(output, before, rtol=0, atol=0)
    row_shrink(x[:0], a, z[:0], routes[:0], meta["ranks"])
    row_expand(z[:0], b, output[:0], **meta)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two visible GPUs required")
def test_device_guard():
    x, a, z, b, _, output, routes, meta = case(device="cuda:1")
    with torch.cuda.device(0):
        row_shrink(x, a, z, routes, meta["ranks"])
        row_expand(z, b, output, **meta)
        assert torch.cuda.current_device() == 0
