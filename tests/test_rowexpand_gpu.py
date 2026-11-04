# SPDX-License-Identifier: Apache-2.0
"""All Fleet projections, phases, unsupported ranks and real SGLang graph routing."""

from itertools import accumulate
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]

from loraforge.integrations._sglang_fleet import create_backend  # noqa: E402
from loraforge.runtime.retained import RetainedLayout, packed_segment_capacity  # noqa: E402


@pytest.fixture(
    scope="module",
    params=[(False, False), (True, False), (False, True)],
    ids=["frozen", "active-prefill", "hybrid"],
)
def fleet_class(request):
    from sglang.srt.lora.mem_pool import LoRAMemoryPool

    from loraforge.ops.rowexpand import rowexpand_extension

    init, load = LoRAMemoryPool.init_buffers, LoRAMemoryPool.load_lora_weight_to_buffer
    rowexpand_extension()
    cls = create_backend({}, active_prefill=request.param[0], hybrid=request.param[1])
    yield cls
    LoRAMemoryPool.init_buffers, LoRAMemoryPool.load_lora_weight_to_buffer = init, load
    del LoRAMemoryPool._loraforge_fleet


def forward(tokens, decode):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        batch_size=tokens,
        extend_num_tokens=tokens,
        extend_seq_lens_cpu=[1] * tokens,
        extend_seq_lens=torch.ones(tokens, dtype=torch.int32, device="cuda"),
        return_logprob=False,
    )


@pytest.mark.parametrize(
    "name,slices",
    [
        ("o_proj", 1),
        ("down_proj", 1),
        ("gate_up_proj", 2),
        ("qkv_proj", 3),
        ("in_proj_qkvz", 4),
        ("out_proj", 1),
        ("in_proj_ba", 2),
    ],
)
@pytest.mark.parametrize("decode,tokens", [(True, 4), (True, 33), (False, 67)])
def test_all_projection_paths(fleet_class, monkeypatch, name, slices, decode, tokens):
    import sglang.srt.lora.backend.chunked_backend as native

    monkeypatch.setattr(
        native, "get_lora", lambda: SimpleNamespace(max_lora_chunk_size=16)
    )
    backend = fleet_class(
        max_loras_per_batch=4, device=torch.device("cuda:0"), server_args=None
    )
    torch.manual_seed(274)
    a = torch.randn(4, slices * 128, 128, device="cuda", dtype=torch.bfloat16) / 20
    b = torch.randn(4, slices * 64, 128, device="cuda", dtype=torch.bfloat16) / 20
    # Check lower effective ranks than checkpoint stride, plus native rank-128 fallback.
    names = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }.get(name, [name])
    catalog = {
        str(k): {
            f"base_model.model.model.layers.0.block.{n}.lora_A.weight": k for n in names
        }
        for k in (3, 16, 32, 64, 128)
    }
    e = RetainedLayout(a, b, name, 0, catalog)
    fleet_class.bind_layout(e)
    raw_a = a.clone()
    for slot, k in enumerate((0, 3, 32, 128)):
        e.load(str(k) if k else None, slot, 128 if k else 0)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16) / 10
    base = torch.randn(tokens, slices * 64, device="cuda", dtype=torch.bfloat16) / 10
    offsets_cpu = torch.arange(slices + 1, dtype=torch.int32) * 64
    offsets = offsets_cpu.to("cuda")
    scales = [0.0, 0.5, 1.5, 0.75]
    routes = ([0, 1, 3, 2, 1, 2, 0] * ((tokens + 6) // 7))[:tokens]
    fb = forward(tokens, decode)
    if decode:
        backend.init_cuda_graph_batch_info(tokens, 1)

    def prepare():
        backend.prepare_lora_batch(fb, routes, [0, 128, 128, 128], scales, decode)

    def compute():
        # Match the real SGLang layer's keyword calls, not just private helpers.
        common = dict(
            output_offset=offsets,
            output_offset_cpu=offsets_cpu,
            base_output=base.clone(),
        )
        if slices >= 3:
            return backend.run_qkv_lora(
                x=x,
                qkv_lora_a=a,
                qkv_lora_b=b,
                max_qkv_out_dim=64,
                n_slices=slices,
                **common,
            )
        if slices == 2:
            return backend.run_gate_up_lora(
                x=x, gate_up_lora_a=a, gate_up_lora_b=b, **common
            )
        z = backend.run_lora_a_sgemm(x=x, weights=a)
        return backend.run_lora_b_sgemm(x=z, weights=b, **common)

    def oracle():
        out = base.clone()
        for row, g in enumerate(routes):
            k = int(e.rank[g])
            if not k:
                continue
            for sl in range(slices):
                z = (x[row].float() @ a[g, sl * k : (sl + 1) * k].float().T).to(x.dtype)
                delta = (
                    z.float() @ b[g, sl * 64 : (sl + 1) * 64, :k].float().T * scales[g]
                ).to(x.dtype)
                out[row, sl * 64 : (sl + 1) * 64] = (
                    out[row, sl * 64 : (sl + 1) * 64].float() + delta.float()
                ).to(x.dtype)
        return out

    prepare()
    compute()
    torch.cuda.synchronize()
    if decode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = compute()
        for k in (16, 0, 64, 3):
            a[1].copy_(raw_a[1])
            e.load(str(k) if k else None, 1, 128 if k else 0)
            routes.reverse()
            prepare()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, oracle(), rtol=0.03, atol=0.003)
    else:
        torch.testing.assert_close(compute(), oracle(), rtol=0.03, atol=0.003)
        mixed = routes[:]
        for active in ([1], [2], [3], [0], mixed):
            routes[:] = (active * tokens)[:tokens]
            prepare()
            slots = backend._fleet_prefill_slots
            backend._fleet_prefill_slots = None
            reference = compute()
            backend._fleet_prefill_slots = slots
            torch.testing.assert_close(compute(), reference, rtol=0, atol=0)
        # Replace the same active slot across both rank classes and inactive.
        routes[:] = [1] * tokens
        for k in (16, 32, 0, 3):
            a[1].copy_(raw_a[1])
            e.load(str(k) if k else None, 1, 128 if k else 0)
            prepare()
            slots = backend._fleet_prefill_slots
            backend._fleet_prefill_slots = None
            reference = compute()
            backend._fleet_prefill_slots = slots
            torch.testing.assert_close(compute(), reference, rtol=0, atol=0)
        if backend._fleet_active_prefill and name != "in_proj_ba":
            assert backend._fleet_hits["skipped_native_high_prefill"] > 0
            assert backend._fleet_hits["skipped_rowexpand_low_prefill"] > 0
        # Capture with only low-rank requests, then replay high/mixed/null routes.
        # Both expand branches must remain in the real static prefill buffers.
        backend.init_prefill_cuda_graph_batch_info(tokens, tokens)

        def prepare_graph():
            backend.prepare_lora_batch(
                fb,
                routes,
                [0, 128, 128, 128],
                scales,
                use_cuda_graph=False,
                use_prefill_cuda_graph=True,
            )
            assert backend._fleet_prefill_slots is None

        prepare_graph()
        compute()
        torch.cuda.synchronize()
        # The capture guard is also safe if an external caller has retained
        # eager metadata before beginning capture.
        backend._fleet_prefill_slots = (1,)
        skipped = backend._fleet_hits["skipped_native_high_prefill"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = compute()
        assert backend._fleet_hits["skipped_native_high_prefill"] == skipped
        for active in ([2], [3], [0], mixed, [1]):
            routes[:] = (active * tokens)[:tokens]
            a[1].copy_(raw_a[1])
            e.load("64", 1, 128)
            prepare_graph()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, compute(), rtol=0, atol=0)
            torch.testing.assert_close(actual, oracle(), rtol=0.03, atol=0.003)
    hits = backend.fleet_snapshot()["python_capture_eager_calls"]
    if name != "in_proj_ba" and (decode or not backend._fleet_hybrid):
        assert hits["rowexpand_decode" if decode else "rowexpand_prefill"] > 0
    if not decode:
        assert hits["rank_aware_shrink"] > 0


def test_four_slice_unequal_widths_and_strided_output():
    from loraforge.ops.rowexpand import direct_shrink, row_expand

    torch.manual_seed(15)
    a = torch.randn(3, 64, 64, device="cuda", dtype=torch.bfloat16) / 10
    b = torch.randn(3, 28, 16, device="cuda", dtype=torch.bfloat16) / 10
    catalog = {
        str(k): {"base_model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight": k}
        for k in (3, 16)
    }
    layout = RetainedLayout(a, b, "in_proj_qkvz", 0, catalog)
    for slot, k in enumerate((0, 3, 16)):
        layout.load(str(k) if k else None, slot, 16 if k else 0)
    x = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16) / 10
    z = torch.zeros(6, 64, device="cuda", dtype=torch.bfloat16)
    output = torch.zeros(6, 32, device="cuda", dtype=torch.bfloat16)[:, :28]

    def ints(data):
        return torch.tensor(data, device="cuda", dtype=torch.int32)

    routes = [0, 1, 2, 1, 0, 2]
    direct_shrink(x, a, z, ints(routes), layout.rank, 4, 16)
    offsets = [0, 6, 12, 20, 28]
    row_expand(
        z,
        layout.store,
        layout.boff,
        layout.bucket,
        output,
        ints([0, 2, 4, 6]),
        ints([0, 1, 2]),
        ints([0, 4, 1, 3, 2, 5]),
        layout.rank,
        torch.ones(3, device="cuda"),
        ints(offsets),
        ints(3),
        16,
        blocks=8,
        threads=128,
    )
    expected = torch.zeros_like(output)
    for row, slot in enumerate(routes):
        k = [0, 3, 16][slot]
        if not k:
            continue
        for sl, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            zz = (x[row].float() @ a[slot, sl * k : (sl + 1) * k].float().T).bfloat16()
            expected[row, start:end] = (
                zz.float() @ b[slot, start:end, :k].float().T
            ).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.003)


@pytest.mark.parametrize("tokens", [1, 8, 33])
@pytest.mark.parametrize(
    "slices,width,hidden",
    [(1, 1024, 2048), (2, 7168, 1024), (3, 5120, 1024), (4, 8192, 1024)],
)
def test_bounded_launches_are_bitwise_equal_after_graph_route_changes(
    tokens, slices, width, hidden
):
    from loraforge.integrations._sglang_rank_aware import shrink
    from loraforge.ops.rowexpand import row_expand

    torch.manual_seed(612)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    a = torch.randn(4, slices * 64, hidden, **kw) / 20
    b = torch.randn(4, width, 64, **kw) / 20
    catalog = {
        str(k): {"model.layers.0.proj.lora_A.weight": k} for k in (3, 16, 32, 64)
    }
    layout = RetainedLayout(a, b, "proj", 0, catalog)
    raw_a = a.clone()
    for slot, k in enumerate((0, 3, 32, 64)):
        layout.load(str(k) if k else None, slot, 64 if k else 0)
    x = torch.randn(tokens, hidden, **kw) / 10
    base = torch.randn(tokens, width, **kw) / 10
    info = SimpleNamespace(
        use_cuda_graph=True,
        max_len=16,
        num_segments=0,
        weight_indices=torch.zeros(128, device="cuda", dtype=torch.int32),
        seg_indptr=torch.zeros(129, device="cuda", dtype=torch.int32),
        permutation=torch.zeros(tokens, device="cuda", dtype=torch.int32),
        lora_ranks=layout.rank,
        scalings=torch.tensor([0.0, 0.75, 1.25, 0.5], device="cuda"),
    )
    nseg = torch.zeros((), device="cuda", dtype=torch.int32)
    widths = [width // slices] * slices
    widths[-1] += width - sum(widths)
    offsets = torch.tensor(
        [0] + list(accumulate(widths)),
        device="cuda",
        dtype=torch.int32,
    )

    def prepare(pattern):
        routes = (pattern * tokens)[:tokens]
        permutation, indptr, slots = [], [0], []
        for slot in range(4):
            rows = [i for i, s in enumerate(routes) if s == slot]
            for start in range(0, len(rows), 16):
                permutation.extend(rows[start : start + 16])
                indptr.append(len(permutation))
                slots.append(slot)
        info.weight_indices.zero_()
        info.weight_indices[: len(slots)].copy_(
            torch.tensor(slots, device="cuda", dtype=torch.int32)
        )
        info.seg_indptr.fill_(tokens)
        info.seg_indptr[: len(indptr)].copy_(
            torch.tensor(indptr, device="cuda", dtype=torch.int32)
        )
        info.permutation.copy_(
            torch.tensor(permutation, device="cuda", dtype=torch.int32)
        )
        nseg.fill_(len(slots))

    def compute(bounded):
        z = shrink(x, a, info, slices, limit_segments=bounded)
        return row_expand(
            z,
            layout.store,
            layout.boff,
            layout.bucket,
            base.clone(),
            info.seg_indptr,
            info.weight_indices,
            info.permutation,
            layout.rank,
            info.scalings,
            offsets,
            nseg,
            64,
            blocks=(
                min(
                    880, packed_segment_capacity(tokens, 4, 16) * ((width + 255) // 256)
                )
                if bounded
                else 880
            ),
        )

    prepare([1])
    torch.testing.assert_close(compute(True), compute(False), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, reference = compute(True), compute(False)
    pointers = [
        t.data_ptr()
        for t in (
            info.weight_indices,
            info.seg_indptr,
            info.permutation,
            layout.rank,
            nseg,
        )
    ]
    for k, routes in [(16, [0, 1, 2, 3]), (64, [1]), (3, [3, 2, 1, 0]), (0, [0])]:
        a[1].copy_(raw_a[1])
        layout.load(str(k) if k else None, 1, 64 if k else 0)
        prepare(routes)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if tokens == 33:
        prepare([0] * 17 + [1] + [2] + [3] * 14)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert pointers == [
        t.data_ptr()
        for t in (
            info.weight_indices,
            info.seg_indptr,
            info.permutation,
            layout.rank,
            nseg,
        )
    ]


@pytest.mark.parametrize("rank", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("slices", [1, 2, 3])
def test_migrated_kernels_match_pinned_probe(rank, slices):
    import importlib.util
    import os
    from pathlib import Path

    from loraforge.ops.rowexpand import rowexpand_extension

    path = os.environ.get("LORAFORGE_ROWEXPAND_REFERENCE")
    if not path:
        pytest.skip("explicit pinned probe extension needed for migration comparison")
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    ours = rowexpand_extension()
    torch.manual_seed(19 + rank)
    kw = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(17, 128, **kw)
    a = torch.randn(2, slices * 64, 128, **kw) / 20
    ranks = torch.tensor([0, rank], device="cuda", dtype=torch.int32)
    routes = torch.ones(17, device="cuda", dtype=torch.int32)
    z = [torch.zeros(17, slices * 64, **kw) for _ in range(2)]
    for mod, out in zip((probe, ours), z, strict=True):
        mod.shrink(x, a, out, routes, ranks, slices, rank)
    torch.testing.assert_close(z[0], z[1], rtol=0, atol=0)
    width = slices * 73
    store = torch.randn(width * rank, **kw) / 20
    seg = torch.tensor([0, 16, 17], device="cuda", dtype=torch.int32)
    slots = torch.ones(2, device="cuda", dtype=torch.int32)
    perm = torch.arange(17, device="cuda", dtype=torch.int32).flip(0).contiguous()
    boff = torch.zeros(2, device="cuda", dtype=torch.int64)
    buckets = torch.tensor([0, rank], device="cuda", dtype=torch.int32)
    scales = torch.tensor([0, 1.25], device="cuda")
    offsets = torch.arange(slices + 1, device="cuda", dtype=torch.int32) * 73
    nseg = torch.tensor(2, device="cuda", dtype=torch.int32)
    base = torch.randn(17, width + 7, **kw)[:, :width]  # noncontiguous row stride
    outputs = [base.clone(), base.clone()]
    for mod, zz, out in zip((probe, ours), z, outputs, strict=True):
        mod.expand(
            zz,
            store,
            boff,
            buckets,
            out,
            seg,
            slots,
            perm,
            ranks,
            scales,
            offsets,
            nseg,
            width,
            64,
            220,
            128,
        )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
