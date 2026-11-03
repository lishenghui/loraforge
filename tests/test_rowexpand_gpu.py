# SPDX-License-Identifier: Apache-2.0
"""All Fleet projections, phases, unsupported ranks and real SGLang graph routing."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]

from loraforge.integrations._sglang_fleet import create_backend  # noqa: E402
from loraforge.runtime.retained import RetainedLayout  # noqa: E402


@pytest.fixture(scope="module")
def fleet_class():
    from sglang.srt.lora.mem_pool import LoRAMemoryPool

    from loraforge.ops.rowexpand import rowexpand_extension

    init, load = LoRAMemoryPool.init_buffers, LoRAMemoryPool.load_lora_weight_to_buffer
    rowexpand_extension()
    cls = create_backend({})
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
    [("o_proj", 1), ("down_proj", 1), ("gate_up_proj", 2), ("qkv_proj", 3)],
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
        z = backend._a(x, a, backend.batch_info, slices)
        return backend._b(
            z, b, backend.batch_info, offsets, 64, base.clone(), offsets_cpu
        )

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
    hits = backend.fleet_snapshot()["python_capture_eager_calls"]
    assert hits["rowexpand_decode" if decode else "rowexpand_prefill"] > 0
    if not decode:
        assert hits["rank_aware_shrink"] > 0


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
