# SPDX-License-Identifier: Apache-2.0
"""Raw GEMV and real vLLM 0.26.0 execution/graph contracts."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="allocated GPU required"),
]

from loraforge.ops.vllm_gemv import expand, shrink  # noqa: E402


def tensors(dtype, rank=32, m=7, padded=True):
    torch.manual_seed(123)
    device = "cuda"
    x = torch.randn(m, 514, device=device, dtype=dtype)[:, :513]
    a = [torch.randn(4, rank, 513, device=device, dtype=dtype) * 0.1 for _ in range(3)]
    b = [
        torch.randn(4, n, rank, device=device, dtype=dtype) * 0.1 for n in (17, 13, 21)
    ]
    if padded:
        a, b = [w.unsqueeze(1) for w in a], [w.unsqueeze(1) for w in b]
    routes = (torch.arange(m, device=device, dtype=torch.int32) % 5) - 1
    y = torch.randn(m, 57, device=device, dtype=dtype) * 0.1
    return x, a, b, routes, y


def reference(x, a, b, routes, base, *, scale=0.7, offset=0, add=True):
    result = base.double().clone()
    for s, (aa, bb) in enumerate(zip(a, b, strict=True)):
        aa, bb = aa.reshape(4, -1, x.shape[-1]), bb.reshape(4, bb.shape[-2], -1)
        for t, g in enumerate(routes.tolist()):
            if g >= 0:
                delta = (bb[g].double() @ (aa[g].double() @ x[t].double())) * scale
                result[t, offset : offset + bb.shape[-2]] = delta + (
                    base[t, offset : offset + bb.shape[-2]].double() if add else 0
                )
        offset += bb.shape[-2]
    return result.to(base.dtype)


def close(actual, expected):
    # Relative norm avoids unstable elementwise relative errors at zero.
    error = (actual.double() - expected.double()).norm()
    limit = 2e-5 if actual.dtype == torch.float32 else 0.01
    assert (error / expected.double().norm().clamp_min(1e-20)).item() < limit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [1, 7, 33, 128, 256])
def test_raw_gemv(dtype, rank):
    x, a, b, routes, y = tensors(dtype, rank, padded=rank != 7)
    before = y.clone()
    buf = torch.empty(3, x.shape[0], rank, device=x.device, dtype=torch.float32)
    shrink(x, a, buf, routes, scale=0.7)
    expand(buf, b, y, routes, offset=3, add=False)
    close(y, reference(x, a, b, routes, before, offset=3, add=False))
    torch.testing.assert_close(y[routes < 0], before[routes < 0], rtol=0, atol=0)
    torch.testing.assert_close(y[:, :3], before[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(y[:, 54:], before[:, 54:], rtol=0, atol=0)


@pytest.fixture
def framework(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.config import LoRAConfig
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    from loraforge.integrations import vllm as integration

    original = PunicaWrapperGPU.add_lora_linear
    monkeypatch.setattr(PunicaWrapperGPU, "add_lora_linear", original)
    monkeypatch.setattr(integration, "_installed", None)
    integration.install_vllm_backend()
    integration.install_vllm_backend()
    with pytest.raises(RuntimeError, match="conflict"):
        integration.install_vllm_backend(tokens_per_lora=2)
    config = LoRAConfig(max_loras=4, max_lora_rank=32, specialize_active_lora=True)
    wrapper = PunicaWrapperGPU(128, 128, "cuda", lora_config=config)
    wrapper.is_prefill = False
    return wrapper, original, integration


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["gemv", "dense", "prefill", "unspecialized"])
def test_real_framework(framework, monkeypatch, dtype, mode):
    wrapper, native, _ = framework
    x, a, b, routes, y = tensors(dtype, m=64 if mode == "dense" else 7)
    x = x.contiguous()
    wrapper.token_mapping_meta.prepare_tensors(routes)
    wrapper.is_prefill = mode == "prefill"
    wrapper.lora_config.specialize_active_lora = mode != "unspecialized"
    from loraforge.ops import _vllm_gemv_kernels as kernels

    calls = []
    original_kernel = kernels._shrink_kernel

    class ObservedKernel:
        def __getitem__(self, grid):
            calls.append(grid)
            return original_kernel[grid]

    expected = y.clone()
    native(wrapper, expected, x, tuple(a), tuple(b), 0.7, (17, 13, 21))
    monkeypatch.setattr(kernels, "_shrink_kernel", ObservedKernel())
    wrapper.add_lora_linear(y, x, tuple(a), tuple(b), 0.7, (17, 13, 21))
    close(y, expected)
    assert bool(calls) == (mode == "gemv")
    if mode == "gemv":
        reference_output = reference(x, a, b, routes, y.new_zeros(y.shape))
        # Compare isolated LoRA deltas as well as outputs with a base projection.
        custom_delta = y.new_zeros(y.shape)
        wrapper.add_lora_linear(custom_delta, x, tuple(a), tuple(b), 0.7, (17, 13, 21))
        close(custom_delta, reference_output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_framework_graph_routes_and_slot_reuse(framework, dtype):
    wrapper, _, _ = framework
    x, a, b, routes, y = tensors(dtype)
    wrapper.token_mapping_meta.prepare_tensors(routes)

    def run():
        wrapper.add_lora_linear(y, x, tuple(a), tuple(b), 0.7, (17, 13, 21))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(2):
        # Keep the active-count specialization unchanged while routes and
        # weights change at their original addresses, including base routes.
        routes.copy_(routes.roll(1))
        wrapper.token_mapping_meta.prepare_tensors(routes)
        for weight in (*a, *b):
            weight.mul_(0.5)
        y.fill_(0.1)
        expected = reference(x, a, b, routes, y)
        graph.replay()
        close(y, expected)


def test_framework_no_adapter(framework):
    wrapper, _, _ = framework
    x, a, b, routes, y = tensors(torch.float16)
    routes.fill_(-1)
    wrapper.token_mapping_meta.prepare_tensors(routes)
    before = y.clone()
    wrapper.add_lora_linear(y, x, tuple(a), tuple(b), 0.7, (17, 13, 21))
    torch.testing.assert_close(y, before, rtol=0, atol=0)


def test_framework_torch_compile(framework):
    wrapper, _, _ = framework
    x, a, b, routes, y = tensors(torch.float16)
    wrapper.token_mapping_meta.prepare_tensors(routes)
    expected = reference(x, a, b, routes, y)

    def run(y, x):
        wrapper.add_lora_linear(y, x, tuple(a), tuple(b), 0.7, (17, 13, 21))

    compiled = torch.compile(run, fullgraph=True)
    compiled(y, x)
    close(y, expected)
