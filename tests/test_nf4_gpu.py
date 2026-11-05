# SPDX-License-Identifier: Apache-2.0
"""Installed public NF4 tensor API against BNB, including graph replay and streams."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("shape", [(64, 64), (1024, 2048), (8192, 1024)])
def test_exact_nf4_reconstruction_with_owned_out_graph_and_stream(shape):
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    from loraforge.ops.nf4 import dequantize_nf4, prepare_scales

    torch.manual_seed(731)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    packed, state = quantize_4bit(
        weight, blocksize=64, quant_type="nf4", compress_statistics=True
    )
    scales = prepare_scales(state)
    reference = dequantize_4bit(packed, quant_state=state)
    out = torch.empty_like(reference)
    assert dequantize_nf4(packed, scales, state.code, shape, out=out) is out
    assert torch.equal(out, reference)
    address = out.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dequantize_nf4(packed, scales, state.code, shape, out=out)
    graph.replay()
    assert out.data_ptr() == address and torch.equal(out, reference)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        value = dequantize_nf4(packed, scales, state.code, shape)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(value, reference)


def test_nf4_output_cannot_alias_packed_input_storage():
    from loraforge.ops.nf4 import dequantize_nf4

    out = torch.empty((64, 64), device="cuda", dtype=torch.bfloat16)
    packed = out.view(torch.uint8).view(-1)[:2048]
    scales = torch.ones(64, device="cuda", dtype=torch.float32)
    code = torch.zeros(16, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="alias"):
        dequantize_nf4(packed, scales, code, (64, 64), out=out)
