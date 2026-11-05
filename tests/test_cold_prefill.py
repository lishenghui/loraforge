# SPDX-License-Identifier: Apache-2.0
"""Admission, stable graph-owned metadata, and eager/decode fallback contracts."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as N

import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch():
    return N(
        forward_mode=N(name="EXTEND"),
        batch_size=8,
        input_ids=list(range(1024)),
        extend_seq_lens_cpu=[128] * 8,
        extend_prefix_lens_cpu=[0] * 8,
        encoder_lens=None,
        spec_info=None,
        contains_mm_inputs=lambda: False,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_size", 7),
        ("input_ids", list(range(1023))),
        ("extend_seq_lens_cpu", [129, 127] + [128] * 6),
        ("extend_prefix_lens_cpu", [1] + [0] * 7),
        ("extend_seq_lens_cpu", None),
        ("spec_info", object()),
        ("encoder_lens", [1] * 8),
        ("forward_mode", N(name="DECODE")),
        ("forward_mode", N(name="TARGET_VERIFY")),
        ("contains_mm_inputs", lambda: True),
    ],
)
def test_cold_prefill_admission(key, value):
    module = load("qwen35_cold_prefill_metadata")
    candidate = batch()
    assert module.eligible(candidate)
    setattr(candidate, key, value)
    assert not module.eligible(candidate)


def test_cold_metadata_owns_static_storage():
    torch = pytest.importorskip("torch")
    module = load("qwen35_cold_prefill_metadata")
    lengths = torch.tensor([1024] + [0] * 7, dtype=torch.int64)
    meta = module.ColdMetadata(lengths)
    meta.refresh(lengths, lengths)
    assert meta.qo.tolist() == [0] + [1024] * 8
    pointers = meta.pointers
    lengths.fill_(128)
    meta.refresh(lengths, [128] * 8)
    assert meta.qo.tolist() == list(range(0, 1025, 128))
    assert meta.kv.tolist() == [0] * 9
    assert meta.indices.tolist() == [0]
    assert meta.pointers == pointers
    for invalid in ([127] * 8, [1025, -1] + [0] * 6, [128] * 7):
        with pytest.raises(AssertionError):
            meta.refresh(lengths, invalid)
    assert meta.refreshes == 2
    assert not torch.cuda.is_initialized()


def test_cold_graph_metadata_refresh_and_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.models",
        "sglang.srt.models.qwen3_5",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.linear",
        "sglang.srt.layers.attention.linear.gdn_backend",
        "sglang.srt.layers.attention.triton_backend",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.cuda_graph_config",
        "sglang.srt.model_executor.runner",
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
    )
    for name in names:
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, mod)
    sys.modules["sglang"].__version__ = "0.5.22"
    full = [True]
    calls = []

    class MHA:
        dcp_size = 1
        use_mla = False
        sliding_window_size = None
        use_sliding_window_kv_pool = False
        dllm_attention = None
        enable_deterministic = False

        def init_forward_metadata_out_graph(self, batch, in_capture=False):
            calls.append("mha-original")
            self.forward_metadata = "decode"

    class GDN:
        def init_forward_metadata_out_graph(self, batch, in_capture=False):
            calls.append("gdn-original")
            self.forward_metadata = "decode"

        def _forward_metadata(self, batch):
            return N(
                has_mamba_track_mask=False,
                replayssm_write_pos=None,
                replayssm_force_flush=None,
                mamba_cache_indices=batch.req_pool_indices.clone(),
                query_start_loc=torch.arange(0, 1025, 128, dtype=torch.int32),
                logical_num_tokens=1024,
            )

        def _init_target_verify_qkv_routing(self, batch):
            calls.append("gdn-routing")

    candidate = batch()
    candidate.input_ids = torch.zeros(1024, dtype=torch.int64)
    for name in ("seq_lens", "orig_seq_lens", "extend_seq_lens"):
        setattr(candidate, name, torch.tensor([1024] + [0] * 7))
    candidate.seq_lens_cpu = candidate.seq_lens.clone()
    candidate.extend_start_loc = torch.zeros(8, dtype=torch.int64)
    candidate.positions = torch.zeros(1024, dtype=torch.int64)
    candidate.req_pool_indices = torch.arange(8, dtype=torch.int32)

    class Runner:
        def capture_prepare(self, num_tokens):
            return candidate, "backend"

        def can_run_graph(self, batch):
            calls.append("native-admission")
            return True

    sys.modules["sglang.srt.layers.attention.triton_backend"].TritonAttnBackend = MHA
    sys.modules["sglang.srt.layers.attention.triton_backend"].ForwardMetadata = N
    sys.modules["sglang.srt.layers.attention.linear.gdn_backend"].GDNAttnBackend = GDN
    sys.modules[
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner"
    ].PrefillCudaGraphRunner = Runner
    sys.modules[
        "sglang.srt.model_executor.cuda_graph_config"
    ].check_cuda_graph_backend = lambda *a: full[0]
    metadata = load("qwen35_cold_prefill_metadata")
    monkeypatch.setitem(sys.modules, "qwen35_cold_prefill_metadata", metadata)
    mod = load("qwen35_cold_prefill_graph")
    mod.install()
    wrapped = MHA.init_forward_metadata_out_graph
    mod.install()
    assert MHA.init_forward_metadata_out_graph is wrapped
    runner, mha, gdn = Runner(), MHA(), GDN()
    captured, _ = runner.capture_prepare(1024)
    assert captured.extend_seq_lens.tolist() == [128] * 8
    assert captured.positions.tolist() == list(range(128)) * 8
    assert captured.extend_start_loc.tolist() == list(range(0, 1024, 128))
    assert runner.can_run_graph(captured)
    mha.init_forward_metadata_out_graph(captured, in_capture=True)
    gdn.init_forward_metadata_out_graph(captured, in_capture=True)
    pointers = mha._lf_cold_metadata.pointers
    saved = gdn._lf_cold_gdn_metadata
    state_pointer = saved.mamba_cache_indices.data_ptr()
    query_pointer = saved.query_start_loc.data_ptr()
    captured.req_pool_indices.add_(10)
    mha.init_forward_metadata_out_graph(captured)
    gdn.init_forward_metadata_out_graph(captured)
    assert mha._lf_cold_metadata.pointers == pointers
    assert gdn.forward_metadata is saved
    assert saved.mamba_cache_indices.tolist() == list(range(10, 18))
    assert saved.mamba_cache_indices.data_ptr() == state_pointer
    assert saved.query_start_loc.data_ptr() == query_pointer
    captured.forward_mode.name = "DECODE"
    mha.init_forward_metadata_out_graph(captured)
    gdn.init_forward_metadata_out_graph(captured)
    assert mha.forward_metadata == gdn.forward_metadata == "decode"
    assert not runner.can_run_graph(captured)
    captured.forward_mode.name = "EXTEND"
    captured.req_pool_indices.add_(10)
    mha.init_forward_metadata_out_graph(captured)
    gdn.init_forward_metadata_out_graph(captured)
    assert gdn.forward_metadata is saved
    assert saved.mamba_cache_indices.tolist() == list(range(20, 28))
    assert saved.mamba_cache_indices.data_ptr() == state_pointer
    captured.extend_prefix_lens_cpu[0] = 1
    assert not runner.can_run_graph(captured)
    with pytest.raises(AssertionError):
        mha.init_forward_metadata_out_graph(captured)
    with pytest.raises(AssertionError):
        gdn.init_forward_metadata_out_graph(captured)
    full[0] = False
    assert runner.can_run_graph(captured)
    mha.init_forward_metadata_out_graph(captured)
    assert mha.forward_metadata == "decode"
    assert calls.count("mha-original") == 2
    assert calls.count("gdn-original") == 1
    assert mod.stats["buffers"] == 1
    assert mod.stats["replay_prepares"] == mod.stats["gdn_replay"] == 2
    sys.modules["sglang"].__version__ = "0.5.23"
    with pytest.raises(RuntimeError, match="0.5.22"):
        mod.install()
