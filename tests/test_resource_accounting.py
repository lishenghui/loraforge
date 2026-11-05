# SPDX-License-Identifier: Apache-2.0
"""Trace accounting must not confuse summed kernels, wall time, or CPU nesting."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "benchmarks/analyze_qwen35_resources.py"
spec = importlib.util.spec_from_file_location("resource_accounting", path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def event(name, ts, dur, cat="kernel", **kwargs):
    return dict(name=name, ts=ts, dur=dur, cat=cat, ph="X", **kwargs)


def test_gpu_overlap_is_not_added_to_wall_time():
    events = [
        event("shrink_kernel", 0, 10),
        event("_dequant", 5, 10),
        event("mystery", 30, 5),
    ]
    r = a.analyze(events)
    assert r["summed_kernel_us"] == 25
    assert r["gpu_activity_union_us"] == 20
    assert r["gpu_activity_span_us"] == 35
    assert r["gpu_activity_gap_us"] == 15
    assert r["groups"]["lora_A"]["fraction_of_summed_kernel_us"] == 0.4
    assert r["phase_groups"] == [
        dict(phase="unattributed", category="base_dequant", count=1, sum_us=10),
        dict(phase="unattributed", category="lora_A", count=1, sum_us=10),
        dict(phase="unattributed", category="unknown_other", count=1, sum_us=5),
    ]


def test_gpu_scope_precedes_cpu_correlation_and_unknowns_stay_unknown():
    events = [
        event(
            "LF/model_forward/mode=EXTEND/bs=8/tokens=1024",
            0,
            30,
            "gpu_user_annotation",
        ),
        event("Cijk_base", 5, 10),
        event("unknown", 50, 10),
    ]
    r = a.analyze(events)
    assert r["gpu_model_forward_spans"] == 1
    assert r["phase_groups"] == [
        dict(phase="EXTEND", category="dense_gemm_including_head", count=1, sum_us=10),
        dict(phase="unattributed", category="unknown_other", count=1, sum_us=10),
    ]


def test_runtime_links_gpu_kernel_to_cpu_forward_without_clock_guessing():
    events = [
        event(
            "LF/model_forward/mode=DECODE/bs=32/tokens=32",
            0,
            20,
            "user_annotation",
            tid=7,
        ),
        event("hipGraphLaunch", 3, 1, "cuda_runtime", tid=7, args={"correlation": 42}),
        event("expand_kernel<4>", 100, 5, args={"correlation": 42}),
    ]
    r = a.analyze(events)
    assert r["phase_groups"] == [
        dict(phase="DECODE", category="lora_B", count=1, sum_us=5)
    ]
    assert r["batches"] == [dict(mode="DECODE", batch=32, tokens=32, count=1)]
    assert r["cpu_inclusive_us"]["model_forward"] == 20


@pytest.mark.parametrize(
    "name,expected",
    [
        ("void gdn_ab_unpack_kernel(...)", "lora_fused_with_consumer"),
        ("row_act_kernel", "lora_fused_with_consumer"),
        ("fused_recurrent_gated_delta_rule_packed_decode_kernel", "GDN_state_conv"),
        ("Cijk_Alik_Bljk", "dense_gemm_including_head"),
        ("cgemm_4bit_bf16", "base_fused_nf4"),
        ("unidentified_custom_kernel", "unknown_other"),
    ],
)
def test_kernel_category_keeps_fused_consumers_separate(name, expected):
    assert a.category(name) == expected


def test_ambiguous_attention_name_needs_verified_source_list():
    assert a.category("_fwd_kernel") == "unknown_other"
    assert a.category("_fwd_kernel", ("_fwd_kernel",)) == "attention"


def test_storage_accounting_deduplicates_views_but_keeps_device_ownership():
    from types import SimpleNamespace as N

    source = Path(__file__).resolve().parents[1] / "benchmarks/qwen35_resource_audit.py"
    spec = importlib.util.spec_from_file_location("resource_memory", source)
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)

    def tensor(pointer, size, device):
        return N(
            device=device,
            untyped_storage=lambda: N(data_ptr=lambda: pointer, nbytes=lambda: size),
        )

    assert (
        memory.tensor_bytes(
            [
                tensor(1, 100, "cuda:0"),
                tensor(1, 100, "cuda:0"),
                tensor(1, 100, "cuda:1"),
                tensor(2, 20, "cuda:0"),
                None,
            ]
        )
        == 220
    )


@pytest.mark.parametrize("mode", ["base", "native", "fleet"])
def test_profile_payload_omits_logprobs_while_validation_can_request_them(
    monkeypatch, mode
):

    benchmarks = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(benchmarks))
    source = benchmarks / "run_qwen35_resources.py"
    spec = importlib.util.spec_from_file_location("resource_requests", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = dict(
        prompt=list(range(128)), adapters=[dict(name=f"a{i}") for i in range(4)]
    )
    prof = runner.batched(manifest, mode, 8)
    validation = runner.batched(manifest, mode, 8, logprobs=True)
    assert not prof["return_logprob"] and validation["return_logprob"]
    assert prof["input_ids"] == validation["input_ids"] == [list(range(128))] * 8
    assert prof["sampling_params"] == validation["sampling_params"]
    if mode == "base":
        assert "lora_path" not in prof
    else:
        assert prof["lora_path"] == [f"a{i % 4}" for i in range(8)]


def test_component_byte_counts_serialize_numpy_scalar_storage_sizes():
    import json
    from types import SimpleNamespace as N

    np = pytest.importorskip("numpy")
    source = Path(__file__).resolve().parents[1] / "benchmarks/qwen35_resource_audit.py"
    spec = importlib.util.spec_from_file_location("resource_bytes", source)
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    tensor = N(
        device="cuda:0",
        untyped_storage=lambda: N(data_ptr=lambda: 1, nbytes=lambda: np.int64(100)),
    )
    value = memory.tensor_bytes([tensor])
    assert type(value) is int and json.loads(json.dumps(dict(bytes=value))) == dict(
        bytes=100
    )
