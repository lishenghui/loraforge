# SPDX-License-Identifier: Apache-2.0
"""Do not turn divergent, nonfinite, or mismatched model outputs into a pass."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "benchmarks" / "qwen35_hybrid.py"
spec = importlib.util.spec_from_file_location("qwen35_protocol", path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def record(tokens, *, adapter=0):
    return dict(serial=[dict(adapter=adapter, length=17, tokens=tokens)], mixed=[])


def test_model_gate_and_common_prefix():
    native = record([[-0.1, 1], [-0.2, 2], [-0.3, 3]])
    assert protocol.numerical_check(native, native)["numerical_pass"]
    drift = record([[-0.11, 1], [-0.2, 7], [-9.0, 3]])
    report = protocol.numerical_check(native, drift)
    assert not report["numerical_pass"]
    assert report["max_common_prefix_logprob_difference"] == pytest.approx(0.01)
    drift = record([[-0.11, 1], [-0.26, 2], [-0.3, 3]])
    assert not protocol.numerical_check(native, drift)["numerical_pass"]


def test_model_gate_rejects_nonfinite_and_changed_cases():
    native = record([[-0.1, 1], [-0.2, 2]])
    with pytest.raises(AssertionError, match="nonfinite"):
        protocol.numerical_check(native, record([[-0.1, 1], [float("nan"), 2]]))
    with pytest.raises(AssertionError, match="cases differ"):
        protocol.numerical_check(native, record([[-0.1, 1], [-0.2, 2]], adapter=1))


def test_timing_opt_in_preserves_failed_logprob_gate_and_rejects_changed_tokens():
    native = record([[-0.1, 1], [-0.2, 2]])
    drift = protocol.numerical_check(native, record([[-0.1, 1], [-0.4, 2]]))
    assert not protocol.timing_allowed(drift)
    assert protocol.timing_allowed(drift, allow_logprob_drift=True)
    assert not drift["numerical_pass"]
    changed = protocol.numerical_check(native, record([[-0.1, 1], [-0.2, 3]]))
    assert not protocol.timing_allowed(changed, allow_logprob_drift=True)
    empty = protocol.numerical_check(record([]), record([]))
    assert not protocol.timing_allowed(empty, allow_logprob_drift=True)


def test_base_payload_omits_adapter_and_preserves_work():
    manifest = dict(
        prompt=list(range(128)), adapters=[dict(name=f"a{i}") for i in range(4)]
    )
    native = protocol.payload(manifest, "native", 5)
    fleet = protocol.payload(manifest, "fleet", 5)
    base = protocol.payload(manifest, "base", 5)
    assert native == fleet
    assert native["lora_path"] == "a1"
    assert "lora_path" not in base
    assert base == {k: v for k, v in native.items() if k != "lora_path"}
    assert len(base["input_ids"]) == 128
    assert base["sampling_params"]["max_new_tokens"] == 64


def test_model_gate_requires_matching_large_batch_validation():
    native = record([[-0.1, 1]])
    candidate = dict(native, mixed32=[[[-0.1, 1]]] * 32)
    with pytest.raises(AssertionError, match="cases differ"):
        protocol.numerical_check(native, candidate)
    native = dict(candidate)
    candidate = dict(candidate, mixed32=[[[-0.1, 2]]] * 32)
    assert not protocol.numerical_check(native, candidate)["numerical_pass"]


@pytest.mark.parametrize("rank", [4, 8, 32, 64])
def test_native_and_fleet_use_the_prepared_pool_capacity(rank):
    manifest = dict(
        model="local-model",
        max_lora_rank=rank,
        target_modules=["qkv_proj", "in_proj_qkvz"],
        adapters=[dict(name=f"a{i}", path=f"local-adapter-{i}") for i in range(4)],
    )
    native = protocol.server_options(manifest, "native", 12345)
    assert native == protocol.server_options(manifest, "fleet", 12345)
    assert native["max_lora_rank"] == rank
    assert native["max_loaded_loras"] == native["max_loras_per_batch"] == 5
    assert "max_lora_rank" not in protocol.server_options(manifest, "base", 12345)


@pytest.mark.parametrize("rank", [4, 8, 32])
def test_fixed_rank_fixture_has_compact_weights_and_unit_scaling(
    tmp_path, monkeypatch, rank
):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _: SimpleNamespace(
            apply_chat_template=lambda *a, **kw: "prompt",
            encode=lambda *a, **kw: list(range(128)),
        ),
    )
    model = tmp_path / "model"
    model.mkdir()
    protocol.dump(
        model / "config.json",
        dict(
            text_config=dict(
                model_type="qwen3_5_text",
                hidden_size=8,
                intermediate_size=16,
                linear_key_head_dim=4,
                linear_num_key_heads=2,
                linear_value_head_dim=4,
                linear_num_value_heads=2,
                head_dim=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                layer_types=["linear_attention", "full_attention"],
            )
        ),
    )
    root = tmp_path / "fixture"
    protocol.prepare(root, model, rank)
    manifest = protocol.read(root / "manifest.json")
    assert manifest["max_lora_rank"] == rank
    assert set(manifest["retained_rank_histogram"]) == {str(rank)}
    count = 0
    for adapter in manifest["adapters"]:
        path = Path(adapter["path"])
        config = protocol.read(path / "adapter_config.json")
        assert config["r"] == config["lora_alpha"] == rank
        weights = safetensors.load_file(path / "adapter_model.safetensors")
        for name, weight in weights.items():
            assert weight.shape[0 if ".lora_A." in name else 1] == rank
            assert weight.dtype == torch.bfloat16 and torch.count_nonzero(weight) > 0
            count += weight.numel()
    fixture = manifest["fixture"]
    assert fixture["physical_parameters"] == fixture["retained_parameters"] == count
    assert fixture["retained_fraction_of_rank64"] == rank / 64


def test_prepare_rejects_invalid_fixed_rank_before_writing(tmp_path):
    with pytest.raises(ValueError, match="rank must be 4, 8 or 32"):
        protocol.prepare(tmp_path / "fixture", tmp_path / "model", rank=0)
    assert not (tmp_path / "fixture").exists()


def test_two_arm_comparison_checks_equal_pool_capacity(tmp_path):
    protocol.dump(
        tmp_path / "manifest.json",
        dict(concurrencies=[1], scope="test", max_lora_rank=4),
    )
    for mode, rate in (("native", 10), ("fleet", 12)):
        attempt = tmp_path / "results" / mode
        attempt.mkdir(parents=True)
        protocol.dump(
            tmp_path / f"{mode}-latest.json",
            dict(state="complete", attempt=str(attempt)),
        )
        protocol.dump(attempt / "correctness.json", record([[-0.1, 1]]))
        protocol.dump(attempt / "server-args.json", dict(max_lora_rank=4))
        protocol.dump(
            attempt / "measurements.json", [dict(concurrency=1, tokens_per_second=rate)]
        )
        protocol.dump(
            attempt / "audit-test.json",
            dict(
                counts={"DECODE/graph": 1},
                loads=[dict(fixed_addresses=True)],
                fleet=dict(python_capture_eager_calls=dict(rowexpand_decode=1)),
            ),
        )
    protocol.compare(tmp_path, ("native", "fleet"))
    report = protocol.read(tmp_path / "comparison.json")
    assert report["numerical_pass"] and report["execution_audit_pass"]
    assert report["performance"][0]["fleet_over_native"] == 1.2
    assert report["adapter_effect_observed"] is None
    graph_args = dict(
        max_lora_rank=4, cuda_graph_config=dict(prefill=dict(backend="full"))
    )
    for mode in ("native", "fleet"):
        protocol.dump(tmp_path / f"results/{mode}/server-args.json", graph_args)
    protocol.compare(tmp_path, ("native", "fleet"))
    assert not protocol.read(tmp_path / "comparison.json")["execution_audit_pass"]
    for mode in ("native", "fleet"):
        audit_file = tmp_path / f"results/{mode}/audit-test.json"
        audit = protocol.read(audit_file)
        audit["counts"]["EXTEND/graph"] = 1
        audit["cold_graph"] = dict(replay_prepares=1, gdn_replay=1)
        protocol.dump(audit_file, audit)
    protocol.compare(tmp_path, ("native", "fleet"))
    assert protocol.read(tmp_path / "comparison.json")["execution_audit_pass"]
    protocol.dump(tmp_path / "results/native/server-args.json", dict(max_lora_rank=64))
    protocol.compare(tmp_path, ("native", "fleet"))
    assert not protocol.read(tmp_path / "comparison.json")["execution_audit_pass"]


def test_cold_graph_configuration_matches_all_baselines(monkeypatch):
    manifest = dict(
        model="local-model",
        max_lora_rank=4,
        target_modules=["qkv_proj"],
        adapters=[dict(name=f"a{i}", path=f"adapter-{i}") for i in range(4)],
    )
    monkeypatch.delenv("LORAFORGE_BENCH_QWEN35_COLD_PREFILL", raising=False)
    assert protocol.server_options(manifest, "native", 1234)[
        "disable_prefill_cuda_graph"
    ]
    monkeypatch.setenv("LORAFORGE_BENCH_QWEN35_COLD_PREFILL", "1")
    for mode in ("native", "fleet", "base"):
        options = protocol.server_options(manifest, mode, 1234)
        assert not options["disable_prefill_cuda_graph"]
        assert options["chunked_prefill_size"] == 1024
        assert options["cuda_graph_config"]["prefill"] == dict(
            backend="full", bs=[1024], full_prefill_max_req=8, max_seq_len=2048
        )
    with pytest.raises(AssertionError, match="cap4"):
        protocol.server_options(dict(manifest, max_lora_rank=8), "fleet", 1234)


def test_long_graph_validation_must_match_reference_cases_and_tokens():
    native = dict(record([[-0.1, 1]]), long32=[[[-0.2, 2]]] * 32)
    assert protocol.numerical_check(native, native)["same_token_cases"] == 33
    candidate = dict(native, long32=[[[-0.2, 3]]] * 32)
    report = protocol.numerical_check(native, candidate)
    assert not protocol.timing_allowed(report, allow_logprob_drift=True)
    with pytest.raises(AssertionError, match="cases differ"):
        protocol.numerical_check(record([[-0.1, 1]]), candidate)
