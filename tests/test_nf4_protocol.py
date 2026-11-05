# SPDX-License-Identifier: Apache-2.0
"""Quantized-base compatibility must keep unrelated validation and adapter dtype."""

import importlib.util
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as N

import pytest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compat = load("qwen35_nf4_compat")
protocol = load("qwen35_hybrid")


def test_platform_extension_retains_other_checks():
    source = """def check(self):
        rocm_supported_quantization = ["awq", "fp8"]
        assert self.quantization in rocm_supported_quantization
        assert self.dtype == "bf16"
        assert self.valid_checkpoint
    """
    result = {}
    exec(compat.extended_verify_source(source), result)
    run = result["check"]
    run(N(quantization="bitsandbytes", dtype="bf16", valid_checkpoint=True))
    run(N(quantization="awq", dtype="bf16", valid_checkpoint=True))
    for obj in (
        N(quantization="bitsandbytes", dtype="fp16", valid_checkpoint=True),
        N(quantization="bitsandbytes", dtype="bf16", valid_checkpoint=False),
        N(quantization="unknown", dtype="bf16", valid_checkpoint=True),
    ):
        with pytest.raises(AssertionError):
            run(obj)


@pytest.mark.parametrize(
    "source",
    [
        "def check():\n pass",
        'def check():\n rocm_supported_quantization=["bitsandbytes"]',
        'def check():\n rocm_supported_quantization=("awq",)',
        'def check():\n rocm_supported_quantization=["awq"]\n rocm_supported_quantization=["fp8"]',
    ],
)
def test_platform_extension_refuses_unknown_source(source):
    with pytest.raises(AssertionError):
        compat.extended_verify_source(source)


class Format(str, Enum):
    BNB = "bitsandbytes"
    SAFE = "safetensors"


@dataclass
class Config:
    load_format: object
    download_dir: str = "keep-download-directory"
    model_loader_extra_config: object = None


@pytest.mark.parametrize("kind", ["bitsandbytes", Format.BNB])
def test_adapter_loader_copies_only_format(kind):
    base = Config(kind, model_loader_extra_config={"owned": "caller"})
    adapter = compat.adapter_load_config(base)
    assert adapter is not base
    assert adapter.load_format == "safetensors"
    assert base.load_format == kind
    assert adapter.download_dir == base.download_dir
    assert adapter.model_loader_extra_config is base.model_loader_extra_config


@pytest.mark.parametrize("kind", ["auto", "pt", "safetensors", Format.SAFE])
def test_nonquantized_adapter_loader_keeps_identity(kind):
    original = Config(kind)
    assert compat.adapter_load_config(original) is original


@pytest.mark.parametrize("rank", [4, 32])
def test_nf4_cli_configuration_preserves_bf16_lora_and_shared_options(
    monkeypatch, rank
):
    manifest = dict(
        model="fused-nf4",
        max_lora_rank=rank,
        target_modules=["qkv_proj"],
        adapters=[dict(name="a0", path="local-bf16-adapter")],
    )
    monkeypatch.setenv("LORAFORGE_BENCH_QWEN35_NF4", "1")
    monkeypatch.setenv("LORAFORGE_BENCH_QWEN35_COLD_PREFILL", "1")
    native = protocol.server_options(manifest, "native", 1234)
    assert native == protocol.server_options(manifest, "fleet", 1234)
    assert native["dtype"] == "bfloat16"
    assert native["quantization"] == native["load_format"] == "bitsandbytes"
    assert native["lora_paths"][0]["lora_path"] == "local-bf16-adapter"
    base = protocol.server_options(manifest, "base", 1234)
    assert base["quantization"] == "bitsandbytes" and not base["enable_lora"]
    assert base["cuda_graph_config"] == native["cuda_graph_config"]
    with pytest.raises(AssertionError, match="rank4"):
        protocol.server_options(dict(manifest, max_lora_rank=8), "fleet", 1234)


@pytest.mark.parametrize(
    "source,wanted",
    [
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.q_proj.weight",
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
        ),
        (
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.blocks.0.attn.qkv.weight",
        ),
    ],
)
def test_packed_checkpoint_normalization(source, wanted):
    assert load("prepare_qwen35_packed").normalize(source) == wanted


def test_checked_loader_intercepts_fused_names_and_checks_storage():
    torch = pytest.importorskip("torch")
    loader = load("qwen35_packed_loader")
    names = []
    for layer in range(24):
        names += [
            f"model.layers.{layer}.mlp.{p}.weight"
            for p in ("gate_up_proj", "down_proj")
        ]
        if layer < 18:
            names += [
                f"model.layers.{layer}.linear_attn.{p}.weight"
                for p in ("in_proj_qkvz", "in_proj_ba", "out_proj")
            ]
        else:
            names += [
                f"model.layers.{layer}.{p}.weight" for p in ("qkv_proj", "o_proj")
            ]
    assert len(names) == 114
    parameters = {
        name: torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        for name in names
    }
    for parameter in parameters.values():
        parameter.weight_loader = lambda p, v: p.data.copy_(v)
    forwarded = []

    class Model:
        def named_parameters(self, **kwargs):
            return parameters.items()

    def original(self, weights):
        for name, value in weights:
            assert not loader.is_projection(name), "fused name reached substring loader"
            forwarded.append(name)
        return {"model.norm.weight"}

    model = Model()
    source = [(name, torch.full((2, 2), i + 1.0)) for i, name in enumerate(names)]
    result = loader.checked_loader(original)(
        model, iter(source + [("model.norm.weight", torch.ones(2))])
    )
    assert len(model._lf_packed_load_report) == 114
    assert set(names).issubset(result) and forwarded == ["model.norm.weight"]
    assert all(torch.equal(parameters[name], value) for name, value in source)
    with pytest.raises(ValueError, match="expected 114"):
        loader.checked_loader(original)(Model(), iter(source[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        loader.checked_loader(original)(Model(), iter([source[0], source[0]]))
    with pytest.raises(ValueError, match="missing from model"):
        loader.checked_loader(original)(
            Model(), iter([("model.layers.999.qkv_proj.weight", torch.ones(2, 2))])
        )
    parameters[names[0]].weight_loader = lambda p, v: None
    with pytest.raises(ValueError, match="differs from checkpoint"):
        loader.checked_loader(original)(
            Model(), iter([(names[0], torch.full((2, 2), -1.0))])
        )


def test_missing_logprob_is_rejected_before_any_speed_gate():
    record = dict(serial=[dict(adapter=0, length=17, tokens=[[None, 1]])], mixed=[])
    with pytest.raises(AssertionError, match="nonfinite logprob"):
        protocol.numerical_check(record, record)


@pytest.mark.parametrize("batch", [8, 32, 1024])
def test_nf4_linear_dispatch_keeps_small_batches_and_other_formats_native(batch):
    hook = load("qwen35_nf4_linear")
    state = N(
        shape=(8192, 1024), dtype="torch.bfloat16", quant_type="nf4", blocksize=64
    )
    weight = N(dtype="torch.uint8", bnb_quant_state={0: state})

    def activation(m, *, dtype="torch.bfloat16", k=1024, stride=1):
        return N(
            dtype=dtype,
            ndim=2,
            shape=(m, k),
            stride=lambda axis: stride,
            numel=lambda: m * k,
        )

    assert hook.select_state(weight, activation(batch)) is state
    for x in [
        activation(1),
        activation(4),
        activation(batch, dtype="torch.float16"),
        activation(batch, k=512),
        activation(batch, stride=2),
    ]:
        assert hook.select_state(weight, x) is None
    for replacement in [
        N(**dict(vars(state), quant_type="fp4")),
        N(**dict(vars(state), blocksize=128)),
        N(**dict(vars(state), shape=(4096, 1024))),
    ]:
        assert (
            hook.select_state(N(bnb_quant_state={0: replacement}), activation(batch))
            is None
        )
    assert (
        hook.select_state(N(bnb_quant_state={0: state, 1: state}), activation(batch))
        is None
    )


def test_nf4_scale_cache_detects_weight_scale_and_state_updates():
    torch = pytest.importorskip("torch")
    hook = load("qwen35_nf4_linear")
    weight = torch.zeros(16, dtype=torch.uint8)
    state = N(
        nested=True,
        absmax=torch.zeros(4, dtype=torch.uint8),
        code=torch.zeros(16),
        state2=N(absmax=torch.ones(1), code=torch.zeros(256)),
        offset=torch.zeros(()),
    )
    stamp = hook.scale_key(weight, state)
    for tensor in (
        weight,
        state.absmax,
        state.code,
        state.state2.absmax,
        state.state2.code,
        state.offset,
    ):
        tensor.add_(1)
        current = hook.scale_key(weight, state)
        assert current != stamp
        stamp = current
    assert hook.scale_key(weight, N(**vars(state))) != stamp
    with torch.inference_mode():
        immutable = torch.zeros(16, dtype=torch.uint8)
    assert hook.scale_key(immutable, state) == hook.scale_key(immutable, state)
