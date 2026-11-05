# SPDX-License-Identifier: Apache-2.0
"""Checked loading of the canonical 114-projection Qwen3.5 benchmark fixture."""

PROJECTIONS = frozenset(
    (
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
    )
)


def is_projection(name):
    return (
        name.startswith("model.layers.")
        and name.endswith(".weight")
        and name.split(".")[-2] in PROJECTIONS
    )


def checked_loader(original):
    """Fused names must not enter the upstream substring shard-name loop."""
    from functools import wraps

    @wraps(original)
    def load(self, weights):
        import torch

        parameters = dict(self.named_parameters(remove_duplicate=False))
        checked = {}

        def remaining():
            for name, value in weights:
                if not is_projection(name):
                    yield name, value
                    continue
                if name in checked:
                    raise ValueError(f"duplicate packed projection: {name}")
                if name not in parameters:
                    raise ValueError(f"packed projection missing from model: {name}")
                parameter = parameters[name]
                # Native TP1 full-checkpoint loading handles the packed uint8
                # offsets for NF4 and the native BF16 projection layouts.
                parameter.weight_loader(parameter, value)
                expected = value.to(device=parameter.device)
                if parameter.shape != expected.shape or not torch.equal(
                    parameter, expected
                ):
                    raise ValueError(
                        f"loaded projection differs from checkpoint: {name}"
                    )
                checked[name] = dict(
                    shape=list(parameter.shape), dtype=str(parameter.dtype), exact=True
                )

        result = original(self, remaining())
        if len(checked) != 114:
            raise ValueError(f"expected 114 checked projections, loaded {len(checked)}")
        self._lf_packed_load_report = checked
        return result if result is None else set(result) | checked.keys()

    return load


def install():
    from sglang import __version__
    from sglang.srt.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    if __version__ != "0.5.22":
        raise RuntimeError("packed fixture loader requires SGLang 0.5.22")
    for cls in (Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration):
        if cls.__dict__.get("_lf_checked_packed_loader", False):
            continue
        cls.load_weights = checked_loader(cls.load_weights)
        cls._lf_checked_packed_loader = True
