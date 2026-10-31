# SPDX-License-Identifier: Apache-2.0
"""Supported standard-LoRA configuration and PEFT pattern semantics."""

import math
import re

_FLAGS = {"fan_in_fan_out", "use_rslora", "use_dora", "use_qalora", "lora_bias"}
_EMPTY = {
    "modules_to_save",
    "layer_replication",
    "target_parameters",
    "trainable_token_indices",
    "megatron_config",
    "loftq_config",
    "eva_config",
    "corda_config",
    "arrow_config",
    "alora_invocation_tokens",
}
_STANDARD = {
    "peft_type",
    "task_type",
    "base_model_name_or_path",
    "revision",
    "auto_mapping",
    "inference_mode",
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "target_modules",
    "exclude_modules",
    "layers_to_transform",
    "layers_pattern",
    "rank_pattern",
    "alpha_pattern",
    "init_lora_weights",
    "megatron_core",
    "qalora_group_size",
    "peft_version",
    "ensure_weight_tying",
    "runtime_config",
}


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _config(config: dict) -> None:
    if config.get("peft_type") != "LORA":
        raise ValueError("only PEFT LORA adapters are supported")
    unknown = config.keys() - (_STANDARD | _FLAGS | _EMPTY)
    if unknown:
        raise ValueError(f"unsupported configuration fields: {sorted(unknown)}")
    for field in _FLAGS:
        if config.get(field, False) is not False:
            raise ValueError(f"unsupported LoRA configuration: {field}")
    for field in _EMPTY:
        if config.get(field) not in (None, [], {}):
            raise ValueError(f"unsupported LoRA configuration: {field}")
    if config.get("ensure_weight_tying", False) is not False:
        raise ValueError("ensure_weight_tying is not supported")
    if config.get("runtime_config") not in (None, {}, {"ephemeral_gpu_offload": False}):
        raise ValueError("non-default runtime_config is not supported")
    if config.get("bias", "none") != "none":
        raise ValueError("bias weights are not supported")
    if not isinstance(config.get("init_lora_weights", True), bool):
        raise ValueError("special LoRA initialization schemes are not supported")
    _positive_int(config.get("r"), "r")
    _finite(config.get("lora_alpha"), "lora_alpha")
    if not 0 <= _finite(config.get("lora_dropout", 0), "lora_dropout") <= 1:
        raise ValueError("lora_dropout must be between zero and one")
    for field in ("rank_pattern", "alpha_pattern"):
        patterns = config.get(field, {})
        if not isinstance(patterns, dict):
            raise ValueError(f"{field} must be an object")
        for key, value in patterns.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} keys must be nonempty strings")
            re.compile(rf"(?:.*\.)?(?:{key})$")
            (_positive_int if field == "rank_pattern" else _finite)(value, field)
    targets = config.get("target_modules")
    if not isinstance(targets, (str, list)) or not targets:
        raise ValueError("explicit target_modules are required")
    if isinstance(targets, list) and not all(isinstance(x, str) and x for x in targets):
        raise ValueError("target_modules must contain nonempty strings")


def _pattern_value(patterns: dict, name: str, default):
    # Preserve PEFT's ordered, suffix-regex matching, including anchored keys.
    for pattern, value in patterns.items():
        if re.fullmatch(rf"(?:.*\.)?(?:{pattern})", name):
            return value
    return patterns.get(name, default)


def _matches(targets, name: str) -> bool:
    if isinstance(targets, str):
        return re.fullmatch(targets, name) is not None
    return any(
        name == target or name.endswith("." + target) for target in targets or ()
    )
