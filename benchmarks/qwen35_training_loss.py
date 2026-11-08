# SPDX-License-Identifier: Apache-2.0
"""Isolate pinned Apache-2.0 Unsloth CE without importing its model patches.

The caller supplies the original upstream files; none are vendored. Selected
definitions execute unchanged. Only dependency bindings and an explicitly
reported optional HIP launch configuration are supplied by this experiment.
"""

import ast
import functools
import hashlib
from pathlib import Path

REVISION = "a0197ca266e14d22f66c0dcffc452fe1ce42bb0f"
SOURCES = {
    "cross_entropy_loss.py": "5f69f987fe160f83228b89da5119f5322225941e925dfcdf19f481f8d7b6f017",
    "utils.py": "a9e0e6618db083ca677b248fdd526a78ec3884de3a2e5df43d7bfa86991e7830",
    "rope_embedding.py": "6ac0a10f8c056319470e2c7702e1f25bbd128f96ec7363cca3d06f780c1b29ed",
}


def checked_sources(directory):
    result = {}
    for name, digest in SOURCES.items():
        path = Path(directory) / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"pinned Unsloth source hash mismatch: {name}")
        result[name] = (path, ast.parse(raw, filename=str(path)))
    return result


def load_unsloth_ce(directory, *, launch="hip"):
    sources = checked_sources(directory)
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    if torch.version.hip is None or not triton.__version__.startswith("3.6."):
        raise RuntimeError("isolated CE probe requires ROCm and Triton 3.6.x")
    if launch not in {"upstream", "hip"}:
        raise ValueError("launch must be upstream or hip")
    namespace = dict(
        __name__="loraforge_isolated_unsloth_ce",
        functools=functools,
        torch=torch,
        triton=triton,
        tl=tl,
        next_power_of_2=triton.next_power_of_2,
        is_hip=lambda: torch.version.hip is not None,
        triton_tanh=libdevice.tanh,
        triton_cast=tl.cast,
        torch_gpu_device=torch.cuda.device,
    )

    def execute(name, keep):
        path, tree = sources[name]
        selected = ast.Module(
            body=[node for node in tree.body if keep(node)], type_ignores=[]
        )
        exec(compile(selected, str(path), "exec"), namespace)

    execute(
        "utils.py",
        lambda n: (
            isinstance(n, ast.FunctionDef)
            and n.name in {"calculate_settings", "is_cdna"}
            or isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "MAX_FUSED_SIZE"
        ),
    )
    execute(
        "rope_embedding.py",
        lambda n: (
            isinstance(n, ast.FunctionDef)
            and n.name in {"_eviction_hints_ok", "_eviction_hints_ok_at"}
        ),
    )
    execute(
        "cross_entropy_loss.py",
        lambda n: (
            isinstance(n, (ast.Assign, ast.ClassDef))
            or isinstance(n, ast.FunctionDef)
            and n.name != "patch_loss_functions"
        ),
    )
    original = namespace["_chunked_forward_config"]
    upstream_config = original(torch.device("cuda"), 248320)
    if launch == "hip":
        # The pinned upstream defaults to (65536, 32) on gfx90a. Keep kernel
        # equations intact and explicitly choose a smaller workgroup here.
        namespace["_chunked_forward_config"] = lambda device, vocab: (16384, 8)
    return namespace["fast_cross_entropy_loss"], dict(
        repository="unslothai/unsloth",
        revision=REVISION,
        source_sha256=SOURCES,
        scope="isolated original CE definitions, not the complete Unsloth trainer",
        upstream_forward_config=list(upstream_config),
        selected_forward_config=list(
            namespace["_chunked_forward_config"](torch.device("cuda"), 248320)
        ),
        launch=launch,
        dependency_bindings="ROCm torch.cuda.device, tl.cast/libdevice, native is_hip",
        global_model_patches=False,
    )


def make_causal_loss(fast_ce):
    def loss(
        logits,
        labels,
        vocab_size,
        num_items_in_batch=None,
        ignore_index=-100,
        shift_labels=None,
        **kwargs,
    ):
        import torch

        if ignore_index != -100 or logits.shape[-1] != vocab_size:
            raise ValueError("fixed -100 ignore index and actual vocabulary required")
        if logits.ndim != 3 or labels.shape != logits.shape[:2]:
            raise ValueError("rectangular causal logits and labels required")
        allowed_model_options = {
            "output_attentions",
            "output_hidden_states",
            "return_dict",
        }
        if shift_labels is not None or set(kwargs) - allowed_model_options:
            raise ValueError(
                "custom loss options/shifted labels are outside this probe"
            )
        targets = torch.empty_like(labels)
        targets[..., :-1].copy_(labels[..., 1:])
        targets[..., -1].fill_(-100)
        return fast_ce(logits, targets, n_items=num_items_in_batch)

    return loss
