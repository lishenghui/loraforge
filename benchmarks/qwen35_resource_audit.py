# SPDX-License-Identifier: Apache-2.0
"""Process-local stage memory and profiler annotations for Qwen3.5 studies.

No device/framework imports until install(). Allocator statistics are separate
from device-wide usage. Runtime peaks exclude profiler windows; graph-pool
buffers are retained. A 9-token untimed sentinel handles control-file commands.
"""

import json
import os
from pathlib import Path


def tensor_bytes(tensors):
    storages = {}
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        storages[(storage.data_ptr(), str(tensor.device))] = storage.nbytes()
    return int(sum(storages.values()))


def install():
    import torch
    from sglang import __version__
    from sglang.srt.model_executor.model_runner import ModelRunner

    assert __version__ == "0.5.22"
    if getattr(ModelRunner, "_lf_resource_audit", False):
        return
    ModelRunner._lf_resource_audit = True
    folder = Path(os.environ["HYBRID_AUDIT_DIR"])
    report = dict(
        pid=os.getpid(),
        stages=[],
        windows=[],
        scope="torch allocator bytes; device usage includes context; profiling excluded from runtime peaks",
    )
    handled = set()

    def components(self):
        model = getattr(self, "model", None)
        params = list(model.parameters()) if model is not None else []
        packed = (
            [getattr(layer, "_lf_packed", None) for _, layer in model.named_modules()]
            if model is not None
            else []
        )
        scales = [p._lf_nf4_scales[1] for p in params if hasattr(p, "_lf_nf4_scales")]
        manager = getattr(self, "lora_manager", None)
        lora = []
        if manager is not None:
            for group in (manager.memory_pool.A_buffer, manager.memory_pool.B_buffer):
                for layers in group.values():
                    lora.extend(t for t in layers if t is not None)
        req = getattr(self, "req_to_token_pool", None)
        state = getattr(getattr(req, "mamba_pool", None), "mamba_cache", None)
        mamba = [] if state is None else [state.temporal, *state.conv]
        kv = getattr(self, "token_to_kv_pool", None)
        return dict(
            parameter_storage_bytes=tensor_bytes(params),
            parameter_and_gdn_storage_bytes=tensor_bytes(params + packed),
            gdn_extra_storage_bytes=tensor_bytes(params + packed)
            - tensor_bytes(params),
            lora_pool_bytes=tensor_bytes(lora),
            gdn_packed_base_bytes=tensor_bytes(packed),
            nf4_scale_cache_bytes=tensor_bytes(scales),
            mamba_state_tensor_bytes=tensor_bytes(mamba),
            kv_tensor_bytes=int(sum(kv.get_kv_size_bytes()))
            if kv is not None and hasattr(kv, "get_kv_size_bytes")
            else None,
        )

    def snapshot(self, label, *, window=False):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        req = getattr(self, "req_to_token_pool", None)
        mamba = getattr(req, "mamba_pool", None)
        row = dict(
            label=label,
            components=components(self),
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            device_used=total - free,
            device_total=total,
            max_total_tokens=getattr(self, "max_total_num_tokens", None),
            max_running_requests=getattr(self, "max_running_requests", None),
            request_pool_size=getattr(req, "size", None),
            mamba_pool_size=getattr(mamba, "size", None),
        )
        report["windows" if window else "stages"].append(row)
        temp = folder / f"resource-audit-{os.getpid()}.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(folder / f"resource-audit-{os.getpid()}.json")

    def wrap(method, label):
        old = getattr(ModelRunner, method)

        def call(self, *args, **kwargs):
            if method == "load_model":
                torch.cuda.reset_peak_memory_stats()
                snapshot(self, "before_model_load")
            result = old(self, *args, **kwargs)
            snapshot(self, label)
            return result

        setattr(ModelRunner, method, call)

    for method, label in [
        ("load_model", "weights_loaded"),
        ("maybe_init_lora_manager", "lora_loaded"),
        ("alloc_memory_pool", "caches_allocated"),
        ("init_cuda_graphs", "graphs_captured"),
    ]:
        wrap(method, label)
    old_forward = ModelRunner._forward_raw

    def forward(self, batch, *args, **kwargs):
        sentinel = batch.forward_mode.is_extend() and len(batch.input_ids) == 9
        if sentinel:
            control = folder / "resource-control.json"
            if control.exists():
                cmd = json.loads(control.read_text())
                if cmd["id"] not in handled:
                    handled.add(cmd["id"])
                    if cmd["action"] == "reset":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    snapshot(self, cmd["label"], window=True)
        if torch.autograd._profiler_enabled():
            name = f"LF/model_forward/mode={batch.forward_mode.name}/bs={batch.batch_size}/tokens={len(batch.input_ids)}"
            with torch.profiler.record_function(name):
                return old_forward(self, batch, *args, **kwargs)
        return old_forward(self, batch, *args, **kwargs)

    ModelRunner._forward_raw = forward

    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.managers.scheduler import Scheduler

    def annotate(cls, method, label):
        previous = getattr(cls, method)

        def call(self, *args, **kwargs):
            if not torch.autograd._profiler_enabled():
                return previous(self, *args, **kwargs)
            with torch.profiler.record_function("LF/" + label):
                return previous(self, *args, **kwargs)

        setattr(cls, method, call)

    annotate(LoRAManager, "prepare_lora_batch", "lora_manager")
    annotate(Scheduler, "run_batch", "scheduler_run")
    annotate(Scheduler, "process_batch_result", "process_result")
