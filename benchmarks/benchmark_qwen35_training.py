# SPDX-License-Identifier: Apache-2.0
"""Allocated-GPU whole Qwen3.5 text-model LoRA training step measurements.

No inference cache or manual CUDA-graph replay. Timed steps include zero_grad,
the complete causal-LM forward/loss/backward and AdamW. Data stays on device.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--backend",
        choices=[
            "peft",
            "packed",
            "compiled-mlp",
            "fused",
            "compact",
            "compiled-model",
        ],
        required=True,
    )
    p.add_argument("--tokens", type=int, nargs="+", default=[128, 1024, 4096])
    p.add_argument(
        "--checkpointing", type=int, nargs="+", choices=[0, 1], default=[0, 1]
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--samples", type=int, default=5)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    import importlib.metadata

    import peft
    import qwen35_training_mlp as wrapper
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(773)
    report = dict(
        state="loading",
        backend=args.backend,
        cases=[],
        scope="Whole 24-layer text-only BF16 Qwen3.5-0.8B; all-linear rank4 LoRA FP32 masters; dropout0; zero_grad+forward+causal_loss+backward+AdamW; synthetic fixed text, not convergence",
        versions={
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "transformers",
                "peft",
                "fla-core",
                "flash-linear-attention",
            ]
        },
        device=torch.cuda.get_device_name(),
        cuda_graph_replay=False,
        gradient_accumulation=1,
        batch=1,
        rank=4,
        alpha=4,
        lr=0.0001,
        warmup=args.warmup,
        samples=args.samples,
    )
    report["source_sha256"] = {
        n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
        for n in [
            "benchmark_qwen35_training.py",
            "qwen35_training_mlp.py",
            "lora_swiglu_training.py",
            "lora_swiglu_kernels.py",
        ]
    }
    report["input_receipt"] = json.loads(
        (args.checkpoint / "preparation.json").read_text()
    )

    def save() -> None:
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )

    save()
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    report["loading"] = loading
    assert not any(
        loading.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), loading
    assert len(model.model.layers) == 24
    assert model.config.layer_types.count("linear_attention") == 18
    assert model.config.layer_types.count("full_attention") == 6
    # One full actual load audit per worker, outside the timed steps.
    checked = 0
    with safe_open(
        args.checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as f:
        for name, t in model.state_dict().items():
            key = "model.embed_tokens.weight" if name == "lm_head.weight" else name
            source = f.get_tensor(key)
            assert torch.equal(t.detach().cpu(), source.to(t.dtype)), name
            checked += 1
    report["checked_loaded_tensors"] = checked
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    gdn = [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
    assert len(gdn) == 18
    assert all(m.chunk_gated_delta_rule.__module__.startswith("fla.") for m in gdn)
    assert all(m.causal_conv1d_fn is not None for m in gdn)
    report["gdn"] = dict(
        layers=len(gdn),
        chunk_function=gdn[0].chunk_gated_delta_rule.__module__,
        norm=type(gdn[0].norm).__name__,
        conv_function=gdn[0].causal_conv1d_fn.__module__,
    )
    model.config.use_cache = False
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0,
            target_modules="all-linear",
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    initial = {
        k: v.detach().cpu().clone()
        for k, v in wrapper.adapter_parameters(model).items()
    }
    assert len(initial) == 372, len(initial)
    report["adapter_tensors"] = len(initial)
    report["adapter_parameters"] = sum(p.numel() for p in initial.values())
    initial_hash = hashlib.sha256()
    for name, value in initial.items():
        initial_hash.update(name.encode())
        initial_hash.update(value.numpy().tobytes())
    report["initial_adapter_sha256"] = initial_hash.hexdigest()
    if args.backend in ("packed", "compiled-mlp", "fused", "compact"):
        backend = args.backend if args.backend in ("fused", "compact") else "native"
        count = wrapper.replace_mlps(model, backend)
        assert count == 24, count
        report["replaced_mlps"] = count
    if args.backend == "compiled-mlp":
        for module in model.modules():
            if isinstance(module, wrapper.TrainingMLP):
                module.forward = torch.compile(
                    module.forward, fullgraph=True, dynamic=False
                )
    params = wrapper.adapter_parameters(model)
    assert params.keys() == initial.keys()
    assert all(p.dtype == torch.float32 for p in params.values())
    base = model.get_base_model()
    train_model = (
        torch.compile(model, fullgraph=False, dynamic=False)
        if args.backend == "compiled-model"
        else model
    )
    report["state"] = "running"
    save()
    input_ids = json.loads((args.checkpoint / "input-tokens.json").read_text())

    def reset():
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(initial[name])
                p.grad = None
        return torch.optim.AdamW(
            list(params.values()),
            lr=0.0001,
            foreach=True,
        )

    for checkpointing in args.checkpointing:
        if checkpointing:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            base.gradient_checkpointing_disable()
        model.train()
        assert base.is_gradient_checkpointing == bool(checkpointing)
        for tokens in args.tokens:
            ids = torch.tensor(
                (input_ids * ((tokens + len(input_ids) - 1) // len(input_ids)))[
                    :tokens
                ],
                dtype=torch.long,
                device="cuda",
            ).reshape(1, tokens)
            batch = dict(input_ids=ids, labels=ids, use_cache=False)
            case = dict(
                tokens=tokens,
                checkpointing=bool(checkpointing),
                losses=[],
                sample_losses=[],
            )
            optim = reset()
            # Save all parameter gradients and the third-step update for CPU
            # cross-method checks; these copies are outside the timing interval.
            for step in range(3):
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = train_model(**batch)
                    loss = output.loss
                if step == 2:
                    positions = [0, tokens // 2, tokens - 2]
                    logits = (
                        output.logits.detach()[0, positions, ::257]
                        .float()
                        .cpu()
                        .contiguous()
                    )
                del output
                loss.backward()
                assert torch.isfinite(loss).item()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all().item()
                    for p in params.values()
                )
                case["losses"].append(loss.item())
                if step == 2:
                    grads = {
                        k: p.grad.detach().cpu().contiguous() for k, p in params.items()
                    }
                    save_file(
                        grads,
                        str(
                            args.output
                            / f"grad-gc{checkpointing}-t{tokens}.safetensors"
                        ),
                    )
                    save_file(
                        {"logits": logits},
                        str(
                            args.output
                            / f"logits-gc{checkpointing}-t{tokens}.safetensors"
                        ),
                    )
                optim.step()
                del loss
            save_file(
                {k: p.detach().cpu().contiguous() for k, p in params.items()},
                str(args.output / f"updated-gc{checkpointing}-t{tokens}.safetensors"),
            )
            del optim
            optim = reset()

            def step(optimizer=optim, inputs=batch):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = train_model(**inputs).loss
                loss.backward()
                optimizer.step()
                return loss.detach()

            for _ in range(args.warmup):
                loss = step()
            torch.cuda.synchronize()
            gc.collect()
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            wall, gpu = [], []
            for _ in range(args.samples):
                torch.cuda.synchronize()
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start = time.perf_counter()
                a.record()
                loss = step()
                b.record()
                torch.cuda.synchronize()
                wall.append((time.perf_counter() - start) * 1000)
                gpu.append(a.elapsed_time(b))
                case["sample_losses"].append(loss.item())
            assert all(torch.isfinite(p.grad).all().item() for p in params.values())
            assert all(
                torch.isfinite(torch.tensor(v)).item() for v in case["sample_losses"]
            )
            case.update(
                wall_ms=wall,
                gpu_ms=gpu,
                median_wall_ms=statistics.median(wall),
                median_gpu_ms=statistics.median(gpu),
                tokens_per_second=tokens * 1000 / statistics.median(wall),
                resident_allocated_bytes=resident,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            report["cases"].append(case)
            report["compiler_counters"] = {
                k: dict(v) for k, v in torch._dynamo.utils.counters.items()
            }
            save()
            print(
                "STEP_CASE",
                args.backend,
                checkpointing,
                tokens,
                round(case["median_wall_ms"], 3),
                round(case["peak_allocated_bytes"] / 2**30, 3),
                flush=True,
            )
            del optim, step, loss, grads, logits, ids, batch
            for parameter in params.values():
                parameter.grad = None
            gc.collect()
            torch.cuda.empty_cache()
    report["state"] = "complete"
    save()
    print("MODEL_TRAINING_COMPLETE", args.backend, flush=True)


if __name__ == "__main__":
    main()
