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
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def make_input_ids(token_ids, *, batch_size, tokens, device):
    """Distinct deterministic full sequences; batch one preserves old inputs."""
    import torch

    if not token_ids or batch_size < 1 or tokens < 2:
        raise ValueError(
            "nonempty tokens, positive batch and sequence length >=2 required"
        )
    base = torch.tensor(
        (token_ids * ((tokens + len(token_ids) - 1) // len(token_ids)))[:tokens],
        dtype=torch.long,
        device=device,
    )
    return torch.stack([base.roll(17 * row) for row in range(batch_size)])


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
            "native-activation",
            "native-consumer",
            "native-expand",
            "native-graph",
        ],
        required=True,
    )
    p.add_argument("--tokens", type=int, nargs="+", default=[128, 1024, 4096])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--loss-backend", choices=["native", "unsloth-ce"], default="native")
    p.add_argument("--loss-source", type=Path)
    p.add_argument("--audit-initial-step", action="store_true")
    p.add_argument(
        "--checkpointing", type=int, nargs="+", choices=[0, 1], default=[0, 1]
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--emulate-precision-casts", action="store_true")
    p.add_argument("--recompile-limit", type=int)
    p.add_argument(
        "--compile-backend",
        choices=["inductor", "cudagraphs", "eager", "aot_eager"],
        default="inductor",
    )
    p.add_argument("--check-input-change", action="store_true")
    p.add_argument("--implicit-causal", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or min(args.tokens) < 2:
        p.error("positive batch and sequence lengths >=2 required")
    if (args.loss_backend == "unsloth-ce") != (args.loss_source is not None):
        p.error("unsloth-ce requires --loss-source; native takes no external source")
    if args.audit_initial_step and args.backend != "peft":
        p.error("initial-step diagnostics are eager-only")
    if args.audit_initial_step and (
        len(args.tokens) != 1 or len(args.checkpointing) != 1
    ):
        p.error("initial-step diagnostics require one length/checkpointing setting")
    args.output.mkdir(exist_ok=False)
    import importlib.metadata

    import peft
    import qwen35_training_mlp as wrapper
    import torch
    import torch._inductor.config as inductor_config
    from qwen35_training_graph import sample_logits
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    if args.emulate_precision_casts:
        inductor_config.emulate_precision_casts = True
    if args.recompile_limit is not None:
        if args.recompile_limit < 1:
            raise ValueError("recompile limit must be positive")
        torch._dynamo.config.recompile_limit = args.recompile_limit
    torch.manual_seed(773)
    report = dict(
        state="loading",
        argv=sys.argv,
        backend=args.backend,
        cases=[],
        scope="Whole text-only BF16 Qwen3.5; all-linear rank4 LoRA FP32 masters; dropout0; zero_grad+forward+causal_loss+backward+AdamW; synthetic fixed text, not convergence",
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
        cuda_graph_replay=(
            None
            if args.backend == "native-graph"
            else (
                None
                if args.backend.startswith("compiled")
                and args.compile_backend == "cudagraphs"
                else False
            )
        ),
        manual_cuda_graph_replay=False,
        native_autograd_graph_requested=args.backend == "native-graph",
        implicit_causal_requested=args.implicit_causal,
        cuda_graph_backend_requested=(
            args.backend.startswith("compiled") and args.compile_backend == "cudagraphs"
        ),
        gradient_accumulation=1,
        batch=args.batch_size,
        loss_backend=args.loss_backend,
        audit_initial_step=args.audit_initial_step,
        rank=4,
        alpha=4,
        lr=0.0001,
        warmup=args.warmup,
        samples=args.samples,
        compiler_settings=dict(
            backend=args.compile_backend
            if args.backend.startswith("compiled")
            else None,
            emulate_precision_casts=inductor_config.emulate_precision_casts,
            recompile_limit=torch._dynamo.config.recompile_limit,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
        ),
    )
    report["source_sha256"] = {
        n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
        for n in [
            "benchmark_qwen35_training.py",
            "run_qwen35_training.py",
            "qwen35_training_mlp.py",
            "lora_swiglu_training.py",
            "lora_swiglu_kernels.py",
            "lora_native_training.py",
            "qwen35_training_graph.py",
            "qwen35_training_mask.py",
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
    config = model.config
    assert (
        len(model.model.layers) == config.num_hidden_layers == len(config.layer_types)
    )
    assert set(config.layer_types) == {"linear_attention", "full_attention"}
    report["model"] = dict(
        layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        gdn_layers=config.layer_types.count("linear_attention"),
        attention_layers=config.layer_types.count("full_attention"),
        base_parameters=sum(p.numel() for p in model.parameters()),
    )
    expected_adapters = {
        f"base_model.model.{name}.lora_{factor}.default.weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name != "lm_head"
        for factor in ("A", "B")
    }
    # One full actual load audit per worker, outside the timed steps.
    checked = 0
    with safe_open(
        args.checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as f:
        for name, t in model.state_dict().items():
            key = (
                "model.embed_tokens.weight"
                if name == "lm_head.weight"
                and name not in f.keys()
                and config.tie_word_embeddings
                else name
            )
            source = f.get_tensor(key)
            assert torch.equal(t.detach().cpu(), source.to(t.dtype)), name
            checked += 1
    report["checked_loaded_tensors"] = checked
    if config.tie_word_embeddings:
        assert (
            model.lm_head.weight.data_ptr()
            == model.model.embed_tokens.weight.data_ptr()
        )
    gdn = [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
    assert len(gdn) == report["model"]["gdn_layers"]
    assert all(m.chunk_gated_delta_rule.__module__.startswith("fla.") for m in gdn)
    assert all(m.causal_conv1d_fn is not None for m in gdn)
    report["gdn"] = dict(
        layers=len(gdn),
        chunk_function=gdn[0].chunk_gated_delta_rule.__module__,
        norm=type(gdn[0].norm).__name__,
        conv_function=gdn[0].causal_conv1d_fn.__module__,
    )
    model.config.use_cache = False
    if args.loss_backend == "unsloth-ce":
        from qwen35_training_loss import load_unsloth_ce, make_causal_loss

        fast_ce, report["loss_implementation"] = load_unsloth_ce(args.loss_source)
        model.loss_function = make_causal_loss(fast_ce)
        report["source_sha256"]["qwen35_training_loss.py"] = hashlib.sha256(
            Path(__file__).with_name("qwen35_training_loss.py").read_bytes()
        ).hexdigest()
    if args.implicit_causal:
        from qwen35_training_mask import install

        report["attention_policy"] = install(model)
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
    assert len(initial) == len(expected_adapters), (
        len(initial),
        len(expected_adapters),
    )
    assert set(initial) == expected_adapters, (
        "unexpected PEFT trainable parameter ownership"
    )
    report["adapter_tensors"] = len(initial)
    report["adapter_parameters"] = sum(p.numel() for p in initial.values())
    initial_hash = hashlib.sha256()
    for name, value in initial.items():
        initial_hash.update(name.encode())
        initial_hash.update(value.numpy().tobytes())
    report["initial_adapter_sha256"] = initial_hash.hexdigest()
    if args.backend in (
        "packed",
        "compiled-mlp",
        "fused",
        "compact",
        "native-activation",
        "native-consumer",
        "native-expand",
    ):
        backend = (
            args.backend if args.backend not in ("packed", "compiled-mlp") else "native"
        )
        count = wrapper.replace_mlps(model, backend)
        assert count == config.num_hidden_layers, count
        report["replaced_mlps"] = count
    if args.backend == "compiled-mlp":
        for module in model.modules():
            if isinstance(module, wrapper.TrainingMLP):
                module.forward = torch.compile(
                    module.forward,
                    backend=args.compile_backend,
                    fullgraph=True,
                    dynamic=False,
                )
    params = wrapper.adapter_parameters(model)
    assert params.keys() == initial.keys()
    assert all(p.dtype == torch.float32 for p in params.values())
    base = model.get_base_model()
    train_model = (
        torch.compile(
            model, backend=args.compile_backend, fullgraph=False, dynamic=False
        )
        if args.backend == "compiled-model"
        else model
    )
    report["state"] = "running"
    save()
    input_ids = json.loads((args.checkpoint / "input-tokens.json").read_text())

    def run(inputs):
        if args.backend == "native-graph":
            return train_model(inputs["input_ids"])
        if args.implicit_causal:
            from qwen35_training_mask import fixed_positions

            inputs = dict(inputs, position_ids=fixed_positions(inputs["input_ids"]))
        result = train_model(**inputs)
        if args.audit_initial_step:
            path = args.output / "initial-head-logits.safetensors"
            if not path.exists():
                positions = (
                    0,
                    inputs["input_ids"].shape[1] // 2,
                    inputs["input_ids"].shape[1] - 2,
                )
                save_file(
                    {
                        "logits": torch.stack(
                            [result.logits.detach()[:, p] for p in positions], dim=1
                        )
                        .cpu()
                        .contiguous(),
                        "targets": torch.stack(
                            [inputs["labels"][:, p + 1] for p in positions], dim=1
                        )
                        .cpu()
                        .contiguous(),
                    },
                    str(path),
                )
        return (
            result.loss,
            sample_logits(result.logits.detach()),
        )

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
            ids = make_input_ids(
                input_ids, batch_size=args.batch_size, tokens=tokens, device="cuda"
            )
            batch = dict(input_ids=ids, labels=ids, use_cache=False)
            if args.implicit_causal and model.config.pad_token_id is not None:
                assert not (ids == model.config.pad_token_id).any().item(), (
                    "padding unsupported"
                )
            case = dict(
                batch=args.batch_size,
                tokens=tokens,
                input_tokens_per_step=args.batch_size * tokens,
                supervised_tokens_per_step=args.batch_size * (tokens - 1),
                input_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                checkpointing=bool(checkpointing),
                losses=[],
                sample_losses=[],
            )
            optim = reset()
            graph_context = nullcontext(None)
            if args.backend == "native-graph":
                if checkpointing:
                    raise ValueError(
                        "native graph probe does not support checkpointing"
                    )
                from qwen35_training_graph import capture, graph_counters

                graph_context = graph_counters()
            with graph_context as graph_counts:
                if args.backend == "native-graph":
                    train_model = capture(
                        model, ids, implicit_causal=args.implicit_causal
                    )
                    case["native_graph_setup"] = dict(graph_counts)
                # Keep the common correctness/timing procedure below outside
                # this temporary capture-only instrumentation context.
            if graph_counts is not None:
                case["native_graph_api_counts"] = graph_counts
            # Save all parameter gradients and the third-step update for CPU
            # cross-method checks; these copies are outside the timing interval.
            for step in range(3):
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, audit_logits = run(batch)
                if step == 2:
                    logits = audit_logits.detach().float().cpu().contiguous()
                del audit_logits
                loss.backward()
                assert torch.isfinite(loss).item()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all().item()
                    for p in params.values()
                )
                case["losses"].append(loss.item())
                if step == 0 and args.audit_initial_step:
                    save_file(
                        {
                            k: p.grad.detach().cpu().contiguous()
                            for k, p in params.items()
                        },
                        str(args.output / "initial-grad.safetensors"),
                    )
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
                if step == 0 and args.audit_initial_step:
                    save_file(
                        {k: p.detach().cpu().contiguous() for k, p in params.items()},
                        str(args.output / "initial-updated.safetensors"),
                    )
                del loss
            save_file(
                {k: p.detach().cpu().contiguous() for k, p in params.items()},
                str(args.output / f"updated-gc{checkpointing}-t{tokens}.safetensors"),
            )
            if args.check_input_change:
                original_ids = ids.clone()
                changed = ids.roll(1, dims=-1)
                assert not torch.equal(changed, ids)
                case["changed_input_sha256"] = hashlib.sha256(
                    changed.cpu().numpy().tobytes()
                ).hexdigest()
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    changed_loss, changed_logits = run(
                        dict(input_ids=changed, labels=changed, use_cache=False)
                    )
                changed_loss.backward()
                assert torch.equal(ids, original_ids), (
                    "graph changed caller input storage"
                )
                case["changed_input_loss"] = changed_loss.item()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all().item()
                    for p in params.values()
                )
                save_file(
                    {k: p.grad.detach().cpu().contiguous() for k, p in params.items()},
                    str(
                        args.output
                        / f"changed-grad-gc{checkpointing}-t{tokens}.safetensors"
                    ),
                )
                save_file(
                    {"logits": changed_logits.detach().float().cpu().contiguous()},
                    str(
                        args.output
                        / f"changed-logits-gc{checkpointing}-t{tokens}.safetensors"
                    ),
                )
                del changed, changed_loss, changed_logits, original_ids
            del optim
            optim = reset()

            def step(optimizer=optim, inputs=batch):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, audit_logits = run(inputs)
                del audit_logits
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
            timed_replays_before = (
                graph_counts["replays"] if graph_counts is not None else 0
            )
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
                tokens_per_second=args.batch_size
                * tokens
                * 1000
                / statistics.median(wall),
                supervised_tokens_per_second=args.batch_size
                * (tokens - 1)
                * 1000
                / statistics.median(wall),
                sequences_per_second=args.batch_size * 1000 / statistics.median(wall),
                resident_allocated_bytes=resident,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            if graph_counts is not None:
                case["native_graph_timed_replays"] = (
                    graph_counts["replays"] - timed_replays_before
                )
                assert case["native_graph_timed_replays"] == 2 * args.samples
                assert graph_counts["captures"] == 2
                report["cuda_graph_replay"] = True
            report["cases"].append(case)
            report["compiler_counters"] = {
                k: dict(v) for k, v in torch._dynamo.utils.counters.items()
            }
            save()
            print(
                "STEP_CASE",
                args.backend,
                checkpointing,
                args.batch_size,
                tokens,
                round(case["median_wall_ms"], 3),
                round(case["peak_allocated_bytes"] / 2**30, 3),
                flush=True,
            )
            del optim, step, loss, grads, logits, ids, batch
            if args.backend == "native-graph":
                train_model = model
            for parameter in params.values():
                parameter.grad = None
            gc.collect()
            torch.cuda.empty_cache()
    report["state"] = "complete"
    save()
    print("MODEL_TRAINING_COMPLETE", args.backend, flush=True)


if __name__ == "__main__":
    main()
