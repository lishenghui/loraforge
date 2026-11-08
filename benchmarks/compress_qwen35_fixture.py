# SPDX-License-Identifier: Apache-2.0
"""Compress a generated rank-32 Qwen3.5 fixture to physical BF16 rank 4.

Run on an allocated GPU. Uses the public FlashTSQR spectral truncation API;
synthetic fixtures measure execution cost, not trained-adapter quality.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    from loraforge.compression import truncate_lora

    source, target = args.source.resolve(), args.destination.resolve()
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["max_lora_rank"] == 32
    target.mkdir(parents=True)
    groups, originals, outputs, configs = defaultdict(list), {}, {}, {}
    for index, adapter in enumerate(manifest["adapters"]):
        path = Path(adapter["path"])
        file = path / "adapter_model.safetensors"
        assert hashlib.sha256(file.read_bytes()).hexdigest() == adapter["sha256"]
        cfg = json.loads((path / "adapter_config.json").read_text())
        assert cfg["r"] == cfg["lora_alpha"] == 32 and not cfg.get("use_rslora", False)
        assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern")
        weights = load_file(str(file))
        configs[index], originals[index], outputs[index] = cfg, weights, {}
        for key, a in weights.items():
            if not key.endswith(".lora_A.weight"):
                continue
            bkey = key.replace(".lora_A.weight", ".lora_B.weight")
            b = weights[bkey]
            assert (
                a.shape[0] == b.shape[1] == 32 and a.dtype == b.dtype == torch.bfloat16
            )
            groups[(a.shape[1], b.shape[0])].append((index, key, bkey))
    receipt = dict(
        state="running",
        scope="synthetic rank32->4 via FlashTSQR, then BF16 export; scaling stays1; no trained quality claim",
        groups=[],
        modules=[],
    )
    for (nin, nout), entries in groups.items():
        for start in range(0, len(entries), 32):
            batch = entries[start : start + 32]
            a = torch.stack([originals[i][ka] for i, ka, kb in batch]).cuda()
            b = torch.stack([originals[i][kb] for i, ka, kb in batch]).cuda()
            # b/a gates have only16 outputs. Add zero output rows solely for the
            # rank32 QR API, then remove them from the exported rank4 factor.
            padded_b = torch.nn.functional.pad(b, (0, 0, 0, max(0, 32 - nout)))
            result = truncate_lora(a, padded_b, keep=4)
            assert result.info.qr.backend == "FlashTSQR" and not result.info.padded
            ca, cb = result.a.to(torch.bfloat16), result.b[:, :nout].to(torch.bfloat16)
            aa, bb, ac, bc = a.double(), b.double(), ca.double(), cb.double()
            source_energy = ((aa @ aa.transpose(1, 2)) * (bb.transpose(1, 2) @ bb)).sum(
                (1, 2)
            )
            output_energy = ((ac @ ac.transpose(1, 2)) * (bc.transpose(1, 2) @ bc)).sum(
                (1, 2)
            )
            cross = ((aa @ ac.transpose(1, 2)) * (bb.transpose(1, 2) @ bc)).sum((1, 2))
            error = (source_energy + output_energy - 2 * cross).clamp_min(0)
            expected = result.singular_values[:, 4:].double().square().sum(1)
            assert torch.isfinite(ca).all() and torch.isfinite(cb).all()
            assert torch.all(error <= expected * 1.01 + source_energy * 1e-4), (
                "export deviates from spectral truncation"
            )
            receipt["groups"].append(
                dict(
                    input=nin,
                    output=nout,
                    count=len(batch),
                    zero_output_padding=max(0, 32 - nout),
                    backend=asdict(result.info),
                )
            )
            for j, (index, ka, kb) in enumerate(batch):
                outputs[index][ka], outputs[index][kb] = (
                    ca[j].cpu().contiguous(),
                    cb[j].cpu().contiguous(),
                )
                receipt["modules"].append(
                    dict(
                        adapter=index,
                        key=ka,
                        input=nin,
                        output=nout,
                        original_rank=32,
                        retained_rank=4,
                        source_energy=source_energy[j].item(),
                        retained_spectral_energy_fraction=(
                            1 - expected[j] / source_energy[j]
                        ).item(),
                        bf16_export_relative_frobenius_error=(
                            error[j] / source_energy[j]
                        )
                        .sqrt()
                        .item(),
                    )
                )
            print("COMPRESSED", nin, nout, start, len(batch), flush=True)
    adapters = []
    for index in originals:
        path = target / "adapters" / f"a{index}"
        path.mkdir(parents=True)
        assert set(outputs[index]) == set(originals[index])
        cfg = dict(configs[index], r=4, lora_alpha=4)
        (path / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        save_file(outputs[index], str(path / "adapter_model.safetensors"))
        report = dict(
            module_ranks={k: 4 for k in outputs[index] if k.endswith(".lora_A.weight")},
            source_sha256=manifest["adapters"][index]["sha256"],
            method="FlashTSQR spectral truncation 32->4 then BF16 export",
            scope=receipt["scope"],
        )
        (path / "compression.json").write_text(json.dumps(report, indent=2) + "\n")
        adapters.append(
            dict(
                name=f"a{index}",
                path=str(path),
                sha256=hashlib.sha256(
                    (path / "adapter_model.safetensors").read_bytes()
                ).hexdigest(),
            )
        )
    source_bytes = sum(
        v.numel() * v.element_size() for w in originals.values() for v in w.values()
    )
    output_bytes = sum(
        v.numel() * v.element_size() for w in outputs.values() for v in w.values()
    )
    assert output_bytes * 8 == source_bytes
    receipt.update(
        state="complete",
        source_adapters=manifest["adapters"],
        compressed_adapters=adapters,
        module_count=len(receipt["modules"]),
        source_tensor_bytes=source_bytes,
        compressed_tensor_bytes=output_bytes,
    )
    (target / "compression-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    fixture = dict(
        kind="compressed_compact",
        physical_rank=4,
        fixed_retained_rank=4,
        physical_parameters=output_bytes // 2,
        retained_parameters=output_bytes // 2,
        scaling=1.0,
    )
    manifest.update(
        adapters=adapters,
        max_lora_rank=4,
        fixture=fixture,
        retained_rank_histogram={"4": receipt["module_count"]},
        scope=receipt["scope"],
    )
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPRESSION_COMPLETE", source_bytes, output_bytes, flush=True)


if __name__ == "__main__":
    main()
