# SPDX-License-Identifier: Apache-2.0
"""CPU-only Chrome-trace accounting; keep unknowns, overlaps and phase gaps.

Kernel-duration shares are not end-to-end wall-time fractions. Fused LoRA
consumers include unpack/activation work and cannot be labeled pure LoRA cost.
"""

import argparse
import bisect
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def union_us(intervals):
    intervals = sorted((a, b) for a, b in intervals if b > a)
    if not intervals:
        return 0.0
    begin, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a > end:
            total += end - begin
            begin, end = a, b
        else:
            end = max(end, b)
    return total + end - begin


def category(name, attention_names=()):
    n = name.lower()
    if name in attention_names:
        return "attention"
    if any(
        s in n
        for s in (
            "gdn_ab_unpack",
            "row_act_kernel",
            "expand_act_kernel",
            "segmented_ab_kernel",
            "fused_ab_kernel",
            "wide_ab_kernel",
            "rank4_ab_kernel",
        )
    ):
        return "lora_fused_with_consumer"
    if "shrink" in n and any(s in n for s in ("lora", "sgmv", "shrink_kernel")):
        return "lora_A"
    if any(
        s in n
        for s in (
            "lora_expand",
            "lora_b",
            "sgmv_expand",
            "expand_tiled_kernel",
            "expand_kernel<",
        )
    ):
        return "lora_B"
    if "dequant" in n:
        return "base_dequant"
    if any(s in n for s in ("gemm_4bit", "gemv_4bit", "gemv_nf4")):
        return "base_fused_nf4"
    if any(
        s in n
        for s in (
            "gated_delta",
            "causal_conv",
            "recurrent",
            "chunk_fwd",
            "recompute_w_u",
            "conv1d",
            "gdn",
            "qkvzba_split",
            "chunk_local_cumsum",
        )
    ):
        return "GDN_state_conv"
    if any(s in n for s in ("attention", "flash_attn", "paged_attn", "attn_fwd")):
        return "attention"
    if any(s in n for s in ("gemm", "cijk_", "cij_", "tensile", "rocblas")):
        return "dense_gemm_including_head"
    if any(
        s in n
        for s in (
            "rmsnorm",
            "rms_norm",
            "layernorm",
            "layer_norm",
            "rotary",
            "rope",
            "silu",
            "gelu",
            "act_and_mul",
            "l2norm",
            "sigmoid_mul",
        )
    ):
        return "norm_position_activation"
    if "store_kvcache" in n:
        return "KV_cache_store"
    if any(
        s in n
        for s in ("sampling", "topk", "top_k", "argmax", "multinomial", "softmax")
    ):
        return "sampling"
    if any(
        s in n
        for s in (
            "copybuffer",
            "memcpy",
            "memset",
            "copy_kernel",
            "catarraybatchedcopy",
        )
    ):
        return "copy_memory"
    return "unknown_other"


def analyze(events, attention_names=()):
    kernels = [
        e
        for e in events
        if e.get("ph") == "X"
        and "kernel" in e.get("cat", "").lower()
        and e.get("dur", 0) > 0
    ]
    copies = [
        e
        for e in events
        if e.get("ph") == "X"
        and any(s in e.get("cat", "").lower() for s in ("gpu_memcpy", "gpu_memset"))
        and e.get("dur", 0) > 0
    ]
    spans = [
        e
        for e in events
        if e.get("ph") == "X"
        and not e.get("cat", "").startswith("gpu_")
        and e.get("name", "").startswith("LF/model_forward/")
    ]
    gpu_spans = sorted(
        [
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") == "gpu_user_annotation"
            and e.get("name", "").startswith("LF/model_forward/")
        ],
        key=lambda e: e["ts"],
    )
    gpu_starts = [e["ts"] for e in gpu_spans]
    by_thread = defaultdict(list)
    for e in spans:
        by_thread[e.get("tid")].append(e)
    starts = {}
    for tid, values in by_thread.items():
        values.sort(key=lambda e: e["ts"])
        starts[tid] = [e["ts"] for e in values]
    correlations = {}
    external = {}
    for e in events:
        cat = e.get("cat", "").lower()
        if e.get("ph") != "X" or "kernel" in cat or cat.startswith("gpu_"):
            continue
        args = e.get("args", {})
        if "External id" in args:
            external[str(args["External id"])] = e
        if "runtime" in cat and "correlation" in args:
            correlations[str(args["correlation"])] = e

    def phase(kernel):
        i = bisect.bisect_right(gpu_starts, kernel["ts"]) - 1
        if (
            i >= 0
            and kernel["ts"] + kernel["dur"]
            <= gpu_spans[i]["ts"] + gpu_spans[i].get("dur", 0) + 1
        ):
            return gpu_spans[i]["name"].split("/mode=")[1].split("/")[0]
        args = kernel.get("args", {})
        cpu = correlations.get(str(args.get("correlation"))) or external.get(
            str(args.get("External id"))
        )
        if cpu is None:
            return "unattributed"
        values = by_thread.get(cpu.get("tid"), [])
        i = bisect.bisect_right(starts.get(cpu.get("tid"), []), cpu["ts"]) - 1
        if i < 0 or cpu["ts"] > values[i]["ts"] + values[i].get("dur", 0):
            return "unattributed"
        return values[i]["name"].split("/mode=")[1].split("/")[0]

    groups = defaultdict(lambda: dict(count=0, sum_us=0.0))
    names = defaultdict(lambda: dict(count=0, sum_us=0.0))
    phases = defaultdict(lambda: dict(count=0, sum_us=0.0))
    for k in kernels:
        group = category(k["name"], attention_names)
        p = phase(k)
        duration = float(k["dur"])
        for v in (groups[group], names[(group, k["name"])], phases[(p, group)]):
            v["count"] += 1
            v["sum_us"] += duration
    total = sum(k["dur"] for k in kernels)
    for v in groups.values():
        v["fraction_of_summed_kernel_us"] = v["sum_us"] / total if total else None
    activity = kernels + copies
    span = max((k["ts"] + k["dur"] for k in activity), default=0) - min(
        (k["ts"] for k in activity), default=0
    )
    busy = union_us((k["ts"], k["ts"] + k["dur"]) for k in activity)
    histogram = Counter()
    for e in spans:
        m = re.search(r"mode=([^/]+)/bs=(\d+)/tokens=(\d+)", e["name"])
        if m:
            histogram[(m[1], int(m[2]), int(m[3]))] += 1
    cpu_labels = {
        label: sum(
            e.get("dur", 0)
            for e in events
            if e.get("ph") == "X"
            and not e.get("cat", "").startswith("gpu_")
            and e.get("name", "").startswith("LF/" + label)
        )
        for label in (
            "model_forward",
            "scheduler_run",
            "process_result",
            "lora_manager",
        )
    }
    return dict(
        kernel_count=len(kernels),
        summed_kernel_us=total,
        gpu_activity_union_us=busy,
        gpu_activity_span_us=span,
        gpu_activity_gap_us=max(0, span - busy),
        groups=dict(groups),
        phase_groups=[
            dict(phase=k[0], category=k[1], **v) for k, v in sorted(phases.items())
        ],
        top_kernels=[
            dict(category=k[0], name=k[1], **v)
            for k, v in sorted(names.items(), key=lambda kv: -kv[1]["sum_us"])[:60]
        ],
        model_forward_spans=len(spans),
        gpu_model_forward_spans=len(gpu_spans),
        batches=[
            dict(mode=k[0], batch=k[1], tokens=k[2], count=v)
            for k, v in sorted(histogram.items())
        ],
        cpu_inclusive_us=cpu_labels,
        scope="summed kernel durations; overlap removed only for activity union; CPU spans are inclusive; unknown phases retained; LoRA consumers include non-LoRA work; not serving latency fractions",
    )


def read_trace(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(read_trace(args.trace)["traceEvents"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("kernel_count", "summed_kernel_us", "groups", "batches")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
