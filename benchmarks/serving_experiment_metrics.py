# SPDX-License-Identifier: Apache-2.0
"""CPU-only summaries; numerical parity and task quality are separate contracts."""

import math
import re


def answer_digit(text):
    matches = re.findall(r"(?:^|\n)\s*Answer:\s*([0-3])\s*[.]?\s*(?=\n|$)", text)
    return matches[-1] if matches else None


def wilson(successes, total, z=2.241402727604947):
    """97.5% marginal interval; two intervals give conservative 95% pairing bounds."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid count")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_quality(native, fleet):
    if not native or native.keys() != fleet.keys():
        raise ValueError("paired case sets differ or are empty")
    for key in native:
        if native[key]["answer"] != fleet[key]["answer"]:
            raise ValueError("paired labels differ")
    n = len(native)
    good_native = {k: v["prediction"] == v["answer"] for k, v in native.items()}
    good_fleet = {k: v["prediction"] == v["answer"] for k, v in fleet.items()}
    lost = sum(good_native[k] and not good_fleet[k] for k in native)
    gained = sum(good_fleet[k] and not good_native[k] for k in native)
    lo_loss, hi_loss = wilson(lost, n)
    lo_gain, hi_gain = wilson(gained, n)
    return dict(
        cases=n,
        native_accuracy=sum(good_native.values()) / n,
        fleet_accuracy=sum(good_fleet.values()) / n,
        lost=lost,
        gained=gained,
        degradation=(lost - gained) / n,
        conservative_95pct_degradation_interval=[lo_loss - hi_gain, hi_loss - lo_gain],
        native_invalid=sum(v["prediction"] is None for v in native.values()),
        fleet_invalid=sum(v["prediction"] is None for v in fleet.values()),
        scope="pilot; does not establish general quality equivalence",
    )
