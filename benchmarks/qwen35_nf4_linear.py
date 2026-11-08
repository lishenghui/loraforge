# SPDX-License-Identifier: Apache-2.0
"""Legacy benchmark entrypoint; prepared dispatch lives in the installed package."""

import os

from loraforge.integrations._sglang_nf4_linear import scale_key, select_state  # noqa: F401


def install():
    from loraforge.integrations.sglang_experimental import install_qwen35_nf4

    if os.environ.get("LORAFORGE_BENCH_QWEN35_NF4") != "1":
        raise RuntimeError("enable the NF4 experiment explicitly")
    mode = (
        "byte-pairs"
        if os.environ.get("LORAFORGE_BENCH_QWEN35_NF4_PAIRS") == "1"
        else "cached-scales"
    )
    install_qwen35_nf4(mode=mode)
