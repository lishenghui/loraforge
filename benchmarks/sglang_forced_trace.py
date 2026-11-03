# SPDX-License-Identifier: Apache-2.0
"""Diagnostic processor: record logits before forcing a shared continuation.

Used only by numerical probes, never by quality or timing requests.
The caller controls the local artifact directory; no remote service is exposed.
"""

import re
from pathlib import Path

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class ForcedTrace(CustomLogitProcessor):
    def __init__(self):
        self.steps = {}

    def __call__(self, logits, custom_param_list=None):
        if logits.shape[0] != 1 or len(custom_param_list or []) != 1:
            raise ValueError("numerical trace requires a single request")
        params = custom_param_list[0]
        case = params["trace_case"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case):
            raise ValueError("invalid trace case")
        root = Path(params["trace_root"])
        mode = params["trace_mode"]
        if mode not in ("native", "fleet"):
            raise ValueError("invalid trace mode")
        step = self.steps.get(case, 0)
        self.steps[case] = step + 1
        raw = logits[0].detach().float().cpu()
        if not torch.isfinite(raw).all():
            raise ValueError("nonfinite pre-intervention logits")
        native_path = root / "native" / case / f"{step:03d}.pt"
        if mode == "native":
            forced = int(raw.argmax())
        else:
            native = torch.load(native_path, map_location="cpu", weights_only=True)
            forced = native["forced_token"]
        path = root / mode / case / f"{step:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError("trace artifact already exists; use a fresh attempt")
        torch.save(dict(logits=raw, forced_token=forced, step=step), path)
        logits.fill_(-float("inf"))
        logits[0, forced] = 0
        return logits
