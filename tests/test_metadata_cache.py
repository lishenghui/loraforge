# SPDX-License-Identifier: Apache-2.0
"""Consecutive graph metadata reuse must invalidate on every owner transition."""


def test_metadata_reuse_invalidation_contracts(monkeypatch):
    import collections
    import importlib.util
    import sys
    import types
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in (
        "loraforge",
        "loraforge.integrations",
        "loraforge.integrations._sglang_fleet",
        "sglang",
        "sglang.srt",
        "sglang.srt.lora",
        "sglang.srt.lora.mem_pool",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.model_runner",
        "sglang.srt.model_executor.runner",
    ):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        if "." in name:
            p, k = name.rsplit(".", 1)
            setattr(sys.modules[p], k, mod)
    capture = [False]
    stream_capture = [False]
    calls = []

    class Backend:
        def __init__(self):
            self._fleet_hits = collections.Counter()
            self.cuda_graph_batch_info = object()
            self.batch_info = None
            self.max_chunk_size = 16
            self.fail = False

        def prepare_lora_batch(self, fb, wi, ranks, scales, graph, prefill=False):
            calls.append((fb.forward_mode.decode, graph, prefill))
            if self.fail:
                self.fail = False
                raise RuntimeError("prepare failure")
            self.batch_info = self.cuda_graph_batch_info if graph else object()

        def reset_batch_state(self):
            self.batch_info = None

    class Pool:
        def load_lora_weight_to_buffer(self, *args):
            return "loaded"

    class Runner:
        def __init__(self, b):
            self.lora_manager = types.SimpleNamespace(
                lora_backend=b, attn_dp_enabled=False
            )

    sys.modules["loraforge.integrations._sglang_fleet"].create_backend = (
        lambda *a, **k: Backend
    )
    sys.modules["sglang.srt.lora.mem_pool"].LoRAMemoryPool = Pool
    sys.modules["sglang.srt.model_executor.model_runner"].ModelRunner = Runner
    sys.modules["sglang.srt.model_executor.runner"].get_is_capture_mode = lambda: (
        capture[0]
    )
    spec = importlib.util.spec_from_file_location(
        "metadata_cache", root / "benchmarks/qwen35_metadata_cache.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install()
    b = sys.modules["loraforge.integrations._sglang_fleet"].create_backend()()
    mode = types.SimpleNamespace(decode=True)
    mode.is_decode = lambda: mode.decode
    fb = types.SimpleNamespace(forward_mode=mode, batch_size=2)
    wi = [1, 2]
    ranks = [0, 4, 4]
    scales = [0.0, 1.0, 1.0]

    def prepare(graph=True, prefill=False):
        b.prepare_lora_batch(fb, wi, ranks, scales, graph, prefill)

    checks = []

    def changed(label, action):
        prepare()
        before = len(calls)
        action()
        prepare()
        assert len(calls) == before + 1, label
        prepare()
        assert len(calls) == before + 1, label
        checks.append(label)

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: stream_capture[0]
    )
    prepare()
    prepare()
    assert len(calls) == 2
    Runner(b)
    prepare()
    n = len(calls)
    prepare()
    assert len(calls) == n
    changed("route", lambda: wi.reverse())
    changed("rank", lambda: ranks.__setitem__(1, 2))
    changed("scale", lambda: scales.__setitem__(1, 0.5))
    changed("batch size", lambda: setattr(fb, "batch_size", 3))
    changed("weight load epoch", lambda: Pool().load_lora_weight_to_buffer())
    changed("reset", lambda: b.reset_batch_state())
    changed(
        "graph buffer replacement",
        lambda: setattr(b, "cuda_graph_batch_info", object()),
    )
    changed("batch_info replacement", lambda: setattr(b, "batch_info", object()))
    for label, flag in [
        ("capture context", capture),
        ("stream capture", stream_capture),
    ]:
        before = len(calls)
        flag[0] = True
        prepare()
        prepare()
        assert len(calls) == before + 2
        flag[0] = False
        prepare()
        before = len(calls)
        prepare()
        assert len(calls) == before
        checks.append(label)
    mode.decode = False
    before = len(calls)
    prepare()
    prepare()
    assert len(calls) == before + 2
    mode.decode = True
    prepare()
    n = len(calls)
    prepare()
    assert len(calls) == n
    checks.append("prefill invalidates")
    before = len(calls)
    prepare(False)
    prepare(False)
    assert len(calls) == before + 2
    prepare()
    checks.append("non-graph fallback")
    before = len(calls)
    prepare(True, True)
    prepare(True, True)
    assert len(calls) == before + 2
    prepare()
    checks.append("prefill graph fallback")
    b.fail = True
    wi.reverse()
    try:
        prepare()
    except RuntimeError:
        pass
    else:
        raise AssertionError("failure not propagated")
    before = len(calls)
    prepare()
    assert len(calls) == before + 1
    checks.append("exception invalidates")
    assert len(checks) == 14
