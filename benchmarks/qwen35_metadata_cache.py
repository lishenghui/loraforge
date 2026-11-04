# SPDX-License-Identifier: Apache-2.0
"""Process-local consecutive-decode metadata reuse, pinned to dense TP1 tests.

Only reuse the most recently prepared static graph metadata, with the same
route/rank/scale/batch/capture buffer and no intervening prepare/reset/load.
Token, position, KV or recurrent-state tensors are never cached here.
"""


def install():
    import torch
    from sglang.srt.lora.mem_pool import LoRAMemoryPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner import get_is_capture_mode

    import loraforge.integrations._sglang_fleet as fleet

    old_factory = fleet.create_backend
    epoch = [0]

    def factory(*args, **kwargs):
        cls = old_factory(*args, **kwargs)
        original = cls.prepare_lora_batch
        reset = cls.reset_batch_state
        load = LoRAMemoryPool.load_lora_weight_to_buffer

        def loaded(*args, **kwargs):
            epoch[0] += 1  # Invalidate even a failed replacement.
            return load(*args, **kwargs)

        LoRAMemoryPool.load_lora_weight_to_buffer = loaded

        def prepare(
            self,
            forward_batch,
            weight_indices,
            lora_ranks,
            scalings,
            use_cuda_graph,
            use_prefill_cuda_graph=False,
        ):
            eligible = (
                getattr(self, "_hc_metadata_ready", False)
                and forward_batch.forward_mode.is_decode()
                and use_cuda_graph
                and not use_prefill_cuda_graph
                and not get_is_capture_mode()
                and not torch.cuda.is_current_stream_capturing()
                and isinstance(weight_indices, (list, tuple))
                and isinstance(lora_ranks, (list, tuple))
                and isinstance(scalings, (list, tuple))
                and all(type(v) is int for v in weight_indices)
                and all(type(v) is int for v in lora_ranks)
                and all(type(v) in (int, float) for v in scalings)
            )
            key = (
                (
                    epoch[0],
                    forward_batch.batch_size,
                    tuple(weight_indices),
                    tuple(lora_ranks),
                    tuple(scalings),
                    id(self.cuda_graph_batch_info),
                    self.max_chunk_size,
                )
                if eligible
                else None
            )
            info = getattr(self, "_hc_cached_info", None)
            if (
                eligible
                and key == getattr(self, "_hc_metadata_key", None)
                and info is not None
                and self.batch_info is info
                and info is self.cuda_graph_batch_info
            ):
                self._fleet_hits["metadata_cache_hit"] += 1
                return
            self._hc_metadata_key = None
            self._hc_cached_info = None
            self._fleet_hits["metadata_rebuild"] += 1
            result = original(
                self,
                forward_batch,
                weight_indices,
                lora_ranks,
                scalings,
                use_cuda_graph,
                use_prefill_cuda_graph,
            )
            if eligible and self.batch_info is self.cuda_graph_batch_info:
                self._hc_metadata_key = key
                self._hc_cached_info = self.batch_info
            return result

        def cleared(self, *args, **kwargs):
            self._hc_metadata_key = None
            self._hc_cached_info = None
            return reset(self, *args, **kwargs)

        cls.prepare_lora_batch = prepare
        cls.reset_batch_state = cleared
        return cls

    fleet.create_backend = factory
    old_init = ModelRunner.__init__

    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        manager = getattr(self, "lora_manager", None)
        if manager is not None and hasattr(manager.lora_backend, "_fleet_hits"):
            assert not manager.attn_dp_enabled, (
                "metadata prototype requires non-DP dense TP1"
            )
            manager.lora_backend._hc_metadata_ready = True

    ModelRunner.__init__ = init
