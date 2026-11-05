# SPDX-License-Identifier: Apache-2.0
"""Process-local SGLang0.5.22 FullCG metadata adapter for cold8x128 BF16 TP1."""

from qwen35_cold_prefill_metadata import ColdMetadata, cpu_list, eligible

stats = dict(buffers=0, capture_prepares=0, replay_prepares=0, scope_rejections=0)


def install():
    import torch
    from sglang import __version__

    if __version__ != "0.5.22":
        raise RuntimeError("cold-prefill prototype requires SGLang 0.5.22")
    import sglang.srt.models.qwen3_5 as model
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.triton_backend import (
        ForwardMetadata,
        TritonAttnBackend,
    )
    from sglang.srt.model_executor.cuda_graph_config import check_cuda_graph_backend
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    if getattr(TritonAttnBackend, "_lf_cold_installed", False):
        return
    old = TritonAttnBackend.init_forward_metadata_out_graph

    def metadata(self, batch, in_capture=False):
        if not (
            check_cuda_graph_backend("prefill", "full")
            and batch.forward_mode.name == "EXTEND"
        ):
            return old(self, batch, in_capture=in_capture)
        # This implementation follows eager EXTEND's prefix-empty metadata contract.
        # Cached-prefix, SWA, MLA, DCP, DLLM and deterministic modes are not handled.
        assert (
            self.dcp_size == 1
            and not self.use_mla
            and self.sliding_window_size in (None, -1, 0)
        )
        assert (
            not self.use_sliding_window_kv_pool
            and self.dllm_attention is None
            and not self.enable_deterministic
        )
        assert batch.batch_size == 8 and len(batch.input_ids) == 1024
        assert (
            cpu_list(batch.extend_prefix_lens_cpu) == [0] * 8
            and batch.encoder_lens is None
            and batch.spec_info is None
        )
        if not in_capture:
            assert eligible(batch), "ineligible replay escaped cold-prefill guard"
        assert not torch.cuda.is_current_stream_capturing(), (
            "refresh metadata outside graph capture"
        )
        state = getattr(self, "_lf_cold_metadata", None)
        if state is None:
            assert in_capture, "initialize cold graph buffers before replay"
            state = ColdMetadata(batch.extend_seq_lens[:8])
            self._lf_cold_metadata = state
            stats["buffers"] += 1
        state.refresh(batch.extend_seq_lens[:8], batch.extend_seq_lens_cpu)
        self._dense_one_shot_kv_indptr = None
        self.forward_metadata = ForwardMetadata(
            attn_logits=None,
            attn_lse=None,
            max_extend_len=128,
            num_kv_splits=None,
            kv_indptr=state.kv,
            kv_indices=state.indices,
            qo_indptr=state.qo,
            custom_mask=None,
            mask_indptr=None,
            window_kv_indptr=None,
            window_kv_indices=None,
            window_num_kv_splits=None,
            window_kv_offsets=None,
        )
        stats["capture_prepares" if in_capture else "replay_prepares"] += 1

    TritonAttnBackend.init_forward_metadata_out_graph = metadata
    # Capture the exact 8x128 shape. Native capture uses 1024+seven empty
    # sequences, whose convolution grids/chunk tables cannot represent 8x128.
    original_prepare = PrefillCudaGraphRunner.capture_prepare

    def prepare(self, num_tokens):
        batch, backend = original_prepare(self, num_tokens)
        if check_cuda_graph_backend("prefill", "full"):
            assert num_tokens == 1024 and batch.batch_size == 8
            for name in ("seq_lens", "orig_seq_lens", "extend_seq_lens"):
                getattr(batch, name).fill_(128)
            batch.extend_start_loc.copy_(
                torch.arange(8, device=batch.input_ids.device) * 128
            )
            batch.seq_lens_cpu.fill_(128)
            batch.extend_seq_lens_cpu = [128] * 8
            batch.extend_logprob_start_lens_cpu = [128] * 8
            batch.positions.copy_(
                torch.arange(128, device=batch.input_ids.device).repeat(8)
            )
            assert eligible(batch)
        return batch, backend

    PrefillCudaGraphRunner.capture_prepare = prepare
    old_gdn = GDNAttnBackend.init_forward_metadata_out_graph

    def gdn_metadata(self, batch, in_capture=False):
        if not (
            check_cuda_graph_backend("prefill", "full")
            and batch.forward_mode.name == "EXTEND"
        ):
            return old_gdn(self, batch, in_capture=in_capture)
        assert eligible(batch) and not torch.cuda.is_current_stream_capturing()
        # Reuse eager's physical state-slot translation; never borrow decode buffers.
        fresh = self._forward_metadata(batch)
        assert not fresh.has_mamba_track_mask
        assert fresh.replayssm_write_pos is None and fresh.replayssm_force_flush is None
        if in_capture:
            self._lf_cold_gdn_metadata = fresh
        else:
            saved = self._lf_cold_gdn_metadata
            saved.mamba_cache_indices.copy_(fresh.mamba_cache_indices)
            saved.query_start_loc.copy_(fresh.query_start_loc)
        self.forward_metadata = self._lf_cold_gdn_metadata
        assert self.forward_metadata.logical_num_tokens == 1024
        self._init_target_verify_qkv_routing(batch)
        stats["gdn_capture" if in_capture else "gdn_replay"] = (
            stats.get("gdn_capture" if in_capture else "gdn_replay", 0) + 1
        )

    GDNAttnBackend.init_forward_metadata_out_graph = gdn_metadata
    old_can = PrefillCudaGraphRunner.can_run_graph

    def can_run(self, batch):
        if check_cuda_graph_backend("prefill", "full") and not eligible(batch):
            stats["scope_rejections"] += 1
            return False
        return old_can(self, batch)

    PrefillCudaGraphRunner.can_run_graph = can_run
    model._lf_cold_stats = stats
    TritonAttnBackend._lf_cold_installed = True
