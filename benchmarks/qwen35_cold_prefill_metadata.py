# SPDX-License-Identifier: Apache-2.0
"""Cold8x128 attention metadata. No framework imports or device work at import."""


def cpu_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if (
        value is not None
        and getattr(getattr(value, "device", None), "type", None) == "cpu"
    ):
        return value.tolist()
    return None


def eligible(batch):
    mode = getattr(batch, "forward_mode", None)
    if (
        getattr(mode, "name", None) != "EXTEND"
        or batch.batch_size != 8
        or len(batch.input_ids) != 1024
    ):
        return False
    q = cpu_list(getattr(batch, "extend_seq_lens_cpu", None))
    p = cpu_list(getattr(batch, "extend_prefix_lens_cpu", None))
    return (
        q == [128] * 8
        and p == [0] * 8
        and getattr(batch, "encoder_lens", None) is None
        and getattr(batch, "spec_info", None) is None
        and not batch.contains_mm_inputs()
    )


class ColdMetadata:
    def __init__(self, query_lengths):
        import torch

        self.qo = torch.zeros(9, device=query_lengths.device, dtype=torch.int32)
        self.kv = torch.zeros_like(self.qo)
        # Prefix length is zero. A resident sentinel avoids a null kernel argument.
        self.indices = torch.zeros(1, device=query_lengths.device, dtype=torch.int64)
        self.pointers = (
            self.qo.data_ptr(),
            self.kv.data_ptr(),
            self.indices.data_ptr(),
        )
        self.refreshes = 0

    def refresh(self, lengths, cpu_lengths):
        import torch

        assert (
            lengths.ndim == 1
            and lengths.shape[0] == 8
            and lengths.dtype in (torch.int32, torch.int64)
        )
        values = cpu_list(cpu_lengths)
        assert (
            values is not None
            and len(values) == 8
            and all(type(v) is int and v >= 0 for v in values)
            and sum(values) == 1024
        )
        self.qo[0].zero_()
        torch.cumsum(lengths, dim=0, dtype=torch.int32, out=self.qo[1:])
        self.kv.zero_()
        assert self.pointers == (
            self.qo.data_ptr(),
            self.kv.data_ptr(),
            self.indices.data_ptr(),
        )
        self.refreshes += 1
