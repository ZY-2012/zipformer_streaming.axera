"""Minimal icefall.utils stub for standalone streaming zipformer export.

Only the symbols used by the vendored model code and by checkpoint
unpickling (AttributeDict) are provided. No k2/lhotse dependency.
"""
import contextlib

import torch


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


@contextlib.contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    with torch.amp.autocast(device_type=device_type, **kwargs):
        yield


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0, pad_left: bool = False) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, int(lengths.max()))
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    if pad_left:
        return expanded_lengths < (max_len - lengths).unsqueeze(1)
    return expanded_lengths >= lengths.unsqueeze(-1)


def subsequent_chunk_mask(size: int, chunk_size: int, num_left_chunks: int = -1,
                          device: torch.device = torch.device("cpu")) -> torch.Tensor:
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        start = 0 if num_left_chunks < 0 else max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret
