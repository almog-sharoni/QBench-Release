from __future__ import annotations

import torch
import torch.nn.functional as F


def chunk_tensor_by_context(tensor: torch.Tensor, chunk_size: int):
    """
    Split a tensor into chunk rows without crossing logical contexts.

    The context is every index except the last dimension. For example:
    [N, C, H, W] is treated as [N*C*H, W], so padding/chunking happens
    independently for each spatial row and never spills into the next row,
    channel, or batch element.
    """
    original_shape = tensor.shape
    if tensor.dim() <= 1:
        flat = tensor.contiguous().reshape(1, -1)
    else:
        flat = tensor.contiguous().reshape(-1, tensor.shape[-1])

    context_len = flat.shape[-1]
    pad_len = 0
    if context_len % chunk_size != 0:
        pad_len = chunk_size - (context_len % chunk_size)
        flat = F.pad(flat, (0, pad_len))

    num_chunks = flat.shape[-1] // chunk_size
    chunked = flat.reshape(flat.shape[0], num_chunks, chunk_size)
    return chunked, original_shape, pad_len


def unchunk_tensor_by_context(
    chunked: torch.Tensor,
    original_shape: torch.Size | tuple[int, ...],
    pad_len: int,
):
    flat = chunked.reshape(chunked.shape[0], -1)
    if pad_len > 0:
        flat = flat[:, :-pad_len]
    return flat.reshape(original_shape)


def count_context_chunks(tensor: torch.Tensor, chunk_size: int) -> int:
    chunked, _, _ = chunk_tensor_by_context(tensor, chunk_size)
    return int(chunked.shape[0] * chunked.shape[1])
