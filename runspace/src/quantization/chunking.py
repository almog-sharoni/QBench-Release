from __future__ import annotations

import torch
import torch.nn.functional as F


def _get_greedy_structure(shape: torch.Size | tuple[int, ...], chunk_size: int):
    """
    Determine the flattened context structure for greedy chunking.
    """
    if not shape:
        return 1, 1, 1
    if len(shape) == 1:
        return 1, 1, shape[0]

    n = shape[0]
    dims = list(shape[1:])
    w = dims.pop()

    # Greedily flatten trailing dimensions into width if they fit in chunk_size
    while dims and w * dims[-1] <= chunk_size:
        w *= dims.pop()

    h_local = 1
    for d in dims:
        h_local *= d

    return n, h_local, w


def chunk_tensor_by_context(tensor: torch.Tensor, chunk_size: int):
    """
    Split a tensor into chunk rows, optionally merging context rows for efficiency.

    This implementation is batch-aware and greedily flattens trailing dimensions
    into the width if they fit within a single chunk (e.g. flattening HxW).
    """
    original_shape = tensor.shape
    n, h_local, w = _get_greedy_structure(original_shape, chunk_size)

    flat = tensor.contiguous().reshape(n, h_local, w)

    # Determine row merging factor based on context per batch element
    if h_local > 64:
        factor = 1
    elif h_local > 32:
        factor = 2
    elif h_local > 16:
        factor = 4
    else:
        factor = 8

    # Merge rows independently within each batch element
    if factor > 1:
        pad_h = (factor - (h_local % factor)) % factor
        if pad_h > 0:
            flat = F.pad(flat, (0, 0, 0, pad_h))
        h_padded = h_local + pad_h
        flat = flat.reshape(n, h_padded // factor, factor * w)

    # Flatten Batch and internal context rows into a single dimension for chunking
    flat = flat.reshape(-1, flat.shape[-1])

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
    """
    Reverse the batch-aware greedy chunking and merging process.
    """
    chunk_size = chunked.shape[-1]
    n, h_local, w = _get_greedy_structure(original_shape, chunk_size)

    # Re-derive factor and padded structure
    if h_local > 64:
        factor = 1
    elif h_local > 32:
        factor = 2
    elif h_local > 16:
        factor = 4
    else:
        factor = 8

    h_padded = ((h_local + factor - 1) // factor) * factor
    
    # Calculate the merged width and its padded version
    merged_width = factor * w
    merged_width_padded = merged_width + pad_len

    # Reshape back to merged batch-aware structure
    flat = chunked.reshape(n, h_padded // factor, merged_width_padded)

    # Remove width padding from the merged width
    if pad_len > 0:
        flat = flat[:, :, :-pad_len]

    # Reshape back to unmerged context rows [n, h_padded, w]
    flat = flat.reshape(n, h_padded, w)

    # Remove per-batch height padding
    if h_padded > h_local:
        flat = flat[:, :h_local, :]

    return flat.reshape(original_shape)


def count_context_chunks(tensor: torch.Tensor, chunk_size: int) -> int:
    chunked, _, _ = chunk_tensor_by_context(tensor, chunk_size)
    return int(chunked.shape[0] * chunked.shape[1])
