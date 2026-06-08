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
    # Conv activations are N, C, spatial...; keep each channel's spatial plane
    # contiguous instead of treating every spatial row as a mostly padded chunk.
    if len(shape) >= 4:
        n = int(shape[0])
        h_local = int(shape[1])
        w = 1
        for d in shape[2:]:
            w *= int(d)
        return n, h_local, w

    n = shape[0]
    dims = list(shape[1:])
    w = dims.pop()

    # Greedily flatten trailing dimensions into width if they fit in chunk_size
    while dims and w * dims[-1] <= chunk_size:
        w *= dims.pop()

    h_local = 1
    for d in dims:
        h_local *= d

    return int(n), int(h_local), int(w)


def _merge_factor(h_local: int) -> int:
    """How many context rows to merge into one chunk, by context count.

    Packs more rows together as the per-batch context count shrinks, capped at
    16 (the 128-PE array packs up to 16 small contexts into one 128-wide chunk):
        h_local > 64 -> 1, (32,64] -> 2, (16,32] -> 4, (8,16] -> 8, <=8 -> 16.
    """
    factor = 1
    if h_local <= 8:
        factor = 16
    elif h_local <= 16:
        factor = 8
    elif h_local <= 32:
        factor = 4
    elif h_local <= 64:
        factor = 2
    return min(factor, max(1, int(h_local)))


def _spatial_activation_meta(shape: torch.Size | tuple[int, ...], chunk_size: int):
    """Return whole-context or row-preserving layout metadata for N,C,spatial activations."""
    if len(shape) < 4:
        return None
    n = int(shape[0])
    c = int(shape[1])
    row_width = int(shape[-1])
    rows_per_context = 1
    for d in shape[2:-1]:
        rows_per_context *= int(d)
    context_width = rows_per_context * row_width
    if context_width <= chunk_size:
        contexts = n * c
        contexts_per_chunk = max(1, chunk_size // max(context_width, 1))
        pad_contexts = (contexts_per_chunk - (contexts % contexts_per_chunk)) % contexts_per_chunk
        context_groups = (contexts + pad_contexts) // contexts_per_chunk
        group_width = contexts_per_chunk * context_width
        pad_width = chunk_size - group_width
        return {
            "kind": "packed_spatial_contexts",
            "n": n,
            "c": c,
            "contexts": contexts,
            "context_width": context_width,
            "contexts_per_chunk": contexts_per_chunk,
            "pad_contexts": pad_contexts,
            "context_groups": context_groups,
            "group_width": group_width,
            "pad_width": pad_width,
            "chunk_size": chunk_size,
        }
    rows_per_chunk = max(1, chunk_size // max(row_width, 1))
    rows_per_chunk = min(rows_per_chunk, max(1, rows_per_context))
    pad_rows = (rows_per_chunk - (rows_per_context % rows_per_chunk)) % rows_per_chunk
    rows_padded = rows_per_context + pad_rows
    row_groups = rows_padded // rows_per_chunk
    group_width = rows_per_chunk * row_width
    pad_width = (chunk_size - (group_width % chunk_size)) % chunk_size
    chunks_per_group = (group_width + pad_width) // chunk_size
    return {
        "kind": "spatial_rows",
        "n": n,
        "c": c,
        "contexts": n * c,
        "rows_per_context": rows_per_context,
        "row_width": row_width,
        "rows_per_chunk": rows_per_chunk,
        "pad_rows": pad_rows,
        "row_groups": row_groups,
        "group_width": group_width,
        "pad_width": pad_width,
        "chunks_per_group": chunks_per_group,
        "chunk_size": chunk_size,
    }


def chunk_tensor_by_context(tensor: torch.Tensor, chunk_size: int):
    """
    Split a tensor into chunk rows, preserving activation contexts.

    N,C,spatial activations pack whole contexts together when they fit inside
    one chunk; larger contexts are split only on spatial-row boundaries.
    Other tensors use the existing batch-aware greedy trailing-dimension layout.
    """
    original_shape = tensor.shape
    spatial_meta = _spatial_activation_meta(original_shape, chunk_size)
    if spatial_meta is not None:
        if spatial_meta["kind"] == "packed_spatial_contexts":
            contexts = spatial_meta["contexts"]
            context_width = spatial_meta["context_width"]
            contexts_per_chunk = spatial_meta["contexts_per_chunk"]
            context_groups = spatial_meta["context_groups"]

            flat = tensor.contiguous().reshape(contexts, context_width)
            if spatial_meta["pad_contexts"] > 0:
                flat = F.pad(flat, (0, 0, 0, spatial_meta["pad_contexts"]))
            flat = flat.reshape(context_groups, contexts_per_chunk * context_width)
            if spatial_meta["pad_width"] > 0:
                flat = F.pad(flat, (0, spatial_meta["pad_width"]))
            chunked = flat.reshape(context_groups, 1, chunk_size)
            return chunked, original_shape, spatial_meta

        contexts = spatial_meta["contexts"]
        rows_per_context = spatial_meta["rows_per_context"]
        row_width = spatial_meta["row_width"]
        rows_per_chunk = spatial_meta["rows_per_chunk"]
        row_groups = spatial_meta["row_groups"]
        chunks_per_group = spatial_meta["chunks_per_group"]

        flat = tensor.contiguous().reshape(contexts, rows_per_context, row_width)
        if spatial_meta["pad_rows"] > 0:
            flat = F.pad(flat, (0, 0, 0, spatial_meta["pad_rows"]))
        flat = flat.reshape(contexts, row_groups, rows_per_chunk * row_width)
        if spatial_meta["pad_width"] > 0:
            flat = F.pad(flat, (0, spatial_meta["pad_width"]))
        chunked = flat.reshape(contexts, row_groups * chunks_per_group, chunk_size)
        return chunked, original_shape, spatial_meta

    n, h_local, w = _get_greedy_structure(original_shape, chunk_size)

    flat = tensor.contiguous().reshape(n, h_local, w)

    # Determine row merging factor based on context per batch element
    factor = _merge_factor(h_local)

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
    if isinstance(pad_len, dict) and pad_len.get("kind") == "packed_spatial_contexts":
        meta = pad_len
        flat = chunked.reshape(meta["context_groups"], chunk_size)
        if meta["pad_width"] > 0:
            flat = flat[:, :-meta["pad_width"]]
        flat = flat.reshape(meta["context_groups"] * meta["contexts_per_chunk"],
                            meta["context_width"])
        if meta["pad_contexts"] > 0:
            flat = flat[:-meta["pad_contexts"], :]
        return flat.reshape(original_shape)

    if isinstance(pad_len, dict) and pad_len.get("kind") == "spatial_rows":
        meta = pad_len
        contexts = meta["contexts"]
        row_groups = meta["row_groups"]
        chunks_per_group = meta["chunks_per_group"]
        rows_per_chunk = meta["rows_per_chunk"]
        row_width = meta["row_width"]

        flat = chunked.reshape(contexts, row_groups, chunks_per_group * chunk_size)
        if meta["pad_width"] > 0:
            flat = flat[:, :, :-meta["pad_width"]]
        flat = flat.reshape(contexts, row_groups * rows_per_chunk, row_width)
        if meta["pad_rows"] > 0:
            flat = flat[:, :-meta["pad_rows"], :]
        return flat.reshape(original_shape)

    n, h_local, w = _get_greedy_structure(original_shape, chunk_size)

    # Re-derive factor and padded structure (must match chunk_tensor_by_context)
    factor = _merge_factor(h_local)

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


# ---------------------------------------------------------------------------
# Weight chunker (128-PE weight rule)
#
# For a weight (c_out, c_in, f, f): fan_in = c_in*f*f, kernel = f*f (1 for Linear).
#   * fan_in <= chunk_size : pack floor(chunk_size/fan_in) WHOLE output channels
#                            into one chunk (one shared scale), never splitting a
#                            channel.  e.g. fan_in=27 -> 4 channels (108) per chunk.
#   * fan_in  > chunk_size and kernel <= chunk_size :
#                            within each output channel, pack floor(chunk_size/kernel)
#                            WHOLE kernel (f*f) blocks per chunk, never splitting an
#                            f*f block and never crossing output channels.
#   * kernel  > chunk_size : split each input-channel kernel into the minimum number
#                            of evenly sized kernel slices, padded to chunk_size.
#                            This handles patch/projection convolutions such as ViT
#                            [out, 3, 16, 16] with 128-wide chunks.
# Returns (chunked[n_ctx, n_chunk, chunk_size], original_shape, meta) with a matching
# unchunk_weight_by_context. Implemented with regular reshapes + zero padding.
# ---------------------------------------------------------------------------
def _weight_dims(shape):
    shape = tuple(int(s) for s in shape)
    c_out = shape[0] if shape else 1
    kernel = 1
    for d in shape[2:]:
        kernel *= int(d)
    numel = 1
    for d in shape:
        numel *= int(d)
    fan_in = numel // c_out if c_out else numel
    return c_out, kernel, fan_in


def chunk_weight_by_context(weight: torch.Tensor, chunk_size: int):
    original_shape = tuple(weight.shape)
    c_out, kernel, fan_in = _weight_dims(original_shape)
    cs = chunk_size

    if fan_in <= cs:
        k = max(1, cs // fan_in)                      # whole channels per chunk
        W = weight.contiguous().reshape(c_out, fan_in)
        pad_ch = (k - c_out % k) % k
        if pad_ch:
            W = F.pad(W, (0, 0, 0, pad_ch))
        n_groups = (c_out + pad_ch) // k
        W = W.reshape(n_groups, k * fan_in)
        pad_w = cs - k * fan_in
        if pad_w:
            W = F.pad(W, (0, pad_w))
        chunked = W.reshape(n_groups, 1, cs)
        meta = {"case": "A", "c_out": c_out, "fan_in": fan_in, "k": k,
                "real_w": k * fan_in, "cs": cs}
        return chunked, original_shape, meta

    if kernel > cs:
        c_in = fan_in // kernel
        n_kernel_chunks = -(-kernel // cs)
        split_w = -(-kernel // n_kernel_chunks)
        padded_kernel = n_kernel_chunks * split_w

        W = weight.contiguous().reshape(c_out, c_in, kernel)
        pad_kernel = padded_kernel - kernel
        if pad_kernel:
            W = F.pad(W, (0, pad_kernel))
        W = W.reshape(c_out, c_in, n_kernel_chunks, split_w)
        pad_w = cs - split_w
        if pad_w:
            W = F.pad(W, (0, pad_w))
        chunked = W.reshape(c_out, c_in * n_kernel_chunks, cs)
        meta = {
            "case": "C",
            "c_out": c_out,
            "c_in": c_in,
            "kernel": kernel,
            "n_kernel_chunks": n_kernel_chunks,
            "split_w": split_w,
            "padded_kernel": padded_kernel,
            "cs": cs,
        }
        return chunked, original_shape, meta

    b = max(1, cs // kernel)                           # whole f*f blocks per chunk
    c_in = fan_in // kernel
    W = weight.contiguous().reshape(c_out, c_in, kernel)
    pad_blk = (b - c_in % b) % b
    if pad_blk:
        W = F.pad(W, (0, 0, 0, pad_blk))
    n_chunk = (c_in + pad_blk) // b
    W = W.reshape(c_out, n_chunk, b * kernel)
    pad_w = cs - b * kernel
    if pad_w:
        W = F.pad(W, (0, pad_w))
    chunked = W.reshape(c_out, n_chunk, cs)
    meta = {"case": "B", "c_out": c_out, "c_in": c_in, "kernel": kernel, "b": b,
            "n_chunk": n_chunk, "real_w": b * kernel, "cs": cs}
    return chunked, original_shape, meta


def unchunk_weight_by_context(chunked: torch.Tensor, original_shape, meta) -> torch.Tensor:
    cs = meta["cs"]
    if meta["case"] == "A":
        c_out, fan_in, k, real_w = meta["c_out"], meta["fan_in"], meta["k"], meta["real_w"]
        n_groups = chunked.shape[0]
        W = chunked.reshape(n_groups, cs)[:, :real_w].reshape(n_groups * k, fan_in)
        return W[:c_out].reshape(original_shape)

    if meta["case"] == "C":
        c_out = meta["c_out"]
        c_in = meta["c_in"]
        kernel = meta["kernel"]
        n_kernel_chunks = meta["n_kernel_chunks"]
        split_w = meta["split_w"]
        padded_kernel = meta["padded_kernel"]
        W = chunked.reshape(c_out, c_in, n_kernel_chunks, cs)[:, :, :, :split_w]
        W = W.reshape(c_out, c_in, padded_kernel)[:, :, :kernel]
        return W.reshape(original_shape)

    c_out, c_in, kernel = meta["c_out"], meta["c_in"], meta["kernel"]
    n_chunk, real_w = meta["n_chunk"], meta["real_w"]
    b = meta["b"]
    W = chunked.reshape(c_out, n_chunk, cs)[:, :, :real_w].reshape(c_out, n_chunk * b, kernel)
    return W[:, :c_in, :].reshape(original_shape)


def count_weight_chunks(weight: torch.Tensor, chunk_size: int) -> int:
    c_out, kernel, fan_in = _weight_dims(tuple(weight.shape))
    if fan_in <= chunk_size:
        k = max(1, chunk_size // fan_in)
        return -(-c_out // k)
    if kernel > chunk_size:
        c_in = fan_in // kernel
        n_kernel_chunks = -(-kernel // chunk_size)
        return c_out * c_in * n_kernel_chunks
    b = max(1, chunk_size // kernel)
    c_in = fan_in // kernel
    return c_out * (-(-c_in // b))
