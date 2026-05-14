import torch
import torch.nn.functional as F

def chunk_tensor_by_context(tensor: torch.Tensor, chunk_size: int):
    original_shape = tensor.shape
    
    # Identify batch, context_per_batch, and width
    if tensor.dim() <= 1:
        n = 1
        h_local = 1
        w = original_shape[0] if original_shape else 1
        flat = tensor.contiguous().reshape(1, 1, w)
    else:
        n = original_shape[0]
        w = original_shape[-1]
        h_local = 1
        for d in original_shape[1:-1]:
            h_local *= d
        flat = tensor.contiguous().reshape(n, h_local, w)

    # Determine factor based on h_local (simulate inference of 1 element)
    if h_local > 64: factor = 1
    elif h_local > 32: factor = 2
    elif h_local > 16: factor = 4
    else: factor = 8

    # Merge rows within each batch element
    if factor > 1:
        pad_h = (factor - (h_local % factor)) % factor
        if pad_h > 0:
            flat = F.pad(flat, (0, 0, 0, pad_h))
        h_padded = h_local + pad_h
        # Reshape to [n, h_padded // factor, factor * w]
        flat = flat.reshape(n, h_padded // factor, factor * w)

    # Now flatten N and H_prime into a single dimension for chunking
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
    if len(original_shape) <= 1:
        n, h_local = 1, 1
        w = original_shape[0] if original_shape else 1
    else:
        n = original_shape[0]
        w = original_shape[-1]
        h_local = 1
        for d in original_shape[1:-1]:
            h_local *= d

    # Determine factor based on h_local
    if h_local > 64: factor = 1
    elif h_local > 32: factor = 2
    elif h_local > 16: factor = 4
    else: factor = 8
    
    h_padded = ((h_local + factor - 1) // factor) * factor
    w_padded = w + pad_len
    
    # Back to merged rows: (N * h_padded // factor, factor * w_padded)
    flat = chunked.reshape(chunked.shape[0], -1)
    
    # Reshape to (N, h_padded // factor, factor * w_padded)
    flat = flat.reshape(n, h_padded // factor, -1)
    
    # Remove width padding
    if pad_len > 0:
        flat = flat[:, :, :-pad_len]
        
    # Reshape back to (N, h_padded, w)
    flat = flat.reshape(n, h_padded, w)
    
    # Remove height padding
    flat = flat[:, :h_local, :]
    
    return flat.reshape(original_shape)

# Test cases
for shape in [(2, 3, 10), (1, 10), (10,), (4, 100, 16)]:
    t = torch.randn(shape)
    c, s, p = chunk_tensor_by_context(t, 16)
    u = unchunk_tensor_by_context(c, s, p)
    
    print(f"Shape {shape} -> chunked: {c.shape} -> match: {torch.allclose(t, u)}")
    
    # Verify no merging across batch for (2, 3, 10)
    if shape == (2, 3, 10):
        # h_local = 3, factor = 8.
        # h_padded = 8.
        # h_prime = 8/8 = 1.
        # total rows = 2 * 1 = 2.
        print(f"  Rows (expect 2): {c.shape[0]}")
