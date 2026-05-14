import torch
import torch.nn.functional as F

def get_structure(shape, chunk_size):
    if not shape: return 1, 1, 1
    if len(shape) == 1: return 1, 1, shape[0]
    
    n = shape[0]
    dims = list(shape[1:])
    w = dims.pop()
    while dims and w * dims[-1] <= chunk_size:
        w *= dims.pop()
    
    h_local = 1
    for d in dims:
        h_local *= d
    return n, h_local, w

def chunk_tensor_by_context(tensor: torch.Tensor, chunk_size: int):
    original_shape = tensor.shape
    n, h_local, w = get_structure(original_shape, chunk_size)
    
    flat = tensor.contiguous().reshape(n, h_local, w)

    # Determine factor based on h_local
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
        flat = flat.reshape(n, h_padded // factor, factor * w)

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
    chunk_size: int # Added for greedy logic re-run
):
    n, h_local, w = get_structure(original_shape, chunk_size)

    if h_local > 64: factor = 1
    elif h_local > 32: factor = 2
    elif h_local > 16: factor = 4
    else: factor = 8
    
    h_padded = ((h_local + factor - 1) // factor) * factor
    w_padded = w + pad_len
    
    flat = chunked.reshape(n, h_padded // factor, -1)
    if pad_len > 0:
        flat = flat[:, :, :-pad_len]
    flat = flat.reshape(n, h_padded, w)
    flat = flat[:, :h_local, :]
    return flat.reshape(original_shape)

# Test cases
chunk_size = 128
test_cases = [
    (2, 64, 7, 7),  # h_local=64, w=49
    (4, 20, 5),     # h_local=1, w=100
    (1, 1024),      # h_local=1, w=1024
    (2, 32, 10, 10) # 10*10=100 <= 128. h_local=32, w=100.
]

for shape in test_cases:
    t = torch.randn(shape)
    c, s, p = chunk_tensor_by_context(t, chunk_size)
    u = unchunk_tensor_by_context(c, s, p, chunk_size)
    
    n, h_l, w = get_structure(shape, chunk_size)
    print(f"Shape {shape} -> h_local={h_l}, w={w} -> chunked: {c.shape} -> match: {torch.allclose(t, u)}")
