import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.quantization.chunking import chunk_tensor_by_context
from runspace.src.quantization.constants import get_format_params
from runspace.src.quantization.cuda import search_best_chunk_format
from runspace.src.quantization.quantizer import quantize

def python_search_best_chunk_format(flat_tensor, chunk_size, candidate_formats):
    chunked, unchunk, pad_len = chunk_tensor_by_context(flat_tensor, chunk_size)
    orig_shape = chunked.shape
    x = chunked.view(-1, chunk_size)
    num_chunks = x.shape[0]

    # Compute amax and scale per chunk
    amax = x.abs().max(dim=1, keepdim=True).values
    s = torch.pow(2.0, torch.floor(torch.log2(amax.clamp(min=1e-12))))
    
    # Handle zeros
    zero_mask = (amax == 0)
    s[zero_mask] = 2.0 ** -126  # Smallest positive normal in FP32
    
    x_scaled = x / s

    best_err = torch.full((num_chunks, 1), float('inf'), device=x.device)
    best_idx = torch.zeros((num_chunks, 1), dtype=torch.long, device=x.device)
    best_q = torch.zeros_like(x)

    for c_idx, fmt in enumerate(candidate_formats):
        e, m = get_format_params(fmt)
        sgn = 0 if fmt.startswith("ufp") else 1
        
        # Quantize in python
        q = quantize(x_scaled.contiguous(), fmt)
        if isinstance(q, tuple):
            print(f"q is a tuple! len={len(q)} type(q[0])={type(q[0])}")
            q = q[0]
        
        err = ((x_scaled - q) ** 2).sum(dim=1, keepdim=True)
        
        mask = err < best_err
        best_err[mask] = err[mask]
        best_idx[mask] = c_idx
        
        mask_q = mask.expand_as(q)
        best_q[mask_q] = q[mask_q]

    return best_idx.view(-1), s.view(-1), best_q

def test_equivalence():
    device = torch.device('cuda')
    torch.manual_seed(42)
    
    # Test cases
    shapes = [
        (1, 640),
        (128, 64),
        (1, 3, 224, 224),
        (10, 10, 10)
    ]
    
    candidate_formats = ['fp8_e4m3', 'fp6_e3m2', 'fp4_e2m1', 'ufp8_e4m4']
    
    all_match = True
    for shape in shapes:
        print(f"\nTesting shape: {shape}")
        x = torch.randn(*shape, device=device)
        # Randomly inject some exact zeros
        x[torch.rand_like(x) < 0.1] = 0.0
        
        flat_x = x.view(-1)
        
        # Python
        py_idx, py_s, py_q = python_search_best_chunk_format(flat_x, 128, candidate_formats)
        
        # CUDA
        cu_idx, cu_s, _, cu_q = search_best_chunk_format(flat_x, 128, candidate_formats)
        
        # Compare
        idx_match = torch.equal(py_idx, cu_idx)
        
        s_match = torch.allclose(py_s, cu_s, rtol=0, atol=0)
        q_match = torch.allclose(py_q, cu_q, rtol=0, atol=0)
        
        print(f"Indices match: {idx_match}")
        print(f"Scales match:  {s_match}")
        print(f"Values match:  {q_match}")
        
        if not (idx_match and s_match and q_match):
            all_match = False
            # Find mismatches
            mismatch_idx = (py_idx != cu_idx).nonzero()
            if len(mismatch_idx) > 0:
                print(f"  First 5 mismatched indices: py={py_idx[mismatch_idx[:5]].flatten()} cu={cu_idx[mismatch_idx[:5]].flatten()}")
            
            mismatch_q = (py_q != cu_q).nonzero()
            if len(mismatch_q) > 0:
                print(f"  Mismatched values detected! Max diff: {(py_q - cu_q).abs().max().item()}")

    if all_match:
        print("\nSUCCESS: CUDA Kernel is exactly bit-for-bit identical to Python implementation!")
    else:
        print("\nFAILURE: Mismatches detected!")

if __name__ == "__main__":
    test_equivalence()
