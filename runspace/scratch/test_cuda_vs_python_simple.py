import torch
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

from runspace.src.quantization.cuda import search_best_chunk_format
from runspace.src.quantization.quantizer import quantize
from runspace.src.quantization.constants import get_format_params

def py_search(x, cands):
    x_chunk = x.view(-1, 128)
    amax = x_chunk.abs().max(dim=1, keepdim=True).values
    s = torch.pow(2.0, torch.floor(torch.log2(amax.clamp(min=1e-12))))
    s[amax == 0] = 2.0 ** -126
    
    x_scaled = x_chunk / s
    
    best_err = torch.full((x_chunk.shape[0], 1), float('inf'), device=x.device)
    best_idx = torch.zeros((x_chunk.shape[0], 1), dtype=torch.long, device=x.device)
    
    for i, fmt in enumerate(cands):
        q = quantize(x_scaled.contiguous(), q_type=fmt)
        err = ((x_scaled - q) ** 2).sum(dim=1, keepdim=True)
        mask = err < best_err
        best_err[mask] = err[mask]
        best_idx[mask] = i
        
    return best_idx.view(-1), s.view(-1)

def main():
    torch.manual_seed(42)
    x = torch.randn(128 * 10, device='cuda')
    cands = ['fp8_e4m3', 'fp6_e3m2', 'fp4_e2m1', 'ufp8_e4m4']
    
    py_idx, py_s = py_search(x, cands)
    e_list, m_list, sgn_list = [], [], []
    for fmt in cands:
        e, m = get_format_params(fmt)
        sgn = 0 if fmt.startswith("ufp") else 1
        e_list.append(e)
        m_list.append(m)
        sgn_list.append(sgn)
        
    cu_idx, cu_s, _, cu_q = search_best_chunk_format(x, e_list, m_list, sgn_list, True)
    
    print("Indices match:", torch.equal(py_idx, cu_idx))
    print("Scales match: ", torch.allclose(py_s, cu_s))
    if not torch.equal(py_idx, cu_idx):
        print(f"py={py_idx.tolist()}")
        print(f"cu={cu_idx.tolist()}")

if __name__ == '__main__':
    main()
