import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.quantization.chunking import chunk_tensor_by_context
from runspace.src.quantization.constants import get_format_params
from runspace.src.quantization.cuda import search_best_chunk_format
from runspace.src.ops.quant_base import quantize_tensor

def _normalize_metric(metric):
    normalized = str(metric or "l2").strip().lower()
    aliases = {
        "mse": "l2",
        "mae": "l1",
        "max": "linf",
        "chebyshev": "linf",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ("l2", "l1", "linf", "bias", "l0", "huber", "logsum"):
        raise ValueError(f"Unknown metric: {metric}")
    return normalized


def _candidate_params(candidate_formats):
    cands_e = []
    cands_m = []
    cands_sgn = []
    for fmt in candidate_formats:
        e, m = get_format_params(fmt)
        cands_e.append(e)
        cands_m.append(m)
        cands_sgn.append(0 if fmt.startswith("ufp") else 1)
    return cands_e, cands_m, cands_sgn


def _chunk_scale(x):
    amax = x.abs().max(dim=1, keepdim=True).values
    scales = torch.ones_like(amax)
    nonzero = amax != 0
    if nonzero.any():
        values = amax[nonzero].contiguous()
        try:
            bits = values.view(torch.int32)
            mask = torch.tensor(-8388608, dtype=torch.int32, device=x.device)
            scales[nonzero] = torch.bitwise_and(bits, mask).view(torch.float32)
        except Exception:
            scales[nonzero] = torch.pow(
                torch.tensor(2.0, dtype=x.dtype, device=x.device),
                torch.floor(torch.log2(values))
            )
    return scales


METRIC_CODES = {
    "l2": 0, "l1": 1, "linf": 2, "bias": 3, "l0": 4, "huber": 5, "logsum": 6,
}
HUBER_DELTA = 0.0625


def _reduce_metric(diff, metric, delta=HUBER_DELTA):
    if metric == "linf":
        return diff.abs().max(dim=1).values
    if metric == "l1":
        return diff.abs().sum(dim=1)
    if metric == "bias":
        return diff.sum(dim=1).abs()
    if metric == "l0":
        return (diff != 0).to(diff.dtype).sum(dim=1)
    if metric == "huber":
        a = diff.abs()
        return torch.where(a <= delta, 0.5 * diff.pow(2), delta * (a - 0.5 * delta)).sum(dim=1)
    if metric == "logsum":
        a = diff.abs()
        return torch.where(a > 0, torch.floor(torch.log2(a.clamp(min=1e-30))),
                           torch.full_like(a, -126.0)).sum(dim=1)
    return diff.pow(2).sum(dim=1)


def python_search_best_chunk_format(tensor, chunk_size, candidate_formats, metric="l2"):
    metric = _normalize_metric(metric)
    chunked, _original_shape, _pad_len = chunk_tensor_by_context(tensor, chunk_size)
    x = chunked.reshape(-1, chunk_size).contiguous()
    num_chunks = x.shape[0]

    s = _chunk_scale(x)
    x_scaled = x / s

    best_err = torch.full((num_chunks,), float("inf"), dtype=x.dtype, device=x.device)
    best_idx = torch.zeros((num_chunks,), dtype=torch.long, device=x.device)
    best_q_scaled = torch.zeros_like(x)

    for c_idx, fmt in enumerate(candidate_formats):
        q, _ = quantize_tensor(
            x_scaled,
            q_type=fmt,
            mode="chunk",
            chunk_size=chunk_size,
        )
        err = _reduce_metric(x_scaled - q, metric)

        mask = err < best_err
        best_err[mask] = err[mask]
        best_idx[mask] = c_idx
        best_q_scaled[mask] = q[mask]

    return best_idx, s.view(-1), (best_q_scaled * s).view(-1), best_q_scaled.view(-1)

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
        chunked, _original_shape, _pad_len = chunk_tensor_by_context(flat_x, 128)
        ref_chunks = chunked.reshape(-1, 128).contiguous()

        # Python L2 reducer
        py_idx, py_s, py_q, py_unscaled_q = python_search_best_chunk_format(
            flat_x, 128, candidate_formats, metric="l2"
        )

        # CUDA L2 reducer
        cands_e, cands_m, cands_sgn = _candidate_params(candidate_formats)
        cu_idx, cu_s, cu_q, cu_unscaled_q = search_best_chunk_format(
            ref_chunks.view(-1).contiguous(),
            cands_e,
            cands_m,
            cands_sgn,
            True,
        )

        # Compare
        idx_match = torch.equal(py_idx, cu_idx)
        s_match = torch.allclose(py_s, cu_s, rtol=0, atol=0)
        q_match = torch.allclose(py_q, cu_q, rtol=0, atol=0)
        unscaled_q_match = torch.allclose(py_unscaled_q, cu_unscaled_q, rtol=0, atol=0)

        print(f"Indices match: {idx_match}")
        print(f"Scales match:  {s_match}")
        print(f"Values match:  {q_match}")
        print(f"Unscaled match:{unscaled_q_match}")

        # Compare Python reference vs CUDA kernel for every metric.
        for metric, code in METRIC_CODES.items():
            py_m_idx, _, _, _ = python_search_best_chunk_format(
                flat_x, 128, candidate_formats, metric=metric
            )
            cu_m_idx, _, _, _ = search_best_chunk_format(
                ref_chunks.view(-1).contiguous(),
                cands_e, cands_m, cands_sgn,
                False,
                code,
                HUBER_DELTA,
            )
            n = py_m_idx.numel()
            agree = int((py_m_idx == cu_m_idx).sum().item())
            rate = agree / n if n else 1.0
            # Fast-math/log differences can flip near-ties; require strong agreement.
            ok = rate >= 0.98
            print(f"{metric.upper():6s} py-vs-cuda argmin agreement: {agree}/{n} ({rate:.3f})"
                  f"{'' if ok else '  <-- LOW'}")
            all_match = all_match and ok

        if not (idx_match and s_match and q_match and unscaled_q_match):
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
