"""
Phase 0 gate: encode -> decode round trip is bit equivalent to
quantize_fp_generic on the active (e, m, b) presets.

Run:
    ./apptainer.sh python runspace/tests/test_fp8_storage.py

The bias is set to QBench's existing convention (1 << e) - 1 to match
quantize_fp_generic, which does not yet take an explicit bias argument
(R4 mitigation pending; tracked in plan §3.2).
"""

import torch

from runspace.src.quantization.cuda import (
    encode_fp8_emb_chunk, decode_fp8_emb_chunk,
)
from runspace.src.quantization.quantizer import quantize_fp_generic


def power_of_two_floor(amax: torch.Tensor) -> torch.Tensor:
    """Match the kernel's pow2_floor_nonneg in Python."""
    bits = amax.view(torch.int32) & 0x7F800000
    bits = torch.where(amax == 0, torch.full_like(bits, 0x3F800000), bits)  # 1.0f
    return bits.view(torch.float32)


def run_phase0(e: int, m: int, b: int, N: int = 8192, K: int = 128, seed: int = 0):
    """Single (e, m, b) preset. Returns (n_mismatch, max_abs_diff)."""
    torch.manual_seed(seed)
    x = torch.randn(N, device='cuda', dtype=torch.float32) * 2.0

    # 1. Per-chunk power-of-two scale.
    x_chunks = x.view(-1, K)
    s = power_of_two_floor(x_chunks.abs().amax(dim=1))

    # 2. Reference path: normalize, quantize_fp_generic, denormalize.
    y_norm  = x_chunks / s.unsqueeze(1)
    y_ref   = quantize_fp_generic(y_norm.flatten(), e, m).view_as(x_chunks)
    x_ref   = (y_ref * s.unsqueeze(1)).flatten()

    # 3. CUDA path: encode, decode (decode rescales internally).
    out_enc     = torch.empty(N,       dtype=torch.uint8,  device='cuda')
    scales_cuda = torch.empty(N // K,  dtype=torch.float32, device='cuda')
    encode_fp8_emb_chunk(x, out_enc, scales_cuda, e, m, b, K)
    x_cuda      = torch.empty(N, dtype=torch.float32, device='cuda')
    decode_fp8_emb_chunk(out_enc, scales_cuda, x_cuda, e, m, b, K)

    # 4. Scale agreement (a stricter prerequisite than reconstruction match).
    assert torch.equal(scales_cuda, s), \
        f'scale mismatch on (e={e}, m={m}, b={b})'

    # 5. Reconstructed value comparison.
    diff       = (x_ref - x_cuda).abs()
    n_mismatch = int((diff > 0).sum().item())
    max_diff   = float(diff.max().item())
    return n_mismatch, max_diff, x, x_ref, x_cuda, diff


def report(e, m, b, n_mismatch, max_diff, x, x_ref, x_cuda, diff, N):
    pct = 100.0 * n_mismatch / N
    print(f'(e={e}, m={m}, b={b}): {N - n_mismatch}/{N} match  '
          f'({pct:.3f}% mismatch), max |diff| = {max_diff:.6e}')
    if n_mismatch > 0:
        idx = diff.nonzero().flatten()[:5].tolist()
        print('  first 5 mismatches:')
        for i in idx:
            print(f'    x[{i:5d}] = {x[i].item():+.7f}  '
                  f'ref = {x_ref[i].item():+.7f}  '
                  f'cuda = {x_cuda[i].item():+.7f}  '
                  f'|diff| = {diff[i].item():.3e}')


if __name__ == '__main__':
    print('Phase 0 bit-exact gate vs quantize_fp_generic')
    print('  using QBench bias convention b = (1 << e) - 1')
    print()
    N, K = 8192, 128

    # E4M3 with QBench bias 15.
    n, mx, x, xr, xc, d = run_phase0(e=4, m=3, b=15, N=N, K=K)
    report(4, 3, 15, n, mx, x, xr, xc, d, N)

    # E5M2 with QBench bias 31.
    n, mx, x, xr, xc, d = run_phase0(e=5, m=2, b=31, N=N, K=K)
    report(5, 2, 31, n, mx, x, xr, xc, d, N)
