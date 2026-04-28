"""
compare_paths.py — side-by-side comparison of the CUDA and PyTorch quantization
paths inside quantize_fp_generic, to identify where they diverge.

Run with:
    python -m runspace.src.quantization.cuda_tests.compare_paths
"""

import os
import torch
from runspace.src.quantization.quantizer import quantize_fp_generic

E, M = 4, 3  # FP8 E4M3
BIAS = (1 << E) - 1  # = 15


def cuda_path(x: torch.Tensor) -> torch.Tensor:
    return quantize_fp_generic(x, exp_bits=E, mant_bits=M)


def python_path(x: torch.Tensor) -> torch.Tensor:
    os.environ["QBENCH_DISABLE_CUDA_QUANTIZE"] = "1"
    result = quantize_fp_generic(x, exp_bits=E, mant_bits=M)
    del os.environ["QBENCH_DISABLE_CUDA_QUANTIZE"]
    return result


def compare(label: str, x: torch.Tensor) -> None:
    x_cuda = x.cuda() if x.device.type == "cpu" else x
    yc = cuda_path(x_cuda)
    yp = python_path(x_cuda)

    matches = torch.equal(yc, yp)
    diff = (yc - yp).abs()
    n_diff = int((diff > 0).sum())
    print(f"\n=== {label} ===")
    print(f"  shape        : {x.shape}")
    print(f"  bit-exact    : {matches}")
    print(f"  n_different  : {n_diff} / {x.numel()}")
    if n_diff > 0:
        print(f"  max |diff|   : {diff.max().item():.6g}")
        # Show the first few differing values
        mask = diff > 0
        idxs = mask.nonzero(as_tuple=False)[:8, 0]
        print("  first diffs  (input  →  cuda  /  python):")
        for i in idxs.tolist():
            print(f"    [{i:5d}]  {x_cuda[i].item():+.8f}  →  "
                  f"{yc[i].item():+.8f}  /  {yp[i].item():+.8f}")


# ── 1. Tie-value sweep: exact midpoints between adjacent FP8 grid points ──────
# FP8 E4M3 b=15 grid in [1, 2): spacing = 0.125 → ties at 1.0625, 1.1875, ...
tie_values = torch.tensor(
    [1.0 + (i + 0.5) * 0.125 for i in range(8)],
    dtype=torch.float32,
)
compare("Exact tie values (midpoints between FP8 grid points)", tie_values)

# ── 2. Dense sweep across the representable range ─────────────────────────────
dense = torch.linspace(0.0, 2.0, steps=10001, dtype=torch.float32)
compare("Dense sweep [0, 2]", dense)

# ── 3. Random weights-like distribution ───────────────────────────────────────
torch.manual_seed(42)
weights = torch.randn(8192, dtype=torch.float32) * 0.5
compare("Random normal (std=0.5, models weight-like values)", weights)

# ── 4. Values above FP8 max (should both saturate) ────────────────────────────
above_max = torch.tensor([2.0, 3.5, 10.0, 100.0, -2.0, -5.0], dtype=torch.float32)
compare("Values above FP8 max (should clip to ±1.875)", above_max)

# ── 5. Very small values near/below the flush-to-zero threshold 2^-14 ─────────
small = torch.tensor(
    [2**-13, 2**-14, 2**-15, 2**-16, 1e-5, 0.0],
    dtype=torch.float32,
)
compare("Near-subnormal values (flush-to-zero boundary at 2^-14)", small)

# ── 6. Large random sweep ─────────────────────────────────────────────────────
torch.manual_seed(0)
large = torch.randn(100_000, dtype=torch.float32) * 1.5
compare("Large random (std=1.5, many values exceed FP8 max)", large)
