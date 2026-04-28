"""
Compare CUDA encode_fp8_{tensor,chunk,channel} variants against quantize_tensor
for every valid fp8_eXmY format (e + m = 7, e in 1..7).

12 CUDA variants tested (4 rounding modes × 3 granularities):
  tensor  : encode_fp8_tensor{,_ARU,_nf,_ARU_nf}
  chunk   : encode_fp8_chunk{,_ARU,_nf,_ARU_nf}   (chunk_size=128, N % 128 == 0)
  channel : encode_fp8_channel{,_ARU,_nf,_ARU_nf} (2D input [C, K], K % 4 == 0)

Usage:
  python compare_cuda_quant.py [--seed 42]
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ops.quant_base import quantize_tensor, calculate_scale
from src.quantization.quantizer import quantize
from src.quantization.constants import get_format_params, get_quantization_bias
from src.quantization.cuda import (
    encode_fp8_tensor,     encode_fp8_tensor_ARU,
    encode_fp8_tensor_nf,  encode_fp8_tensor_ARU_nf,
    decode_fp8_tensor,
    encode_fp8_chunk,      encode_fp8_chunk_ARU,
    encode_fp8_chunk_nf,   encode_fp8_chunk_ARU_nf,
    decode_fp8_chunk,
    encode_fp8_channel,    encode_fp8_channel_ARU,
    encode_fp8_channel_nf, encode_fp8_channel_ARU_nf,
    decode_fp8_channel,
)

ALL_FP8_FORMATS = [f"fp8_e{e}m{7 - e}" for e in range(1, 8)]

TENSOR_VARIANTS = [
    ("encode_fp8_tensor",        encode_fp8_tensor),
    ("encode_fp8_tensor_ARU",    encode_fp8_tensor_ARU),
    ("encode_fp8_tensor_nf",     encode_fp8_tensor_nf),
    ("encode_fp8_tensor_ARU_nf", encode_fp8_tensor_ARU_nf),
]

CHUNK_VARIANTS = [
    ("encode_fp8_chunk",         encode_fp8_chunk),
    ("encode_fp8_chunk_ARU",     encode_fp8_chunk_ARU),
    ("encode_fp8_chunk_nf",      encode_fp8_chunk_nf),
    ("encode_fp8_chunk_ARU_nf",  encode_fp8_chunk_ARU_nf),
]

CHANNEL_VARIANTS = [
    ("encode_fp8_channel",        encode_fp8_channel),
    ("encode_fp8_channel_ARU",    encode_fp8_channel_ARU),
    ("encode_fp8_channel_nf",     encode_fp8_channel_nf),
    ("encode_fp8_channel_ARU_nf", encode_fp8_channel_ARU_nf),
]

CHUNK_SIZE = 128

# ---------------------------------------------------------------------------
# Test tensor factories
# ---------------------------------------------------------------------------

# Tensor-mode: any shape
TENSOR_TESTS = [
    ("small 1D [256]",        lambda: torch.randn(256)),
    ("medium 1D [4096]",      lambda: torch.randn(4096)),
    ("large 1D [65536]",      lambda: torch.randn(65536)),
    ("non-power-of-4 [1001]", lambda: torch.randn(1001)),
    ("2D [128,256]",          lambda: torch.randn(128, 256)),
    ("3D [4,64,128]",         lambda: torch.randn(4, 64, 128)),
    ("all zeros [512]",       lambda: torch.zeros(512)),
    ("near-zero [512]",       lambda: torch.randn(512) * 1e-6),
    ("large magnitude [512]", lambda: torch.randn(512) * 1e3),
    ("single element [1]",    lambda: torch.randn(1)),
]

# Chunk-mode: 1D, N must be multiple of 128
CHUNK_TESTS = [
    ("1D [128]",              lambda: torch.randn(128)),
    ("1D [1024]",             lambda: torch.randn(1024)),
    ("1D [16384]",            lambda: torch.randn(16384)),
    ("all zeros [512]",       lambda: torch.zeros(512)),
    ("near-zero [512]",       lambda: torch.randn(512) * 1e-6),
    ("large magnitude [512]", lambda: torch.randn(512) * 1e3),
]

# Channel-mode: 2D [C, K], K must be multiple of 4
CHANNEL_TESTS = [
    ("2D [32,128]",           lambda: torch.randn(32, 128)),
    ("2D [64,256]",           lambda: torch.randn(64, 256)),
    ("2D [128,512]",          lambda: torch.randn(128, 512)),
    ("2D [1,256]",            lambda: torch.randn(1, 256)),
    ("all zeros [32,128]",    lambda: torch.zeros(32, 128)),
    ("near-zero [32,128]",    lambda: torch.randn(32, 128) * 1e-6),
    ("large mag [32,128]",    lambda: torch.randn(32, 128) * 1e3),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compare(ref: torch.Tensor, cuda_out: torch.Tensor, label: str) -> float:
    diff = (ref - cuda_out).abs()
    match_pct = 100.0 * (diff == 0).sum().item() / diff.numel()
    print(
        f"    {label:<30}  match={match_pct:6.2f}%  "
        f"max_ae={diff.max().item():.3e}  "
        f"mae={diff.mean().item():.3e}  "
        f"mse={diff.pow(2).mean().item():.3e}"
    )
    return match_pct


# ---------------------------------------------------------------------------
# Tensor-mode comparison
# ---------------------------------------------------------------------------

def _run_tensor_variant(x_flat_cuda, scale_val, e, m, b, encode_fn):
    N = x_flat_cuda.numel()
    out = torch.zeros(N, dtype=torch.uint8, device="cuda")
    decoded = torch.zeros(N, dtype=torch.float32, device="cuda")
    encode_fn(x_flat_cuda, out, scale_val, e, m, b)
    decode_fp8_tensor(out, decoded, scale_val, e, m, b)
    return decoded.cpu()


def run_tensor_tests(q_type, e, m, b) -> bool:
    print(f"\n  [tensor mode]")
    ok = True
    for name, tensor_fn in TENSOR_TESTS:
        t = tensor_fn()
        ref = quantize_tensor(t, q_type=q_type, mode="tensor")[0].flatten()
        max_val = t.abs().amax().clamp(min=1e-5)
        scale_val = float(calculate_scale(max_val, q_type).item())
        x_cuda = t.flatten().cuda().contiguous()
        print(f"\n    {name}  scale={scale_val:.3e}")
        for label, fn in TENSOR_VARIANTS:
            dec = _run_tensor_variant(x_cuda, scale_val, e, m, b, fn)
            if _compare(ref, dec.reshape(ref.shape), label) < 100.0:
                ok = False
    return ok


# ---------------------------------------------------------------------------
# Chunk-mode comparison
# ---------------------------------------------------------------------------

def _run_chunk_variant(x_cuda, e, m, b, encode_fn):
    N = x_cuda.numel()
    out = torch.zeros(N, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(N // CHUNK_SIZE, dtype=torch.float32, device="cuda")
    decoded = torch.zeros(N, dtype=torch.float32, device="cuda")
    encode_fn(x_cuda, out, scales, e, m, b, CHUNK_SIZE)
    decode_fp8_chunk(out, scales, decoded, e, m, b, CHUNK_SIZE)
    return decoded.cpu()


def run_chunk_tests(q_type, e, m, b) -> bool:
    print(f"\n  [chunk mode  chunk_size={CHUNK_SIZE}]")
    ok = True
    for name, tensor_fn in CHUNK_TESTS:
        t = tensor_fn()
        ref = quantize_tensor(t, q_type=q_type, mode="chunk", chunk_size=CHUNK_SIZE)[0].flatten()
        x_cuda = t.flatten().cuda().contiguous()
        print(f"\n    {name}")
        for label, fn in CHUNK_VARIANTS:
            dec = _run_chunk_variant(x_cuda, e, m, b, fn)
            if _compare(ref, dec.reshape(ref.shape), label) < 100.0:
                ok = False
    return ok


# ---------------------------------------------------------------------------
# Channel-mode comparison
#
# CUDA encode_fp8_channel uses [C, K] input with one scale per ROW (C scales).
# quantize_tensor(mode='channel') keeps dim 1 and reduces over all other dims,
# so for [C, K] it produces K column-wise scales — a different convention.
# The Python reference here manually applies the same row-wise scaling as CUDA.
# ---------------------------------------------------------------------------

def _run_channel_variant(x_cuda, scales_cuda, C, K, e, m, b, encode_fn):
    out = torch.zeros(C * K, dtype=torch.uint8, device="cuda")
    decoded = torch.zeros(C, K, dtype=torch.float32, device="cuda")
    encode_fn(x_cuda, scales_cuda, out, e, m, b)
    decode_fp8_channel(out, scales_cuda, decoded, e, m, b)
    return decoded.cpu()


def _python_channel_ref(t: torch.Tensor, scales: torch.Tensor, q_type: str) -> torch.Tensor:
    """Row-wise quantize [C, K] using pre-computed per-row scales (matches CUDA convention)."""
    scaled = t / scales.unsqueeze(1)
    return quantize(scaled, q_type=q_type) * scales.unsqueeze(1)


def run_channel_tests(q_type, e, m, b) -> bool:
    print(f"\n  [channel mode]")
    ok = True
    for name, tensor_fn in CHANNEL_TESTS:
        t = tensor_fn()
        C, K = t.shape
        # One scale per row — matches CUDA encode_fp8_channel's convention
        scales = calculate_scale(t.abs().amax(dim=1).clamp(min=1e-5), q_type)  # [C]
        ref = _python_channel_ref(t, scales, q_type).flatten()
        scales_cuda = scales.cuda().contiguous()
        x_cuda = t.cuda().contiguous()
        print(f"\n    {name}")
        for label, fn in CHANNEL_VARIANTS:
            dec = _run_channel_variant(x_cuda, scales_cuda, C, K, e, m, b, fn)
            if _compare(ref, dec.flatten(), label) < 100.0:
                ok = False
    return ok


# ---------------------------------------------------------------------------
# Per-format driver
# ---------------------------------------------------------------------------

def run_format(q_type: str, seed: int) -> bool:
    e, m = get_format_params(q_type)
    b = get_quantization_bias(q_type)
    print(f"\n{'#'*72}")
    print(f"  FORMAT: {q_type}   e={e}  m={m}  b={b}")
    print(f"{'#'*72}")
    torch.manual_seed(seed)

    tensor_ok  = run_tensor_tests(q_type, e, m, b)
    chunk_ok   = run_chunk_tests(q_type, e, m, b)
    channel_ok = run_channel_tests(q_type, e, m, b)

    all_ok = tensor_ok and chunk_ok and channel_ok
    parts = []
    if not tensor_ok:  parts.append("tensor")
    if not chunk_ok:   parts.append("chunk")
    if not channel_ok: parts.append("channel")
    status = "ALL MATCH" if all_ok else f"DIFF in: {', '.join(parts)}"
    print(f"\n  >>> {q_type}: {status}")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    print("Comparing CUDA encode_fp8_{tensor,chunk,channel} variants vs quantize_tensor")
    print(f"Formats: {ALL_FP8_FORMATS}")

    summary = {q: run_format(q, args.seed) for q in ALL_FP8_FORMATS}

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    for q_type, ok in summary.items():
        print(f"  {'OK  ' if ok else 'DIFF'}  {q_type}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
