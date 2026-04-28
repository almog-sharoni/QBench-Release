"""
Single-value FP8 trace: Python `quantize_fp_generic` vs CUDA `_ARU_nf` variants.

Spec: of the 4 CUDA rounding variants, only `_ARU_nf` (Always Round Up + No
subnormal Flush) is intended to match Python `quantize_fp_generic`. The
other variants flush subnormals or change rounding on purpose.

For every fp8_eXmY format (e in 1..7), this script feeds a curated set of
single FP32 inputs through:
  - Python `quantize_fp_generic` (the spec, called directly to bypass scaling)
  - CUDA `encode_fp8_tensor_ARU_nf`   + decode  (scale=1.0)
  - CUDA `encode_fp8_chunk_ARU_nf`    + decode  (kernel computes pow2 scale)
  - CUDA `encode_fp8_channel_ARU_nf`  + decode  (scales=[1.0])

For chunk, the kernel internally normalizes by pow2_floor(|v|), so we
compare against a chunk-aware Python reference: `Q(v / s) * s`.

Each input prints one row showing the four results and a MATCH/DIFF flag.
At end of each format, a summary lists which granularities diverged.

Usage:
  ./apptainer.sh python3.13 runspace/trace_arunf_vs_python.py [--diff-only]
                                                              [--format fp8_e4m3]
"""

import sys
import os
import math
import struct
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantization.quantizer import quantize_fp_generic
from src.quantization.constants import get_format_params, get_quantization_bias
from src.ops.quant_base import calculate_scale
from src.quantization import cuda as cuda_mod
from src.quantization.cuda import (
    decode_fp8_tensor, decode_fp8_chunk, decode_fp8_channel,
)

# Resolved at runtime from --variant flag
_ENC_TENSOR  = None
_ENC_CHUNK   = None
_ENC_CHANNEL = None

ALL_FP8_FORMATS = [f"fp8_e{e}m{7 - e}" for e in range(1, 8)]
CHUNK_SIZE = 128


# ---------------------------------------------------------------------------
# Reference: decode an FP8 byte using the CUDA decode formula (host-side mirror)
# ---------------------------------------------------------------------------
def fp8_byte_to_float(byte: int, e: int, m: int, b: int) -> float:
    sign = (byte >> 7) & 1
    exp_t = (byte >> m) & ((1 << e) - 1)
    mant = byte & ((1 << m) - 1) if m > 0 else 0
    if exp_t == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        v = mant * (2.0 ** (1 - b - m))
        return -v if sign else v
    exp_f = exp_t - b
    if m == 0:
        v = 2.0 ** exp_f
    else:
        v = (1.0 + mant / (1 << m)) * (2.0 ** exp_f)
    return -v if sign else v


def pow2_floor(x: float) -> float:
    """Mirror of CUDA pow2_floor_nonneg (zero the FP32 mantissa)."""
    if x == 0.0:
        return 1.0
    bits = struct.unpack("<I", struct.pack("<f", abs(x)))[0]
    bits &= 0xFF800000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# ---------------------------------------------------------------------------
# Build curated test values per format
# ---------------------------------------------------------------------------
def build_test_values(e: int, m: int, b: int) -> list[tuple[str, float]]:
    """Return list of (label, value) covering every FP8 representable, midpoints, edges."""
    seen: dict[float, str] = {}
    out: list[tuple[str, float]] = []

    def add(label: str, v: float):
        # Dedupe by exact float value, keep first label
        key = v
        if key in seen:
            return
        seen[key] = label
        out.append((label, v))

    # Zero
    add("zero", 0.0)

    # Every representable value (positive + negative) decoded from the byte
    representables_pos: list[float] = []
    for byte in range(256):
        v = fp8_byte_to_float(byte, e, m, b)
        if v > 0 and math.isfinite(v):
            representables_pos.append(v)
    representables_pos = sorted(set(representables_pos))

    for v in representables_pos:
        add(f"repr+{v:.6e}", v)
        add(f"repr-{v:.6e}", -v)

    # Midpoints between consecutive positive representables (rounding tests)
    for i in range(len(representables_pos) - 1):
        a, b_ = representables_pos[i], representables_pos[i + 1]
        mid = (a + b_) / 2.0
        # Just below midpoint and just above — to test rounding direction
        add(f"mid+{mid:.6e}", mid)
        add(f"mid-{mid:.6e}", -mid)

    # Saturation: above max representable
    max_v = representables_pos[-1]
    add("over_max+", max_v * 2.0)
    add("over_max-", -max_v * 2.0)

    # Underflow: below smallest positive representable
    min_v = representables_pos[0]
    add("under_min+", min_v * 0.25)
    add("under_min-", -min_v * 0.25)

    return out


# ---------------------------------------------------------------------------
# CUDA wrappers (single-value)
# ---------------------------------------------------------------------------
def host_scale(v: float, q_type: str) -> float:
    """Mirror what `quantize_tensor` computes for tensor/channel modes:
    pow2_floor of |v|, with the same min-clamp it uses to avoid /0."""
    max_val = torch.tensor(max(abs(v), 1e-5), dtype=torch.float32)
    return float(calculate_scale(max_val, q_type).item())


def cuda_tensor_arunf(v: float, e: int, m: int, b: int, scale: float) -> float:
    x = torch.tensor([v], dtype=torch.float32, device="cuda")
    out = torch.zeros(1, dtype=torch.uint8, device="cuda")
    decoded = torch.zeros(1, dtype=torch.float32, device="cuda")
    _ENC_TENSOR(x, out, scale, e, m, b)
    decode_fp8_tensor(out, decoded, scale, e, m, b)
    return float(decoded[0].item())


def cuda_chunk_arunf(v: float, e: int, m: int, b: int) -> tuple[float, float]:
    """Returns (decoded[0], internal_scale_used_by_kernel)."""
    buf = torch.zeros(CHUNK_SIZE, dtype=torch.float32, device="cuda")
    buf[0] = v
    out = torch.zeros(CHUNK_SIZE, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(1, dtype=torch.float32, device="cuda")
    decoded = torch.zeros(CHUNK_SIZE, dtype=torch.float32, device="cuda")
    _ENC_CHUNK(buf, out, scales, e, m, b, CHUNK_SIZE)
    decode_fp8_chunk(out, scales, decoded, e, m, b, CHUNK_SIZE)
    return float(decoded[0].item()), float(scales[0].item())


def cuda_channel_arunf(v: float, e: int, m: int, b: int, scale: float) -> float:
    x = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
    x[0, 0] = v
    scales = torch.tensor([scale], dtype=torch.float32, device="cuda")
    out = torch.zeros(1 * 4, dtype=torch.uint8, device="cuda")
    decoded = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
    _ENC_CHANNEL(x, scales, out, e, m, b)
    decode_fp8_channel(out, scales, decoded, e, m, b)
    return float(decoded[0, 0].item())


# ---------------------------------------------------------------------------
# Python references
# ---------------------------------------------------------------------------
def python_ref(v: float, exp_bits: int, mant_bits: int) -> float:
    """Direct call — no scaling."""
    t = torch.tensor([v], dtype=torch.float32)
    out = quantize_fp_generic(t, exp_bits, mant_bits)
    return float(out[0].item())


def python_chunk_ref(v: float, exp_bits: int, mant_bits: int, internal_scale: float) -> float:
    """Python equivalent of what the chunk kernel computes: Q(v / s) * s."""
    if internal_scale == 0.0:
        return 0.0
    return python_ref(v / internal_scale, exp_bits, mant_bits) * internal_scale


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------
def fp32_bits(v: float) -> str:
    return "0x" + struct.pack("<f", v).hex()


def floats_equal(a: float, b: float) -> bool:
    """Bit-exact comparison (handles +0/-0 separately)."""
    if math.isnan(a) and math.isnan(b):
        return True
    return struct.pack("<f", a) == struct.pack("<f", b)


def trace_format(q_type: str, diff_only: bool) -> dict:
    e, m = get_format_params(q_type)
    b = get_quantization_bias(q_type)

    print(f"\n{'#' * 96}")
    print(f"  FORMAT: {q_type}   e={e}  m={m}  b={b}   "
          f"(tensor/channel use host-supplied scale = pow2_floor(max(|v|, 1e-5)))")
    print(f"{'#' * 96}")
    print(f"  {'label':<24}  {'v':>13}  {'bits':<10}  "
          f"{'python':>13}  {'tensor':>13}  {'chunk(s,Q)':>20}  {'channel':>13}   flag")
    print(f"  {'-' * 24}  {'-' * 13}  {'-' * 10}  "
          f"{'-' * 13}  {'-' * 13}  {'-' * 20}  {'-' * 13}   ----")

    tests = build_test_values(e, m, b)
    diff_tensor = diff_chunk = diff_channel = 0
    total = 0

    for label, v in tests:
        s_host = host_scale(v, q_type)
        py_host = python_chunk_ref(v, e, m, s_host)   # mirror the divide-quantize-multiply
        ct = cuda_tensor_arunf(v, e, m, b, s_host)
        cc, s_chunk = cuda_chunk_arunf(v, e, m, b)
        ch = cuda_channel_arunf(v, e, m, b, s_host)
        py_chunk = python_chunk_ref(v, e, m, s_chunk)

        t_match = floats_equal(py_host, ct)
        c_match = floats_equal(py_chunk, cc)
        h_match = floats_equal(py_host, ch)

        if not t_match: diff_tensor += 1
        if not c_match: diff_chunk += 1
        if not h_match: diff_channel += 1
        total += 1

        all_match = t_match and c_match and h_match
        if diff_only and all_match:
            continue

        flags = []
        if not t_match: flags.append("T")
        if not c_match: flags.append("C")
        if not h_match: flags.append("H")
        flag = "MATCH" if all_match else "DIFF " + ",".join(flags)

        chunk_str = f"s={s_chunk:.3e},Q={cc:.3e}"
        print(f"  {label:<24}  {v:>13.6e}  {fp32_bits(v):<10}  "
              f"{py_host:>13.6e}  {ct:>13.6e}  {chunk_str:>20}  {ch:>13.6e}   {flag}")

    print(f"\n  --- {q_type} summary: {total} tests, "
          f"tensor_diff={diff_tensor}  chunk_diff={diff_chunk}  channel_diff={diff_channel}")
    return {"total": total, "tensor": diff_tensor, "chunk": diff_chunk, "channel": diff_channel}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff-only", action="store_true",
                        help="Only print rows where at least one CUDA variant differs from Python")
    parser.add_argument("--format", default=None,
                        help="Trace only one format (e.g. fp8_e4m3). Default: all 7.")
    parser.add_argument("--variant", default="std",
                        choices=["ARU_nf", "ARU", "nf", "std"],
                        help="Which CUDA variant suffix to compare against Python "
                             "(std = the unsuffixed encode_fp8_*). Default: ARU_nf.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    # Resolve CUDA encoder names from the variant flag
    suffix_map = {"ARU_nf": "_ARU_nf", "ARU": "_ARU", "nf": "_nf", "std": ""}
    sfx = suffix_map[args.variant]
    global _ENC_TENSOR, _ENC_CHUNK, _ENC_CHANNEL
    _ENC_TENSOR  = getattr(cuda_mod, f"encode_fp8_tensor{sfx}")
    _ENC_CHUNK   = getattr(cuda_mod, f"encode_fp8_chunk{sfx}")
    _ENC_CHANNEL = getattr(cuda_mod, f"encode_fp8_channel{sfx}")
    print(f"Comparing CUDA encode_fp8_{{tensor,chunk,channel}}{sfx} vs Python quantize_fp_generic")

    formats = [args.format] if args.format else ALL_FP8_FORMATS
    summary = {q: trace_format(q, args.diff_only) for q in formats}

    print(f"\n{'=' * 96}")
    print("FINAL SUMMARY  (DIFF counts per granularity)")
    print(f"{'=' * 96}")
    print(f"  {'format':<12}  {'tests':>6}  {'tensor':>8}  {'chunk':>8}  {'channel':>8}")
    for q, s in summary.items():
        print(f"  {q:<12}  {s['total']:>6}  {s['tensor']:>8}  {s['chunk']:>8}  {s['channel']:>8}")
    print(f"{'=' * 96}")


if __name__ == "__main__":
    main()
