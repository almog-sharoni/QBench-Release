"""Minimal usage example for the QBench low-precision CUDA codec.

The codec packs each w-bit element into a uint32 storage word, fitting
n_per_word(w) = floor(32 / w) elements per word; remaining bits are
zero padding.  Element width is

    w = signed_bit + e + m,  2 <= w <= 16

with signed_bit = 1 for signed formats (q_type prefix 'fp') and 0 for
unsigned formats (q_type prefix 'ufp').  Unsigned formats flush negative
inputs to +0; they are intended for nonnegative tensors (post-ReLU
activations, attention probabilities, magnitudes, etc.).
"""
from __future__ import annotations

import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runspace.src.quantization.cuda import (
    encode_tensor,  decode_tensor,
    encode_chunk,   decode_chunk,
    encode_channel, decode_channel,
    resolve_format, n_per_word, element_width,
)


# ----------------------------------------------------------------------------
# Format pickers (comprehension)
# ----------------------------------------------------------------------------
#
# For each width w in [2, 16] we pick a "balanced" canonical split.  For
# signed formats with width w we want e + m == w - 1; for unsigned we
# want e + m == w.  We choose e such that the format covers a useful
# dynamic range without crowding mantissa or exponent.

def _balanced_em(w_value: int, is_signed: bool) -> tuple[int, int]:
    """Pick a balanced (e, m) for width w_value, respecting host bounds."""
    payload = w_value - (1 if is_signed else 0)            # bits for exp + mant
    e = max(2, payload // 2 + 1)                           # leans on exponent
    m = payload - e
    # Host check: e in [1, 15], m in [0, 14].
    e = min(e, 15)
    m = max(0, min(m, 14))
    return e, m


def _label(e: int, m: int, is_signed: bool) -> str:
    prefix = "fp" if is_signed else "ufp"
    width  = (1 if is_signed else 0) + e + m
    return f"{prefix}{width}_e{e}m{m}"


SIGNED_FORMATS = [_label(*_balanced_em(w, True ),  True ) for w in range(4, 17, 2)]
UNSIGN_FORMATS = [_label(*_balanced_em(w, False), False) for w in range(4, 17, 2)]
ALL_FORMATS    = SIGNED_FORMATS + UNSIGN_FORMATS


def section(title: str) -> None:
    print("\n" + "=" * 80 + f"\n  {title}\n" + "=" * 80)


# ----------------------------------------------------------------------------
# Mode demo helpers (one per mode)
# ----------------------------------------------------------------------------

def _demo_tensor(q_type: str, x: torch.Tensor) -> torch.Tensor:
    e, m, sgn = resolve_format(q_type)
    data, scale = encode_tensor(x, e, m, sgn)
    return decode_tensor(data, scale, x.numel(), e, m, sgn).reshape(x.shape)


def _demo_chunk(q_type: str, x: torch.Tensor) -> torch.Tensor:
    e, m, sgn = resolve_format(q_type)
    data, scales = encode_chunk(x, e, m, sgn)
    return decode_chunk(data, scales, x.numel(), e, m, sgn).reshape(x.shape)


def _demo_channel(q_type: str, x: torch.Tensor, channel_dim: int) -> torch.Tensor:
    e, m, sgn = resolve_format(q_type)
    data, scales = encode_channel(x, channel_dim, e, m, sgn)
    return decode_channel(data, scales, list(x.shape), channel_dim, e, m, sgn)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    print(f"Device : {torch.cuda.get_device_name(0)}")

    # ---- 1. Tensor mode (signed) -----------------------------------------
    section("Tensor mode, signed (fp8_e4m3)")
    x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    y = _demo_tensor("fp8_e4m3", x)
    print(f"x.shape={tuple(x.shape)}   max|x - y| = {(x - y).abs().max().item():.3e}")

    # ---- 2. Tensor mode (unsigned) on a nonnegative input ----------------
    section("Tensor mode, unsigned (ufp8_e4m4) on a nonnegative tensor")
    x = torch.randn(4, 64, device="cuda", dtype=torch.float32).abs()
    y = _demo_tensor("ufp8_e4m4", x)
    print(f"x is nonnegative : x.min()={x.min().item():.3e}  "
          f"x.max()={x.max().item():.3e}")
    print(f"max|x - y|       = {(x - y).abs().max().item():.3e}")

    # ---- 3. Unsigned path on mixed-sign input flushes negatives ----------
    section("Unsigned (ufp8_e4m4) on mixed-sign input: negatives -> +0")
    x = torch.randn(8, device="cuda", dtype=torch.float32)
    y = _demo_tensor("ufp8_e4m4", x)
    for xi, yi in zip(x.tolist(), y.tolist()):
        flushed = "(flushed)" if xi < 0.0 else ""
        print(f"  x = {xi:+.6f}   y = {yi:+.6f}  {flushed}")
    neg_mask = x < 0
    print(f"\n  negative inputs flushed to 0: "
          f"{int(neg_mask.sum().item())} / {x.numel()}")

    # ---- 4. Chunk mode, full-sized chunk ---------------------------------
    section("Chunk mode (signed fp8_e4m3, exact multiple of 128)")
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    y = _demo_chunk("fp8_e4m3", x)
    print(f"x.shape={tuple(x.shape)}   N={x.numel()} = 8 * 128")
    print(f"max|x - y| = {(x - y).abs().max().item():.3e}")

    # ---- 5. Chunk mode, partial last chunk (N = 120) ---------------------
    section("Chunk mode with a partial chunk (N = 120, single 128-slot chunk)")
    x = torch.randn(120, device="cuda", dtype=torch.float32)
    y = _demo_chunk("fp8_e4m3", x)
    print(f"x.shape={tuple(x.shape)}   N=120 -> 1 chunk holding 128 slots")
    print(f"output y.shape={tuple(y.shape)} (only the real 120 entries)")
    print(f"max|x - y| = {(x - y).abs().max().item():.3e}")

    # ---- 6. Chunk mode crossing chunk boundary (N = 130) ----------------
    section("Chunk mode crossing one chunk boundary (N = 130)")
    x = torch.randn(130, device="cuda", dtype=torch.float32)
    y = _demo_chunk("fp8_e4m3", x)
    print(f"x.shape={tuple(x.shape)}   N=130 -> 2 chunks (128 + 2 real elements)")
    print(f"max|x - y| = {(x - y).abs().max().item():.3e}")

    # ---- 7. Channel mode -------------------------------------------------
    section("Channel mode (signed fp8_e4m3, channel_dim = 1)")
    x = torch.randn(4, 16, 8, 8, device="cuda", dtype=torch.float32)
    y = _demo_channel("fp8_e4m3", x, channel_dim=1)
    print(f"x.shape={tuple(x.shape)}   channel_dim=1   y.shape={tuple(y.shape)}")
    print(f"max|x - y| = {(x - y).abs().max().item():.3e}")

    # ---- 8. Packing geometry sweep --------------------------------------
    section("Packing geometry across signed and unsigned formats")
    print(f"  {'q_type':<14s}  {'(e, m, sgn)':<14s}  {'w':>3s}  {'n/word':>7s}")
    for q in ALL_FORMATS:
        e, m, sgn = resolve_format(q)
        w   = element_width(e, m, sgn)
        npw = n_per_word(e, m, sgn)
        print(f"  {q:<14s}  ({e}, {m}, {int(sgn)})       {w:>3d}  {npw:>7d}")

    # ---- 9. Round-trip self-consistency, all listed formats -------------
    section("Round-trip self-consistency (tensor mode)")
    torch.manual_seed(0)
    x_signed = torch.randn(256, 512, device="cuda", dtype=torch.float32)
    x_unsign = x_signed.abs()                                 # nonnegative

    print(f"  {'q_type':<14s}  {'w':>3s}  {'max_err':>11s}  {'rmse':>11s}")
    rows = [
        (q, *_self_check(q, x_signed if resolve_format(q)[2] else x_unsign))
        for q in ALL_FORMATS
    ]
    for q, w, max_err, rmse in rows:
        print(f"  {q:<14s}  {w:>3d}  {max_err:>11.3e}  {rmse:>11.3e}")


def _self_check(q_type: str, x: torch.Tensor) -> tuple[int, float, float]:
    e, m, sgn = resolve_format(q_type)
    w = element_width(e, m, sgn)
    data, scale = encode_tensor(x, e, m, sgn)
    y = decode_tensor(data, scale, x.numel(), e, m, sgn).reshape(x.shape)
    return w, (x - y).abs().max().item(), (x - y).pow(2).mean().sqrt().item()


if __name__ == "__main__":
    main()