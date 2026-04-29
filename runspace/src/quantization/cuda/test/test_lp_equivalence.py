"""Bit-exact equivalence test: CUDA codec vs. the Python reference.

Compares the round-trip x -> encode -> decode produced by the CUDA codec
against `quantize_tensor` (runspace/src/quantization/quant_base.py), which
calls the unmodified `quantize_fp_generic`.  The CUDA encode kernel is a
bit-exact restatement of `quantize_fp_generic` and includes subnormal
encoding, so every entry must match to FP32 bit identity for every format
and every shape.

Pass criterion: max_err == 0 across all combinations.
"""
from __future__ import annotations

import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runspace.src.ops.quant_base   import quantize_tensor
from runspace.src.quantization.cuda         import (
    encode_tensor,  decode_tensor,
    encode_chunk,   decode_chunk,
    encode_channel, decode_channel,
    resolve_format,
)


# ----------------------------------------------------------------------------
# Reference helpers
# ----------------------------------------------------------------------------

def _unpack(ret):
    """quantize_tensor returns (input_fp8, max_val) by default; we need the first."""
    return ret[0] if isinstance(ret, tuple) else ret


def ref_tensor(x: torch.Tensor, q_type: str) -> torch.Tensor:
    return _unpack(quantize_tensor(x, q_type=q_type, mode="tensor"))


def ref_chunk(x: torch.Tensor, q_type: str, chunk_size: int = 128) -> torch.Tensor:
    return _unpack(quantize_tensor(
        x, q_type=q_type,
        mode="chunk",
        chunk_size=chunk_size,
    ))


def ref_channel(x: torch.Tensor, q_type: str, channel_dim: int) -> torch.Tensor:
    # The Python reference hard-codes the channel axis at dim 1 in the
    # `mode='channel'` branch and accepts no `channel_dim` keyword.  All
    # tests in this module use channel_dim == 1 to match.
    assert channel_dim == 1, (
        "ref_channel: the Python quantize_tensor reference reduces over all "
        "dims except dim 1, so channel_dim must be 1 to compare meaningfully."
    )
    return _unpack(quantize_tensor(x, q_type=q_type, mode="channel"))


# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
#
# This test compares decode(encode(x)) against the Python quantize_tensor
# reference, which is bit-exact only for element widths w = 1 + e + m
# such that w <= 8 with e <= 8.  Wider formats are tested for round-trip
# self-consistency in example_lp_usage.py.


_E_MAX = 16
_M_MAX = 16
_W_MIN = 2
_W_MAX = 16

_SIGNED: dict[str, tuple[int, int]] = {
    f"fp{1 + e + m}_e{e}m{m}": (e, m)
    for e in range(1, _E_MAX + 1)
    for m in range(0, _M_MAX + 1)
    if _W_MIN <= 1 + e + m <= _W_MAX
}

_UNSIGNED: dict[str, tuple[int, int]] = {
    f"ufp{e + m}_e{e}m{m}": (e, m)
    for e in range(1, _E_MAX + 1)
    for m in range(0, _M_MAX + 1)
    if _W_MIN <= e + m <= _W_MAX
}

FORMATS_DEFAULT = _SIGNED # dict union; fallback to _SIGNED for Python < 3.9
FORMATS_DEFAULT.update(_UNSIGNED)

SHAPES_TENSOR  = [(32,),     (8, 16),  (256, 512)]
SHAPES_CHUNK   = [(128,),    (8, 128), (256, 512)]
SHAPES_CHANNEL = [(8, 128),  (4, 16, 8, 8), (256, 256)]
CHANNEL_DIM    = 1


# ----------------------------------------------------------------------------
# Per-mode runners
# ----------------------------------------------------------------------------

def _run_tensor(formats, shapes):
    print("=" * 88)
    print("  tensor mode")
    print("=" * 88)
    pass_n = fail_n = 0
    for q_type in formats:
        e, m, is_signed = resolve_format(q_type)
        for shape in shapes:
            torch.manual_seed(0)
            x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
            y_ref  = ref_tensor(x, q_type)
            data, scale = encode_tensor(x, e, m, is_signed)
            y_cuda = decode_tensor(data, scale, x.numel(), e, m, is_signed).reshape(shape)
            err = (y_cuda - y_ref).abs().max().item()
            ok  = (err == 0.0)
            print(f"  {'PASS' if ok else 'FAIL':4s}  {q_type:10s} shape={str(shape):28s} "
                  f"max_err={err:.3e}")
            pass_n += int(ok); fail_n += int(not ok)
    return pass_n, fail_n


def _run_chunk(formats, shapes):
    print("=" * 88)
    print("  chunk mode (chunk_size = 128)")
    print("=" * 88)
    pass_n = fail_n = 0
    for q_type in formats:
        e, m, is_signed = resolve_format(q_type)
        for shape in shapes:
            torch.manual_seed(0)
            x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
            y_ref  = ref_chunk(x, q_type, chunk_size=128)
            data, scales = encode_chunk(x, e, m, is_signed)
            y_cuda = decode_chunk(data, scales, x.numel(), e, m, is_signed).reshape(shape)
            err = (y_cuda - y_ref).abs().max().item()
            ok  = (err == 0.0)
            print(f"  {'PASS' if ok else 'FAIL':4s}  {q_type:10s} shape={str(shape):28s} "
                  f"max_err={err:.3e}")
            pass_n += int(ok); fail_n += int(not ok)
    return pass_n, fail_n


def _run_channel(formats, shapes, channel_dim=CHANNEL_DIM):
    print("=" * 88)
    print(f"  channel mode (channel_dim = {channel_dim})")
    print("=" * 88)
    pass_n = fail_n = 0
    for q_type in formats:
        e, m, is_signed = resolve_format(q_type)
        for shape in shapes:
            torch.manual_seed(0)
            x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
            y_ref = ref_channel(x, q_type, channel_dim=channel_dim)
            data, scales = encode_channel(x, channel_dim, e, m, is_signed)
            y_cuda = decode_channel(data, scales, list(x.shape), channel_dim, e, m, is_signed)
            err = (y_cuda - y_ref).abs().max().item()
            ok  = (err == 0.0)
            print(f"  {'PASS' if ok else 'FAIL':4s}  {q_type:10s} shape={str(shape):28s} "
                  f"max_err={err:.3e}")
            pass_n += int(ok); fail_n += int(not ok)
    return pass_n, fail_n


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Note   : codec is bit-exact against quantize_fp_generic; "
          f"every case must report max_err = 0.0e+00.")

    pt, ft = _run_tensor (FORMATS_DEFAULT, SHAPES_TENSOR)
    pc, fc = _run_chunk  (FORMATS_DEFAULT, SHAPES_CHUNK)
    ph, fh = _run_channel(FORMATS_DEFAULT, SHAPES_CHANNEL)

    pass_total = pt + pc + ph
    fail_total = ft + fc + fh
    print("=" * 88)
    print(f"TOTAL  pass={pass_total}  fail={fail_total}")
    print("=" * 88)
    sys.exit(0 if fail_total == 0 else 1)


if __name__ == "__main__":
    main()
