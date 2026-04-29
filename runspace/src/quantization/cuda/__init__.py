# runspace/src/quantization/cuda/__init__.py
#
# JIT-loaded CUDA extension exposing the QBench low-precision codec.
#
# Element width is w = signed_bit + e + m bits, with signed_bit = 1 for
# signed formats and 0 for unsigned formats.  Supported range 2 <= w <= 16.
# Each uint32 storage word packs n_per_word(w) = floor(32 / w) consecutive
# elements; remaining low bits are zero padding.
#
# Format keys follow the conventions:
#   * Signed   formats begin with 'fp':   fp4_e2m1, fp8_e4m3, ...
#   * Unsigned formats begin with 'ufp':  ufp4_e2m2, ufp8_e4m4, ...

from __future__ import annotations
import os
from torch.utils.cpp_extension import load

_THIS = os.path.dirname(os.path.abspath(__file__))

_ext = load(
    name="qbench_lp_codec",
    sources=[
        os.path.join(_THIS, "ops_host.cpp"),
        os.path.join(_THIS, "ops_tensor.cu"),
        os.path.join(_THIS, "ops_chunk.cu"),
        os.path.join(_THIS, "ops_channel.cu"),
    ],
    extra_include_paths=[_THIS],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

encode_tensor  = _ext.encode_tensor
decode_tensor  = _ext.decode_tensor
encode_chunk   = _ext.encode_chunk
decode_chunk   = _ext.decode_chunk
encode_channel = _ext.encode_channel
decode_channel = _ext.decode_channel

roundtrip_tensor  = _ext.roundtrip_tensor
roundtrip_chunk   = _ext.roundtrip_chunk
roundtrip_channel = _ext.roundtrip_channel

n_per_word     = _ext.n_per_word
element_width  = _ext.element_width


# ----------------------------------------------------------------------------
# Format tables (generated)
# ----------------------------------------------------------------------------
#
# Width range: 2 <= w <= 16.  Signed widths satisfy 1 + e + m in [2, 16];
# unsigned widths satisfy e + m in [2, 16].  In both tables we require
# e >= 1 (an exponent field of width 0 is degenerate) and m >= 0.

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

def resolve_format(q_type: str) -> tuple[int, int, bool]:
    """Return (e, m, is_signed) for a q_type string."""
    if q_type in _SIGNED:
        e, m = _SIGNED[q_type]
        return e, m, True
    if q_type in _UNSIGNED:
        e, m = _UNSIGNED[q_type]
        return e, m, False
    raise ValueError(
        f"unknown q_type {q_type!r}; valid keys: "
        f"{sorted(_SIGNED) + sorted(_UNSIGNED)}"
    )


__all__ = [
    "encode_tensor", "decode_tensor",
    "encode_chunk",  "decode_chunk",
    "encode_channel", "decode_channel",
    "roundtrip_tensor", "roundtrip_chunk", "roundtrip_channel",
    "resolve_format",
    "n_per_word", "element_width",
]