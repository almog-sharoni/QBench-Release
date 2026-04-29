"""test_lp_performance.py

Wall-clock comparison between the Python `quantize_tensor` reference path
and the CUDA encode + decode round trip exposed by the Rev3 codec.

For each (mode, q_type, shape) triple the test runs:

    Reference   :  y = quantize_tensor(x, q_type=q_type, mode=mode)[0]
    CUDA round  :  data, s = encode_*(x, ...);  y = decode_*(data, s, ...)

Both produce a dequantized FP32 tensor; the timed work is identical at the
mathematical level, so the ratio reflects implementation overhead only.

Timing protocol:
    1. one untimed warmup pass.
    2. WARMUP iterations whose timing is discarded.
    3. ITERS timed iterations bracketed by torch.cuda.Event.

Output is a small table per mode.  Run from the runspace root:

    ./apptainer.sh python3.13 runspace/src/quantization/cuda/test/test_lp_performance.py
    ./apptainer.sh python3.13 runspace/src/quantization/cuda/test/test_lp_performance.py --iters 200 --formats fp8_e4m3
"""

from __future__ import annotations

import argparse
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runspace.src.quantization.cuda import (
    encode_tensor,  decode_tensor,
    encode_chunk,   decode_chunk,
    encode_channel, decode_channel,
    resolve_format,
)
from runspace.src.ops.quant_base import quantize_tensor


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
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

DEFAULT_FORMATS = _SIGNED # dict union; fallback to _SIGNED for Python < 3.9
DEFAULT_FORMATS.update(_UNSIGNED)

TENSOR_SHAPES  = [(256, 512), (1024, 1024), (4096, 4096)]
CHUNK_SHAPES   = [(256, 512), (1024, 1024), (4096, 4096)]
CHANNEL_SHAPES = [(256, 512), (1024, 1024), (4096, 4096)]

CHUNK_SIZE  = 128
CHANNEL_DIM = 1                  # fixed by the Python reference path
DEVICE      = "cuda"
DTYPE       = torch.float32

WARMUP_DEFAULT = 10
ITERS_DEFAULT  = 100


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _unpack(ret):
    """quantize_tensor returns (input_fp8, max_val) by default; we need the first."""
    return ret[0] if isinstance(ret, tuple) else ret


def time_callable(fn, warmup: int, iters: int) -> float:
    """Return median wall time in milliseconds over `iters` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def make_tensor(shape, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=DTYPE, device=DEVICE).contiguous()


# ----------------------------------------------------------------------------
# Per-mode benchmarks
# ----------------------------------------------------------------------------

_HEADER = (f"  {'q_type':<10s} {'shape':<18s}  "
           f"{'ref ms':>10s}  {'cuda ms':>10s}  {'speedup':>8s}")


def bench_tensor(formats, shapes, warmup, iters):
    print("\n=== tensor mode ===")
    print(_HEADER)
    for q_type in formats:
        e, m, is_signed = resolve_format(q_type)
        for shape in shapes:
            x = make_tensor(shape)
            n = x.numel()

            ref_fn = lambda: _unpack(quantize_tensor(x, q_type=q_type, mode="tensor"))
            def cuda_fn():
                data, s = encode_tensor(x, e, m, is_signed)
                _ = decode_tensor(data, s, n, e, m, is_signed)

            t_ref  = time_callable(ref_fn,  warmup, iters)
            t_cuda = time_callable(cuda_fn, warmup, iters)
            print(f"  {q_type:<10s} {str(tuple(shape)):<18s}  "
                  f"{t_ref:>10.3f}  {t_cuda:>10.3f}  {t_ref / t_cuda:>7.2f}x")


def bench_chunk(formats, shapes, warmup, iters):
    print("\n=== chunk mode ===")
    print(_HEADER)
    for q_type in formats:
        e, m, is_signed = resolve_format(q_type)
        for shape in shapes:
            x = make_tensor(shape)
            n = x.numel()

            ref_fn = lambda: _unpack(quantize_tensor(
                x, q_type=q_type, mode="chunk", chunk_size=CHUNK_SIZE))
            def cuda_fn():
                data, scales = encode_chunk(x, e, m, is_signed)
                _ = decode_chunk(data, scales, n, e, m, is_signed)

            t_ref  = time_callable(ref_fn,  warmup, iters)
            t_cuda = time_callable(cuda_fn, warmup, iters)
            print(f"  {q_type:<10s} {str(tuple(shape)):<18s}  "
                  f"{t_ref:>10.3f}  {t_cuda:>10.3f}  {t_ref / t_cuda:>7.2f}x")


def bench_channel(formats, shapes, warmup, iters):
    print(f"\n=== channel mode (channel_dim = {CHANNEL_DIM}) ===")
    print(_HEADER)
    for q_type in formats:
        e, m, is_signed = resolve_format(q_type)
        for shape in shapes:
            x = make_tensor(shape)

            ref_fn = lambda: _unpack(quantize_tensor(x, q_type=q_type, mode="channel"))
            def cuda_fn():
                data, scales = encode_channel(x, CHANNEL_DIM, e, m, is_signed)
                _ = decode_channel(data, scales, list(x.shape), CHANNEL_DIM, e, m, is_signed)

            t_ref  = time_callable(ref_fn,  warmup, iters)
            t_cuda = time_callable(cuda_fn, warmup, iters)
            print(f"  {q_type:<10s} {str(tuple(shape)):<18s}  "
                  f"{t_ref:>10.3f}  {t_cuda:>10.3f}  {t_ref / t_cuda:>7.2f}x")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS)
    parser.add_argument("--warmup",  type=int, default=WARMUP_DEFAULT)
    parser.add_argument("--iters",   type=int, default=ITERS_DEFAULT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        sys.exit(2)

    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"Formats : {args.formats}")
    print(f"Warmup  : {args.warmup}    Iters : {args.iters}")

    bench_tensor (args.formats, TENSOR_SHAPES,  args.warmup, args.iters)
    bench_chunk  (args.formats, CHUNK_SHAPES,   args.warmup, args.iters)
    bench_channel(args.formats, CHANNEL_SHAPES, args.warmup, args.iters)


if __name__ == "__main__":
    main()
