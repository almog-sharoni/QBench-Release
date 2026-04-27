"""
Performance: QFP8Tensor CUDA path vs quantize_fp_generic PyTorch path.

Both paths produce the same FP32 output bit-exactly under matching bias
(verified by Phase 0 gate). This script times the round trip across
several tensor sizes.

What is timed:
  CUDA   : x -> encode_fp8_emb_chunk -> decode_fp8_emb_chunk -> x_recon
  Python : x -> chunk amax -> pow2_floor scale -> divide -> quantize_fp_generic
                -> multiply -> x_recon

Both round trips include scale computation. The CUDA path stores the
intermediate as uint8 + scales (~1.03 bytes/elem); the Python path keeps
everything in FP32 throughout. Time is the metric, memory footprint is
separately a 4x storage advantage for CUDA.

Run:
    ./apptainer.sh python runspace/src/quantization/cuda_tests/perf_test.py
"""

import torch
from runspace.src.quantization.qfp8_tensor import QFP8Tensor
from runspace.src.quantization.quantizer import quantize_fp_generic


# ---------------------------------------------------------------- helpers

def pow2_floor(amax: torch.Tensor) -> torch.Tensor:
    """Match the kernel's pow2_floor_nonneg in Python."""
    bits = amax.view(torch.int32) & 0x7F800000
    bits = torch.where(amax == 0, torch.full_like(bits, 0x3F800000), bits)
    return bits.view(torch.float32)


def cuda_round_trip(x, e, m, b, K=128):
    q = QFP8Tensor.from_float(x, e=e, m=m, b=b, mode='chunk', chunk_size=K)
    return q.to_float()


def python_round_trip(x, e, m, K=128):
    chunks = x.view(-1, K)
    amax   = chunks.abs().amax(dim=1)
    s      = pow2_floor(amax)
    y      = chunks / s.unsqueeze(1)
    y_q    = quantize_fp_generic(y.flatten(), e, m).view_as(chunks)
    return (y_q * s.unsqueeze(1)).flatten()


def time_fn(fn, n_iters, n_warmup=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters    # ms per iter


# ---------------------------------------------------------------- main

def main():
    e, m, b = 4, 3, 15                       # E4M3, QBench bias convention
    K = 128
    n_iters = 200

    sizes = [
        ('4 K',      1 <<  12),               # 16 KB FP32
        ('64 K',     1 <<  16),               # 256 KB
        ('1 M',      1 <<  20),               # 4 MB
        ('4 M',      1 <<  22),               # 16 MB
        ('16 M',     1 <<  24),               # 64 MB
        ('64 M',     1 <<  26),               # 256 MB
    ]

    # 1. Verify bit-exact on a small case before timing.
    torch.manual_seed(0)
    x_v = torch.randn(8192, device='cuda', dtype=torch.float32) * 2.0
    x_cu = cuda_round_trip(x_v, e, m, b, K)
    x_py = python_round_trip(x_v, e, m, K)
    assert torch.equal(x_cu, x_py), 'numerical mismatch between paths'
    print('bit-exact verification on 8192 elements: OK')
    print()

    # 2. Benchmark across sizes.
    print(f'{"size":>6}  {"N":>10}  {"CUDA (ms)":>10}  {"Python (ms)":>12}'
          f'  {"speedup":>8}  {"CUDA GB/s":>10}')
    print('-' * 70)

    for label, N in sizes:
        torch.manual_seed(0)
        x = torch.randn(N, device='cuda', dtype=torch.float32) * 2.0

        t_cu = time_fn(lambda: cuda_round_trip(x, e, m, b, K),  n_iters)
        t_py = time_fn(lambda: python_round_trip(x, e, m, K),   n_iters)

        # CUDA round trip moves ~10 bytes/element through DRAM:
        #   encode: read 4 (FP32) + write 1 (uint8) + write ~0.03 (scales)
        #   decode: read 1 (uint8) + read ~0.03 (scales) + write 4 (FP32)
        bytes_per_elem = 4 + 1 + 1 + 4
        gbs_cu = (N * bytes_per_elem / 1e9) / (t_cu / 1000.0)

        print(f'{label:>6}  {N:>10}  {t_cu:>10.3f}  {t_py:>12.3f}'
              f'  {t_py / t_cu:>7.2f}x  {gbs_cu:>9.1f}')

    print()
    print('Notes:')
    print('  * CUDA round trip = encode + decode through 1-byte/elem storage.')
    print('  * Python round trip = chunk normalize + quantize_fp_generic + denormalize,')
    print('    all in FP32 throughout.')
    print('  * H100 LPDDR5 peak ~ 3 TB/s; CUDA throughput is DRAM-bound for large N.')


if __name__ == '__main__':
    main()
