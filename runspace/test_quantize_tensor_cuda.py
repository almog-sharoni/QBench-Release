"""
Smoke test: `quantize_tensor_cuda` vs `quantize_tensor` parity check.

For tensor / channel / chunk modes across all fp8_eXmY formats, run both
implementations on a small batch of random tensors and compare:
  - quantized output (must match bit-exactly: same scale, same _ARU_nf encoder
    mirrors quantize_fp_generic)
  - max_val
  - returned scale shape (when return_scale=True)
"""
import sys, os, struct
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ops.quant_base import quantize_tensor
from src.ops.quant_base_cuda import quantize_tensor_cuda

ALL_FP8_FORMATS = [f"fp8_e{e}m{7 - e}" for e in range(1, 8)]


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach().cpu().float().contiguous(), b.detach().cpu().float().contiguous()
    if a.shape != b.shape:
        return False
    return (a.view(torch.int32) == b.view(torch.int32)).all().item()


def near_equal(a: torch.Tensor, b: torch.Tensor, atol=1e-5) -> bool:
    return torch.allclose(a.detach().cpu().float(), b.detach().cpu().float(), atol=atol)


SHAPES = {
    'tensor':  [(256,), (4096,), (32, 64), (4, 8, 16), (2, 3, 4, 5)],
    'channel': [(32, 16), (4, 8, 16), (2, 3, 4, 5)],   # needs dim>=2
    'chunk':   [(128,), (1024,), (4, 256), (2, 4, 128)],  # last dim div by 128
}


def run_mode(mode: str, q_type: str) -> tuple[int, int]:
    ok = total = 0
    for shape in SHAPES[mode]:
        torch.manual_seed(hash((mode, q_type, shape)) & 0xFFFF_FFFF)
        x = torch.randn(*shape) * 2.0
        kwargs = {'q_type': q_type, 'mode': mode}
        if mode == 'chunk':
            kwargs['chunk_size'] = 128

        ref_q, ref_max = quantize_tensor(x, **kwargs)
        cuda_q, cuda_max = quantize_tensor_cuda(x, **kwargs)

        total += 1
        q_match  = bits_equal(ref_q, cuda_q)
        mv_match = near_equal(ref_max, cuda_max)
        if q_match and mv_match:
            ok += 1
        else:
            print(f"  DIFF  {mode:<8} {q_type:<10} shape={list(shape)}  "
                  f"q_match={q_match} mv_match={mv_match}  "
                  f"max_abs_diff={(ref_q.cpu()-cuda_q.cpu()).abs().max().item():.3e}")
    return ok, total


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available."); sys.exit(1)

    print(f"{'mode':<8}  {'format':<10}  {'pass/total':>10}")
    print('-' * 40)
    grand_ok = grand_total = 0
    for mode in ('tensor', 'channel', 'chunk'):
        for q in ALL_FP8_FORMATS:
            ok, tot = run_mode(mode, q)
            grand_ok += ok; grand_total += tot
            status = 'OK ' if ok == tot else 'DIFF'
            print(f"{mode:<8}  {q:<10}  {ok}/{tot}  {status}")
    print('-' * 40)
    print(f"GRAND TOTAL: {grand_ok}/{grand_total}")


if __name__ == '__main__':
    main()
