# scratch_dispatch.py
import torch
from runspace.src.quantization.quantizer import quantize_fp_generic

torch.manual_seed(0)
N = 8192

# CUDA input: hits the fast path.
x_gpu = torch.randn(N, device='cuda', dtype=torch.float32) * 1.5
y_cuda_path = quantize_fp_generic(x_gpu, exp_bits=4, mant_bits=3)

# Same input forced through the Python fallback.
import os
os.environ['QBENCH_DISABLE_CUDA_QUANTIZE'] = '1'
y_python_path = quantize_fp_generic(x_gpu, exp_bits=4, mant_bits=3)
del os.environ['QBENCH_DISABLE_CUDA_QUANTIZE']

print('bit-exact:', torch.equal(y_cuda_path, y_python_path))
print('max diff :', (y_cuda_path - y_python_path).abs().max().item())

# CPU input: never hits the fast path.
x_cpu = x_gpu.cpu()
y_cpu = quantize_fp_generic(x_cpu, exp_bits=4, mant_bits=3)
print('CPU still works:', y_cpu.shape == x_cpu.shape)