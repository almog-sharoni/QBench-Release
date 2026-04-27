"""
End-to-end FP8 storage workflow on a small MLP.

Demonstrates the plan §2.3 pattern:
  1. Encode FP32 weights once at model load time (per-tensor mode).
  2. Encode each activation to FP8 between layers (chunk mode).
  3. Inside each layer: decode operands, FP32 compute, encode output.
  4. Compare end-to-end output to a pure FP32 reference path.

The fp8_linear helper here is a stand-in. The real Phase 1 path will fuse
the decode prologue, FP32 GEMM, bias, and encode epilogue into one CUDA
kernel (gemm_fp8_fp32acc_encode, plan §2.2). The data flow is identical;
only the materialization of the FP32 intermediate to DRAM differs.

Run:
    ./apptainer.sh python runspace/src/quantization/cuda_tests/example_inference.py
"""

import torch
import torch.nn.functional as F
from runspace.src.quantization.qfp8_tensor import QFP8Tensor


def hr(title): print('\n' + '=' * 72 + '\n' + title + '\n' + '=' * 72)


# ============================================================================
# Helper: simulated FP8 Linear layer.
# ============================================================================

def fp8_linear(
    x_q:    QFP8Tensor,
    w_q:    QFP8Tensor,
    bias:   torch.Tensor,
    e: int = 4, m: int = 3, b: int = 15,
) -> QFP8Tensor:
    """
    Decode -> FP32 matmul -> encode. Output is per-chunk encoded for
    consumption by the next layer.
    """
    x_fp32 = x_q.to_float()
    w_fp32 = w_q.to_float()
    y_fp32 = F.linear(x_fp32, w_fp32, bias)
    return QFP8Tensor.from_float(y_fp32, e=e, m=m, b=b, mode='chunk')


# ============================================================================
# Build a small MLP.
# ============================================================================

torch.manual_seed(0)
DEV = 'cuda'

IN, HID, OUT, BATCH = 512, 1024, 128, 32

fc1_w = torch.randn(HID, IN,  device=DEV) * 0.02
fc1_b = torch.zeros(HID,      device=DEV)
fc2_w = torch.randn(OUT, HID, device=DEV) * 0.02
fc2_b = torch.zeros(OUT,      device=DEV)


# ============================================================================
# 1. Encode weights once. Plan §2.4 default: per-tensor mode for weights.
# ============================================================================

hr('1. Encoding weights (per-tensor mode)')

fc1_w_q = QFP8Tensor.from_float(fc1_w, e=4, m=3, b=15, mode='tensor')
fc2_w_q = QFP8Tensor.from_float(fc2_w, e=4, m=3, b=15, mode='tensor')

print(f'  fc1: {fc1_w_q}')
print(f'  fc2: {fc2_w_q}')

fp32_w_bytes = (fc1_w.numel() + fc2_w.numel()) * 4
fp8_w_bytes  = fc1_w_q.nbytes + fc2_w_q.nbytes
print(f'\n  weight memory: FP32 {fp32_w_bytes:>9,d} B  '
      f'-> FP8 {fp8_w_bytes:>9,d} B  ({fp32_w_bytes / fp8_w_bytes:.2f}x)')


# ============================================================================
# 2. Forward pass through the FP8 path.
#    Activations between layers are stored as chunk-mode FP8 (plan §2.4).
# ============================================================================

hr('2. FP8 forward pass')

x = torch.randn(BATCH, IN, device=DEV)

# Input activation -> chunk mode FP8.
x_q = QFP8Tensor.from_float(x, e=4, m=3, b=15, mode='chunk')

# fc1
h_q = fp8_linear(x_q, fc1_w_q, fc1_b)

# ReLU between the two layers. In the real fused kernel this lives in the
# encode epilogue; for the simulation we decode, apply, re-encode.
h_fp32 = F.relu(h_q.to_float())
h_q    = QFP8Tensor.from_float(h_fp32, e=4, m=3, b=15, mode='chunk')

# fc2
y_q = fp8_linear(h_q, fc2_w_q, fc2_b)
y   = y_q.to_float()

print(f'  x_q : {x_q}')
print(f'  h_q : {h_q}')
print(f'  y_q : {y_q}')
print(f'  output shape: {tuple(y.shape)}')


# ============================================================================
# 3. FP32 reference forward pass.
# ============================================================================

hr('3. FP32 reference forward pass')

y_ref = F.linear(F.relu(F.linear(x, fc1_w, fc1_b)), fc2_w, fc2_b)
print(f'  output shape: {tuple(y_ref.shape)}')


# ============================================================================
# 4. End-to-end accuracy.
# ============================================================================

hr('4. FP8 vs FP32 end-to-end')

err = (y - y_ref).abs()
cos = F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0).item()
l2r = ((y - y_ref).norm() / y_ref.norm()).item()

print(f'  max abs error    : {err.max().item():.6f}')
print(f'  mean abs error   : {err.mean().item():.6f}')
print(f'  cosine similarity: {cos:.6f}')
print(f'  L2 relative error: {l2r:.6f}')

# Plan §1.1 numerical fidelity gate is "per layer cosine similarity > 0.9999".
# This is end-to-end, two layers deep, so we expect a slightly lower number.
status = 'PASS' if cos > 0.9999 else 'review'
print(f'  cosine > 0.9999  : {status}')


# ============================================================================
# 5. Per-tensor vs per-channel for fc1 weight.
# ============================================================================

hr('5. Mode comparison on fc1 weight')

q_t = QFP8Tensor.from_float(fc1_w, e=4, m=3, b=15, mode='tensor')
q_c = QFP8Tensor.from_float(fc1_w, e=4, m=3, b=15, mode='channel', channel_dim=0)

err_t = (fc1_w - q_t.to_float()).abs()
err_c = (fc1_w - q_c.to_float()).abs()

print(f'  per-tensor   : nbytes={q_t.nbytes:>6,d}  '
      f'max err={err_t.max().item():.4f}  mean err={err_t.mean().item():.4f}')
print(f'  per-channel  : nbytes={q_c.nbytes:>6,d}  '
      f'max err={err_c.max().item():.4f}  mean err={err_c.mean().item():.4f}')

# For randn weights with similar per-channel ranges, per-tensor and
# per-channel give nearly identical accuracy. Per-channel pays off only when
# channel magnitudes vary by enough to push some channels into flush or
# saturation under per-tensor. See example_qfp8_modes.py Example 6.

print('\nDone.')
