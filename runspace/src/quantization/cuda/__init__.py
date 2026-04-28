# runspace/src/quantization/cuda/__init__.py
"""JIT loader for QBench FP8 CUDA kernels."""

import os
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

_module = load(
    name='qbench_fp8_cuda',
    sources=[
        os.path.join(_HERE, 'fp8_encode.cu'),
        os.path.join(_HERE, 'fp8_decode.cu'),
    ],
    extra_cuda_cflags=['-O3', '-lineinfo'],
    verbose=True,
)

# Per-chunk codec.
encode_fp8_chunk = _module.encode_fp8_chunk
decode_fp8_chunk = _module.decode_fp8_chunk

# Per-tensor codec (scalar scale).
encode_fp8_tensor    = _module.encode_fp8_tensor
decode_fp8_tensor    = _module.decode_fp8_tensor

# Per-channel codec (one scale per channel of [C, K] input).
encode_fp8_channel   = _module.encode_fp8_channel
decode_fp8_channel   = _module.decode_fp8_channel

# ARU variants (Always Round Up — round-half-up, no sticky bit).
encode_fp8_chunk_ARU    = _module.encode_fp8_chunk_ARU
encode_fp8_tensor_ARU   = _module.encode_fp8_tensor_ARU
encode_fp8_channel_ARU  = _module.encode_fp8_channel_ARU

# nf variants (No subnormal Flush — RNTE rounding).
encode_fp8_chunk_nf     = _module.encode_fp8_chunk_nf
encode_fp8_tensor_nf    = _module.encode_fp8_tensor_nf
encode_fp8_channel_nf   = _module.encode_fp8_channel_nf

# ARU_nf variants (Always Round Up + No subnormal Flush).
encode_fp8_chunk_ARU_nf    = _module.encode_fp8_chunk_ARU_nf
encode_fp8_tensor_ARU_nf   = _module.encode_fp8_tensor_ARU_nf
encode_fp8_channel_ARU_nf  = _module.encode_fp8_channel_ARU_nf