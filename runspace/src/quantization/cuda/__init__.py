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
encode_fp8_emb_chunk = _module.encode_fp8_emb_chunk
decode_fp8_emb_chunk = _module.decode_fp8_emb_chunk
encode_fp8_emb_chunk_rhup         = _module.encode_fp8_emb_chunk_rhup
encode_fp8_emb_chunk_noflush      = _module.encode_fp8_emb_chunk_noflush
encode_fp8_emb_chunk_rhup_noflush = _module.encode_fp8_emb_chunk_rhup_noflush

# Per-tensor codec (scalar scale).
encode_fp8_tensor    = _module.encode_fp8_tensor
decode_fp8_tensor    = _module.decode_fp8_tensor
encode_fp8_tensor_rhup         = _module.encode_fp8_tensor_rhup
encode_fp8_tensor_noflush      = _module.encode_fp8_tensor_noflush
encode_fp8_tensor_rhup_noflush = _module.encode_fp8_tensor_rhup_noflush

# Per-channel codec (one scale per channel of [C, K] input).
encode_fp8_channel     = _module.encode_fp8_channel
decode_fp8_channel     = _module.decode_fp8_channel
encode_fp8_channel_rhup           = _module.encode_fp8_channel_rhup
encode_fp8_channel_noflush        = _module.encode_fp8_channel_noflush
encode_fp8_channel_rhup_noflush   = _module.encode_fp8_channel_rhup_noflush
