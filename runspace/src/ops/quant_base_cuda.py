"""
CUDA-backed drop-in replacement for `quant_base.quantize_tensor`.

Uses the `encode_fp8_*_ARU_nf` device functions, which (post the
fp8_codec.cuh subnormal fix) bit-exactly mirror Python `quantize_fp_generic`.
The semantics for tensor / channel / chunk modes match `quantize_tensor`:

  - Same `calculate_scale` (pow2_floor of clamped amax) for tensor/channel.
  - Same dim-1-as-channel convention for `mode='channel'` (the CUDA channel
    kernel's row-as-channel layout is sidestepped by pre-dividing on host
    and calling `encode_fp8_tensor_ARU_nf` instead).
  - Same chunk_size=128 power-of-two per-chunk scale for `mode='chunk'`.

Not supported (raises NotImplementedError):
  - `mode='dynamic_oracle'` and `q_type` starting with `'dynamic_oracle'`
  - `chunk_formats` per-chunk format selection
  - `chunk_size` other than 128

`rounding` and `validate` are accepted for API parity but ignored — the
encoder is fixed at ARU + no-flush.
"""
from __future__ import annotations

import torch

from ..quantization.constants import get_format_params, get_quantization_bias
from ..quantization.cuda import (
    encode_fp8_tensor_ARU_nf, decode_fp8_tensor,
    encode_fp8_chunk_ARU_nf,  decode_fp8_chunk,
)
from .quant_base import calculate_scale


CHUNK_SIZE = 128


def _to_cuda_float32_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        t = t.float()
    if not t.is_cuda:
        t = t.cuda()
    return t.contiguous()


def _encode_decode_tensor(x_flat_cuda: torch.Tensor, scale_val: float,
                          e: int, m: int, b: int) -> torch.Tensor:
    """Run `encode_fp8_tensor_ARU_nf` + `decode_fp8_tensor` on a 1-D float32
    CUDA tensor. Returns the dequantized 1-D CUDA tensor."""
    N = x_flat_cuda.numel()
    encoded = torch.empty(N, dtype=torch.uint8,   device='cuda')
    decoded = torch.empty(N, dtype=torch.float32, device='cuda')
    encode_fp8_tensor_ARU_nf(x_flat_cuda, encoded, scale_val, e, m, b)
    decode_fp8_tensor(encoded, decoded, scale_val, e, m, b)
    return decoded


def quantize_tensor_cuda(
    input: torch.Tensor,
    q_type: str = 'fp8_e4m3',
    return_unscaled: bool = False,
    return_scale: bool = False,
    mode: str = 'tensor',
    chunk_size: int | None = None,
    rounding: str = 'nearest',     # accepted for parity, ignored
    validate: bool = False,        # accepted for parity, ignored
    chunk_formats=None,
):
    """CUDA-backed `quantize_tensor`. See module docstring for caveats."""
    if q_type == 'fp32':
        max_val = input.abs().max()
        scale = torch.tensor(1.0, device=input.device)
        if return_unscaled and return_scale:
            return input, input, max_val, scale, scale
        if return_unscaled:
            return input, input, max_val
        if return_scale:
            return input, scale, scale
        return input, max_val

    if q_type.startswith('dynamic_oracle') or mode == 'dynamic_oracle':
        raise NotImplementedError("dynamic_oracle is not supported by quantize_tensor_cuda")
    if chunk_formats is not None:
        raise NotImplementedError("per-chunk `chunk_formats` is not supported by quantize_tensor_cuda")

    e, m = get_format_params(q_type)
    b    = get_quantization_bias(q_type)

    # =============================================================
    # chunk mode: kernel computes its own per-chunk pow2 scale
    # =============================================================
    if mode == 'chunk':
        cs = chunk_size if chunk_size is not None else CHUNK_SIZE
        if cs != CHUNK_SIZE:
            raise NotImplementedError(
                f"CUDA chunk kernel is hardcoded to chunk_size={CHUNK_SIZE}; got {cs}")

        orig_shape = input.shape
        if input.dim() > 1:
            flat_input = input.flatten(1)            # [B, K]
            batch_size = input.shape[0]
        else:
            flat_input = input.unsqueeze(0)          # [1, K]
            batch_size = 1

        num_elements = flat_input.shape[-1]
        pad_len = (-num_elements) % cs
        if pad_len:
            flat_input = torch.nn.functional.pad(flat_input, (0, pad_len))

        x_cuda = _to_cuda_float32_contig(flat_input.reshape(-1))
        N = x_cuda.numel()
        encoded = torch.empty(N,        dtype=torch.uint8,   device='cuda')
        scales  = torch.empty(N // cs,  dtype=torch.float32, device='cuda')
        decoded = torch.empty(N,        dtype=torch.float32, device='cuda')
        encode_fp8_chunk_ARU_nf(x_cuda, encoded, scales, e, m, b, cs)
        decode_fp8_chunk(encoded, scales, decoded, e, m, b, cs)

        # Reshape, drop padding, restore original shape, return on input's device
        decoded = decoded.view(batch_size, -1)
        if pad_len:
            decoded = decoded[:, :num_elements]
        quantized = decoded.reshape(orig_shape).to(input.device)

        # Build a broadcastable scale tensor (one scale per chunk, expanded
        # to the original element layout). `s_packed` keeps the per-chunk vector.
        s_packed = scales.view(batch_size, -1)       # [B, num_chunks]
        s_expanded = (s_packed.unsqueeze(-1)
                              .expand(-1, -1, cs)
                              .reshape(batch_size, -1))
        if pad_len:
            s_expanded = s_expanded[:, :num_elements]
        s = s_expanded.reshape(orig_shape).to(input.device)
        s_packed = s_packed.to(input.device)

        # Match `quantize_tensor` chunk path: max over per-chunk input amaxes
        # (clamped to 1e-9). Equivalent to `input.abs().max()` for any
        # non-trivial input.
        max_val = input.abs().max().clamp(min=1e-9)
        if return_unscaled and return_scale:
            return quantized, quantized, max_val, s, s_packed
        if return_unscaled:
            return quantized, quantized, max_val
        if return_scale:
            return quantized, s, s_packed
        return quantized, max_val

    # =============================================================
    # tensor / channel mode: host computes scale (pow2_floor of amax)
    # =============================================================
    if mode == 'channel':
        if input.dim() >= 2:
            reduce_dims = tuple(d for d in range(input.dim()) if d != 1)
        else:
            reduce_dims = tuple(range(input.dim()))
    else:  # 'tensor'
        reduce_dims = tuple(range(input.dim()))

    max_val = input.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-5)
    s = calculate_scale(max_val, q_type)             # broadcastable to input.shape

    # Pre-divide on host so we can use the scalar-scale tensor encoder for
    # both modes (sidesteps the channel kernel's row-as-channel convention).
    input_scaled = input / s
    x_cuda = _to_cuda_float32_contig(input_scaled.reshape(-1))
    decoded_flat = _encode_decode_tensor(x_cuda, 1.0, e, m, b)
    input_fp8_unscaled = decoded_flat.view(input.shape).to(input.device)
    input_fp8 = input_fp8_unscaled * s

    if return_unscaled:
        if return_scale:
            return input_fp8, input_fp8_unscaled, max_val, s, s
        return input_fp8, input_fp8_unscaled, max_val
    if return_scale:
        return input_fp8, s, s
    return input_fp8, max_val
