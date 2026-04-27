# runspace/src/quantization/qfp8_tensor.py
"""
Storage container for FP8(e, m, b) tensors. Implements plan §2.1.

Three modes:
  'chunk'  : per-chunk power-of-two scale, chunk_size = 128. Default for
             activations between matmul ops (plan §2.4).
  'tensor' : single power-of-two scale for the whole tensor. Default for
             weights (plan §2.4 weight_mode = tensor).
  'channel': per-channel power-of-two scale. Optional weight mode where
             a `channel_dim` is given and one scale is allocated per slice
             along that axis.

Footprint:
  chunk   : 1 + 4/128 = 1.03125 bytes/element
  tensor  : 1 + 4/N   ~ 1 byte/element for any reasonable N
  channel : 1 + 4*C/N where C is the channel count

Scales are always FP32 powers of two, computed via pow2_floor of the
chunk/tensor/channel amax. The same convention applies in all three modes,
so the kernel encode_fp8_emb device function is shared across all of them.
"""

from typing import Tuple, Optional, List
import torch

from .cuda import (
    encode_fp8_emb_chunk, decode_fp8_emb_chunk,
    encode_fp8_tensor,    decode_fp8_tensor,
    encode_fp8_channel,   decode_fp8_channel,
)


_VALID_MODES = ('chunk', 'tensor', 'channel')


def _pow2_floor(amax: torch.Tensor) -> torch.Tensor:
    """Bitwise pow2_floor matching the kernel's pow2_floor_nonneg."""
    bits = amax.view(torch.int32) & 0x7F800000
    bits = torch.where(amax == 0, torch.full_like(bits, 0x3F800000), bits)
    return bits.view(torch.float32)


class QFP8Tensor:
    """FP8(e, m, b) tensor with associated scales."""

    def __init__(
        self,
        storage_uint8:  torch.Tensor,
        scales:         torch.Tensor,
        e:              int,
        m:              int,
        b:              int,
        mode:           str            = 'chunk',
        chunk_size:     int            = 128,
        original_shape: Optional[Tuple[int, ...]]    = None,
        original_dtype: torch.dtype                  = torch.float32,
        channel_perm:   Optional[List[int]]          = None,
    ):
        assert 1 + e + m == 8, f'1 + e + m must be 8, got {1 + e + m}'
        assert mode in _VALID_MODES, f'mode must be one of {_VALID_MODES}'
        assert storage_uint8.dtype == torch.uint8
        assert scales.dtype        == torch.float32
        assert storage_uint8.device == scales.device
        assert storage_uint8.is_contiguous() and scales.is_contiguous()

        self.storage_uint8  = storage_uint8
        self.scales         = scales
        self.e              = e
        self.m              = m
        self.b              = b
        self.mode           = mode
        self.chunk_size     = chunk_size
        self.original_shape = (tuple(original_shape) if original_shape is not None
                               else (storage_uint8.numel(),))
        self.original_dtype = original_dtype
        # For channel mode, this is the permutation that moved channel_dim to
        # axis 0 during from_float. Inverse is applied in to_float.
        self.channel_perm   = list(channel_perm) if channel_perm is not None else None

    # ------------------------------------------------------------------ ctors

    @classmethod
    def from_float(
        cls,
        x:           torch.Tensor,
        e:           int,
        m:           int,
        b:           int,
        mode:        str = 'chunk',
        chunk_size:  int = 128,
        channel_dim: Optional[int] = None,
    ) -> 'QFP8Tensor':
        """Encode an FP32 (or castable) CUDA tensor."""
        assert x.is_cuda, 'QFP8Tensor.from_float requires a CUDA tensor'
        original_shape = tuple(x.shape)
        original_dtype = x.dtype
        x_fp32 = x.to(torch.float32) if x.dtype != torch.float32 else x

        if mode == 'chunk':
            return cls._from_float_chunk(
                x_fp32, e, m, b, chunk_size, original_shape, original_dtype)
        if mode == 'tensor':
            return cls._from_float_tensor(
                x_fp32, e, m, b, original_shape, original_dtype)
        if mode == 'channel':
            if channel_dim is None:
                raise ValueError('channel mode requires channel_dim argument')
            return cls._from_float_channel(
                x_fp32, e, m, b, channel_dim, original_shape, original_dtype)
        raise ValueError(f'unknown mode {mode!r}')

    # --------------------------------------------------------- chunk mode

    @classmethod
    def _from_float_chunk(cls, x, e, m, b, chunk_size, original_shape, original_dtype):
        x_flat = x.contiguous().flatten()
        N = x_flat.numel()
        if N % chunk_size != 0:
            raise ValueError(
                f'chunk mode: N={N} not a multiple of chunk_size={chunk_size}')
        storage = torch.empty(N,                 dtype=torch.uint8,   device=x.device)
        scales  = torch.empty(N // chunk_size,   dtype=torch.float32, device=x.device)
        encode_fp8_emb_chunk(x_flat, storage, scales, e, m, b, chunk_size)
        return cls(storage, scales, e, m, b, 'chunk', chunk_size,
                   original_shape, original_dtype)

    # -------------------------------------------------------- tensor mode

    @classmethod
    def _from_float_tensor(cls, x, e, m, b, original_shape, original_dtype):
        x_flat = x.contiguous().flatten()
        N = x_flat.numel()


        amax = x_flat.abs().max()
        s    = _pow2_floor(amax.unsqueeze(0)).squeeze(0)        # FP32 scalar tensor
        s_py = float(s.item())
        if s_py == 0.0:
            s_py = 1.0                                          # all-zero tensor

        storage = torch.empty(N, dtype=torch.uint8, device=x.device)
        encode_fp8_tensor(x_flat, storage, s_py, e, m, b)
        scales = torch.tensor([s_py], dtype=torch.float32, device=x.device)
        return cls(storage, scales, e, m, b, 'tensor', 1,
                   original_shape, original_dtype)

    # ------------------------------------------------------- channel mode

    @classmethod
    def _from_float_channel(cls, x, e, m, b, channel_dim, original_shape, original_dtype):
        ndim = x.dim()
        if not (-ndim <= channel_dim < ndim):
            raise ValueError(f'channel_dim {channel_dim} out of range for ndim={ndim}')
        channel_dim = channel_dim % ndim

        # Permute channel axis to 0, flatten the rest.
        perm = [channel_dim] + [i for i in range(ndim) if i != channel_dim]
        x_perm = x.permute(perm).contiguous()                   # [C, ...]
        C = x_perm.size(0)
        K = x_perm.numel() // C
        if K % 4 != 0:
            raise ValueError(
                f'channel mode: per-channel size K={K} must be a multiple of 4')
        x_2d = x_perm.view(C, K)

        # Per-channel pow2_floor scales.
        amax   = x_2d.abs().amax(dim=1)
        scales = _pow2_floor(amax)
        # If a whole channel is zero, _pow2_floor returns 1.0; the kernel's
        # divide-by-scale stays well-defined.

        storage_2d = torch.empty(C, K, dtype=torch.uint8, device=x.device).contiguous()
        encode_fp8_channel(x_2d, scales, storage_2d, e, m, b)
        return cls(storage_2d.flatten(), scales, e, m, b, 'channel', 1,
                   original_shape, original_dtype, channel_perm=perm)

    # ------------------------------------------------------------------ exit

    def to_float(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if   self.mode == 'chunk':   out = self._to_float_chunk()
        elif self.mode == 'tensor':  out = self._to_float_tensor()
        elif self.mode == 'channel': out = self._to_float_channel()
        else: raise NotImplementedError

        target = dtype if dtype is not None else self.original_dtype
        return out if target == torch.float32 else out.to(target)

    def _to_float_chunk(self) -> torch.Tensor:
        N = self.storage_uint8.numel()
        out = torch.empty(N, dtype=torch.float32, device=self.storage_uint8.device)
        decode_fp8_emb_chunk(
            self.storage_uint8, self.scales, out,
            self.e, self.m, self.b, self.chunk_size)
        return out.reshape(self.original_shape)

    def _to_float_tensor(self) -> torch.Tensor:
        N = self.storage_uint8.numel()
        out = torch.empty(N, dtype=torch.float32, device=self.storage_uint8.device)
        s_py = float(self.scales[0].item())
        decode_fp8_tensor(self.storage_uint8, out, s_py, self.e, self.m, self.b)
        return out.reshape(self.original_shape)

    def _to_float_channel(self) -> torch.Tensor:
        if self.channel_perm is None:
            raise RuntimeError('channel mode tensor missing channel_perm metadata')
        # Reconstruct (C, K) layout from flat storage.
        C = self.scales.numel()
        K = self.storage_uint8.numel() // C
        in_2d = self.storage_uint8.view(C, K)

        out_2d = torch.empty(C, K, dtype=torch.float32, device=self.storage_uint8.device)
        decode_fp8_channel(in_2d, self.scales, out_2d, self.e, self.m, self.b)

        # Reshape to permuted original shape, then invert the permutation.
        permuted_shape = [self.original_shape[i] for i in self.channel_perm]
        out_perm = out_2d.view(permuted_shape)
        inv_perm = [self.channel_perm.index(i) for i in range(len(self.channel_perm))]
        return out_perm.permute(inv_perm).contiguous()

    # --------------------------------------------------------- introspection

    @property
    def shape(self)  -> torch.Size:    return torch.Size(self.original_shape)
    @property
    def device(self) -> torch.device:  return self.storage_uint8.device
    @property
    def fmt(self)    -> str:           return f'fp8_e{self.e}m{self.m}_b{self.b}'

    def numel(self) -> int:            return self.storage_uint8.numel()

    @property
    def nbytes(self) -> int:
        return self.storage_uint8.numel() + self.scales.numel() * 4

    def fp32_nbytes(self) -> int:      return self.storage_uint8.numel() * 4

    def compression_ratio(self) -> float:
        return self.fp32_nbytes() / self.nbytes

    def __repr__(self) -> str:
        return (f'QFP8Tensor(shape={tuple(self.shape)}, fmt={self.fmt}, '
                f'mode={self.mode!r}, n_scales={self.scales.numel()}, '
                f'nbytes={self.nbytes}, compression={self.compression_ratio():.3f}x)')
