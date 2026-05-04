from runspace.src.quantization.cuda import roundtrip_chunk
from runspace.src.quantization.cuda import roundtrip_channel
from runspace.src.quantization.cuda import roundtrip_tensor
import torch
try:
    import torch.fx
except ImportError:
    pass
from runspace.src.quantization.quantizer import quantize
from runspace.src.quantization.constants import get_format_params

def _get_reduce_dims(input: torch.Tensor):
    return tuple(range(input.dim()))


# ----------------------------------------------------------------------------
# CUDA fast path for quantize_tensor's standard mode={tensor,chunk,channel}.
# Lazy-imported because the codec extension is JIT-built on first import.
# ----------------------------------------------------------------------------

_CUDA_CODEC = None

def _cuda_codec():
    global _CUDA_CODEC
    if _CUDA_CODEC is None:
        from runspace.src.quantization.cuda import (
            roundtrip_tensor, roundtrip_chunk, roundtrip_channel,
            resolve_format,
        )
        _CUDA_CODEC = (
            roundtrip_tensor, roundtrip_chunk, roundtrip_channel,
            resolve_format,
        )
    return _CUDA_CODEC


def _can_use_cuda(input: torch.Tensor, q_type: str, mode: str, chunk_size):
    """Validate the standard-path preconditions for the CUDA codec.

    Returns (e, m, is_signed) on success; raises RuntimeError on any
    unsupported attribute.  The orthogonal special q_type paths (fp32,
    tf32, chunk_formats != None) are filtered out by the caller before
    this function is invoked.
    """
    if not input.is_cuda:
        raise RuntimeError(
            f"quantize_tensor: input must be on CUDA (got device {input.device}); "
            f"the CUDA codec is the only supported backend for the standard path."
        )
    if input.dtype != torch.float32:
        raise RuntimeError(
            f"quantize_tensor: input must be float32 (got {input.dtype})."
        )
    if mode not in ("tensor", "chunk", "channel"):
        raise RuntimeError(
            f"quantize_tensor: mode must be tensor/chunk/channel (got {mode!r})."
        )
    _, _, _, resolve_format = _cuda_codec()
    try:
        e, m, is_signed = resolve_format(q_type)
    except ValueError as exc:
        raise RuntimeError(
            f"quantize_tensor: q_type {q_type!r} is not supported by the CUDA "
            f"codec (efp/uefp formats and unknown q_types fall here): {exc}"
        ) from exc

    if mode == "chunk":
        if chunk_size != 128:
            raise RuntimeError(
                f"quantize_tensor: chunk mode requires chunk_size=128 "
                f"(CUDA codec hardcodes 128); got chunk_size={chunk_size}."
            )
    if mode == "channel" and input.dim() < 2:
        raise RuntimeError(
            f"quantize_tensor: channel mode requires input.dim() >= 2; "
            f"got shape {tuple(input.shape)}."
        )
    return e, m, is_signed


def _quantize_tensor_cuda(
    input: torch.Tensor,
    q_type: str,
    e: int, m: int, is_signed: bool,
    return_unscaled: bool,
    return_scale: bool,
    mode: str,
    chunk_size,
    validate: bool,
):
    """Codec-backed implementation of quantize_tensor's standard path.

    Returns the same variable-length tuple shape as the Python tail
    (lines below quantize_tensor's special-q_type branches).

    Behavioural note: the codec does not clamp non-zero amax (only
    amax==0 -> 1.0).  The Python tail clamps to min=1e-5 (tensor/channel)
    or min=1e-9 (chunk).  The two paths therefore diverge for tiny
    inputs with 0 < amax < 1e-5; for natural inputs (amax ~ 1) results
    are bit-exact.
    """
    roundtrip_tensor, roundtrip_chunk, roundtrip_channel, _ = _cuda_codec()

    x = input.contiguous()

    # `scale_b` (the input-shape broadcast scale) is only materialised
    # if a caller actually needs it (return_unscaled or return_scale).
    # For chunk mode this avoids an input-sized tensor allocation per
    # call (~64MB for 4096²); the codec's per-chunk scale is sufficient
    # everywhere else.  For tensor/channel modes scale_b is a view of the
    # codec's scale tensor (no allocation).
    needs_scale_b = return_unscaled or return_scale

    if mode == "tensor":
        input_fp8, scale_b, scale_p, max_val = roundtrip_tensor(x, e, m, is_signed)
    elif mode == "chunk":
        input_fp8, scale_b, scale_p, max_val = roundtrip_chunk(
            x, e, m, is_signed, needs_scale_b
        )
    else:  # channel, dim>=2, channel_dim=1
        input_fp8, scale_b, scale_p, max_val = roundtrip_channel(
            x, 1, e, m, is_signed
        )

    if validate:
        from runspace.src.quantization.quantizer import assert_fp8_valid
        assert_fp8_valid(input_fp8, q_type=q_type)

    if return_unscaled:
        input_fp8_unscaled = input_fp8 / scale_b
        if return_scale:
            return input_fp8, input_fp8_unscaled, max_val, scale_b, scale_p
        return input_fp8, input_fp8_unscaled, max_val
    if return_scale:
        return input_fp8, scale_b, scale_p
    return input_fp8, max_val

  

def calculate_scale(max_val: torch.Tensor, q_type: str):
    bias = 0
            
    if q_type in ['int8', 'int4']:
        # # power-of-two scale so it can use the same log2/floor/pow2 (shift) hardware
        # e = torch.floor(torch.log2(max_val)) - bias #TODO: Check this
        # # For int8: equivalently: e = floor(log2(max_val / 127.0))
        # # For int4: equivalently: e = floor(log2(max_val / 7.0))
        # s = torch.pow(2.0, e)
        raise NotImplementedError("int8 and int4 quantization not supported anymore")
    else:
        # User defined strategy: Scale to be exp of x_max (with no bias)
        # s = 2^(floor(log2(max_val)))
        # This normalizes the max value to the range [1, 2)
        e = torch.floor(torch.log2(max_val))
        s = torch.pow(2, e)
    return s


def quantize_tensor(
    input: torch.Tensor, 
    q_type: str = 'fp8_e4m3', 
    return_unscaled: bool = False, 
    return_scale: bool = False, 
    mode: str = 'tensor', 
    chunk_size: int | None = None, 
    validate: bool = False, 
    chunk_formats: list[str] | None = None
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] |
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] |
    tuple[torch.Tensor, torch.Tensor]
):
    """
    Quantizes a tensor to FP8 or other supported formats.
    Args:
        input: Input tensor
        q_type: Default quantization type
        return_unscaled: If True, returns (scaled_input, unscaled_input)
        return_scale: If True, returns (quantized_tensor, scale) or (quantized_tensor, unscaled, scale)
        mode: Quantization mode ('tensor', 'channel', 'chunk')
        chunk_size: Size of chunk for 'chunk' mode
        rounding: Rounding mode ('nearest')
        validate: If True, perform expensive FP8 validation checks
        chunk_formats: Optional list of formats per chunk. If provided, overrides q_type for chunk mode.
    Returns:
        Quantized tensor (scaled back to original range)
    """
    # # Check for FP8 support
    # if q_type == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
    #         raise RuntimeError("FP8 E4M3 support (torch.float8_e4m3fn) is required.")
    # if q_type == 'fp8_e5m2' and not hasattr(torch, 'float8_e5m2'):
    #         raise RuntimeError("FP8 E5M2 support (torch.float8_e5m2) is required.")
    # FP4 types are manually implemented, so no check needed
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
    
    # Standard path (mode ∈ {tensor, chunk, channel}, single q_type, no
    # chunk_formats list): the CUDA codec is mandatory.
    if chunk_formats is not None:
        raise NotImplementedError(
            "quantize_tensor: per-chunk chunk_formats path was removed and is "
            "not currently implemented."
        )
    if mode not in ('tensor', 'chunk', 'channel'):
        raise ValueError(f'Unknown mode {mode}')
    e, m, is_signed = _can_use_cuda(input, q_type, mode, chunk_size)
    return _quantize_tensor_cuda(
        input, q_type, e, m, is_signed,
        return_unscaled, return_scale,
        mode, chunk_size,
        validate,
    )




class QuantizedLayerMixin:
    """
    Mixin for quantized layers (Conv2d, Linear, BatchNorm2d) that handles:
    1. Weight calibration (per-output-channel)
    2. Input quantization (per-channel/feature)
    """
    
    def calibrate_weights(self):
        """
        Calibrates weights: computes scale per output channel and quantizes weights.
        """
        # Check support based on q_type
        q_type = getattr(self, 'q_type', 'fp8_e4m3')
        
        # if q_type == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
        #      raise RuntimeError("FP8 E4M3 support (torch.float8_e4m3fn) is required for calibration.")
        # if q_type == 'fp8_e5m2' and not hasattr(torch, 'float8_e5m2'):
        #      raise RuntimeError("FP8 E5M2 support (torch.float8_e5m2) is required for calibration.")

        if not hasattr(self, 'weight') or self.weight is None:
            return

        # Keep weights in FP32 path while still using quantized layer wrappers.
        # This is useful for input-only quantization experiments where op replacement
        # must remain identical to deployment code paths.
        if not getattr(self, 'weight_quantization', True):
            self.weight_fp8 = self.weight.detach()
            self.weight_scale = torch.tensor(1.0, device=self.weight.device)
            self.weight_scale_packed = self.weight_scale
            if getattr(self, 'capture_activations', False):
                self.last_quant_weight = self.weight.detach()
                self.last_quant_weight_scale = self.weight_scale.detach()
            return



        # Calculate max per output channel (dim 0)
        # Flatten all dims except 0: [out_channels, -1]
        
        # Use weight_mode and weight_chunk_size
        mode = getattr(self, 'weight_mode', 'channel') # Default to channel for weights
        chunk_size = getattr(self, 'weight_chunk_size', None)
        rounding = getattr(self, 'rounding', 'nearest') # Default to nearest for weights
        chunk_formats = getattr(self, 'chunk_formats', None) # Per-chunk format list
        
        if chunk_formats is not None and chunk_size is not None:
            # Flatten weight
            if self.weight.dim() > 1:
                flat_weight = self.weight.flatten(1)
                batch_size = self.weight.shape[0]
            else:
                flat_weight = self.weight.flatten(0)
                batch_size = 1
            
            num_elements = flat_weight.shape[-1]
            pad_len = 0
            if num_elements % chunk_size != 0:
                pad_len = chunk_size - (num_elements % chunk_size)
                flat_weight = torch.nn.functional.pad(flat_weight, (0, pad_len))
            
            num_chunks = flat_weight.shape[-1] // chunk_size
            chunked = flat_weight.reshape(batch_size, num_chunks, chunk_size)
            
            # chunk_formats handling
            total_chunks = batch_size * num_chunks
            
            # Align chunk_formats
            final_formats = []
            if len(chunk_formats) == total_chunks:
                final_formats = chunk_formats
            elif len(chunk_formats) == num_chunks:
                # Broadcast across batch
                final_formats = []
                for _ in range(batch_size):
                    final_formats.extend(chunk_formats)
            else:
                print(f"Warning: chunk_formats length ({len(chunk_formats)}) mismatch. Expected {total_chunks} or {num_chunks}.")
                # Fallback: use first format
                final_formats = [chunk_formats[0]] * total_chunks
                
            # Flatten chunks for processing: [total_chunks, chunk_size]
            chunked_flat = chunked.reshape(-1, chunk_size)
            
            quantized_flat = torch.zeros_like(chunked_flat)
            scale_flat = torch.zeros_like(chunked_flat) # Expanded scale
            
            # Group by format to vectorize quantization
            unique_fmts = set(final_formats)
            
            for fmt in unique_fmts:
                # Find indices for this format
                indices = [i for i, f in enumerate(final_formats) if f == fmt]
                
                if not indices:
                    continue
                    
                indices_tensor = torch.tensor(indices, device=self.weight.device)
                
                # Extract sub-tensor
                sub_chunks = chunked_flat[indices_tensor] # [M, chunk_size]
                
                # Calculate scale
                max_val = sub_chunks.abs().amax(dim=1, keepdim=True).clamp(min=1e-9)
                scale = calculate_scale(max_val, fmt) # [M, 1]
                
                # Quantize
                scaled = sub_chunks / scale
                # Note: quantize returns floats for fp8/int8 simulation
                # Codec-routed: scaled has per-row amax ∈ [1, 2) so the codec's
                # internal scale is 1.0 and unscaled == quantize(scaled).
                quant = quantize_tensor(
                    scaled.contiguous(), q_type=fmt, mode='tensor',
                    return_unscaled=True,
                )[1].to(chunked_flat.dtype)
                
                # Store back
                quantized_flat[indices_tensor] = quant
                
                # Expand scale to chunk size and store
                scale_expanded = scale.expand(-1, chunk_size).contiguous()
                scale_flat[indices_tensor] = scale_expanded

            # Reshape back to [batch, num_chunks, chunk_size]
            weight_fp8_chunked = quantized_flat.reshape(batch_size, num_chunks, chunk_size)
            weight_scale_chunked = scale_flat.reshape(batch_size, num_chunks, chunk_size)
            
            # Reshape to flat [batch, num_elements_padded]
            weight_fp8_flat = weight_fp8_chunked.reshape(batch_size, -1)
            weight_scale_flat = weight_scale_chunked.reshape(batch_size, -1)
            
            # Remove padding
            if pad_len > 0:
                weight_fp8_flat = weight_fp8_flat[:, :num_elements]
                weight_scale_flat = weight_scale_flat[:, :num_elements]
            
            # View as original shape (clone to ensure contiguous, non-overlapping memory)
            self.weight_fp8 = weight_fp8_flat.reshape_as(self.weight).clone()
            self.weight_scale = weight_scale_flat.reshape_as(self.weight).clone()
            
            # Store chunk formats for reference
            self.weight_chunk_formats = final_formats
            
            return
        
        # We can reuse quantize_tensor logic to get scale and quantized weight
        # But we need to be careful about the "channel" definition.
        # For weights, "channel" usually means output channel (dim 0).
        # quantize_tensor 'channel' mode reduces all except dim 1.
        # So if we want output channel quantization (dim 0), we might need to transpose or handle it.
        
        # If mode is 'channel' and weight is [Out, In, K, K], we want scale per Out (dim 0).
        # quantize_tensor 'channel' preserves dim 1.
        # So we should probably transpose weight to [In, Out, K, K] before calling if we use 'channel' mode?
        # Or just implement custom logic here as before but updated for modes.
        
        # Let's stick to custom logic here to be safe and precise about weight structure.
        
        if mode == 'chunk':
             # Use quantize_tensor for chunking
             # It handles arbitrary shapes
             # Modified to return scales (broadcast and packed)
             _, w_unscaled, _, scale_b, scale_p = quantize_tensor(self.weight, q_type=q_type, return_unscaled=True, return_scale=True, mode='chunk', chunk_size=chunk_size)
             self.weight_scale = scale_b
             self.weight_scale_packed = scale_p
             if q_type == 'int8':
                 self.weight_fp8 = w_unscaled.to(torch.int8)
             elif q_type == 'int4':
                 # Store as int8 (no native int4 type), but values are in [-8, 7]
                 self.weight_fp8 = w_unscaled.to(torch.int8)
             elif q_type in ['fp8_e4m3', 'fp8_e5m2', "fp8_e2m5", "fp8_e3m4", "fp8_e1m6", "fp8_e6m1", "fp8_e7m0"]:
                 # Use custom quantize to ensure no inf/nan and consistent behavior
                 # w_unscaled is already quantized by quantize_tensor, just store as float32
                 self.weight_fp8 = w_unscaled
             else:
                 self.weight_fp8 = w_unscaled # FP4 simulated
                 
             return

        # Default/Channel/Tensor logic
        if mode == 'tensor':
             reduce_dims = tuple(range(self.weight.dim()))
        elif mode == 'channel':
             # Per output channel (dim 0)
             reduce_dims = tuple(range(1, self.weight.dim()))
        else:
             # Fallback
             reduce_dims = tuple(range(self.weight.dim()))

        
        if q_type == 'fp32':
            self.weight_fp8 = self.weight
            self.weight_scale = torch.tensor(1.0, device=self.weight.device)
            return

        max_val = self.weight.abs().amax(dim=reduce_dims, keepdim=True)
        
        # Avoid log of 0
        max_val = max_val.clamp(min=1e-9)
 
        # Calculate scale
        s = calculate_scale(max_val, q_type)
        
        self.weight_scale = s
        
        # Quantize: w_fp8 = (w / s).to(fp8) * s
        # We store the quantized version (as fp8 type)
        w_scaled = self.weight / self.weight_scale
        
        # if q_type in ['fp8_e4m3', 'fp8_e5m2', "fp8_e2m5", "fp8_e3m4", "fp8_e1m6", "fp8_e6m1", "fp8_e7m0", "fp8_e0m7"]:
            # Use custom quantize to ensure no inf/nan and consistent behavior
            # self.weight_fp8 = quantize(w_scaled, q_type=q_type, bias=bias, rounding=rounding)
        # elif q_type in ['fp4_e2m1', 'fp4_e3m0']:
        #     # FP4 types are stored as FP32 (simulated) or we could pack them if we had a packer.
        #     # For now, we store them as FP32 but with values quantized to FP4.
        #     # We already computed w_scaled, but it's not quantized yet!
        #     # Wait, the logic above for FP8 casts to the type which does the quantization (if native).
        #     # For FP4, we need to call quantize() explicitly.
            
        #     # Recalculate w_scaled using quantize function to ensure it snaps to valid values
        #     # Note: quantize() returns float32 tensor with valid FP4 values.
        #     self.weight_fp8 = quantize(w_scaled, q_type=q_type, bias=bias, rounding=rounding)
        # elif q_type == 'int8':
        #     # Store as int8
        #     # w_scaled is float, we need to round and clamp
        #     # quantize(..., q_type='int8') does exactly that (returns float with int values)
        #     # But we can store as torch.int8 to save memory if we want.
        #     # However, for consistency with other simulated types (FP4) and to avoid dequantization complexity in forward,
        #     # let's store as float (simulated) or int8?
        #     # If we store as int8, we need to cast back to float in forward.
        #     # Let's store as int8 to be true to the type.
        #     w_quant = quantize(w_scaled, q_type='int8') # Returns float with int values
        #     self.weight_fp8 = w_quant.to(torch.int8)
        # elif q_type == 'int4':
        #     # Store as int8 (no native int4 type), but values are in [-8, 7]
        #     w_quant = quantize(w_scaled, q_type='int4') # Returns float with int values
        #     self.weight_fp8 = w_quant.to(torch.int8)
        if q_type == 'fp32':
            # Identity quantization
            self.weight_fp8 = self.weight
            self.weight_scale = torch.tensor(1.0, device=self.weight.device)
        else:
              # Codec-routed: w_scaled has amax ∈ [1, 2), so the codec's
              # internal scale is 1.0 and unscaled == quantize(w_scaled).
              self.weight_fp8 = quantize_tensor(
                  w_scaled.contiguous(), q_type=q_type, mode='tensor',
                  return_unscaled=True
              )[1]

        # Capture weight and weight_scale if capture_activations is enabled
        if getattr(self, 'capture_activations', False):
            self.last_quant_weight = (self.weight_fp8.float() * self.weight_scale).detach()
            self.last_quant_weight_scale = self.weight_scale.detach()

    def quantize_input(self, input: torch.Tensor, override_q_type: str = None):
        """
        Quantizes input tensor to FP8.
        Returns: (input_fp8, scale)
        """
        if not isinstance(input, torch.Tensor):
            return input
        if getattr(self, 'capture_activations', False):
            # Capture the raw input format on every call (regardless of input_quantization),
            # so the comparator can runtime-detect what format actually arrived at this layer.
            self.last_pre_quant_input = input.detach()
            
        # Use input_q_type if available, otherwise fallback to q_type
        q_type = override_q_type or getattr(self, 'input_q_type', getattr(self, 'q_type', 'fp8_e4m3'))
        capture = getattr(self, 'capture_activations', False)
        

        mode = getattr(self, 'input_mode', getattr(self, 'quant_mode', 'chunk')) # Use input_mode if set, else quant_mode
        chunk_size = getattr(self, 'input_chunk_size', getattr(self, 'chunk_size', None))
        rounding = getattr(self, 'rounding', 'nearest') # Default to nearest for inputs
        chunk_formats = getattr(self, 'input_chunk_formats', None) # Per-chunk input formats
        
        # Check if input quantization is enabled
        if not getattr(self, 'input_quantization', True):
            # If disabled, return input as-is (cast to float if needed) and scale=1.0
            # But we still need to return max_val for stats?
            # The signature is (input_fp8, max_val) or (input_fp8, scale) depending on return_scale
            # Wait, quantize_input signature is just (input_fp8, scale) according to docstring?
            # No, let's check usage.
            # In QuantConv2d: input_fp8 = self.quantize_input(input)
            # It expects a single return value?
            # Let's check quantize_input implementation below.
            
            # Implementation below calls quantize_tensor with return_unscaled=True
            # input_fp8, input_fp8_unscaled, max_val = quantize_tensor(...)
            
            # And returns:
            # return input_fp8
            
            # So it returns just the quantized input.
            
            return input
            
        # Hot path: when capture_activations is off, only `input_fp8` is
        # actually consumed — the unscaled / scale_b / scale_p / max_val
        # outputs would be thrown away.  In chunk mode this matters: scale_b
        # is an input-sized expand+contiguous+slice allocation per layer per
        # batch.  Gate the flags on `capture` so we skip that work in eval.
        if capture:
            input_fp8, input_fp8_unscaled, max_val, scale_b, scale_p = quantize_tensor(
                input, q_type=q_type, return_unscaled=True, return_scale=True,
                mode=mode, chunk_size=chunk_size,
                chunk_formats=chunk_formats,
            )
        else:
            input_fp8, _ = quantize_tensor(
                input, q_type=q_type,
                mode=mode, chunk_size=chunk_size,
                chunk_formats=chunk_formats,
            )

        if capture:
            self.last_quant_input_unscaled = input_fp8_unscaled.detach()
            self.last_quant_input = input_fp8.detach()
            self.last_quant_input_max = max_val.detach()
            self.last_quant_input_scale = scale_b.detach()
            self.last_quant_input_scale_packed = scale_p.detach()

            # Calculate dequantized max
            # We need to re-calculate max from the quantized-then-dequantized tensor
            # But input_fp8 is already scaled back! So we just take max of it.
            # However, we need to use the same reduction dims.
            reduce_dims = _get_reduce_dims(input_fp8)
            self.last_quant_input_dequant_max = input_fp8.abs().amax(dim=reduce_dims, keepdim=True).detach()

        return input_fp8

    def quantize_output(self, output, override_q_type: str = None):
        """
        Quantize a layer's output tensor.
        Pass-through when self.output_quantization is False (default).
        Always captures last_natural_output for report-side runtime format detection
        when capture_activations is on, regardless of whether quantization runs.
        """
        if not isinstance(output, torch.Tensor):
            return output
        if getattr(self, 'capture_activations', False):
            self.last_natural_output = output.detach()
        if not getattr(self, 'output_quantization', False):
            return output

        q_type = override_q_type or getattr(self, 'output_q_type',
                                            getattr(self, 'q_type', 'fp8_e4m3'))
        mode = getattr(self, 'output_mode', 'tensor')
        chunk_size = getattr(self, 'output_chunk_size', None)
        capture = getattr(self, 'capture_activations', False)

        if capture:
            out_q, out_q_unscaled, max_val, scale_b, scale_p = quantize_tensor(
                output, q_type=q_type, return_unscaled=True, return_scale=True,
                mode=mode, chunk_size=chunk_size,
            )
            self.last_pre_quant_output = output.detach()
            self.last_quant_output = out_q.detach()
            self.last_quant_output_unscaled = out_q_unscaled.detach()
            self.last_quant_output_max = max_val.detach()
            self.last_quant_output_scale = scale_b.detach()
            self.last_quant_output_scale_packed = scale_p.detach()
        else:
            out_q, _ = quantize_tensor(
                output, q_type=q_type, mode=mode, chunk_size=chunk_size,
            )
        return out_q


    def quantized_forward(self, input, *args, **kwargs):
        """
        Generic forward pass for quantized layers.
        1. Quantize input (or cast if first layer).
        2. Dequantize weights (simulated).
        3. Run original forward pass with dequantized weights.
        """
        # Check for FP8 support
        q_type = getattr(self, 'q_type', 'fp8_e4m3')
        if q_type == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
             raise RuntimeError("FP8 support (torch.float8_e4m3fn) is required but not available.")
        
        # 1. Handle Input
        if getattr(self, 'is_first_layer', False):
            if not getattr(self, 'quantize_first_layer', False):
                # Do NOT quantize to FP8. Just cast to float for the operation.
                input_fp8 = input.float()
            else:
                input_fp8 = self.quantize_input(input.float())
        else:
            input_fp8 = self.quantize_input(input)

        # 2. Handle Weights
        # We need to temporarily replace self.weight with the dequantized version
        # so that the super().forward() uses it.
        
        original_weight = None
        w_decomp = None
        
        if hasattr(self, 'weight'):
            original_weight = self.weight
            
            if (
                hasattr(self, 'weight_fp8') and self.weight_fp8 is not None and
                hasattr(self, 'weight_scale') and self.weight_scale is not None
            ):
                w_decomp = self.weight_fp8.float() * self.weight_scale
                self.weight = torch.nn.Parameter(w_decomp)
        
        try:
            # 3. Run original forward
            # We assume the original forward signature starts with input.
            # If the layer has other required args, they must be passed in *args.
            output = super().forward(input_fp8, *args, **kwargs)
            
        finally:
            # Restore original weight
            if original_weight is not None:
                self.weight = original_weight
            
        # Capture activations if enabled
        if getattr(self, 'capture_activations', False):
            # last_quant_input is already captured in quantize_input if enabled
            if w_decomp is not None:
                self.last_quant_weight = w_decomp.detach()

        return self.quantize_output(output)

# Prevent tracing into quantize_tensor to avoid Proxy errors with dynamic shapes
if hasattr(torch, 'fx') and hasattr(torch.fx, 'wrap'):
    torch.fx.wrap('quantize_tensor')

if __name__ == "__main__":
    # Corrected main block for basic testing
    tensor = torch.randn(1, 1, 128, 128).cuda()
    # Note: _quantize_tensor_cuda expects e, m, is_signed from resolve_format
    from runspace.src.quantization.cuda import resolve_format
    e, m, is_signed = resolve_format('fp8_e4m3')
    
    cuda_quant, _ = _quantize_tensor_cuda(tensor, 'fp8_e4m3', e, m, is_signed, False, False, 'tensor', 128, False)
    ref_quant, _ = quantize_tensor(tensor, 'fp8_e4m3', mode='tensor')
    
    print(f"CUDA Quantization Difference Norm: {torch.norm(cuda_quant - ref_quant).item()}")
