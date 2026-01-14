import torch
try:
    import torch.fx
except ImportError:
    pass
from ..quantization.quantizer import quantize

def _get_reduce_dims(input: torch.Tensor):
    return tuple(range(input.dim()))

def calculate_scale(max_val: torch.Tensor, q_type: str, bias: int = None):
    """
    Calculates the scale factor for quantization.
    Args:
        max_val: Maximum absolute value (tensor or scalar)
        q_type: Quantization type
        bias: Exponent bias (optional)
    Returns:
        Scale factor
    """
    if bias is None:
        if q_type == 'fp8_e5m2':
            bias = 15
        elif q_type == 'fp4_e2m1':
            bias = 1
        elif q_type == 'fp4_e3m0':
            bias = 3
        elif q_type == 'int8':
            bias = 0
        else:
            bias = 8 # Default for e4m3
            
    if q_type == 'int8':
        # power-of-two scale so it can use the same log2/floor/pow2 (shift) hardware
        e = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(torch.tensor(127.0, device=max_val.device))) #TODO: Check this
        # equivalently: e = floor(log2(max_val / 127.0))
        s = torch.pow(2.0, e)
    else:
        e = torch.floor(torch.log2(max_val)) - bias
        s = torch.pow(2, e)
    return s


def quantize_tensor(input: torch.Tensor, q_type: str = 'fp8_e4m3', bias: int = None, return_unscaled: bool = False, return_scale: bool = False, mode: str = 'tensor', chunk_size: int = None):
    """
    Quantizes a tensor to FP8.
    Args:
        input: Input tensor
        q_type: Quantization type ('fp8_e4m3' or 'fp8_e5m2')
        bias: Exponent bias (default: 7 for e4m3, 15 for e5m2 if None)
        bias: Exponent bias (default: 7 for e4m3, 15 for e5m2 if None)
        return_unscaled: If True, returns (scaled_input, unscaled_input)
        return_scale: If True, returns (quantized_tensor, scale) or (quantized_tensor, unscaled, scale) if return_unscaled is True
        mode: Quantization mode ('tensor', 'channel', 'chunk')
        chunk_size: Size of chunk for 'chunk' mode
    Returns:
        Quantized tensor (scaled back to original range)
    """
    # Check for FP8 support
    if q_type == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError("FP8 E4M3 support (torch.float8_e4m3fn) is required.")
    if q_type == 'fp8_e5m2' and not hasattr(torch, 'float8_e5m2'):
            raise RuntimeError("FP8 E5M2 support (torch.float8_e5m2) is required.")
    # FP4 types are manually implemented, so no check needed

    s = None
    
    if mode == 'chunk':
        if chunk_size is None:
            raise ValueError("chunk_size must be provided for mode='chunk'")
        
        # Flatten all dimensions except batch (if present)
        # We assume dim 0 is batch if dim > 1
        if input.dim() > 1:
            flat_input = input.flatten(1)
            batch_size = input.shape[0]
        else:
            flat_input = input.flatten(0)
            batch_size = 1
            
        num_elements = flat_input.shape[-1]
        
        # Pad if necessary
        if num_elements % chunk_size != 0:
            pad_len = chunk_size - (num_elements % chunk_size)
            flat_input = torch.nn.functional.pad(flat_input, (0, pad_len))
            
        # Reshape to [N, num_chunks, chunk_size]
        num_chunks = flat_input.shape[-1] // chunk_size
        chunked = flat_input.view(batch_size, num_chunks, chunk_size)
        
        # Max per chunk
        max_val_chunk = chunked.abs().amax(dim=-1, keepdim=True) # [N, num_chunks, 1]
        max_val_chunk = torch.clamp(max_val_chunk, min=1e-5)
        
        # Calculate scale per chunk
        # Calculate scale per chunk
        s_chunk = calculate_scale(max_val_chunk, q_type, bias)
            
        # Expand scale back to original shape
        # s_chunk is [N, num_chunks, 1]
        # We need to broadcast it to [N, num_chunks, chunk_size]
        s_expanded = s_chunk.expand(-1, -1, chunk_size).reshape(flat_input.shape)
        
        # Remove padding
        if num_elements % chunk_size != 0:
            s_expanded = s_expanded[..., :num_elements]
            
        # Reshape back to original input shape
        if input.dim() > 1:
            s = s_expanded.view(input.shape)
        else:
            s = s_expanded.view(input.shape)
            
        # For return_unscaled, we need a single max_val? 
        # Or should we return the max_val tensor?
        # The caller expects max_val to be useful for stats.
        # We'll return the mean max_val or something? 
        # Or just return the max_val tensor (which might be large).
        # Let's return the max of max_vals for simplicity in reporting, 
        # or we update the report to handle tensors.
        # Current report expects scalar or small tensor.
        # Let's return the global max for reporting purposes if it's chunked.
        max_val = max_val_chunk.max() 

    else:
        # Determine reduction dimensions
        if mode == 'channel':
            # Reduce all except dim 1 (channel)
            if input.dim() >= 2:
                reduce_dims = tuple(d for d in range(input.dim()) if d != 1)
            else:
                reduce_dims = tuple(range(input.dim())) # Fallback for 1D
        else: # 'tensor'
            reduce_dims = tuple(range(input.dim()))

        # Find max absolute value
        max_val = input.abs().amax(dim=reduce_dims, keepdim=True)
        # Avoid division by zero
        max_val = torch.clamp(max_val, min=1e-5)
        
        # Calculate exponent: 2^e >= max_val
        # Calculate exponent: 2^e >= max_val
        s = calculate_scale(max_val, q_type, bias)
    
    # Scale input
    input_scaled = input / s
    
    # Quantize to FP8 (simulated)
    # Capture unscaled quantized values (as float for simulation/verification)
    input_fp8_unscaled = quantize(input_scaled, q_type=q_type, bias=bias)
    input_fp8 = input_fp8_unscaled * s
    
    if return_unscaled:
        if return_scale:
            return input_fp8, input_fp8_unscaled, max_val, s
        return input_fp8, input_fp8_unscaled, max_val
        
    if return_scale:
        return input_fp8, s
        
    return input_fp8, max_val

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
        
        if q_type == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
             raise RuntimeError("FP8 E4M3 support (torch.float8_e4m3fn) is required for calibration.")
        if q_type == 'fp8_e5m2' and not hasattr(torch, 'float8_e5m2'):
             raise RuntimeError("FP8 E5M2 support (torch.float8_e5m2) is required for calibration.")

        if not hasattr(self, 'weight') or self.weight is None:
            return

        # Default bias: 7 for E4M3, 15 for E5M2
        bias = getattr(self, 'quantization_bias', None)
        if bias is None:
             if q_type == 'fp8_e5m2':
                 bias = 15
             elif q_type == 'fp4_e2m1':
                 bias = 1
             elif q_type == 'fp4_e3m0':
                 bias = 3
             elif q_type == 'int8':
                 bias = 0
             else:
                 bias = 7

        # Calculate max per output channel (dim 0)
        # Flatten all dims except 0: [out_channels, -1]
        
        # Use weight_mode and weight_chunk_size
        mode = getattr(self, 'weight_mode', 'channel') # Default to channel for weights
        chunk_size = getattr(self, 'weight_chunk_size', None)
        
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
             w_quant, scale = quantize_tensor(self.weight, q_type=q_type, bias=bias, return_scale=True, mode='chunk', chunk_size=chunk_size)
             self.weight_scale = scale
             # w_quant is already scaled back (simulated).
             # We need to store the quantized integer/fp8 form?
             # quantize_tensor returns dequantized float.
             # We need the unscaled version to store as weight_fp8.
             # Let's call with return_unscaled=True
             _, w_unscaled, _, scale = quantize_tensor(self.weight, q_type=q_type, bias=bias, return_unscaled=True, return_scale=True, mode='chunk', chunk_size=chunk_size)
             self.weight_scale = scale
             
             if q_type == 'int8':
                 self.weight_fp8 = w_unscaled.to(torch.int8)
             elif q_type == 'fp8_e4m3':
                 self.weight_fp8 = w_unscaled.to(torch.float8_e4m3fn)
             elif q_type == 'fp8_e5m2':
                 self.weight_fp8 = w_unscaled.to(torch.float8_e5m2)
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

        
        max_val = self.weight.abs().amax(dim=reduce_dims, keepdim=True)
        
        # Avoid log of 0
        max_val = max_val.clamp(min=1e-9)
 
        # Calculate scale
        s = calculate_scale(max_val, q_type, bias)
        
        self.weight_scale = s
        
        # Quantize: w_fp8 = (w / s).to(fp8) * s
        # We store the quantized version (as fp8 type)
        w_scaled = self.weight / self.weight_scale
        
        if q_type == 'fp8_e4m3':
            # Clamp to valid range to avoid NaNs (max 448)
            w_scaled = w_scaled.clamp(-448.0, 448.0)
            self.weight_fp8 = w_scaled.to(torch.float8_e4m3fn)
        elif q_type == 'fp8_e5m2':
            # E5M2 has Inf, so clamping is less critical but good for safety if we want to avoid Inf
            # Max normal is 57344.
            self.weight_fp8 = w_scaled.to(torch.float8_e5m2)
        elif q_type in ['fp4_e2m1', 'fp4_e3m0']:
            # FP4 types are stored as FP32 (simulated) or we could pack them if we had a packer.
            # For now, we store them as FP32 but with values quantized to FP4.
            # We already computed w_scaled, but it's not quantized yet!
            # Wait, the logic above for FP8 casts to the type which does the quantization (if native).
            # For FP4, we need to call quantize() explicitly.
            
            # Recalculate w_scaled using quantize function to ensure it snaps to valid values
            # Note: quantize() returns float32 tensor with valid FP4 values.
            self.weight_fp8 = quantize(w_scaled, q_type=q_type, bias=bias)
        elif q_type == 'int8':
            # Store as int8
            # w_scaled is float, we need to round and clamp
            # quantize(..., q_type='int8') does exactly that (returns float with int values)
            # But we can store as torch.int8 to save memory if we want.
            # However, for consistency with other simulated types (FP4) and to avoid dequantization complexity in forward,
            # let's store as float (simulated) or int8?
            # If we store as int8, we need to cast back to float in forward.
            # Let's store as int8 to be true to the type.
            w_quant = quantize(w_scaled, q_type='int8') # Returns float with int values
            self.weight_fp8 = w_quant.to(torch.int8)
        else:
            raise ValueError(f"Unsupported q_type: {q_type}")

    def quantize_input(self, input: torch.Tensor):
        """
        Quantizes input tensor to FP8.
        Returns: (input_fp8, scale)
        """
        # Use input_q_type if available, otherwise fallback to q_type
        q_type = getattr(self, 'input_q_type', getattr(self, 'q_type', 'fp8_e4m3'))
        capture = getattr(self, 'capture_activations', False)
        
        bias = getattr(self, 'quantization_bias', None)
        mode = getattr(self, 'input_mode', getattr(self, 'quant_mode', 'tensor')) # Use input_mode if set, else quant_mode
        chunk_size = getattr(self, 'input_chunk_size', getattr(self, 'chunk_size', None))
        
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
            
        input_fp8, input_fp8_unscaled, max_val = quantize_tensor(input, q_type=q_type, bias=bias, return_unscaled=True, mode=mode, chunk_size=chunk_size)
        
        if capture:
            self.last_quant_input_unscaled = input_fp8_unscaled.detach()
            self.last_quant_input = input_fp8.detach()
            self.last_quant_input_max = max_val.detach()
            
            # Calculate dequantized max
            # We need to re-calculate max from the quantized-then-dequantized tensor
            # But input_fp8 is already scaled back! So we just take max of it.
            # However, we need to use the same reduction dims.
            reduce_dims = _get_reduce_dims(input_fp8)
            self.last_quant_input_dequant_max = input_fp8.abs().amax(dim=reduce_dims, keepdim=True).detach()
            
        return input_fp8


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
                input_fp8, _ = self.quantize_input(input.float())
        else:
            input_fp8, _ = self.quantize_input(input)

        # 2. Handle Weights
        # We need to temporarily replace self.weight with the dequantized version
        # so that the super().forward() uses it.
        
        original_weight = None
        w_decomp = None
        
        if hasattr(self, 'weight'):
            original_weight = self.weight
            
            if hasattr(self, 'weight_fp8') and self.weight_fp8 is not None:
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
                
        return output

# Prevent tracing into quantize_tensor to avoid Proxy errors with dynamic shapes
if hasattr(torch, 'fx') and hasattr(torch.fx, 'wrap'):
    torch.fx.wrap('quantize_tensor')
