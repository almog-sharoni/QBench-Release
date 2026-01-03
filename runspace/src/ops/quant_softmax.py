import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from .quant_base import quantize_tensor

# @OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax)
@OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax)
class QuantSoftmax(nn.Softmax):
    """
    Quantized Softmax operation following a hardware-friendly paradigm.
    
    Paradigm:
    1. Quantize input at the softmax boundary.
    2. Subtract maximum for stability.
    3. Approximate exponential using base-2 (split int/frac, LUT for frac).
    4. Accumulate sum.
    5. Compute reciprocal using Newton-Raphson.
    6. Normalize.
    7. Quantize output.
    """
    def __init__(self, dim: int = None, q_type: str = "fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None):
        super().__init__(dim=dim)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.capture_activations = False
        self.last_quant_input = None
        self.last_quant_output_unscaled = None
        self.exp2_lut = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1. Quantize input at the softmax boundary
        # Compute a power-of-two scale from the maximum absolute input value.
        # Quantize the input tensor to the target format (FP8 / INT8), then dequantize it back to float for computation.
        input_dequant, _ = quantize_tensor(
            input, 
            q_type=self.q_type, 
            bias=self.quantization_bias, 
            mode=self.quant_mode, 
            chunk_size=self.chunk_size
        )
        
        # 2. Subtract maximum for stability
        # For each softmax vector, subtract the maximum value so all elements are <= 0.
        dim = self.dim if self.dim is not None else -1
        max_val = input_dequant.amax(dim=dim, keepdim=True)
        x = input_dequant - max_val
        
        # 3. Approximate exponential using base-2
        # Convert exp(x) to 2^(x * log2(e)).
        log2e = 1.4426950408889634
        y = x * log2e
        
        # Split the result into integer and fractional parts.
        y_int = torch.floor(y)
        y_frac = y - y_int
        
        # Compute 2^integer using bit shifts (simulated via exp2)
        pow2_int = torch.exp2(y_int)
        
        # Compute 2^fraction using a small LUT
        # Initialize LUT if needed (size 16)
        if self.exp2_lut is None or self.exp2_lut.device != input.device:
             steps = torch.arange(16, device=input.device, dtype=input.dtype) / 16.0
             self.exp2_lut = torch.exp2(steps)
             
        # Index into LUT: floor(y_frac * 16)
        # y_frac is in [0, 1), so index is in [0, 15]
        lut_idx = (y_frac * 16).long().clamp(0, 15)
        pow2_frac = self.exp2_lut[lut_idx]
        
        # Multiply both to get the exponential approximation.
        exp_val = pow2_int * pow2_frac
        
        # 4. Accumulate sum
        # Sum all exponentials using higher-precision arithmetic.
        sum_exp = exp_val.sum(dim=dim, keepdim=True)
        
        # Clamp the sum to a small positive value to avoid division by zero.
        sum_exp = torch.clamp(sum_exp, min=1e-12)
        
        # 5. Compute reciprocal using Newton-Raphson
        # Compute an initial reciprocal estimate (using standard division here as a placeholder for low-precision estimate)
        r = 1.0 / sum_exp
        
        # Refine it using one Newton-Raphson iteration: r = r * (2 - sum * r).
        r = r * (2.0 - sum_exp * r)
        
        # 6. Normalize
        # Multiply each exponential by the reciprocal to get probabilities.
        prob = exp_val * r
        
        # Clamp results to the range [0, 1].
        prob = torch.clamp(prob, 0.0, 1.0)
        
        # 7. Quantize output
        # Quantize the probability tensor back to the target format using the same power-of-two scaling method.
        output_fp8, output_fp8_unscaled, max_val_out = quantize_tensor(
            prob, 
            q_type=self.q_type, 
            bias=self.quantization_bias, 
            return_unscaled=True,
            mode=self.quant_mode,
            chunk_size=self.chunk_size
        )
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = output_fp8_unscaled.detach()
            
        return output_fp8
