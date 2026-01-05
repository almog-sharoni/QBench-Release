"""
Quantized Activation Functions with Look-Up Table (LUT)

These modules wrap standard activations and use a precomputed LUT for FP8 inference.
This ensures the entire dataflow through the network stays in FP8 and is optimized.
"""

import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from ..quantization.quantizer import quantize, get_fp8_e4m3_table
from .quant_base import quantize_tensor


class LUTActivation:
    """
    Mixin for LUT-based FP8 activations.
    Precomputes outputs for all 256 possible FP8 inputs.
    """
    def build_lut(self, activation_fn, q_type="fp8_e4m3", bias=None, quant_mode="tensor", chunk_size=None):
        """
        Builds the Look-Up Table (LUT) for the given activation function.
        """
        self.q_type = q_type
        self.bias = bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        
        # For FP4 (simulated), we compute on-the-fly instead of using a LUT
        if q_type.startswith("fp4"):
            self.activation_fn = activation_fn
            return

        if q_type == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
            raise RuntimeError("LUTActivation requires torch.float8_e4m3fn support.")
        if q_type == 'fp8_e5m2' and not hasattr(torch, 'float8_e5m2'):
            raise RuntimeError("LUTActivation requires torch.float8_e5m2 support.")

        # 1. Generate all 256 bit patterns
        indices = torch.arange(256, dtype=torch.uint8)
        
        # 2. Convert to Float to get the actual values
        if q_type == 'fp8_e4m3':
            dummy = torch.zeros(256, dtype=torch.float8_e4m3fn)
            dummy_u8 = dummy.view(torch.uint8)
            dummy_u8[:] = indices
            input_values = dummy.float()
        elif q_type == 'fp8_e5m2':
            dummy = torch.zeros(256, dtype=torch.float8_e5m2)
            dummy_u8 = dummy.view(torch.uint8)
            dummy_u8[:] = indices
            input_values = dummy.float()
        elif q_type == 'int8':
            # For int8, indices 0-255 map to -128 to 127
            # We treat uint8 as int8 bit pattern
            input_values = indices.view(torch.int8).float()
        else:
            raise ValueError(f"Unsupported q_type: {q_type}")
        
        # 3. Apply activation function (in FP32)
        output_values = activation_fn(input_values)
        
        # 4. Quantize output to FP8
        output_quant = quantize(output_values, q_type=q_type, bias=bias)
        
        # Register as buffer so it's saved with state_dict but not trained
        self.register_buffer('lut', output_quant)

    def apply_lut(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply the LUT to the input tensor.
        Input is expected to be a float tensor.
        """
        # For FP4 (simulated), compute directly
        if self.q_type.startswith("fp4"):
            return quantize(self.activation_fn(input), q_type=self.q_type, bias=self.bias)

        # Dynamic Quantization
        # We need to scale input to target range.
        input_fp8, input_fp8_unscaled, max_val, s = quantize_tensor(
            input, 
            q_type=self.q_type, 
            bias=self.bias, 
            return_unscaled=True,
            return_scale=True,
            mode=self.quant_mode,
            chunk_size=self.chunk_size
        )
        
        # s is now the scale tensor returned by quantize_tensor
        # It handles chunking/channel/tensor modes correctly.

        # 1. Cast input to FP8/INT8 to get the indices
        if self.q_type == 'int8':
            # input_fp8_unscaled is float [-127, 127]
            input_q = input_fp8_unscaled.to(torch.int8)
        elif self.q_type == 'fp8_e4m3':
            input_q = input_fp8_unscaled.to(torch.float8_e4m3fn)
        elif self.q_type == 'fp8_e5m2':
            input_q = input_fp8_unscaled.to(torch.float8_e5m2)
        else:
            raise ValueError(f"Unsupported q_type: {self.q_type}")
        
        # 2. Get indices (uint8 view)
        indices = input_q.view(torch.uint8).long() # Long for indexing
        
        # 3. Lookup
        q_output = self.lut[indices]
        
        # 4. Dequantize
        # Note: This assumes the activation function is scale-invariant (homogeneous of degree 1),
        # like ReLU. For non-homogeneous functions (GELU, SiLU), this is an approximation
        # or technically incorrect with dynamic scaling.
        output = q_output * s
        
        return output, q_output


@OpRegistry.register("QuantReLU", original_cls=nn.ReLU, is_activation=True, compliance_status="FP32 activation")
class QuantReLU(nn.ReLU):
    """
    Quantized ReLU using LUT.
    """
    def __init__(self, inplace: bool = False, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None):
        super().__init__(inplace=inplace)
        self.capture_activations = False
        self.last_quant_output_unscaled = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # User requested modification: instead of quantizing here and use lut to actually forward like this:
        # if x<0 than x=0 else x=x (do nothing)
        output_quant = nn.functional.relu(input)
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = output_quant.detach()
        
        return output_quant


@OpRegistry.register("QuantReLU6", original_cls=nn.ReLU6, is_activation=True, compliance_status="FP32 activation")
class QuantReLU6(nn.ReLU6):
    """
    Quantized ReLU6 using LUT.
    """
    def __init__(self, inplace: bool = False, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None):
        super().__init__(inplace=inplace)
        self.capture_activations = False
        self.last_quant_output_unscaled = None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # User requested modification: use standard relu6 instead of lut
        output_quant = nn.functional.relu6(input)
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = output_quant.detach()
        
        return output_quant


# @OpRegistry.register("QuantSiLU", original_cls=nn.SiLU, is_activation=True)
# class QuantSiLU(nn.SiLU, LUTActivation):
#     """
#     Quantized SiLU (Swish) using LUT.
#     """
#     def __init__(self, inplace: bool = False, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None):
#         super().__init__(inplace=inplace)
#         self.capture_activations = False
#         self.last_quant_output_unscaled = None
#         self.build_lut(nn.functional.silu, q_type=q_type, bias=quantization_bias, quant_mode=quant_mode, chunk_size=chunk_size)
    
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         output_quant, output_unscaled = self.apply_lut(input)
        
#         if self.capture_activations:
#             self.last_quant_input = input.detach()
#             self.last_quant_output_unscaled = output_unscaled.detach()
        
#         return output_quant


@OpRegistry.register("QuantGELU", original_cls=nn.GELU, is_activation=True, compliance_status="FP32 activation")
class QuantGELU(nn.GELU, LUTActivation):
    """
    Quantized GELU using a piecewise approximation with a small LUT.
    
    Approximation:
      if x <= -A: y = 0
      if x >= +A: y = x
      else:       y = LUT[index(x)]
      
    Where index(x) maps [-A, +A] to [0, 255].
    """
    def __init__(self, approximate: str = 'none', q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None, A: float = 3.0):
        super().__init__(approximate=approximate)
        self.capture_activations = False
        self.last_quant_output_unscaled = None
        self.A = A
        
        # Build the specific LUT for this approximation
        self.build_piecewise_lut(A)
        
    def build_piecewise_lut(self, A):
        # L = 256, domain = [-A, +A]
        L = 256
        
        # Bin midpoints
        # x_i = -A + ( (i + 0.5) / L ) * (2*A) for i = 0..255
        i = torch.arange(L, dtype=torch.float32)
        x_i = -A + ((i + 0.5) / L) * (2 * A)
        
        # Calculate GELU(x_i)
        lut_values = nn.functional.gelu(x_i, approximate=self.approximate)
        
        # Continuity enforcement
        # Force LUT[0] = 0
        lut_values[0] = 0.0
        # Force LUT[255] = A
        lut_values[255] = A
        
        self.register_buffer('piecewise_lut', lut_values)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input x (FP32)
        x = input
        A = self.A
        
        # Calculate index for LUT region
        # t = (x + A) / (2*A)
        # We clamp t to [0, 1] to ensure safety against Inf and to keep index in bounds
        t = (x + A) / (2 * A)
        t = torch.clamp(t, 0.0, 1.0)
        i = torch.floor(t * 255.0).long()
        
        # LUT lookup
        y_lut = self.piecewise_lut[i]
        
        # Apply piecewise logic
        # if x <= -A: y = 0
        # if x >= +A: y = x
        # else:       y = LUT[i]
        
        y = torch.where(x <= -A, torch.tensor(0.0, device=x.device, dtype=x.dtype),
                        torch.where(x >= A, x, y_lut))
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = y.detach()
            
        return y