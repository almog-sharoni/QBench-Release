"""
Quantized Activation Functions with Look-Up Table (LUT)

These modules wrap standard activations and use a precomputed LUT for FP8 inference.
This ensures the entire dataflow through the network stays in FP8 and is optimized.
"""

import torch
import torch.nn as nn
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import quantize_tensor


@OpRegistry.register("QuantReLU", original_cls=nn.ReLU, is_activation=True, passthrough=True)
class QuantReLU(nn.ReLU):
    """
    Quantized ReLU using LUT.
    """
    def __init__(self, inplace: bool = False, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None, **kwargs):
        super().__init__(inplace=inplace)
        self.capture_activations = False
        self.quantization_bias = quantization_bias
        self.last_quant_output_unscaled = None

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
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
    def __init__(self, inplace: bool = False, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None, **kwargs):
        super().__init__(inplace=inplace)
        self.capture_activations = False
        self.quantization_bias = quantization_bias
        self.last_quant_output_unscaled = None
    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        # User requested modification: use standard relu6 instead of lut
        output_quant = nn.functional.relu6(input)
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = output_quant.detach()
        
        return output_quant


@OpRegistry.register("QuantSiLU", original_cls=nn.SiLU, is_activation=True, compliance_status="FP32 activation")
class QuantSiLU(nn.SiLU):
    """
    Quantized SiLU (Swish) using a piecewise approximation with a small LUT.
    
    Approximation:
      if x <= -A: y = 0
      if x >= +A: y = x
      else:       y = LUT[index(x)]
      
    Where index(x) maps [-A, +A] to [0, 255].
    """
    def __init__(self, inplace: bool = False, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None, A: float = 4.0, **kwargs):
        super().__init__(inplace=inplace)
        self.capture_activations = False
        self.quantization_bias = quantization_bias
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
        
        # Calculate SiLU(x_i)
        lut_values = nn.functional.silu(x_i)
        
        # Continuity enforcement
        # Force LUT[0] = 0
        lut_values[0] = 0.0
        # Force LUT[255] = A
        lut_values[255] = A
        
        self.register_buffer('piecewise_lut', lut_values)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        if not getattr(self, 'input_quantization', True):
            return nn.functional.silu(input, inplace=self.inplace)

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
        
        y = torch.where(x <= -A, 0.0,
                        torch.where(x >= A, x, y_lut))
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = y.detach()
            
        return y


@OpRegistry.register("QuantGELU", original_cls=nn.GELU, is_activation=True, compliance_status="FP32 activation")
class QuantGELU(nn.GELU):
    """
    Quantized GELU using a piecewise approximation with a small LUT.
    
    Approximation:
      if x <= -A: y = 0
      if x >= +A: y = x
      else:       y = LUT[index(x)]
      
    Where index(x) maps [-A, +A] to [0, 255].
    """
    def __init__(self, approximate: str = 'none', q_type="fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None, A: float = 4.0, **kwargs):
        super().__init__(approximate=approximate)
        self.capture_activations = False
        self.quantization_bias = quantization_bias
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
        if not getattr(self, 'input_quantization', True):
            return nn.functional.gelu(input, approximate=self.approximate)

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
        
        y = torch.where(x <= -A, 0.0,
                        torch.where(x >= A, x, y_lut))
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = y.detach()
            
        return y