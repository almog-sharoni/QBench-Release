import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from .quant_base import quantize_tensor

# @OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax)
# @OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax,is_activation=True, compliance_status = "treat softmax like activation", under_construction=True)
class QuantSoftmax(nn.Softmax):
    """
    1 Goal and Notation
    Quantized Softmax operation following a hardware-friendly paradigm.
    
    Paradigm:
    2 Boundary Quantization of the Input
    3 Numerically Stable Softmax (Max Subtraction)
    4 Exponential Approximation Using Base-2 Arithmetic
    5 Accumulation of the Denominator
    6 Reciprocal via Newton–Raphson
    7 Normalization and Output
    8 Output Quantization
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
        # 2 Boundary Quantization of the Input
        # 2.1 Power-of-two scale
        # 2.2 Quantize and dequantize
        input_dequant, _ = quantize_tensor(
            input, 
            q_type=self.q_type, 
            bias=self.quantization_bias, 
            mode=self.quant_mode, 
            chunk_size=self.chunk_size
        )
        
        # 3 Numerically Stable Softmax (Max Subtraction)
        dim = self.dim if self.dim is not None else -1
        max_val = input_dequant.amax(dim=dim, keepdim=True)
        x = input_dequant - max_val
        
        # 4 Exponential Approximation Using Base-2 Arithmetic
        log2e = 1.4426950408889634
        y = x * log2e
        
        # 4.1 Integer-fraction decomposition
        y_int = torch.floor(y)
        y_frac = y - y_int
        
        # 4.2 LUT for the fractional exponential
        pow2_int = torch.exp2(y_int)
        
        if self.exp2_lut is None or self.exp2_lut.device != input.device:
             steps = torch.arange(16, device=input.device, dtype=input.dtype) / 16.0
             self.exp2_lut = torch.exp2(steps)
             
        lut_idx = (y_frac * 16).long().clamp(0, 15)
        pow2_frac = self.exp2_lut[lut_idx]
        
        exp_val = pow2_int * pow2_frac
        
        # 5 Accumulation of the Denominator
        sum_exp = exp_val.sum(dim=dim, keepdim=True)
        sum_exp = torch.clamp(sum_exp, min=1e-12)
        
        # 6 Reciprocal 
        # 6.1 Seed
        r = 1.0 / sum_exp
        
        # # 6.2 Newton–Raphson iterations // currently using div
        # r = r * (2.0 - sum_exp * r)
        
        # 7 Normalization and Output
        prob = exp_val * r
        prob = torch.clamp(prob, 0.0, 1.0)
        
        # 8 Output Quantization #currently not needed
        # output_fp8, output_fp8_unscaled, max_val_out = quantize_tensor(
        #     prob, 
        #     q_type=self.q_type, 
        #     bias=self.quantization_bias, 
        #     return_unscaled=True,
        #     mode=self.quant_mode,
        #     chunk_size=self.chunk_size
        # )
        
        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_output_unscaled = prob.detach()
            
        return prob
