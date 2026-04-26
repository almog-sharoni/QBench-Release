import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from .quant_base import quantize_tensor
from ..quantization.quantizer import round_fractional_part



@OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax)
# @OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax,is_activation=True, compliance_status = "treat softmax like activation", under_construction=True)
class QuantSoftmax(nn.Softmax):
    q_type: str
    quantization_bias: int | None
    quant_mode: str
    chunk_size: int | None
    capture_activations: bool
    last_quant_input: torch.Tensor | None
    last_quant_output_unscaled: torch.Tensor | None
    exp2_lut: torch.Tensor | None

    """
    1 Goal and Notation
    Quantized Softmax operation following a hardware-friendly paradigm.
    
    Paradigm:
    2 Boundary Quantization of the Input
    3 Numerically Stable Softmax (Max Subtraction)
    4 Exponential Approximation Using Base-2 Arithmetic
    5 Accumulation of the Denominator
    7 Normalization and Output
    8 Output Quantization
    """
    def __init__(self, dim: int | None = None, q_type: str = "fp8_e4m3", quantization_bias: int | None = None, quant_mode: str = "tensor", chunk_size: int | None = None):
        super().__init__(dim=dim)
        self.q_type = q_type
        self.quantization_bias = None
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.capture_activations = False
        self.last_quant_input = None
        self.last_quant_output_unscaled = None
        self.exp2_lut = None
        # self.mant_bits = get_mant_bits(q_type)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 2 Boundary Quantization of the Input
        # 2.1 Power-of-two scale
        # 2.2 Quantize and dequantize
        input_dequant, _ = quantize_tensor(
            input,
            q_type=self.q_type,
            mode=self.quant_mode,
            chunk_size=self.chunk_size
        )
        # 3 Numerically Stable Softmax (Max Subtraction)
        dim = self.dim if self.dim is not None else -1
        max_val = input_dequant.amax(dim=dim, keepdim=True)
        x = input_dequant - max_val
        # 4 Exponential Approximation Using Base-2 Arithmetic
        log2e = 1.4453125 # 1.4426950408889634
        y = x * log2e
        
        # 4.1 Integer-fraction decomposition #TODO: check this
        y_int = torch.floor(y)
        y_frac = y - y_int
        
        # 4.2 LUT for the fractional exponential
        pow2_int = torch.exp2(y_int)
        
        # if self.exp2_lut is None or self.exp2_lut.device != input.device:
        #      steps = torch.arange(16, device=input.device, dtype=input.dtype) / 16.0
        #      self.exp2_lut = torch.exp2(steps)

        # y_frac = y_frac.view(torch.int32) & 0x007F8000
        # y_frac = y_frac.to(dtype=torch.float32)

        y_frac = round_fractional_part(y_frac)
        
        pow2_frac = torch.exp2(y_frac)
        
        
        x_val = pow2_int * pow2_frac
        
        
        # 5 Accumulation of the Denominator #TODO: maybe after scaleing
        sum_exp = x_val.sum(dim=dim, keepdim=True)
        sum_exp = torch.clamp(sum_exp, min=1e-14)
        
        x_val, _ = quantize_tensor(
            x_val,
            q_type=self.q_type,
            mode=self.quant_mode,
            chunk_size=self.chunk_size
        )

        prob = x_val / sum_exp
        # prob = torch.clamp(prob, 0.0, 1.0)
        
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
