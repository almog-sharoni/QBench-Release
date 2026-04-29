import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from .quant_base import quantize_tensor
from ..quantization.quantizer import round_fractional_part
from .quant_base import QuantizedLayerMixin

def qtype_to_unsigned_qtype(
    q_type: str,
    add_to_mant: bool = True
):
    if q_type == "fp32":
        return q_type
    
    # If already unsigned, return as is
    if q_type.startswith("u"):
        return q_type
        
    # Parse generic fp formats (e.g., fp8_e4m3)
    # When converting to unsigned, we free the sign bit and can add it to exp or mant.
    # For Softmax (outputs in [0, 1]), adding to mantissa (+1 precision) is generally preferred.
    import re
    match = re.match(r"(fp\d+)_e(\d+)m(\d+)", q_type)
    if match:
        prefix, exp, mant = match.groups()
        exp = int(exp)
        mant = int(mant)
        
        if add_to_mant:
            mant += 1
        else:
            exp += 1
            
        return f"u{prefix}_e{exp}m{mant}"
        
    # Fallback for other types
    return "u" + q_type

@OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax, is_activation=True)
class QuantSoftmax(nn.Softmax):
    q_type: str
    uq_type: str
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
    def __init__(self, dim: int | None = None, q_type: str = "fp8_e4m3", quantization_bias: int | None = None, quant_mode: str = "tensor", chunk_size: int | None = None, unsigned_input_sources: list | None = None):
        super().__init__(dim=dim)
        self.uq_type = qtype_to_unsigned_qtype(q_type, add_to_mant=True)
        self.q_type = q_type
        self.quantization_bias = None
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True
        self.capture_activations = False
        self.last_quant_input = None
        self.last_quant_output_unscaled = None
        self.exp2_lut = None
        self.unsigned_input_sources = [s.lower() for s in (unsigned_input_sources or [])]
        # self.mant_bits = get_mant_bits(q_type)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not getattr(self, 'input_quantization', True):
            prob = super().forward(input)
            if self.capture_activations:
                self.last_quant_input = input.detach()
                self.last_quant_input_unscaled = None
                self.last_quant_inputs_unscaled = []
                self.last_quant_input_formats = []
                self.last_quant_output_unscaled = None
            return prob

        # Quantization of the Input
        input_dequant, input_unscaled, _, _, _ = quantize_tensor(
            input,
            q_type=self.q_type,
            mode=self.quant_mode,
            chunk_size=self.chunk_size,
            return_unscaled=True,
            return_scale=True
        )
        # Numerically Stable Softmax (Max Subtraction)
        dim = self.dim if self.dim is not None else -1
        max_val = input_dequant.amax(dim=dim, keepdim=True)
        x = input_dequant - max_val

        # Exponential Approximation Using Base-2 Arithmetic
        log2e = 1.4453125 # 1.4426950408889634
        y = x * log2e
        
        # Integer-fraction decomposition #TODO: check this
        y_int = torch.floor(y)
        y_frac = y - y_int
        
        # LUT for the fractional exponential
        pow2_int = torch.exp2(y_int)
        y_frac = round_fractional_part(y_frac)
        pow2_frac = torch.exp2(y_frac)
        x_val = pow2_int * pow2_frac
        
        # Accumulation sum for division #TODO: maybe after scaleing
        sum_exp = x_val.sum(dim=dim, keepdim=True)
        sum_exp = torch.clamp(sum_exp, min=1e-14)
        
        # input quant again for second pipeline trough ipu
        # Use ufp if softmax is in unsigned_input_sources
        q_type_x = self.uq_type if any(s in self.unsigned_input_sources for s in ["softmax", "quantsoftmax"]) else self.q_type
        
        x_val, x_val_unscaled, _, _, _ = quantize_tensor(
            x_val,
            q_type=q_type_x,
            mode=self.quant_mode,
            chunk_size=self.chunk_size,
            return_unscaled=True,
            return_scale=True
        )

        
        # Division
        prob = x_val / sum_exp
        
        # # Final output quantization
        # prob, _ = quantize_tensor(
        #     prob,
        #     q_type=self.uq_type,
        #     mode=self.quant_mode,
        #     chunk_size=self.chunk_size
        # )

        if self.capture_activations:
            self.last_quant_input = input.detach()
            self.last_quant_input_unscaled = input_unscaled.detach()
            self.last_quant_inputs_unscaled = [
                input_unscaled.detach(),
                x_val_unscaled.detach(),
            ]
            self.last_quant_input_formats = [self.q_type, q_type_x]
            self.last_quant_output_unscaled = prob.detach()
            
        return prob

# class QuantSoftmax(nn.Softmax , QuantizedLayerMixin):
#     def __init__(self, dim: int | None = None, q_type: str = "fp8_e4m3", quantization_bias: int | None = None, quant_mode: str = "tensor", chunk_size: int | None = None):
#         super().__init__(dim=dim)
#         self.uq_type = qtype_to_unsigned_qtype(q_type)
#         self.q_type = q_type
#         self.quantization_bias = None
#         self.quant_mode = quant_mode
#         self.chunk_size = chunk_size
#         self.capture_activations = False
#         self.last_quant_input = None
#         self.last_quant_output_unscaled = None
#         self.exp2_lut = None
#         # self.mant_bits = get_mant_bits(q_type)

#     def forward(self, x):
#         x = self.quantize_input(x)
#         prob = super().forward(x)
#         # if self.capture_activations:
#         #     self.last_quant_input = x.detach()
#         #     self.last_quant_output_unscaled = prob.detach()
#         return prob
