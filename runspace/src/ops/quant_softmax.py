import torch
import torch.nn as nn
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import quantize_tensor
from runspace.src.quantization.quantizer import round_fractional_part
import re
from runspace.src.ops.quant_base import QuantizedLayerMixin

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
class QuantSoftmax(nn.Softmax, QuantizedLayerMixin):
    q_type: str
    uq_type: str
    quantization_bias: int | None
    quant_mode: str
    chunk_size: int | None
    capture_activations: bool
    last_quant_input: torch.Tensor | None
    last_quant_output_unscaled: torch.Tensor | None

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
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True
        self.capture_activations = False
        self.last_quant_input = None
        self.last_quant_output_unscaled = None
        self.unsigned_input_sources = [s.lower() for s in (unsigned_input_sources or [])]
        # self.mant_bits = get_mant_bits(q_type)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        capture = getattr(self, 'capture_activations', False)
        if not getattr(self, 'input_quantization', True):
            prob = super().forward(input)
            if capture:
                self.last_quant_input = input.detach()
                self.last_quant_input_unscaled = None
                self.last_quant_inputs_unscaled = []
                self.last_quant_input_formats = []
                self.last_quant_output_unscaled = None
                # Initialize other common capture attributes to prevent AttributeErrors in comparator
                self.last_quant_input_max = None
                self.last_quant_input_scale = None
                self.last_quant_output_max = None
                self.last_quant_output_scale = None
            return self.quantize_output(prob)

        # Quantization of the Input
        input_dequant = self.quantize_input(input)
        input_unscaled = getattr(self, 'last_quant_input_unscaled', None) if capture else None

        # Numerically Stable Softmax (Max Subtraction)
        dim = self.dim if self.dim is not None else -1
        max_val = input_dequant.amax(dim=dim, keepdim=True)
        x = input_dequant - max_val

        # Exponential Approximation Using Base-2 Arithmetic
        log2e = 1.4453125 # 1.4426950408889634
        y = x * log2e
        
        # Integer-fraction decomposition
        y_int = torch.floor(y)
        y_frac = y - y_int
        
        # LUT for the fractional exponential
        pow2_int = torch.exp2(y_int)
        y_frac = round_fractional_part(y_frac)
        pow2_frac = torch.exp2(y_frac)
        # Enforce 17-bit precision (1s, 8e, 8m) for the fractional LUT output
        pow2_frac = round_fractional_part(pow2_frac)
        x_val = pow2_int * pow2_frac
        
        # Accumulation sum for division
        sum_exp = x_val.sum(dim=dim, keepdim=True)
        sum_exp = torch.clamp(sum_exp, min=1e-14)
        
        # input quant again for second pipeline trough ipu
        # Use ufp if softmax is in unsigned_input_sources
        q_type_x = self.uq_type if any(s in self.unsigned_input_sources for s in ["softmax", "quantsoftmax"]) else self.q_type
        
        x_val = self.quantize_input(x_val, override_q_type=q_type_x, internal=True)
        x_val_unscaled = getattr(self, 'last_quant_input_unscaled', None) if capture else None

        
        # Division
        prob = x_val / sum_exp
        

        if capture:
            self.last_quant_input = input.detach()
            self.last_quant_inputs_unscaled = [
                input_unscaled.detach() if input_unscaled is not None else None,
                x_val_unscaled.detach() if x_val_unscaled is not None else None,
            ]
            self.last_quant_input_formats = [self.q_type, q_type_x]
            self.last_quant_output_unscaled = prob.detach()

        return self.quantize_output(prob)
