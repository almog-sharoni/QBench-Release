import torch
import torch.nn as nn
import torch.nn.functional as F
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import quantize_tensor, QuantizedLayerMixin
from runspace.src.quantization.quantizer import quantize, round_fractional_part

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

def rsqrt_lut_approx_with_inv_n(x, n, lut_size=256, eps=1e-12):
    """
    Approximates:
        1 / sqrt((1/n) * x)
    """
    x = torch.clamp(x, min=eps)

    # Decompose x = 2^E * M
    E = torch.floor(torch.log2(x))
    M = x / torch.exp2(E)

    # LUT index for M in [1, 2)
    idx = torch.clamp(
        ((M - 1.0) * lut_size).long(),
        min=0,
        max=lut_size - 1
    )

    # LUT contains 1 / sqrt((1/n) * M)
    lut_m = 1.0 + torch.arange(
        lut_size,
        device=x.device,
        dtype=x.dtype
    ) / lut_size

    inv_n = 1.0 / float(n)
    lut = torch.rsqrt(inv_n * lut_m)
    # Enforce 17-bit precision (1s, 8e, 8m)
    lut = round_fractional_part(lut)

    mant_rsqrt_with_inv_n = lut[idx]

    # exponent correction
    exp_corr = torch.exp2(-0.5 * E)

    return exp_corr * mant_rsqrt_with_inv_n

@OpRegistry.register("QuantLayerNorm", original_cls=nn.LayerNorm)
class QuantLayerNorm(nn.LayerNorm, QuantizedLayerMixin):
    q_type: str
    uq_type: str
    quantization_bias: int | None
    quant_mode: str
    chunk_size: int | None
    capture_activations: bool
    last_quant_input: torch.Tensor | None
    last_quant_output_unscaled: torch.Tensor | None

    """
    Quantized LayerNorm operation following the pattern in QuantSoftmax.
    """
    def __init__(
        self, 
        normalized_shape, 
        eps=1e-5, 
        elementwise_affine=True, 
        device=None, 
        dtype=None, 
        q_type="fp8_e4m3", 
        quantization_bias: int | None = None,
        quant_mode: str = "tensor",
        chunk_size: int | None = None,
        unsigned_input_sources: list | None = None
    ):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)
        self.uq_type = qtype_to_unsigned_qtype(q_type, add_to_mant=True)
        self.q_type = q_type
        self.quantization_bias = None # Softmax style
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True
        self.capture_activations = False
        self.last_quant_input = None
        self.last_quant_output_unscaled = None
        self.unsigned_input_sources = [s.lower() for s in (unsigned_input_sources or [])]
        
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)

    # @classmethod
    # def from_native(cls, module: nn.LayerNorm, q_type="fp8_e4m3", quantization_bias: int | None = None, **kwargs):
    #     """
    #     Creates a QuantLayerNorm from a native nn.LayerNorm.
    #     """
    #     created = cls(
    #         normalized_shape=module.normalized_shape,
    #         eps=module.eps,
    #         elementwise_affine=module.elementwise_affine,
    #         q_type=q_type,
    #         quantization_bias=quantization_bias,
    #         **kwargs
    #     )
    #     if module.elementwise_affine:
    #         with torch.no_grad():
    #             created.weight.copy_(module.weight)
    #             if module.bias is not None:
    #                 created.bias.copy_(module.bias)
    #     return created

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_type = getattr(self, 'q_type', 'fp8_e4m3')
        capture = getattr(self, 'capture_activations', False)



        # 1. Quantization of the Input
        input_dequant = self.quantize_input(input)

        # 2. CORE LAYER NORM LOGIC
        # calc mean 
        mean = input_dequant.sum(dim=-1, keepdim=True)
        n = float(input_dequant.shape[-1])
        mean_neg = mean / -n 
        # quantize mean (w2m simulation)
        mean_neg = self.quantize_input(mean_neg, internal=True)

        # x - mean 
        diff = input_dequant + mean_neg
        diff_sq = diff * diff
        
        # calc variance via lut
        sum_diff_sq = diff_sq.sum(dim=-1, keepdim=True)
        variance_inv_rsqrt_n = rsqrt_lut_approx_with_inv_n(sum_diff_sq, n)

        variance_inv_rsqrt_n = self.quantize_input(variance_inv_rsqrt_n, internal=True)

        x_minux_man_div_var = diff * variance_inv_rsqrt_n

        x_minux_man_div_var = self.quantize_input(x_minux_man_div_var, internal=True)

        #quantize gamma and beta
        if self.weight is not None:
            quantized_gamma = self.quantize_input(self.weight, internal=True)
        else:
            quantized_gamma = 1.0

        if self.bias is not None:
            quantized_betta = self.quantize_input(self.bias, internal=True)
        else:
            quantized_betta = 0.0

        # (x - mean) / sqrt(var) * gamma + beta
        out = x_minux_man_div_var * quantized_gamma + quantized_betta
        

        # 3. Activation Capture and Output Quantization
        if capture:
            # We add the intermediate stages to the capture list.
            # Note: self.quantize_input already updated self.last_quant_input_unscaled
            # to the last stage (Beta). We append it here for completeness if needed.
            self.last_quant_inputs_unscaled = [
                getattr(self, 'last_quant_input_unscaled', None)
            ]
            self.last_quant_input_formats = [q_type]
            self.last_quant_output_unscaled = out.detach()

        return self.quantize_output(out)


