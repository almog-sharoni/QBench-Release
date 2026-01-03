import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin
from ..quantization.quantizer import quantize

@OpRegistry.register("QuantLayerNorm", original_cls=nn.LayerNorm)
class QuantLayerNorm(nn.LayerNorm, QuantizedLayerMixin):
    """
    Quantized LayerNorm layer.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None, q_type="fp8_e4m3", quantization_bias: int = None):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)
        
    # calibrate_weights is provided by QuantizedLayerMixin

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Quantize Input (if enabled by mixin)
        # Note: LayerNorm usually needs high precision input, but we support input quantization if requested.
        input_quant = self.quantize_input(input)
        
        # Dequantize weights if they are quantized
        if self.elementwise_affine and self.weight_fp8 is not None:
             w_decomp = self.weight_fp8.float() * self.weight_scale
             b = self.bias
        else:
             w_decomp = self.weight
             b = self.bias
             
        if getattr(self, 'capture_activations', False):
             if self.elementwise_affine:
                 self.last_quant_weight = w_decomp.detach()
        
        out = F.layer_norm(input_quant, self.normalized_shape, w_decomp, b, self.eps)
        
        # Quantize Output
        out = quantize(out, q_type=self.q_type, bias=self.quantization_bias)
        
        # DEBUG
        # print(f"DEBUG: LN out min={out.min()}, max={out.max()}, q_type={self.q_type}, bias={self.quantization_bias}")
        
        return out
