import torch
import torch.nn as nn
import torch.nn.functional as F
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import QuantizedLayerMixin
from runspace.src.quantization.quantizer import quantize

# @OpRegistry.register("QuantLayerNorm", original_cls=nn.LayerNorm, under_construction=True)
class QuantLayerNorm(nn.LayerNorm, QuantizedLayerMixin):
    q_type: str
    quantization_bias: int | None
    weight_scale: torch.Tensor | None
    weight_fp8: torch.Tensor | None

    """
    Quantized LayerNorm layer.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None, q_type="fp8_e4m3", quantization_bias: int | None = None):
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
        if self.elementwise_affine and self.weight_fp8 is not None and self.weight_scale is not None:
             w_decomp = self.weight_fp8.float() * self.weight_scale
             b = self.bias
        else:
             w_decomp = self.weight
             b = self.bias
             
        if getattr(self, 'capture_activations', False):
             if self.elementwise_affine:
                 self.last_quant_weight = w_decomp.detach()
        
        out = F.layer_norm(input_quant, self.normalized_shape, w_decomp, b, self.eps)

        return self.quantize_output(out)

try:
    from torchvision.models.convnext import LayerNorm2d
    
    @OpRegistry.register("QuantLayerNorm2d", original_cls=LayerNorm2d, under_construction=True)
    class QuantLayerNorm2d(QuantLayerNorm):
        q_type: str
        quantization_bias: int | None

        """
        Quantized LayerNorm2d layer (from ConvNext).
        Expects input (N, C, H, W), normalizes over C.
        """
        def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None, q_type="fp8_e4m3", quantization_bias: int | None = None):
            # LayerNorm2d in ConvNext takes normalized_shape as int (channels) or list
            # We pass it to QuantLayerNorm which expects it.
            super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype, q_type=q_type, quantization_bias=quantization_bias)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Input: (N, C, H, W)
            # Permute to (N, H, W, C) for LayerNorm
            x = input.permute(0, 2, 3, 1)

            # Use QuantLayerNorm forward (this captures last_natural_output /
            # last_quant_output in (N, H, W, C) shape via super().quantize_output).
            out = super().forward(x)

            # Permute back to (N, C, H, W) — the wrapper's natural output shape.
            out = out.permute(0, 3, 1, 2)

            # Re-capture in the wrapper's true (N, C, H, W) shape so report
            # columns key off the tensor that actually leaves this module.
            if getattr(self, 'capture_activations', False):
                self.last_natural_output = out.detach()
                if getattr(self, 'last_quant_output', None) is not None:
                    self.last_quant_output = out.detach()
            return out

except ImportError:
    pass
