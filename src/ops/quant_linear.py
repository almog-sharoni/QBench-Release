import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry.op_registry import OpRegistry
from ..quantization.quantizer import quantize
from .quant_base import QuantizedLayerMixin

@OpRegistry.register("QuantLinear", original_cls=nn.Linear)
class QuantLinear(nn.Linear, QuantizedLayerMixin):
    """
    Quantized Linear layer that simulates FP8 quantization.
    """
    def __init__(self, *args, q_type="fp8_e4m3", **kwargs):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)

    # calibrate_weights is provided by QuantizedLayerMixin

    def forward(self, input):
        # Check for FP8 support
        if not hasattr(torch, 'float8_e4m3fn'):
             raise RuntimeError("FP8 support (torch.float8_e4m3fn) is required but not available.")

        # Use shared quantization logic
        input_fp8 = self.quantize_input(input)
        
        # Dequantize weights for operation: w = w_fp8 * s_w
        w_decomp = self.weight_fp8.float() * self.weight_scale
        
        if getattr(self, 'capture_activations', False):
            # last_quant_input is already captured in quantize_input if enabled
            self.last_quant_weight = w_decomp.detach()

        return F.linear(input_fp8, w_decomp, self.bias)

