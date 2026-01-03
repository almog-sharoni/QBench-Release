import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from ..quantization.quantizer import quantize
from .quant_base import QuantizedLayerMixin

@OpRegistry.register("QuantConv2d", original_cls=nn.Conv2d)
class QuantConv2d(nn.Conv2d, QuantizedLayerMixin):
    """
    Quantized Convolution layer that simulates FP8 quantization.
    """
    def __init__(self, *args, is_first_layer=False, q_type="fp8_e4m3", **kwargs):
        super().__init__(*args, **kwargs)
        self.is_first_layer = is_first_layer
        self.q_type = q_type
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)

    # calibrate_weights is provided by QuantizedLayerMixin

    def forward(self, input):
        # Check for FP8 support
        if not hasattr(torch, 'float8_e4m3fn'):
             raise RuntimeError("FP8 support (torch.float8_e4m3fn) is required but not available.")

        if self.is_first_layer:
            if not getattr(self, 'quantize_first_layer', False):
                # Do NOT quantize to FP8. Just cast to float for the operation.
                input_fp8 = input.float() 
                
                # For first layer, we still need w_decomp (weights are quantized)
                w_decomp = self.weight_fp8.float() * self.weight_scale
            else:
                # Use shared quantization logic
                input_fp8 = self.quantize_input(input.float())
            
                # Dequantize weights for operation: w = w_fp8 * s_w
                w_decomp = self.weight_fp8.float() * self.weight_scale

        else:
            # Use shared quantization logic
            input_fp8 = self.quantize_input(input)
        
            # Dequantize weights for operation: w = w_fp8 * s_w
            w_decomp = self.weight_fp8.float() * self.weight_scale
        
        # Capture activations if enabled
        if getattr(self, 'capture_activations', False):
            # last_quant_input is already captured in quantize_input if enabled
            # We just need to capture weight
            self.last_quant_weight = w_decomp.detach()
        
        return nn.functional.conv2d(
            input_fp8, 
            w_decomp, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

