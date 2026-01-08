import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin

#not supported yet
@OpRegistry.register("QuantMaxPool2d", original_cls=nn.MaxPool2d)
class QuantMaxPool2d(nn.MaxPool2d, QuantizedLayerMixin):
    """
    Quantized MaxPool2d layer.
    """
    def __init__(self, *args, q_type="fp8_e4m3", **kwargs):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        # MaxPool doesn't have weights, so no weight_scale/weight_fp8 needed
        # But QuantizedLayerMixin might expect them if we call certain methods?
        # calibrate_weights checks for self.weight, so it's safe.
        
        self.capture_activations = False

    def forward(self, input):
        # Quantize input
        input_fp8 = self.quantize_input(input)
        
        # Run MaxPool
        output = super().forward(input_fp8)
        
        # Capture activations if enabled
        if getattr(self, 'capture_activations', False):
            # last_quant_input is already captured in quantize_input
            self.last_quant_output = output.detach()
            
        return output


@OpRegistry.register("QuantAdaptiveAvgPool2d", original_cls=nn.AdaptiveAvgPool2d)
class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, QuantizedLayerMixin):
    """
    Quantized AdaptiveAvgPool2d layer.
    """
    def __init__(self, *args, q_type="fp8_e4m3", **kwargs):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.capture_activations = False

    def forward(self, input):
        # Quantize input
        input_fp8 = self.quantize_input(input)
        
        # Run AvgPool
        # Note: AvgPool on quantized data (which are floats) works fine mathematically.
        # But the output might not be perfectly quantized anymore (averages of FP8 are not necessarily FP8).
        # Should we requantize the output?
        # For now, let's return the float result of averaging.
        # If the next layer is quantized, it will quantize this input anyway.
        output = super().forward(input_fp8)
        
        # Capture activations if enabled
        if getattr(self, 'capture_activations', False):
            self.last_quant_output = output.detach()
            
        return output
