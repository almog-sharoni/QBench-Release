import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin

@OpRegistry.register("QuantDropout", original_cls=nn.Dropout)
class QuantDropout(nn.Dropout, QuantizedLayerMixin):
    """
    Quantized Dropout layer.
    Quantizes input to FP8 before applying dropout.
    """
    def __init__(self, p: float = 0.5, inplace: bool = False, q_type="fp8_e4m3", quantization_bias: int = None):
        super().__init__(p=p, inplace=inplace)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        # Dropout has no weights, so no weight buffers needed
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Quantize Input
        # This ensures the input values are on the FP8 grid before dropout passes them through (or drops them)
        input_quant = self.quantize_input(input)
        
        if getattr(self, 'capture_activations', False):
             # last_quant_input is already captured in quantize_input if enabled
             pass
        
        return super().forward(input_quant)
