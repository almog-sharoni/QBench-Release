import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin

@OpRegistry.register("QuantStochasticDepth", original_cls=StochasticDepth)
class QuantStochasticDepth(StochasticDepth, QuantizedLayerMixin):
    """
    Quantized StochasticDepth layer.
    Quantizes input to FP8 before applying stochastic depth.
    """
    def __init__(self, p: float = 0.5, mode="row", q_type="fp8_e4m3", quantization_bias: int = None):
        super().__init__(p=p, mode=mode)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        # Dropout has no weights, so no weight buffers needed
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Quantize Input
        # This ensures the input values are on the FP8 grid before stochastic depth passes them through (or drops them)
        input_quant = self.quantize_input(input)
        
        if getattr(self, 'capture_activations', False):
             # last_quant_input is already captured in quantize_input if enabled
             pass
        
        return super().forward(input_quant)
