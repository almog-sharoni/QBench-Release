"""
Quantized pooling ops.

MaxPool preserves its inputs bit-for-bit (the output is one of the input
values), so when the upstream tensor is FP8-representable the pool output is
too. No quantization math is needed — this is a pass-through that participates
in activation capture and registry-based layer replacement.
"""

import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry


@OpRegistry.register("QuantMaxPool2d", original_cls=nn.MaxPool2d)
class QuantMaxPool2d(nn.MaxPool2d):
    """
    Pass-through MaxPool2d that supports activation capture.
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False,
                 q_type="fp8_e4m3", quantization_bias=None,
                 quant_mode="tensor", chunk_size=None):
        super().__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, return_indices=return_indices,
                         ceil_mode=ceil_mode)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.capture_activations = False
        self.last_quant_input_unscaled = None
        self.last_quant_output_unscaled = None

    @classmethod
    def from_native(cls, module: nn.MaxPool2d, q_type="fp8_e4m3", quantization_bias=None):
        return cls(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=module.return_indices,
            ceil_mode=module.ceil_mode,
            q_type=q_type,
            quantization_bias=quantization_bias,
        )

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        output = super().forward(input)

        if self.capture_activations:
            self.last_quant_input_unscaled = input.detach()
            self.last_quant_output_unscaled = (
                output[0].detach() if isinstance(output, tuple) else output.detach()
            )

        return output
