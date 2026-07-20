import torch.nn as nn
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import QuantizedLayerMixin


@OpRegistry.register("QuantMaxPool2d", original_cls=nn.MaxPool2d)
class QuantMaxPool2d(nn.MaxPool2d, QuantizedLayerMixin):
    """MaxPool2d compute wrapper for producer-stage activation transport."""

    def __init__(
        self,
        *args,
        q_type="fp8_e4m3",
        quantization_bias=None,
        quant_mode="tensor",
        chunk_size=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True
        self.output_quantization = False
        self.capture_activations = False

    def forward(self, input):
        if (
            getattr(self, "_qbench_activation_transport_active", False)
            and self.return_indices
        ):
            raise RuntimeError(
                "Producer-stage activation transport does not support "
                "MaxPool2d(return_indices=True): the stage output contains "
                "both activation and index tensors."
            )
        input_fp8 = self.quantize_input(input)
        output = super().forward(input_fp8)
        return self.quantize_output(output)


@OpRegistry.register("QuantAdaptiveAvgPool2d", original_cls=nn.AdaptiveAvgPool2d)
class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, QuantizedLayerMixin):
    """AdaptiveAvgPool2d compute wrapper for producer-stage transport."""

    def __init__(
        self,
        *args,
        q_type="fp8_e4m3",
        quantization_bias=None,
        quant_mode="tensor",
        chunk_size=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True
        self.output_quantization = False
        self.capture_activations = False

    def forward(self, input):
        input_fp8 = self.quantize_input(input)
        output = super().forward(input_fp8)
        return self.quantize_output(output)


@OpRegistry.register("QuantAvgPool2d", original_cls=nn.AvgPool2d)
class QuantAvgPool2d(nn.AvgPool2d, QuantizedLayerMixin):
    """AvgPool2d compute wrapper for producer-stage activation transport."""

    def __init__(
        self,
        *args,
        q_type="fp8_e4m3",
        quantization_bias=None,
        quant_mode="tensor",
        chunk_size=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True
        self.output_quantization = False
        self.capture_activations = False

    def forward(self, input):
        input_fp8 = self.quantize_input(input)
        output = super().forward(input_fp8)
        return self.quantize_output(output)
