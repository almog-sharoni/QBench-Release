import torch
import torch.nn as nn
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import QuantizedLayerMixin


@OpRegistry.register("QuantConv1d", original_cls=nn.Conv1d)
class QuantConv1d(nn.Conv1d, QuantizedLayerMixin):
    """
    Quantized 1D Convolution layer. Mirrors QuantConv2d for Conv1d-based models
    (e.g. SuperGlue's keypoint encoder and attention MLPs).
    """

    def __init__(self, *args, is_first_layer=False, q_type="fp8_e4m3",
                 simulate_tf32_accum=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_first_layer = is_first_layer
        self.q_type = q_type
        self.simulate_tf32_accum = simulate_tf32_accum
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)

    def forward(self, input):
        if self.weight_fp8 is not None and self.weight_scale is not None:
            w_decomp = self.weight_fp8.float() * self.weight_scale
        else:
            w_decomp = self.weight

        if self.is_first_layer and not getattr(self, 'quantize_first_layer', False):
            input_q = input.float()
        else:
            input_q = self.quantize_input(input)

        if getattr(self, 'capture_activations', False):
            self.last_quant_weight = w_decomp.detach()

        if self.simulate_tf32_accum:
            from runspace.src.utils.tf32_patcher import TF32Patcher
            with TF32Patcher():
                output = nn.functional.conv1d(
                    input_q, w_decomp, None,
                    self.stride, self.padding, self.dilation, self.groups
                )
                if self.bias is not None:
                    output = output + self.bias.view(1, -1, 1)
                return self.quantize_output(output)
        else:
            return self.quantize_output(nn.functional.conv1d(
                input_q, w_decomp, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            ))
