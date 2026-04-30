import torch
import torch.nn as nn
import torch.nn.functional as F
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.quantization.quantizer import quantize
from runspace.src.ops.quant_base import QuantizedLayerMixin

@OpRegistry.register("QuantLinear", original_cls=nn.Linear)
class QuantLinear(nn.Linear, QuantizedLayerMixin):
    q_type: str
    weight_scale: torch.Tensor | None
    weight_fp8: torch.Tensor | None

    """
    Quantized Linear layer (simulated weight quantization).
    All computation stays in float32 — weights are stored in dequantized form
    (weight_fp8 * weight_scale), so no native FP8 hardware is required.
    Supports any quantization format (fp2–fp8, ufp, efp, etc.).
    """
    def __init__(self, *args, q_type="fp8_e4m3", **kwargs):
        super().__init__(*args, **kwargs)
        self.q_type = q_type
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_fp8', None)

    # calibrate_weights is provided by QuantizedLayerMixin

    def forward(self, input):
        # This layer uses **simulated** quantization only.
        # weight_fp8 holds the quantized-then-dequantized weight values as float32.
        # No native torch.float8_* types are used; the old unconditional check was
        # wrong because it blocked all non-FP8 formats (fp4, fp7, etc.).

        # Quantize input (gated by self.input_quantization; pass-through if disabled)
        input_q = self.quantize_input(input)

        # Dequantize weights: w = w_fp8_sim * scale.
        # If calibration was skipped or weight quantization is disabled, use FP32 weights.
        if self.weight_fp8 is not None and self.weight_scale is not None:
            w_decomp = self.weight_fp8.float() * self.weight_scale
        else:
            w_decomp = self.weight

        if getattr(self, 'capture_activations', False):
            self.last_quant_weight = w_decomp.detach()

        return F.linear(input_q, w_decomp, self.bias)
