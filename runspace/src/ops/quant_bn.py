import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry.op_registry import OpRegistry
from ..quantization.quantizer import quantize
from .quant_base import QuantizedLayerMixin, quantize_tensor

@OpRegistry.register("QuantBatchNorm2d", original_cls=nn.BatchNorm2d)
class QuantBatchNorm2d(nn.BatchNorm2d, QuantizedLayerMixin):
    """
    Quantized BatchNorm2d layer that simulates FP8 quantization.
    Quantizes input and weight (gamma). Bias, running_mean, running_var remain FP32.
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
    
        # Dequantize weights (gamma)
        if self.weight_fp8 is not None:
            w_decomp = self.weight_fp8.float() * self.weight_scale
            if getattr(self, 'capture_activations', False):
                self.last_quant_weight = w_decomp.detach()
        else:
            w_decomp = self.weight

        # Run BatchNorm
        # Quantize running stats if they exist
        rm_quant = self.running_mean
        rv_quant = self.running_var
        
        if self.running_mean is not None:
             # Use weight_mode for stats? Or just tensor?
             # running_mean is 1D [C]. quantize_tensor 'channel' mode on 1D falls back to 'tensor'.
             # So we use weight_mode to support 'chunk' if set, otherwise it will be 'tensor' (scalar).
             mode = getattr(self, 'weight_mode', 'tensor')
             chunk_size = getattr(self, 'weight_chunk_size', None)
             
             # rm_quant, rm_unscaled, _ = quantize_tensor(self.running_mean, q_type=self.q_type, bias=self.quantization_bias, mode=mode, chunk_size=chunk_size, return_unscaled=True)
             
             if getattr(self, 'capture_activations', False):
                 # if self.q_type == 'fp8_e4m3' and hasattr(torch, 'float8_e4m3fn'):
                 #     self.last_quant_rm = rm_unscaled.to(torch.float8_e4m3fn)
                 # elif self.q_type == 'fp8_e5m2' and hasattr(torch, 'float8_e5m2'):
                 #     self.last_quant_rm = rm_unscaled.to(torch.float8_e5m2)
                 # else:
                 #     self.last_quant_rm = rm_unscaled
                 self.last_quant_rm = self.running_mean
             
        if self.running_var is not None:
             mode = getattr(self, 'weight_mode', 'tensor')
             chunk_size = getattr(self, 'weight_chunk_size', None)
             # rv_quant, rv_unscaled, _ = quantize_tensor(self.running_var, q_type=self.q_type, bias=self.quantization_bias, mode=mode, chunk_size=chunk_size, return_unscaled=True)

             if getattr(self, 'capture_activations', False):
                 # if self.q_type == 'fp8_e4m3' and hasattr(torch, 'float8_e4m3fn'):
                 #     self.last_quant_rv = rv_unscaled.to(torch.float8_e4m3fn)
                 # elif self.q_type == 'fp8_e5m2' and hasattr(torch, 'float8_e5m2'):
                 #     self.last_quant_rv = rv_unscaled.to(torch.float8_e5m2)
                 # else:
                 #     self.last_quant_rv = rv_unscaled
                 self.last_quant_rv = self.running_var

        return F.batch_norm(
            input_fp8, 
            rm_quant, 
            rv_quant, 
            w_decomp, 
            self.bias, 
            self.training, 
            self.momentum, 
            self.eps
        )

