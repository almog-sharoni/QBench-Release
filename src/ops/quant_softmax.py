import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from .quant_base import quantize_tensor

@OpRegistry.register("QuantSoftmax", original_cls=nn.Softmax)
class QuantSoftmax(nn.Softmax):
    """
    Quantized Softmax operation.
    
    Since Softmax requires high precision for the exponentiation and summation,
    we perform the operation in FP32 (or BF16) and then quantize the output.
    
    Flow:
    1. Dequantize input (if it's FP8) to FP32.
    2. Apply Softmax in FP32.
    3. Quantize output to FP8.
    """
    def __init__(self, dim: int = None, q_type: str = "fp8_e4m3", quantization_bias: int = None, quant_mode: str = "tensor", chunk_size: int = None):
        super().__init__(dim=dim)
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.capture_activations = False
        self.last_quant_input = None
        self.last_quant_output_unscaled = None

    def forward(self, input: torch.Tensor) -> torch.Tensor: # TODO: do whole op in quantized space
        # 1. Dequantize input if necessary
        # We assume input might be FP8. If it is, we cast to float.
        # If it's already float, this is a no-op (copy).
        input_float = input.float()
        
        # 2. Apply Softmax in FP32
        output_float = super().forward(input_float)
        
        # 3. Quantize output to FP8
        # We use quantize_tensor from quant_base which handles scaling
        # Softmax output is in [0, 1], so it fits well in FP8 E4M3 (max 448) or E5M2 (max 57344)
        # However, E4M3 has more precision for small numbers which might be good for probabilities.
        # But E5M2 has more dynamic range.
        # Usually Softmax outputs are small, so precision matters.
        
        output_fp8, output_fp8_unscaled, max_val = quantize_tensor(
            output_float, 
            q_type=self.q_type, 
            bias=self.quantization_bias, 
            return_unscaled=True,
            mode=self.quant_mode,
            chunk_size=self.chunk_size
        )
        
        if self.capture_activations:
            self.last_quant_input = input.detach() # We capture the input as it was passed (likely FP8)
            self.last_quant_output_unscaled = output_fp8_unscaled.detach()
            
        return output_fp8
