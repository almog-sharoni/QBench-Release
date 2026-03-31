import torch
import torch.nn as nn
from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin

@OpRegistry.register("QuantAdd", is_activation=False, compliance_status="Fixed-Point Addition")
class QuantAdd(nn.Module, QuantizedLayerMixin):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1 = self.quantize_input(input)
        q2 = self.quantize_input(other)
        return torch.add(q1, q2)

@OpRegistry.register("QuantSub", is_activation=False, compliance_status="Fixed-Point Subtraction")
class QuantSub(nn.Module, QuantizedLayerMixin):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1 = self.quantize_input(input)
        q2 = self.quantize_input(other)
        return torch.sub(q1, q2)

@OpRegistry.register("QuantMul", is_activation=False, compliance_status="Fixed-Point Multiplication")
class QuantMul(nn.Module, QuantizedLayerMixin):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1 = self.quantize_input(input)
        q2 = self.quantize_input(other)
        return torch.mul(q1, q2)

@OpRegistry.register("QuantDiv", is_activation=False, compliance_status="Fixed-Point Division")
class QuantDiv(nn.Module, QuantizedLayerMixin):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1 = self.quantize_input(input)
        q2 = self.quantize_input(other)
        return torch.div(q1, q2)

@OpRegistry.register("QuantCat", is_activation=False, compliance_status="Token Concatenation")
class QuantCat(nn.Module, QuantizedLayerMixin):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="tensor", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = True

    def forward(self, tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        # Quantize all inputs
        quantized = [self.quantize_input(t) for t in tensors]
        return torch.cat(quantized, dim=dim)
