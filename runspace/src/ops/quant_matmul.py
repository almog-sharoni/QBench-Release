import torch
import torch.nn as nn

from ..registry.op_registry import OpRegistry
from .quant_base import QuantizedLayerMixin


@OpRegistry.register("QuantMatMul", is_activation=False, compliance_status="FP8 MatMul")
class QuantMatMul(nn.Module, QuantizedLayerMixin):
    q_type: str
    quantization_bias: int | None
    quant_mode: str
    chunk_size: int | None
    input_quantization: bool

    """Explicitly quantized matrix multiplication."""
    def __init__(self, q_type: str = "fp8_e4m3", quantization_bias: int | None = None, quant_mode: str = "tensor", chunk_size: int | None = None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        q1 = self.quantize_input(input1)
        q2 = self.quantize_input(input2)
        return torch.matmul(q1, q2)


@OpRegistry.register("QuantBMM", is_activation=False, compliance_status="FP8 Batch MatMul")
class QuantBMM(nn.Module, QuantizedLayerMixin):
    q_type: str
    quantization_bias: int | None
    quant_mode: str
    chunk_size: int | None
    input_quantization: bool

    """Explicitly quantized batch matrix multiplication."""
    def __init__(self, q_type: str = "fp8_e4m3", quantization_bias: int | None = None, quant_mode: str = "tensor", chunk_size: int | None = None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        q1 = self.quantize_input(input1)
        q2 = self.quantize_input(input2)
        return torch.bmm(q1, q2)


__all__ = ["QuantMatMul", "QuantBMM"]
