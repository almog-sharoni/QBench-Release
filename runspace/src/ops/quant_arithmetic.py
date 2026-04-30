import torch
import torch.nn as nn
from runspace.src.registry.op_registry import OpRegistry
from runspace.src.ops.quant_base import QuantizedLayerMixin


class _QuantArithmeticBase(nn.Module, QuantizedLayerMixin):
    def _quantize_operands(self, operands, q_types=None):
        q_types = q_types or [None] * len(operands)
        quantized = []
        captured_unscaled = []
        captured_formats = []
        capture = getattr(self, 'capture_activations', False)

        for operand, q_type in zip(operands, q_types):
            if capture:
                self.last_quant_input_unscaled = None
            quantized_operand = self.quantize_input(operand, override_q_type=q_type)
            quantized.append(quantized_operand)

            if capture and getattr(self, 'input_quantization', True) and isinstance(operand, torch.Tensor):
                unscaled = getattr(self, 'last_quant_input_unscaled', None)
                if unscaled is not None:
                    captured_unscaled.append(unscaled.detach())
                    captured_formats.append(
                        q_type or getattr(self, 'input_q_type', getattr(self, 'q_type', 'fp8_e4m3'))
                    )

        if capture:
            self.last_quant_inputs_unscaled = captured_unscaled
            self.last_quant_input_formats = captured_formats

        return quantized

@OpRegistry.register("QuantAdd", is_activation=False, compliance_status="Fixed-Point Addition")
class QuantAdd(_QuantArithmeticBase):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="chunk", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1_type = getattr(self, 'input1_q_type', None)
        q2_type = getattr(self, 'input2_q_type', None)
        q1, q2 = self._quantize_operands([input, other], [q1_type, q2_type])
        return torch.add(q1, q2)

@OpRegistry.register("QuantSub", is_activation=False, compliance_status="Fixed-Point Subtraction")
class QuantSub(_QuantArithmeticBase):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="chunk", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1_type = getattr(self, 'input1_q_type', None)
        q2_type = getattr(self, 'input2_q_type', None)
        q1, q2 = self._quantize_operands([input, other], [q1_type, q2_type])
        return torch.sub(q1, q2)

@OpRegistry.register("QuantMul", is_activation=False, compliance_status="Fixed-Point Multiplication")
class QuantMul(_QuantArithmeticBase):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="chunk", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1_type = getattr(self, 'input1_q_type', None)
        q2_type = getattr(self, 'input2_q_type', None)
        q1, q2 = self._quantize_operands([input, other], [q1_type, q2_type])
        return torch.mul(q1, q2)

@OpRegistry.register("QuantDiv", is_activation=False, compliance_status="Fixed-Point Division")
class QuantDiv(_QuantArithmeticBase):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="chunk", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False

    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        q1_type = getattr(self, 'input1_q_type', None)
        q2_type = getattr(self, 'input2_q_type', None)
        q1, q2 = self._quantize_operands([input, other], [q1_type, q2_type])
        return torch.div(q1, q2)

@OpRegistry.register("QuantCat", is_activation=False, compliance_status="Token Concatenation")
class QuantCat(_QuantArithmeticBase):
    def __init__(self, q_type="fp8_e4m3", quantization_bias: int = None, quant_mode="chunk", chunk_size=None):
        super().__init__()
        self.q_type = q_type
        self.quantization_bias = quantization_bias
        self.quant_mode = quant_mode
        self.chunk_size = chunk_size
        self.input_quantization = False
    
    def forward(self, tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        # Quantize all inputs
        try:
            quantized = self._quantize_operands(tensors)
            return torch.cat(quantized, dim=dim)
        except Exception as e:
            print(f"QuantCat: Error in forward pass: {e}")
            raise e
