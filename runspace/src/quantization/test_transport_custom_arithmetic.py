import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from runspace.src.adapters.generic_adapter import GenericAdapter
from runspace.src.ops.quant_activations import (
    QuantGELU,
    QuantHardsigmoid,
    QuantHardswish,
    QuantReLU,
    QuantReLU6,
    QuantSiLU,
)
from runspace.src.ops.quant_softmax import QuantSoftmax
from runspace.src.quantization.activation_transport import ActivationTransport
from runspace.src.quantization.activation_transport_runtime import (
    ActivationTransportRuntime,
)


def _install_leaf_transport(module, width=4, device="cpu"):
    module.input_quantization = True
    projection = nn.Linear(width, width, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(width))
    model = nn.Sequential(projection, module).to(device).eval()

    class CustomArithmeticLeafTracer(torch.fx.Tracer):
        def is_leaf_module(self, candidate, module_qualified_name):
            if candidate is module:
                return True
            return super().is_leaf_module(candidate, module_qualified_name)

    traced = torch.fx.GraphModule(
        model,
        CustomArithmeticLeafTracer().trace(model),
    )
    model._qbench_activation_trace_provider = lambda: traced
    runtime = ActivationTransportRuntime(
        model,
        ActivationTransport(mode="reference", chunk_size=width),
        lambda _stage, tensor: tensor,
    ).install()
    return model, runtime


def test_adapter_keeps_requested_functional_hardware_wrappers_when_boundaries_are_off():
    class FunctionalArithmetic(nn.Module):
        def forward(self, value):
            activated = F.gelu(value)
            probabilities = F.softmax(activated, dim=-1)
            return torch.add(probabilities, probabilities)

    adapter = GenericAdapter(
        model_name="functional_arithmetic_test",
        model=FunctionalArithmetic(),
        quantized_ops=["all"],
        input_quantization=False,
        weight_quantization=False,
        fold_layers=False,
        fold_input_norm=False,
        skip_calibration=True,
        enable_fx_quantization=True,
    )
    module_types = {type(module).__name__ for module in adapter.model.modules()}

    assert "QuantGELU" in module_types
    assert "QuantSoftmax" in module_types
    assert "QuantAdd" in module_types


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA quantization codec required")
def test_transport_preserves_softmax_internal_hardware_quantization():
    softmax = QuantSoftmax(
        dim=-1,
        q_type="fp4_e1m2",
        quant_mode="chunk",
        chunk_size=128,
    )
    internal_values = []
    original_quantize_input = softmax.quantize_input

    def observe_quantize_input(value, *args, **kwargs):
        result = original_quantize_input(value, *args, **kwargs)
        if kwargs.get("internal", False):
            internal_values.append((value.detach().clone(), result.detach().clone()))
        return result

    softmax.quantize_input = observe_quantize_input
    model, runtime = _install_leaf_transport(
        softmax,
        width=128,
        device="cuda",
    )
    try:
        output = model(torch.linspace(-1.0, 1.0, 128, device="cuda").unsqueeze(0))

        assert output.shape == (1, 128)
        assert not softmax.input_quantization
        assert internal_values
        assert any(
            not torch.equal(value, quantized)
            for value, quantized in internal_values
        )
    finally:
        runtime.cleanup()


@pytest.mark.parametrize(
    "activation",
    [QuantSiLU(), QuantHardswish(), QuantHardsigmoid(), QuantGELU()],
    ids=["silu", "hardswish", "hardsigmoid", "gelu"],
)
def test_transport_preserves_activation_lut_hardware_arithmetic(activation):
    activation.piecewise_lut.fill_(0.25)
    model, runtime = _install_leaf_transport(activation)
    try:
        output = model(torch.zeros(1, 4))

        assert not activation.input_quantization
        assert torch.equal(output, torch.full_like(output, 0.25))
    finally:
        runtime.cleanup()


@pytest.mark.parametrize(
    ("activation", "input_value", "expected"),
    [
        (QuantReLU(), -1.0, 0.0),
        (QuantReLU6(), 8.0, 6.0),
    ],
    ids=["relu", "relu6"],
)
def test_transport_preserves_simple_activation_arithmetic(
    activation,
    input_value,
    expected,
):
    model, runtime = _install_leaf_transport(activation)
    try:
        output = model(torch.full((1, 4), input_value))

        assert not activation.input_quantization
        assert torch.equal(output, torch.full_like(output, expected))
    finally:
        runtime.cleanup()
