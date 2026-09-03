import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from runspace.src.adapters.generic_adapter import GenericAdapter


class ExpandedLinear(nn.Linear):
    """Linear-compatible storage with additional forward arithmetic."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("output_scale", torch.tensor(1.75))

    def forward(self, value):
        projected = F.linear(value, self.weight, self.bias)
        activated = F.relu(projected)
        return activated * self.output_scale


class TransparentLinear(nn.Linear):
    pass


class ProjectionModel(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, value):
        return self.projection(value)


def _build_adapter(model, *, enable_fx_quantization):
    return GenericAdapter(
        model_name="custom_linear_subclass_test",
        model=model,
        quantized_ops=["Linear", "ReLU", "Mul"],
        input_quantization=False,
        weight_quantization=False,
        output_quantization=False,
        fold_layers=False,
        fold_input_norm=False,
        skip_calibration=True,
        enable_fx_quantization=enable_fx_quantization,
    )


def test_overridden_linear_forward_is_preserved_without_fx():
    torch.manual_seed(1)
    source = ProjectionModel(ExpandedLinear(4, 3)).eval()
    expected_source = copy.deepcopy(source)
    value = torch.randn(2, 4)

    converted = _build_adapter(source, enable_fx_quantization=False).model.eval()

    assert isinstance(converted.projection, ExpandedLinear)
    assert type(converted.projection).__name__ != "QuantLinear"
    torch.testing.assert_close(converted(value), expected_source(value))


def test_overridden_linear_forward_is_decomposed_into_quantized_ops():
    torch.manual_seed(2)
    source = ProjectionModel(ExpandedLinear(4, 3)).eval()
    expected_source = copy.deepcopy(source)
    value = torch.randn(2, 4)

    converted = _build_adapter(source, enable_fx_quantization=True).model.cpu().eval()
    module_type_names = {type(module).__name__ for module in converted.modules()}

    assert isinstance(converted, torch.fx.GraphModule)
    assert {"QuantLinear", "QuantReLU", "QuantMul"} <= module_type_names
    assert "ExpandedLinear" not in module_type_names
    torch.testing.assert_close(converted(value), expected_source(value))


def test_transparent_linear_subclass_still_uses_direct_quantized_mapping():
    torch.manual_seed(3)
    source = ProjectionModel(TransparentLinear(4, 3)).eval()
    expected_source = copy.deepcopy(source)
    value = torch.randn(2, 4)

    converted = _build_adapter(source, enable_fx_quantization=False).model.eval()

    assert type(converted.projection).__name__ == "QuantLinear"
    torch.testing.assert_close(converted(value), expected_source(value))
