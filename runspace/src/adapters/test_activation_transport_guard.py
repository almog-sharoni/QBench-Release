import pytest
import torch
import torch.nn as nn

from runspace.src.adapters.base_adapter import BaseAdapter
from runspace.src.adapters.generic_adapter import GenericAdapter
from runspace.src.quantization.activation_transport import ActivationTransport
from runspace.src.quantization.activation_transport_runtime import (
    ActivationTransportRuntime,
)


class _TestAdapter(BaseAdapter):
    def __init__(self, model):
        self.model = model
        self.guard = self._install_activation_transport_guard(model)

    def build_model(self, quantized=False):
        return self.model

    def prepare_batch(self, batch):
        return batch

    def forward(self, model, batch):
        return model(batch)

    def get_layer_names(self, model):
        return [name for name, _ in model.named_modules()]

    def create_metrics(self):
        return None


def _model_with_fake_quant_flag(q_type="fp8_e4m3"):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    model[0].input_quantization = True
    model[0].input_q_type = q_type
    return model


def test_direct_forward_rejects_module_activation_fake_quantization():
    adapter = _TestAdapter(_model_with_fake_quant_flag())

    with pytest.raises(AssertionError, match="producer-stage hardware transport"):
        adapter.forward(adapter.model, torch.randn(2, 4))


def test_fp32_boundary_does_not_require_transport():
    adapter = _TestAdapter(_model_with_fake_quant_flag(q_type="fp32"))

    assert adapter.guard is not None
    assert adapter.model(torch.randn(2, 4)).shape == (2, 2)


def test_stage_transport_disables_guarded_flags_during_forward():
    adapter = _TestAdapter(_model_with_fake_quant_flag())
    runtime = ActivationTransportRuntime(
        adapter.model,
        ActivationTransport(mode="reference", chunk_size=128),
        lambda _stage, tensor: tensor,
    )

    runtime.install()
    try:
        assert adapter.model(torch.randn(2, 4)).shape == (2, 2)
    finally:
        runtime.cleanup()

    with pytest.raises(AssertionError, match="producer-stage hardware transport"):
        adapter.model(torch.randn(2, 4))


def test_output_fake_quantization_is_guarded():
    model = nn.Sequential(nn.Linear(4, 2))
    model[0].output_quantization = True
    model[0].output_q_type = "fp8_e4m3"
    adapter = _TestAdapter(model)

    with pytest.raises(AssertionError, match="output_quantization"):
        adapter.model(torch.randn(2, 4))


def test_generic_adapter_installs_guard_on_built_model(monkeypatch):
    model = _model_with_fake_quant_flag()
    monkeypatch.setattr(
        GenericAdapter,
        "build_model",
        lambda _self, quantized=False: model,
    )

    adapter = GenericAdapter(model_name="guard_test", build_quantized=True)

    with pytest.raises(AssertionError, match="producer-stage hardware transport"):
        adapter.model(torch.randn(2, 4))


def test_guard_rejects_boundary_enabled_after_model_construction():
    model = nn.Sequential(nn.Linear(4, 2))
    adapter = _TestAdapter(model)
    assert adapter.model(torch.randn(2, 4)).shape == (2, 2)

    model[0].input_quantization = True
    model[0].input_q_type = "fp8_e4m3"

    with pytest.raises(AssertionError, match="input_quantization"):
        adapter.model(torch.randn(2, 4))


def test_every_public_generic_model_build_is_guarded(monkeypatch):
    monkeypatch.setattr(
        GenericAdapter,
        "_load_base_model",
        lambda _self: _model_with_fake_quant_flag(),
    )
    adapter = GenericAdapter(
        model_name="guard_rebuild_test",
        model_source="torchvision",
        build_quantized=False,
        fold_input_norm=False,
        fold_layers=False,
    )

    rebuilt = adapter.build_model(quantized=False)

    assert rebuilt._qbench_activation_transport_guarded is True
    with pytest.raises(AssertionError, match="producer-stage hardware transport"):
        rebuilt(torch.randn(2, 4))
