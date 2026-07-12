import pytest
import torch
import torch.nn as nn

from runspace.experiments.mixed_precision_experiment import (
    mixed_precision_experiment as experiment,
)


class _TransportRequiredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.transport_active = False
        self.forward_calls = 0

    def forward(self, inputs):
        assert self.transport_active
        self.forward_calls += 1
        return self.linear(inputs)


def test_fp8_collection_installs_uniform_stage_transport(monkeypatch):
    quant_model = _TransportRequiredModel()
    instances = []

    class FakeUniformInputQuantizer:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self.cleaned = False
            instances.append(self)

        def register_hooks(self):
            self.model.transport_active = True

        def cleanup(self):
            self.model.transport_active = False
            self.cleaned = True

    monkeypatch.setattr(
        experiment,
        "UniformInputQuantizer",
        FakeUniformInputQuantizer,
    )
    collector = experiment.MetricCollector(
        nn.Linear(4, 2),
        quant_model,
        torch.device("cpu"),
        activation_transport="reference",
        activation_chunk_size=128,
    )

    collector.run_lockstep_collection(
        [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))],
        num_batches=1,
        modes=("ref", "fp8"),
    )

    assert quant_model.forward_calls == 1
    assert len(instances) == 1
    assert instances[0].kwargs["fmt"] == "fp8_e4m3"
    assert instances[0].kwargs["transport"] == "reference"
    assert instances[0].cleaned is True
    assert quant_model.transport_active is False


def test_int8_mode_fails_before_model_evaluation(monkeypatch):
    quant_model = _TransportRequiredModel()

    def unexpected_quantizer(*_args, **_kwargs):
        raise AssertionError("INT8 validation must happen before transport setup")

    monkeypatch.setattr(experiment, "UniformInputQuantizer", unexpected_quantizer)
    collector = experiment.MetricCollector(
        nn.Linear(4, 2),
        quant_model,
        torch.device("cpu"),
        activation_transport="reference",
    )

    with pytest.raises(ValueError, match="INT8 activation mode is not supported"):
        collector.run_lockstep_collection(
            [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long))],
            num_batches=1,
            modes=("ref", "int8"),
        )

    assert quant_model.forward_calls == 0
