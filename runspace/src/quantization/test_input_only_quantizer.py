from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from runspace.src.quantization.input_only_quantizer import InputOnlyActivationQuantizer


def test_reference_input_only_transport_patches_and_restores_model_forward():
    class RecordingLinear(nn.Linear):
        def forward(self, value):
            self.last_input = value.detach().clone()
            return super().forward(value)

    model = RecordingLinear(4, 2, bias=False).eval()
    original_forward = model.forward
    quantizer = InputOnlyActivationQuantizer(
        model=model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
    )
    value = torch.tensor([[0.13, -0.27, 0.61, -1.19]], dtype=torch.float32)

    quantizer.register_hooks()
    output = model(value)
    stats = quantizer.get_final_stats()

    assert output.shape == (1, 2)
    assert not torch.equal(model.last_input, value)
    assert stats["transport"] == "reference"
    assert stats["transmission_count"] == 1
    assert stats["packet_count"] == 0
    assert stats["activation_plan"]["model_input"]["kind"] == "input"

    quantizer.cleanup()
    assert model.forward == original_forward


def test_input_only_transport_suspends_and_restores_legacy_boundary_flags():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    model[0].input_quantization = True
    model[0].output_quantization = False
    model[1].input_quantization = False
    model[1].output_quantization = True
    quantizer = InputOnlyActivationQuantizer(
        model=model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
    )

    quantizer.register_hooks()
    assert model[0].input_quantization is False
    assert model[0].output_quantization is False
    assert model[1].input_quantization is False
    assert model[1].output_quantization is False

    model(torch.randn(1, 4))
    quantizer.cleanup()

    assert model[0].input_quantization is True
    assert model[0].output_quantization is False
    assert model[1].input_quantization is False
    assert model[1].output_quantization is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA activation codec required")
def test_encoded_input_only_transport_uses_packet():
    model = nn.Linear(128, 4, bias=False).cuda().eval()
    quantizer = InputOnlyActivationQuantizer(
        model=model,
        fmt="fp8_e1m6",
        chunk_size=128,
        transport="encoded",
        collect_error_stats=False,
    )
    quantizer.register_hooks()
    output = model(torch.randn(1, 128, device="cuda"))
    stats = quantizer.get_final_stats()

    assert output.shape == (1, 4)
    assert stats["packet_count"] == 1
    assert stats["transmission_count"] == 1
    assert stats["encoded_bytes"] > 0
    assert stats["layer_stats"]["model_input"]["total_chunks"] == 1
    quantizer.cleanup()
