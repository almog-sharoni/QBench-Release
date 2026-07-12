from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from runspace.src.quantization.activation_transport import ActivationTransport
from runspace.src.quantization.activation_transport_runtime import ActivationTransportRuntime
from runspace.src.quantization.dynamic_input_quantizer import (
    ActivationProducerPolicyConflict,
    DynamicInputQuantizer,
)
from runspace.src.quantization.uniform_input_quantizer import UniformInputQuantizer
from runspace.src.ops.quant_ln import QuantLayerNorm


class _FanoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.producer = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()
        self.left = nn.Linear(4, 4, bias=False)
        self.right = nn.Linear(4, 4, bias=False)

    def forward(self, value):
        shared = self.relu(self.producer(value))
        return self.left(shared) + self.right(shared)


def test_runtime_encodes_one_shared_producer_for_fanout():
    model = _FanoutModel().eval()
    transport = ActivationTransport(mode="reference", chunk_size=4)
    encoded_stage_ids = []
    observed_stage_ids = []

    def encode(stage, tensor):
        encoded_stage_ids.append(stage.stage_id)
        q_type = "ufp4_e1m3" if stage.is_unsigned else "fp4_e1m2"
        return transport.transmit_uniform(tensor, q_type, producer_id=stage.stage_id)

    runtime = ActivationTransportRuntime(
        model,
        transport,
        encode,
        decode_observer=lambda stage_id, _decoded: observed_stage_ids.append(stage_id),
    ).install()
    relu_stage = next(stage for stage in runtime.plan.stages if "relu" in stage.node_names)

    output = model(torch.randn(2, 4))

    assert output.shape == (2, 4)
    assert relu_stage.node_names == ("producer", "relu")
    assert relu_stage.is_unsigned
    assert relu_stage.has_fanout
    assert encoded_stage_ids.count(relu_stage.stage_id) == 1
    assert observed_stage_ids.count(relu_stage.stage_id) == 1
    assert runtime.decode_reads > len(set(encoded_stage_ids))
    runtime.cleanup()


def test_runtime_handles_graph_module_without_forward_recursion():
    original = _FanoutModel().eval()
    graph_module = torch.fx.symbolic_trace(original)
    transport = ActivationTransport(mode="reference", chunk_size=4)

    def encode(stage, tensor):
        q_type = "ufp4_e1m3" if stage.is_unsigned else "fp4_e1m2"
        return transport.transmit_uniform(tensor, q_type, producer_id=stage.stage_id)

    runtime = ActivationTransportRuntime(graph_module, transport, encode).install()
    output = graph_module(torch.randn(2, 4))

    assert output.shape == (2, 4)
    assert runtime.graph_module is not graph_module
    assert runtime.graph_module.producer is graph_module.producer
    runtime.cleanup()


def test_runtime_uses_model_activation_trace_provider():
    model = _FanoutModel().eval()
    provided_graph = torch.fx.symbolic_trace(model)
    provider_calls = []

    def trace_provider():
        provider_calls.append(True)
        return provided_graph, {}

    model._qbench_activation_trace_provider = trace_provider
    transport = ActivationTransport(mode="reference", chunk_size=4)
    runtime = ActivationTransportRuntime(
        model,
        transport,
        lambda _stage, tensor: transport.transmit_uniform(
            tensor,
            "fp4_e1m2",
        ),
    ).install()

    output = model(torch.randn(2, 4))

    assert output.shape == (2, 4)
    assert provider_calls == [True]
    assert runtime.plan.graph_module is provided_graph
    runtime.cleanup()


def test_runtime_bypasses_integer_model_inputs_before_encoding():
    class TokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 4)
            self.projection = nn.Linear(4, 2)

        def forward(self, token_ids):
            return self.projection(self.embedding(token_ids))

    model = TokenModel().eval()
    transport = ActivationTransport(mode="reference", chunk_size=4)
    encoded_dtypes = []

    def encode(stage, tensor):
        encoded_dtypes.append(tensor.dtype)
        return transport.transmit_uniform(
            tensor,
            "fp4_e1m2",
            producer_id=stage.stage_id,
        )

    runtime = ActivationTransportRuntime(model, transport, encode).install()
    output = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

    assert output.shape == (1, 4, 2)
    assert encoded_dtypes
    assert all(dtype.is_floating_point for dtype in encoded_dtypes)
    assert runtime.transmission_count == len(encoded_dtypes)
    runtime.cleanup()


def test_runtime_blocks_legacy_internal_activation_fake_quantization():
    norm = QuantLayerNorm(4, q_type="fp4_e1m2", quant_mode="chunk", chunk_size=4)
    model = nn.Sequential(nn.Linear(4, 4), norm).eval()
    internal_values = []
    original_quantize_input = norm.quantize_input

    def observe_quantize_input(value, *args, **kwargs):
        result = original_quantize_input(value, *args, **kwargs)
        if kwargs.get("internal", False) and isinstance(value, torch.Tensor):
            internal_values.append((value.detach().clone(), result.detach().clone()))
        return result

    norm.quantize_input = observe_quantize_input

    class LayerNormLeafTracer(torch.fx.Tracer):
        def is_leaf_module(self, module, module_qualified_name):
            if module is norm:
                return True
            return super().is_leaf_module(module, module_qualified_name)

    traced = torch.fx.GraphModule(model, LayerNormLeafTracer().trace(model))
    model._qbench_activation_trace_provider = lambda: traced
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
    )
    quantizer.register_hooks()

    assert norm._qbench_activation_transport_active is True
    output = model(torch.randn(2, 4))

    assert output.shape == (2, 4)
    assert not internal_values

    quantizer.cleanup()
    assert not hasattr(norm, "_qbench_activation_transport_active")
    assert norm.input_quantization is True


def test_activation_off_blocks_layernorm_internal_fake_quantization():
    norm = QuantLayerNorm(4, q_type="fp4_e1m2", quant_mode="chunk", chunk_size=4)
    norm.input_quantization = False
    internal_values = []
    original_quantize_input = norm.quantize_input

    def observe_quantize_input(value, *args, **kwargs):
        result = original_quantize_input(value, *args, **kwargs)
        if kwargs.get("internal", False) and isinstance(value, torch.Tensor):
            internal_values.append((value.detach().clone(), result.detach().clone()))
        return result

    norm.quantize_input = observe_quantize_input
    output = norm(torch.randn(2, 4))

    assert output.shape == (2, 4)
    assert not internal_values


def test_uniform_quantizes_after_relu_and_restores_forward():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    baseline = copy.deepcopy(model)
    original_forward = model.forward
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
    )
    quantizer.register_hooks()

    value = torch.randn(2, 4)
    output = model(value)
    stats = quantizer.get_final_stats()
    relu_entry = next(
        entry
        for entry in stats["layer_stats"].values()
        if entry["unsigned_source"] == "relu"
    )

    assert output.shape == (2, 2)
    assert relu_entry["is_unsigned"] is True
    assert set(relu_entry["format_counts"]) == {"ufp4_e1m3"}
    assert stats["transport"] == "reference"
    assert stats["decode_reads"] > 0

    quantizer.cleanup()
    assert model.forward == original_forward
    torch.testing.assert_close(model(value), baseline(value))


def test_uniform_legacy_output_policy_preserves_input_and_layer_formats():
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 2, bias=False),
    ).eval()
    recorded_inputs = []
    input_hook = model[0].register_forward_pre_hook(
        lambda _module, args: recorded_inputs.append(args[0].detach().clone())
    )
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
        stage_format_policy={
            "producer_default": "fp4_e1m2",
            "consumer_default": None,
            "producer_overrides": {"0": "fp4_e2m1"},
            "consumer_overrides": {},
        },
    )
    value = torch.tensor([[0.13, -0.27, 0.61, -1.19]], dtype=torch.float32)

    quantizer.register_hooks()
    output = model(value)
    stats = quantizer.get_final_stats()

    assert output.shape == (1, 2)
    torch.testing.assert_close(recorded_inputs[-1], value, rtol=0, atol=0)
    assert stats["transmission_count"] == 2
    assert stats["decode_reads"] == 2
    assert all(entry["type"] != "input" for entry in stats["layer_stats"].values())
    assert {
        fmt
        for entry in stats["layer_stats"].values()
        for fmt in entry["format_counts"]
    } == {"fp4_e1m2", "fp4_e2m1"}
    quantizer.cleanup()
    input_hook.remove()


def test_uniform_legacy_policy_rejects_fanout_format_conflict():
    model = _FanoutModel().eval()
    original_forward = model.forward
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
        stage_format_policy={
            "producer_default": None,
            "consumer_default": "fp4_e1m2",
            "producer_overrides": {},
            "consumer_overrides": {"right": "fp4_e2m1"},
        },
    )

    with pytest.raises(ValueError, match="one hardware packet"):
        quantizer.register_hooks()

    assert quantizer._transport_runtime is None
    assert model.forward == original_forward


def test_uniform_legacy_policy_honors_skip_on_fused_softmax_output():
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Softmax(dim=-1),
        nn.Linear(4, 2, bias=False),
    ).eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
        stage_format_policy={
            "producer_default": "fp4_e1m2",
            "consumer_default": None,
            "producer_overrides": {"1": "fp32"},
            "consumer_overrides": {},
        },
    )

    quantizer.register_hooks()
    output = model(torch.randn(1, 4))
    stats = quantizer.get_final_stats()

    assert output.shape == (1, 2)
    assert stats["transmission_count"] == 1
    assert len(stats["layer_stats"]) == 1
    assert all(
        entry.get("unsigned_source") != "softmax"
        for entry in stats["layer_stats"].values()
    )
    quantizer.cleanup()


def test_dynamic_softmax_stage_uses_unsigned_candidate_pair():
    class AttentionLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Linear(4, 4)
            self.softmax = nn.Softmax(dim=-1)
            self.value = nn.Linear(4, 2)

        def forward(self, value):
            return self.value(self.softmax(self.score(value)))

    model = AttentionLike().eval()
    quantizer = DynamicInputQuantizer(
        model,
        metric="mse",
        chunk_size=4,
        candidate_formats=["fp4_e1m2", "fp4_e2m1"],
        transport="reference",
    )
    quantizer.register_hooks()
    output = model(torch.randn(2, 4))
    stats = quantizer.get_final_stats()
    softmax_stage = next(
        stage for stage in quantizer._transport_runtime.plan.stages if "softmax" in stage.node_names
    )
    softmax_stats = stats["layer_stats"][softmax_stage.stage_id]

    assert output.shape == (2, 2)
    assert softmax_stage.is_unsigned
    assert softmax_stage.unsigned_source == "softmax"
    assert set(softmax_stats["format_counts"]).issubset({"ufp4_e1m3", "ufp4_e2m2"})
    quantizer.cleanup()


def test_dynamic_fanout_rejects_conflicting_consumer_policies():
    model = _FanoutModel().eval()
    original_forward = model.forward
    quantizer = DynamicInputQuantizer(
        model,
        metric="mse",
        chunk_size=4,
        candidate_formats=["fp4_e1m2", "fp4_e2m1"],
        transport="reference",
    )

    def consumer_candidates(layer_name, _module=None, input_index=0):
        del input_index
        if layer_name == "left":
            return ["fp4_e1m2"]
        if layer_name == "right":
            return ["fp4_e2m1"]
        return ["fp4_e1m2", "fp4_e2m1"]

    quantizer._candidates_for_layer = consumer_candidates

    with pytest.raises(ActivationProducerPolicyConflict, match="one shared packet"):
        quantizer.register_hooks()

    assert quantizer._transport_runtime is None
    assert model.forward == original_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA activation codec required")
def test_uniform_encoded_runtime_uses_packets():
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU()).cuda().eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp8_e1m6",
        chunk_size=128,
        transport="encoded",
    )
    quantizer.register_hooks()
    output = model(torch.randn(1, 128, device="cuda"))
    stats = quantizer.get_final_stats()

    assert output.shape == (1, 128)
    assert stats["packet_count"] > 0
    assert stats["encoded_bytes"] > 0
    quantizer.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA activation codec required")
def test_uniform_encoded_runtime_quantizes_after_softmax():
    class SoftmaxConsumer(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=-1)
            self.consumer = nn.Linear(128, 8)

        def forward(self, value):
            return self.consumer(self.softmax(value))

    model = SoftmaxConsumer().cuda().eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp8_e1m6",
        chunk_size=128,
        transport="encoded",
    )
    quantizer.register_hooks()
    output = model(torch.randn(1, 128, device="cuda"))
    stats = quantizer.get_final_stats()
    softmax_stats = next(
        entry
        for entry in stats["layer_stats"].values()
        if entry.get("unsigned_source") == "softmax"
    )

    assert output.shape == (1, 8)
    assert set(softmax_stats["format_counts"]) == {"ufp8_e1m7"}
    assert stats["packet_count"] == stats["transmission_count"]
    quantizer.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA activation codec required")
def test_dynamic_encoded_runtime_uses_selector_layout_for_transformer_tensor():
    class TransformerStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(192)
            self.softmax = nn.Softmax(dim=-1)
            self.projection = nn.Linear(192, 64)

        def forward(self, value):
            return self.projection(self.softmax(self.norm(value)))

    model = TransformerStage().cuda().eval()
    quantizer = DynamicInputQuantizer(
        model,
        metric="mse",
        candidate_formats=["fp8_e1m6", "fp8_e2m5"],
        chunk_size=128,
        transport="encoded",
        collect_error_stats=False,
    )
    quantizer.register_hooks()
    output = model(torch.randn(1, 256, 192, device="cuda"))
    stats = quantizer.get_final_stats()

    assert output.shape == (1, 256, 64)
    assert stats["packet_count"] == stats["transmission_count"]
    assert any(
        entry.get("unsigned_source") == "softmax"
        for entry in stats["layer_stats"].values()
    )
    quantizer.cleanup()
