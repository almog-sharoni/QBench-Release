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
from runspace.src.ops.quant_pooling import (
    QuantAdaptiveAvgPool2d,
    QuantAvgPool2d,
    QuantMaxPool2d,
)
from src.eval.comparator import LayerComparator


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


class _ShapeBookkeepingModel(nn.Module):
    def forward(self, value):
        torch._assert(value.dim() == 2, "expected a matrix")
        half_width = value.shape[1] // 2
        full_width = half_width * 2
        expanded = value.expand(value.shape[0], full_width)
        return expanded * 2.0


class _TimmStylePoolHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(4, 2)

    def forward(self, value):
        value = self.conv(value)
        value = self.pool(value)
        value = self.flatten(value)
        return self.head(value)


@pytest.mark.parametrize(
    ("pool", "expected_shape"),
    [
        (QuantMaxPool2d(kernel_size=2), (1, 2, 2)),
        (QuantAvgPool2d(kernel_size=2), (1, 2, 2)),
        (QuantAdaptiveAvgPool2d(output_size=(1, 1)), (1, 1, 1)),
    ],
)
def test_quant_pooling_uses_producer_stage_transport(pool, expected_shape):
    pool.capture_activations = True
    model = nn.Sequential(pool).eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
        collect_error_stats=False,
    )
    quantizer.register_hooks()
    output = model(torch.randn(1, 1, 4, 4))

    runtime = quantizer._transport_runtime
    pool_stage = runtime.plan.stage_for_node("_0")
    assert output.shape[1:] == expected_shape
    assert pool_stage.kind.value == "compute"
    assert runtime.stage_transmissions[pool_stage.stage_id] == 1
    assert runtime.transmission_count == len(runtime.plan.stages)
    assert pool.last_natural_output is not None
    assert getattr(pool, "last_quant_output", None) is None
    assert not pool.input_quantization
    assert not pool.output_quantization
    quantizer.cleanup()


def test_quant_max_pool_indices_fail_before_structured_transport():
    pool = QuantMaxPool2d(kernel_size=2, return_indices=True)
    model = nn.Sequential(pool).eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
        collect_error_stats=False,
    )
    quantizer.register_hooks()

    with pytest.raises(RuntimeError, match="return_indices=True"):
        model(torch.randn(1, 1, 4, 4))
    quantizer.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA activation codec required")
@pytest.mark.parametrize(
    "pool",
    [
        QuantMaxPool2d(kernel_size=2),
        QuantAvgPool2d(kernel_size=2),
        QuantAdaptiveAvgPool2d(output_size=(8, 8)),
    ],
)
def test_quant_pooling_uses_encoded_hardware_packets(pool):
    model = nn.Sequential(pool).cuda().eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp8_e4m3",
        chunk_size=128,
        transport="encoded",
        collect_error_stats=False,
    )
    quantizer.register_hooks()
    output = model(torch.randn(1, 1, 16, 16, device="cuda"))

    runtime = quantizer._transport_runtime
    pool_stage = runtime.plan.stage_for_node("_0")
    assert output.is_cuda
    assert runtime.stage_transmissions[pool_stage.stage_id] == 1
    assert runtime.packet_count == runtime.transmission_count
    assert runtime.transmission_count == len(runtime.plan.stages)
    quantizer.cleanup()


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
    assert runtime.stage_display_name(relu_stage) == "relu"
    assert runtime.stage_module_names(relu_stage) == ("producer", "relu")
    assert relu_stage.is_unsigned
    assert relu_stage.has_fanout
    assert encoded_stage_ids.count(relu_stage.stage_id) == 1
    assert observed_stage_ids.count(relu_stage.stage_id) == 1
    assert runtime.decode_reads > len(set(encoded_stage_ids))
    relu_plan = runtime.transport_stats()["activation_plan"][relu_stage.stage_id]
    assert relu_plan["layer_name"] == "relu"
    assert relu_plan["module_names"] == ["producer", "relu"]
    runtime.cleanup()


def test_comparator_reports_complete_hardware_transport_coverage():
    model = _FanoutModel().eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
    )
    quantizer.register_hooks()
    model(torch.randn(2, 4))

    comparator = object.__new__(LayerComparator)
    comparator.activation_quantizer = quantizer
    layers, summary = comparator._hardware_transport_coverage()

    assert summary["covered_stages"] == summary["total_stages"]
    assert summary["uncovered_stages"] == []
    assert summary["covered_module_inputs"] == summary["total_module_inputs"]
    assert summary["covered_module_outputs"] == summary["total_module_outputs"]
    assert layers["producer"]["input_covered"]
    assert layers["relu"]["output_covered"]
    assert layers["left"]["input_covered"]
    assert layers["right"]["input_covered"]
    quantizer.cleanup()


def test_comparator_excludes_shape_control_and_transparent_nodes_from_hw_coverage():
    model = _ShapeBookkeepingModel().eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
    )
    quantizer.register_hooks()
    model(torch.randn(2, 4))

    comparator = object.__new__(LayerComparator)
    comparator.activation_quantizer = quantizer
    plan = quantizer._transport_runtime.plan
    mul_roles = sorted(
        getattr(plan.node_roles[node.name], "value", plan.node_roles[node.name])
        for node in plan.graph_module.graph.nodes
        if getattr(node.target, "__name__", None) == "mul"
    )
    lines, uncovered_count = comparator._verify_coverage_fx()
    report = "\n".join(lines)

    assert mul_roles == ["compute", "non_tensor"]
    assert "Method: Hardware Activation Stage Plan + Runtime Packets" in report
    assert "100.0%" in report
    assert "Excluded shape/control nodes:" in report
    assert ".dim" in report
    assert "eq" in report
    assert "_assert" in report
    assert "floordiv" in report
    assert "mul" in report
    assert "Transport-preserving nodes:" in report
    assert ".expand" in report
    assert "Unquantized Unsupported Ops" not in report
    assert "All activation stages are runtime-quantized!" in report
    assert uncovered_count == 0
    quantizer.cleanup()


def test_comparator_covers_timm_style_transparent_flatten_route():
    model = _TimmStylePoolHead().eval()
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp4_e1m2",
        chunk_size=4,
        transport="reference",
    )
    quantizer.register_hooks()
    model(torch.randn(2, 3, 8, 8))

    comparator = object.__new__(LayerComparator)
    comparator.activation_quantizer = quantizer
    layers, summary = comparator._hardware_transport_coverage()

    assert layers["flatten"]["input_covered"]
    assert layers["flatten"]["output_covered"]
    assert layers["flatten"]["input_stage_ids"] == layers["flatten"]["output_stage_ids"]
    assert summary["covered_module_inputs"] == summary["total_module_inputs"]
    assert summary["covered_module_outputs"] == summary["total_module_outputs"]
    quantizer.cleanup()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA quantization codec required")
def test_runtime_preserves_layernorm_internal_hardware_quantization():
    norm = QuantLayerNorm(
        128,
        q_type="fp4_e1m2",
        quant_mode="chunk",
        chunk_size=128,
    )
    model = nn.Sequential(nn.Linear(128, 128), norm).cuda().eval()
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
        chunk_size=128,
        transport="reference",
    )
    quantizer.register_hooks()

    assert norm._qbench_activation_transport_active is True
    output = model(torch.randn(2, 128, device="cuda"))

    assert output.shape == (2, 128)
    assert internal_values
    assert any(
        not torch.equal(value, quantized)
        for value, quantized in internal_values
    )

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


def test_dynamic_signed_stage_overrides_unsigned_consumer_metadata():
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 2, bias=False),
    ).eval()
    # Reproduce the ViT failure mode: adapter metadata says this consumer uses
    # unsigned inputs even though its actual producer is a signed compute stage.
    model[1].input_q_type = "ufp2_e1m1"
    quantizer = DynamicInputQuantizer(
        model,
        metric="mse",
        chunk_size=4,
        candidate_formats=["fp2_e1m0"],
        unsigned_input_sources=["relu"],
        transport="reference",
    )

    assert "1" in quantizer.post_unsigned_layers
    quantizer.register_hooks()
    producer_stage = quantizer._transport_runtime.plan.stage_for_node("_0")

    assert not producer_stage.is_unsigned
    assert quantizer._producer_candidate_cache[producer_stage.stage_id] == [
        "fp2_e1m0"
    ]
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

    def consumer_candidates(
        layer_name,
        _module=None,
        input_index=0,
        producer_is_unsigned=None,
    ):
        del input_index, producer_is_unsigned
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
