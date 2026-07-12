import inspect
import operator
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from runspace.core.runner import Runner
from runspace.src.adapters.slm_adapter import (
    SLMAdapter,
    _install_activation_trace_provider,
    _mark_activation_metadata_bypasses,
    _slm_activation_planner_kwargs,
)
from runspace.src.quantization.activation_stage_planner import plan_activation_stages
from runspace.src.quantization.uniform_input_quantizer import UniformInputQuantizer


def test_opt125m_config_defaults_to_encoded_uniform_transport():
    config_path = Path(__file__).parents[1] / "configs" / "opt125m_fp8.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["adapter"]["type"] == "slm"
    assert Runner._activation_transport_mode(config) == "uniform"
    implicit = Runner._implicit_uniform_input_quant_cfg(config)
    assert implicit["enabled"] is True
    assert implicit["mode"] == "uniform"
    assert implicit["transport"] == "encoded"


class _DictOutputLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input_ids = None
        self.last_attention_mask = None

    def forward(self, *, input_ids, attention_mask):
        self.last_input_ids = input_ids
        self.last_attention_mask = attention_mask
        return {"logits": input_ids.float().unsqueeze(-1)}


def test_slm_batch_keeps_token_ids_and_attention_mask_as_integer_metadata():
    adapter = object.__new__(SLMAdapter)
    model_inputs, labels = adapter.prepare_batch(
        {
            "input_ids": [[2, 4, 6]],
            "attention_mask": [[True, True, False]],
        }
    )

    assert set(model_inputs) == {"input_ids", "attention_mask"}
    assert model_inputs["input_ids"].dtype == torch.long
    assert model_inputs["attention_mask"].dtype == torch.long
    assert torch.equal(labels, model_inputs["input_ids"])

    model = _DictOutputLM()
    logits = adapter.forward(model, (model_inputs, labels))
    assert logits.shape == (1, 3, 1)
    assert model.last_input_ids.dtype == torch.long
    assert model.last_attention_mask.dtype == torch.long


def test_slm_batch_synthesizes_mask_and_rejects_wrong_shape():
    adapter = object.__new__(SLMAdapter)
    model_inputs, _ = adapter.prepare_batch(torch.tensor([[2, 3, 4]]))
    assert torch.equal(
        model_inputs["attention_mask"],
        torch.ones_like(model_inputs["input_ids"]),
    )

    with pytest.raises(ValueError, match="same shape"):
        adapter.prepare_batch(
            {
                "input_ids": torch.tensor([[2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        )


class _TinyMaskedEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        mask = 1.0 - attention_mask.to(hidden.dtype)
        return hidden + mask.unsqueeze(-1)


def test_slm_metadata_tag_stops_at_embedding_and_mixed_activation():
    graph_module = torch.fx.symbolic_trace(_TinyMaskedEmbedding().eval())
    _mark_activation_metadata_bypasses(graph_module)
    nodes = {node.name: node for node in graph_module.graph.nodes}

    assert nodes["input_ids"].meta["qbench_activation_bypass"]
    assert nodes["attention_mask"].meta["qbench_activation_bypass"]
    assert nodes["sub"].meta["qbench_activation_bypass"]
    assert "qbench_activation_bypass" not in nodes["embedding"].meta

    add_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is operator.add
    )
    assert "qbench_activation_bypass" not in add_node.meta

    plan = plan_activation_stages(
        graph_module,
        **_slm_activation_planner_kwargs(),
    )
    assert plan.stage_for_node("embedding").kind.value == "compute"
    assert plan.stage_for_node("sub").kind.value == "compute"


def test_hf_opt_trace_provider_is_exact_and_plannable():
    transformers = pytest.importorskip("transformers")
    config = transformers.OPTConfig(
        vocab_size=32,
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        word_embed_proj_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = transformers.OPTForCausalLM(config).eval()
    _install_activation_trace_provider(model)

    graph_module, planner_kwargs = model._qbench_activation_trace_provider()
    assert config.use_cache is True
    assert tuple(inspect.signature(graph_module.forward).parameters) == (
        "input_ids",
        "attention_mask",
    )

    input_ids = torch.tensor([[2, 5, 7, 9]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        actual = graph_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["logits"]
    assert torch.equal(actual, expected)

    plan = plan_activation_stages(graph_module, **planner_kwargs)
    assert len(plan.stages) > 0
    input_stage = plan.stage_for_node("input_ids")
    embedding_stage = next(
        stage
        for stage in plan.stages
        if any("embed_tokens" in name for name in stage.node_names)
    )
    node_map = {node.name: node for node in graph_module.graph.nodes}
    assert node_map[input_stage.output_node].meta["qbench_activation_bypass"]
    assert "qbench_activation_bypass" not in node_map[embedding_stage.output_node].meta


def test_tiny_opt_uniform_transport_bypasses_metadata_and_transmits_activations():
    transformers = pytest.importorskip("transformers")
    config = transformers.OPTConfig(
        vocab_size=32,
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        word_embed_proj_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = transformers.OPTForCausalLM(config).eval()
    _install_activation_trace_provider(model)
    quantizer = UniformInputQuantizer(
        model,
        fmt="fp8_e4m3",
        chunk_size=8,
        quant_mode="chunk",
        transport="reference",
    )
    quantizer.register_hooks()

    adapter = object.__new__(SLMAdapter)
    model_inputs, labels = adapter.prepare_batch(
        {
            "input_ids": torch.tensor([[2, 5, 7, 9]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        }
    )
    try:
        with torch.inference_mode():
            logits = adapter.forward(model, (model_inputs, labels))
        runtime = quantizer._transport_runtime
        nodes = {node.name: node for node in runtime.plan.graph_module.graph.nodes}
        metadata_stage_ids = {
            stage.stage_id
            for stage in runtime.plan.stages
            if nodes[stage.output_node].meta.get("qbench_activation_bypass")
        }
        embedding_stage_ids = {
            stage.stage_id
            for stage in runtime.plan.stages
            if any("embed_tokens" in name for name in stage.node_names)
        }

        assert logits.shape == (1, 4, config.vocab_size)
        assert metadata_stage_ids
        assert all(
            runtime.stage_transmissions.get(stage_id, 0) == 0
            for stage_id in metadata_stage_ids
        )
        assert embedding_stage_ids
        assert all(
            runtime.stage_transmissions.get(stage_id, 0) > 0
            for stage_id in embedding_stage_ids
        )
        assert runtime.transmission_count > 0
        assert runtime.packet_count == 0
    finally:
        quantizer.cleanup()


def test_quantized_tiny_opt_uses_hf_trace_provider_and_uniform_transport():
    transformers = pytest.importorskip("transformers")
    config = transformers.OPTConfig(
        vocab_size=32,
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        word_embed_proj_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    base_model = transformers.OPTForCausalLM(config).eval()
    adapter = SLMAdapter(
        model_name="tiny-random-opt",
        model=base_model,
        model_source="huggingface",
        input_quantization=False,
        output_quantization=False,
        weight_quantization=False,
        quantize_first_layer=False,
        quantized_ops=["all"],
        # A real low-precision q_type verifies that stage transport suppresses
        # every legacy module/internal activation fake-quant path on CPU.
        quantization_type="fp8_e4m3",
        quant_mode="chunk",
        chunk_size=8,
        skip_calibration=True,
        build_quantized=True,
    )
    module_types = {type(module).__name__ for module in adapter.model.modules()}
    assert "QuantLayerNorm" in module_types
    assert "QuantOPTAttention" in module_types

    quantizer = UniformInputQuantizer(
        adapter.model,
        fmt="fp8_e4m3",
        chunk_size=8,
        quant_mode="chunk",
        transport="reference",
    )
    quantizer.register_hooks()
    model_inputs, labels = adapter.prepare_batch(
        {
            "input_ids": torch.tensor([[2, 5, 7, 9]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        }
    )
    try:
        with torch.inference_mode():
            logits = adapter.forward(adapter.model, (model_inputs, labels))
        runtime = quantizer._transport_runtime
        nodes = {node.name: node for node in runtime.plan.graph_module.graph.nodes}
        metadata_stage_ids = {
            stage.stage_id
            for stage in runtime.plan.stages
            if nodes[stage.output_node].meta.get("qbench_activation_bypass")
        }
        assert logits.shape == (1, 4, config.vocab_size)
        assert runtime.transmission_count > 0
        assert metadata_stage_ids
        assert all(
            runtime.stage_transmissions.get(stage_id, 0) == 0
            for stage_id in metadata_stage_ids
        )
    finally:
        quantizer.cleanup()
