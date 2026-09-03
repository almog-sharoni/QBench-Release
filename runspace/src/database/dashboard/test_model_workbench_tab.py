from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

torch = pytest.importorskip("torch")


TAB_PATH = Path(__file__).with_name("model_workbench_tab.py")
APP_STARTUP_TIMEOUT = 90


def _app_source(
    *,
    capture_kind: str = "fx",
    validation_fails: bool = False,
    dataset_loader_fails: bool = False,
    dataset_benchmark_fails: bool = False,
    semantic_alias: bool = False,
    fp32_composite: bool = False,
    explicit_replacement_count: int = 0,
    replacement_validation_fails: bool = False,
) -> str:
    """Build a fast dashboard harness with an injected workbench backend."""

    operation_counts = (
        {}
        if capture_kind == "module_hierarchy"
        else {
            "exact_native_support": 2,
            "functional_support": 1,
            "unsupported": 1,
            "structural_passthrough": 2,
        }
    )
    operation_nodes = (
        []
        if capture_kind == "module_hierarchy"
        else [
            {
                "id": "module:projection",
                "label": "projection",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "exact_native_support",
                "module_path": "projection",
                "module_type": "Linear",
                "reason": "Fixture linear layer is convertible.",
            }
        ]
    )
    target_operation_nodes = (
        []
        if capture_kind == "module_hierarchy"
        else [
            {
                "id": "target:module:projection",
                "label": "QuantLinear",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "proposed_quantized",
                "module_path": "projection",
                "module_type": "QuantLinear",
                "reason": "Fixture converted linear layer.",
            }
        ]
    )
    mappings = (
        []
        if capture_kind == "module_hierarchy"
        else [
            {
                "source_node_ids": ["module:projection"],
                "target_node_ids": ["target:module:projection"],
                "kind": "one_to_one",
                "reason": "Fixture layer mapping.",
            }
        ]
    )
    if fp32_composite:
        operation_nodes = [
            {
                "id": "module:attention",
                "label": "attention",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "mixed_quantized_fp32",
                "module_path": "attention",
                "module_type": "LinearSelfAttention",
                "reason": "Composite attention mixes supported and FP32 operations.",
            },
            {
                "id": "module:attention.qkv",
                "label": "qkv",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "exact_native_support",
                "module_path": "attention.qkv",
                "module_type": "Conv2d",
                "parent": "module:attention",
                "reason": "Supported convolution descendant.",
            },
            {
                "id": "module:attention.dropout",
                "label": "dropout",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "exact_native_support",
                "module_path": "attention.dropout",
                "module_type": "Dropout",
                "parent": "module:attention",
                "reason": "Supported dropout descendant.",
            },
            {
                "id": "op:attention.softmax",
                "label": "softmax",
                "kind": "operation",
                "op": "call_function",
                "status": "functional_support",
                "module_path": "attention",
                "module_type": "LinearSelfAttention",
                "parent": "module:attention",
                "reason": "Supported functional softmax.",
            },
            {
                "id": "op:attention.sum",
                "label": "sum",
                "kind": "operation",
                "op": "call_method",
                "status": "unsupported",
                "module_path": "attention",
                "module_type": "LinearSelfAttention",
                "parent": "module:attention",
                "reason": "Unsupported reduction remains in FP32.",
            },
        ]
        target_operation_nodes = [
            {
                "id": "target:module:attention",
                "label": "Expanded · LinearSelfAttention",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "custom_expanded",
                "module_path": "attention",
                "module_type": "LinearSelfAttention",
                "reason": "Preview-only composite container.",
            },
            {
                "id": "target:module:attention.qkv",
                "label": "QuantConv2d",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "fp32_fallback",
                "module_path": "attention.qkv",
                "module_type": "QuantConv2d",
                "parent": "target:module:attention",
                "reason": "Converted convolution descendant.",
            },
            {
                "id": "target:module:attention.dropout",
                "label": "QuantDropout",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "fp32_fallback",
                "module_path": "attention.dropout",
                "module_type": "QuantDropout",
                "parent": "target:module:attention",
                "reason": "Converted dropout descendant.",
            },
            {
                "id": "target:op:attention.softmax",
                "label": "QuantSoftmax",
                "kind": "operation",
                "op": "call_function",
                "status": "fp32_fallback",
                "module_path": "attention",
                "module_type": "QuantSoftmax",
                "parent": "target:module:attention",
                "reason": "Converted functional softmax.",
            },
            {
                "id": "target:op:attention.sum",
                "label": "FP32 · sum",
                "kind": "operation",
                "op": "call_method",
                "status": "fp32_fallback",
                "module_path": "attention",
                "module_type": "sum",
                "parent": "target:module:attention",
                "reason": "Unsupported sum executes in FP32.",
            },
        ]
        mappings = [
            {
                "source_node_ids": ["module:attention"],
                "target_node_ids": ["target:module:attention"],
                "kind": "fp32_fallback",
                "reason": "Ancestor fallback marks the preview container.",
            },
            {
                "source_node_ids": ["module:attention"],
                "target_node_ids": [
                    "target:module:attention.qkv",
                    "target:module:attention.dropout",
                    "target:op:attention.softmax",
                    "target:op:attention.sum",
                ],
                "kind": "fp32_fallback",
                "reason": "Ancestor fallback taints every decomposed mapping.",
            },
            {
                "source_node_ids": ["module:attention.qkv"],
                "target_node_ids": ["target:module:attention.qkv"],
                "kind": "fp32_fallback",
                "reason": "Supported child mapping inherits ancestor fallback.",
            },
            {
                "source_node_ids": ["module:attention.dropout"],
                "target_node_ids": ["target:module:attention.dropout"],
                "kind": "fp32_fallback",
                "reason": "Supported child mapping inherits ancestor fallback.",
            },
            {
                "source_node_ids": ["op:attention.softmax"],
                "target_node_ids": ["target:op:attention.softmax"],
                "kind": "fp32_fallback",
                "reason": "Supported functional mapping inherits ancestor fallback.",
            },
            {
                "source_node_ids": ["op:attention.sum"],
                "target_node_ids": ["target:op:attention.sum"],
                "kind": "fp32_fallback",
                "reason": "Unsupported reduction remains in FP32.",
            },
        ]
    capture_details = {
        "sample_input_shape": [1, 3, 4, 4],
        "input_specialized": capture_kind == "torch_export",
        "full_fx_succeeded": capture_kind == "fx",
    }
    alias_source_nodes = []
    alias_target_nodes = []
    alias_mappings = []
    if semantic_alias:
        alias_source_nodes = [
            {
                "id": "module:stem.bn",
                "label": "bn",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "exact_native_support",
                "module_path": "stem.bn",
                "module_type": "BatchNormAct2d",
                "reason": "Fixture fused normalization group.",
            },
            {
                "id": "op:batch_norm",
                "label": "BatchNormAct2d",
                "kind": "operation",
                "op": "call_function",
                "status": "custom_expanded",
                "module_path": "stem.bn",
                "module_type": "BatchNormAct2d",
                "reason": "The same fused normalization execution.",
            },
        ]
        alias_target_nodes = [
            {
                "id": "target:module:stem.bn",
                "label": "QuantBatchNormAct2d",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "proposed_quantized",
                "module_path": "stem.bn",
                "module_type": "QuantBatchNormAct2d",
                "reason": "Fixture quantized normalization group.",
            },
            {
                "id": "target:op:batch_norm",
                "label": "QuantBatchNormAct2d",
                "kind": "operation",
                "op": "call_function",
                "status": "proposed_quantized",
                "module_path": "stem.bn",
                "module_type": "QuantBatchNormAct2d",
                "reason": "The same quantized normalization execution.",
            },
        ]
        alias_mappings = [
            {
                "source_node_ids": ["module:stem.bn"],
                "target_node_ids": ["target:module:stem.bn"],
                "kind": "one_to_one",
                "reason": "Fixture group rewrite.",
            },
            {
                "source_node_ids": ["op:batch_norm"],
                "target_node_ids": ["target:op:batch_norm"],
                "kind": "one_to_one",
                "reason": "Fixture operation rewrite.",
            },
        ]

    replacement_rows = []
    if explicit_replacement_count:
        operation_nodes = []
        target_operation_nodes = []
        mappings = []
        for index in range(explicit_replacement_count):
            path = f"blocks.{index}.projection"
            node_id = f"module:{path}"
            target_node_id = f"target:{node_id}"
            operation_nodes.append({
                "id": node_id,
                "label": path,
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "fp32_fallback",
                "module_path": path,
                "module_type": "MyLinear",
                "reason": "Custom projection requires an explicit user decision.",
            })
            target_operation_nodes.append({
                "id": target_node_id,
                "label": "FP32 · MyLinear",
                "kind": "module_group",
                "op": "module_group",
                "is_group": True,
                "status": "fp32_fallback",
                "module_path": path,
                "module_type": "MyLinear",
                "reason": "Analysis-time conservative fallback.",
            })
            mappings.append({
                "source_node_ids": [node_id],
                "target_node_ids": [target_node_id],
                "kind": "fp32_fallback",
                "reason": "Awaiting an explicit replacement recipe.",
            })
            replacement_rows.append({
                "node_id": node_id,
                "path": path,
                "type": "MyLinear",
                "status": "fp32_fallback",
                "reason": "No automatic equivalence is assumed.",
                "candidates": ["fp32"],
                "recommended": "replace",
                "custom_type": True,
            })

    return f"""
import copy
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as fixture_components
import torch

import runspace.src.quantization.model_workbench as backend
import runspace.src.utils.model_input_utils as input_utils


class _FixtureConvertedModel(torch.nn.Identity):
    pass


def _list_model_names(source):
    return ["fixture_model"]


def _load_model(source, model_name, pretrained=False, custom_factory=None):
    return torch.nn.Identity()


def _analyze_model(model, model_name="fixture_model", source="torchvision", sample_input=None):
    source_nodes = [
        {{
            "id": "module:$root",
            "label": model_name,
            "kind": "module_group",
            "type": "module_group",
            "op": "module_group",
            "status": "structural_passthrough",
            "module_path": "",
            "reason": "Fixture root.",
        }}
    ] + {operation_nodes!r} + {alias_source_nodes!r}
    target_nodes = [
        {{
            "id": "target:module:$root",
            "label": model_name,
            "kind": "module_group",
            "type": "module_group",
            "op": "module_group",
            "status": "structural_passthrough",
            "module_path": "",
            "reason": "Fixture target root.",
        }}
    ] + {target_operation_nodes!r} + {alias_target_nodes!r}
    return {{
        "model_name": model_name,
        "source": source,
        "capture_kind": {capture_kind!r},
        "capture_details": {capture_details!r},
        "schema_version": 2,
        "source_graph": {{"nodes": source_nodes, "edges": []}},
        "target_graph": {{"nodes": target_nodes, "edges": []}},
        "mappings": {mappings!r} + {alias_mappings!r},
        "summary": {{
            "supported_modules": 3,
            "unsupported_modules": 1,
            "passthrough_modules": 2,
            "operation_status_counts": {operation_counts!r},
            "source_node_count": len(source_nodes),
            "target_node_count": len(target_nodes),
            "capture_kind": {capture_kind!r},
            "capture_details": {capture_details!r},
            "schema_version": 2,
        }},
        "warnings": [],
        "module_rows": {replacement_rows!r},
        "trace_succeeded": {capture_kind == "fx"!r},
    }}


def _list_replacement_targets():
    return [{{
        "id": "qbench.quant_linear",
        "target_name": "QuantLinear",
        "target_type": "qbench.nn.QuantLinear",
        "native_name": "Linear",
        "native_type": "torch.nn.Linear",
        "constructor_parameters": ["in_features", "out_features", "bias"],
        "realizable": True,
    }}]


def _inspect_replacement_target(
    model,
    source_path,
    target_id,
    *,
    constructor_args=None,
    constructor_kwargs=None,
):
    calls = list(st.session_state.get("_fixture_replacement_inspections", []))
    calls.append({{
        "source_path": source_path,
        "target_id": target_id,
        "constructor_args": list(constructor_args or []),
        "constructor_kwargs": dict(constructor_kwargs or {{}}),
    }})
    st.session_state["_fixture_replacement_inspections"] = calls
    if target_id != "qbench.quant_linear":
        raise ValueError("fixture target is not catalog-backed")
    return {{
        "source": {{
            "path": source_path,
            "type": "fixture.MyLinear",
            "state_fields": [
                {{"key": "weight", "local_key": "weight", "qualified_key": f"{{source_path}}.weight", "kind": "parameter", "shape": [4, 4], "dtype": "torch.float32", "requires_grad": True}},
                {{"key": "bias", "local_key": "bias", "qualified_key": f"{{source_path}}.bias", "kind": "parameter", "shape": [4], "dtype": "torch.float32", "requires_grad": True}},
                {{"key": "legacy_scale", "local_key": "legacy_scale", "qualified_key": f"{{source_path}}.legacy_scale", "kind": "buffer", "shape": [1], "dtype": "torch.float32", "requires_grad": False}},
            ],
        }},
        "target": {{
            "path": source_path,
            "type": "qbench.nn.QuantLinear",
            "state_fields": [
                {{"key": "weight", "local_key": "weight", "qualified_key": "weight", "kind": "parameter", "shape": [4, 4], "dtype": "torch.float32", "requires_grad": True}},
                {{"key": "bias", "local_key": "bias", "qualified_key": "bias", "kind": "parameter", "shape": [4], "dtype": "torch.float32", "requires_grad": True}},
                {{"key": "scale", "local_key": "scale", "qualified_key": "scale", "kind": "buffer", "shape": [1], "dtype": "torch.float32", "requires_grad": False}},
            ],
        }},
        "suggested_state_mapping": {{"weight": "weight", "bias": "bias"}},
        "exact_name_shape_suggestions": {{"weight": ["weight"], "bias": ["bias"]}},
        "spec_template": {{
            "target_id": target_id,
            "constructor_args": list(constructor_args or []),
            "constructor_kwargs": dict(constructor_kwargs or {{}}),
        }},
    }}


def _validate_replacement_spec(model, source_path, spec):
    if {replacement_validation_fails!r}:
        raise ValueError("fixture backend rejected incompatible replacement")
    if not spec.get("confirmed"):
        raise ValueError("fixture requires explicit confirmation")
    target_fields = {{"weight", "bias", "scale"}}
    mapped = set(spec.get("state_mapping", {{}}))
    initialized = set(spec.get("state_initializers", {{}}))
    if mapped & initialized or mapped | initialized != target_fields:
        raise ValueError("fixture target-state coverage must be exact and disjoint")
    normalized = copy.deepcopy(dict(spec))
    calls = list(st.session_state.get("_fixture_validated_replacements", []))
    calls.append({{"source_path": source_path, "spec": normalized}})
    st.session_state["_fixture_validated_replacements"] = calls
    return normalized


def _build_conversion_plan(
    analysis,
    decisions=None,
    quant_options=None,
    replacement_specs=None,
    **overrides,
):
    plan = {{
        "model_name": analysis.get("model_name", "fixture_model"),
        "source": analysis.get("source", "torchvision"),
        "decisions": dict(decisions or {{}}),
        "quant_options": dict(quant_options or {{}}),
        "replacement_specs": copy.deepcopy(dict(replacement_specs or {{}})),
        "capture_kind": analysis.get("capture_kind", "fx"),
        "warnings": [],
    }}
    st.session_state["_fixture_last_plan"] = copy.deepcopy(plan)
    return plan


def _preview_conversion_plan(analysis, plan):
    target_graph = copy.deepcopy(analysis["target_graph"])
    mappings = copy.deepcopy(analysis["mappings"])
    for path, spec in plan.get("replacement_specs", {{}}).items():
        target_id = f"target:module:{{path}}"
        for node in target_graph.get("nodes", []):
            if node.get("id") == target_id:
                node["label"] = "QuantLinear"
                node["module_type"] = "QuantLinear"
                node["status"] = "proposed_quantized"
                node["reason"] = f"Explicit replacement {{spec['target_id']}}."
        for mapping in mappings:
            if target_id in mapping.get("target_node_ids", []):
                mapping["kind"] = "user_replacement"
                mapping["reason"] = f"User-confirmed catalog replacement {{spec['target_id']}}."
    st.session_state["_fixture_preview_plan"] = copy.deepcopy(plan)
    return target_graph, mappings


def _convert_model(model, plan):
    st.session_state["_fixture_built_plan"] = copy.deepcopy(plan)
    return {{
        "model": _FixtureConvertedModel(),
        "warnings": [],
        "recipe": {{
            "version": 1,
            "fixture": True,
            "replacement_specs": copy.deepcopy(plan.get("replacement_specs", {{}})),
        }},
        "realization": {{
            "total": 1,
            "by_type": {{"QuantLinear": 1}},
            "paths": ["projection"],
        }},
    }}


def _run_sample_inference(reference, quantized, sample_input):
    if {validation_fails!r}:
        raise RuntimeError("fixture inference boom")
    return {{
        "reference_output": torch.zeros(1, 2),
        "quantized_output": torch.zeros(1, 2),
        "comparison": {{
            "structure_match": True,
            "allclose": True,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "leaves": [],
        }},
        "runtime_audit": {{
            "quantized_modules_total": 1,
            "executed_quantized_modules": 1,
            "runtime_calls_by_type": {{"QuantLinear": 1}},
        }},
        "reference_summary": {{"type": "tensor", "shape": [1, 2]}},
        "quantized_summary": {{"type": "tensor", "shape": [1, 2]}},
    }}


def _build_classification_validation_loader(**kwargs):
    st.session_state["_fixture_dataset_loader_kwargs"] = kwargs
    if {dataset_loader_fails!r}:
        raise FileNotFoundError("fixture dataset missing")
    loader = [(torch.zeros(8, 3, 4, 4), torch.zeros(8, dtype=torch.long))]
    metadata = {{
        "dataset_kind": kwargs.get("dataset_kind"),
        "dataset_size": kwargs.get("max_samples"),
        "transform_source": "fixture weights",
    }}
    return loader, metadata


def _benchmark_classification_models(
    reference,
    quantized,
    data_loader,
    *,
    max_samples=None,
    device="auto",
    progress_callback=None,
):
    st.session_state["_fixture_benchmark_kwargs"] = {{
        "max_samples": max_samples,
        "device": device,
        "loader_length": len(data_loader),
    }}
    if {dataset_benchmark_fails!r}:
        raise RuntimeError("fixture benchmark boom")
    if progress_callback is not None:
        progress_callback(40, 40)
    return {{
        "device": "cpu",
        "samples": 40,
        "batches": 5,
        "class_dimension": 10,
        "effective_top5_k": 5,
        "reference": {{
            "top1_accuracy_percent": 75.0,
            "top5_accuracy_percent": 95.0,
            "correct_top1": 30,
            "correct_top5": 38,
            "elapsed_seconds": 2.0,
            "throughput_samples_per_second": 20.0,
        }},
        "quantized": {{
            "top1_accuracy_percent": 72.5,
            "top5_accuracy_percent": 92.5,
            "correct_top1": 29,
            "correct_top5": 37,
            "elapsed_seconds": 2.5,
            "throughput_samples_per_second": 16.0,
        }},
        "delta": {{
            "top1_accuracy_percentage_points": -2.5,
            "top5_accuracy_percentage_points": -2.5,
        }},
        "prediction_agreement_percent": 90.0,
        "prediction_agreement_count": 36,
    }}


backend.list_model_names = _list_model_names
backend.load_model = _load_model
backend.analyze_model = _analyze_model
backend.build_conversion_plan = _build_conversion_plan
backend.preview_conversion_plan = _preview_conversion_plan
backend.convert_model = _convert_model
backend.run_sample_inference = _run_sample_inference
backend.list_replacement_targets = _list_replacement_targets
backend.inspect_replacement_target = _inspect_replacement_target
backend.validate_replacement_spec = _validate_replacement_spec
backend.WORKBENCH_DATASET_BENCHMARK_API_VERSION = 1
backend.build_classification_validation_loader = _build_classification_validation_loader
backend.benchmark_classification_models = _benchmark_classification_models
input_utils.resolve_model_input_size = lambda model, batch_size=1: (batch_size, 3, 4, 4)

if not hasattr(fixture_components, "_mw_fixture_original_html"):
    fixture_components._mw_fixture_original_html = fixture_components.html


def _capture_component_html(body, *args, **kwargs):
    st.session_state["_fixture_component_html"] = body
    return fixture_components._mw_fixture_original_html(body, *args, **kwargs)


fixture_components.html = _capture_component_html

tab_workbench = st.tabs(["Workbench"])[0]
tab_path = {str(TAB_PATH)!r}
exec(compile(Path(tab_path).read_text(encoding="utf-8"), tab_path, "exec"))
"""


def _run_app(
    *,
    capture_kind: str = "fx",
    validation_fails: bool = False,
    dataset_loader_fails: bool = False,
    dataset_benchmark_fails: bool = False,
    semantic_alias: bool = False,
    fp32_composite: bool = False,
    explicit_replacement_count: int = 0,
    replacement_validation_fails: bool = False,
) -> AppTest:
    return AppTest.from_string(
        _app_source(
            capture_kind=capture_kind,
            validation_fails=validation_fails,
            dataset_loader_fails=dataset_loader_fails,
            dataset_benchmark_fails=dataset_benchmark_fails,
            semantic_alias=semantic_alias,
            fp32_composite=fp32_composite,
            explicit_replacement_count=explicit_replacement_count,
            replacement_validation_fails=replacement_validation_fails,
        ),
        default_timeout=APP_STARTUP_TIMEOUT,
    ).run(timeout=APP_STARTUP_TIMEOUT)


def _values(elements) -> list[str]:
    return [str(element.value) for element in elements]


def _metric_value(app: AppTest, label: str) -> str:
    return next(str(metric.value) for metric in app.metric if metric.label == label)


def _type_map_html(app: AppTest) -> str:
    return str(app.session_state["_fixture_component_html"])


def _type_map_elements(app: AppTest) -> list[dict]:
    html = _type_map_html(app)
    prefix = "const elements = "
    suffix = ";\n  const shell ="
    start = html.index(prefix) + len(prefix)
    end = html.index(suffix, start)
    return json.loads(html[start:end])


def _widget_with_label(elements, label: str):
    return next(element for element in elements if str(element.label) == label)


def test_model_workbench_initial_render_is_actionable():
    app = _run_app()

    assert not list(app.exception)
    assert not list(app.error)
    assert app.selectbox(key="mw_model_source").value == "torchvision"
    assert app.selectbox(key="mw_torchvision_model_name").value == "fixture_model"
    assert app.checkbox(key="mw_use_provider_input_shape").value is True
    assert app.button(key="mw_analyze").disabled is False
    assert any("Model Quantization Workbench" in value for value in _values(app.markdown))


@pytest.mark.parametrize(
    ("capture_kind", "message_kind", "message", "operation_coverage"),
    [
        ("fx", "success", "Graph capture: full quantization-aware FX", "3/4"),
        ("torch_export", "info", "Graph capture: input-specialized", "3/4"),
        (
            "module_hierarchy",
            "error",
            "Only the module hierarchy could be captured",
            "n/a",
        ),
    ],
)
def test_model_workbench_capture_status_and_coverage_metrics(
    capture_kind: str,
    message_kind: str,
    message: str,
    operation_coverage: str,
):
    app = _run_app(capture_kind=capture_kind)
    app.button(key="mw_analyze").click().run(timeout=30)

    assert not list(app.exception)
    assert any(message in value for value in _values(getattr(app, message_kind)))
    assert _metric_value(app, "Convertible modules") == "3/4"
    assert _metric_value(app, "Quantizable operations") == operation_coverage
    assert _metric_value(app, "FP32 operations") == (
        "0" if capture_kind == "module_hierarchy" else "1"
    )
    assert _metric_value(app, "Structural modules") == "2"
    assert _metric_value(app, "Mappings") == (
        "0" if capture_kind == "module_hierarchy" else "1"
    )
    if capture_kind != "module_hierarchy":
        assert app.selectbox(key="mw_graph_detail").value == "Layer-type overview"
        assert any(
            "Layer map: 1 original layer type → 1 converted layer type, "
            "with 1 conversion arrow." in value
            for value in _values(app.caption)
        )
        type_map_html = _type_map_html(app)
        assert 'data-conversion-edge-count="1"' in type_map_html
        assert type_map_html.count('"classes":"conversion-edge ') == 1


def test_stale_analysis_schema_is_hidden_until_reanalysis():
    app = _run_app()
    app.button(key="mw_analyze").click().run(timeout=30)
    app.session_state["_mw_analysis"]["schema_version"] = 1
    app.session_state["_mw_analysis"]["summary"]["schema_version"] = 1

    app.run(timeout=30)

    assert not list(app.exception)
    assert any(
        "older analysis engine and has been hidden" in value
        for value in _values(app.warning)
    )
    assert not list(app.metric)


def test_layer_type_overview_collapses_semantically_identical_fx_aliases():
    app = _run_app(semantic_alias=True)
    app.button(key="mw_analyze").click().run(timeout=30)

    assert not list(app.exception)
    assert app.selectbox(key="mw_graph_detail").value == "Layer-type overview"
    assert any(
        "Layer map: 2 original layer types → 2 converted layer types, "
        "with 2 conversion arrows." in value
        for value in _values(app.caption)
    )
    type_map_html = _type_map_html(app)
    assert 'data-conversion-edge-count="2"' in type_map_html
    assert type_map_html.count('"classes":"conversion-edge ') == 2


def test_fp32_composite_keeps_supported_children_and_explicit_sum_fallback():
    app = _run_app(fp32_composite=True)
    app.button(key="mw_analyze").click().run(timeout=30)

    assert not list(app.exception)
    assert app.selectbox(key="mw_graph_detail").value == "Layer-type overview"
    assert any(
        "Layer map: 3 original layer types → 4 converted layer types, "
        "with 6 conversion arrows." in value
        for value in _values(app.caption)
    )

    type_map_html = _type_map_html(app)
    assert 'data-conversion-edge-count="6"' in type_map_html
    elements = _type_map_elements(app)
    source_nodes = [
        element["data"]
        for element in elements
        if "source-type" in str(element.get("classes", ""))
    ]
    target_nodes = [
        element["data"]
        for element in elements
        if "target-type" in str(element.get("classes", ""))
    ]
    conversion_edges = [
        element["data"]
        for element in elements
        if "conversion-edge" in str(element.get("classes", ""))
    ]

    assert {node["module_type"] for node in source_nodes} == {
        "Conv2d",
        "Dropout",
        "LinearSelfAttention",
    }
    assert {node["module_type"] for node in target_nodes} == {
        "FP32 / sum",
        "QuantConv2d",
        "QuantDropout",
        "QuantSoftmax",
    }
    assert all(int(node["count"]) == 1 for node in source_nodes + target_nodes)

    edge_kinds = {
        (edge["source_type"], edge["target_type"]): edge["kind"]
        for edge in conversion_edges
    }
    assert edge_kinds == {
        ("Conv2d", "QuantConv2d"): "one_to_one",
        ("Dropout", "QuantDropout"): "one_to_one",
        ("LinearSelfAttention", "FP32 / sum"): "fp32_fallback",
        ("LinearSelfAttention", "QuantConv2d"): "decomposed",
        ("LinearSelfAttention", "QuantDropout"): "decomposed",
        ("LinearSelfAttention", "QuantSoftmax"): "decomposed",
    }
    assert {
        edge["target_type"]
        for edge in conversion_edges
        if edge["source_type"] == "Conv2d"
    } == {"QuantConv2d"}
    assert {
        edge["target_type"]
        for edge in conversion_edges
        if edge["source_type"] == "Dropout"
    } == {"QuantDropout"}


def test_explicit_replacement_is_locked_until_exact_recipe_is_confirmed():
    app = _run_app(explicit_replacement_count=1)
    app.button(key="mw_analyze").click().run(timeout=30)

    assert not list(app.exception)
    assert app.button(key="mw_build_quantized").disabled is True
    assert any(
        "Choose a target from the safe backend catalog" in value
        for value in _values(app.error)
    )

    target = _widget_with_label(app.selectbox, "Safe replacement target")
    target.select("qbench.quant_linear").run(timeout=30)

    assert not list(app.exception)
    assert _widget_with_label(app.selectbox, "Target `weight`").value == "source:weight"
    assert _widget_with_label(app.selectbox, "Target `bias`").value == "source:bias"
    assert _widget_with_label(app.selectbox, "Target `scale`").value == (
        "initializer:target_default"
    )
    assert app.button(key="mw_build_quantized").disabled is True
    assert any(
        "Unused source state (informational)" in value
        and "legacy_scale" in value
        for value in _values(app.caption)
    )

    confirmation = next(
        checkbox
        for checkbox in app.checkbox
        if str(checkbox.label).startswith("I explicitly confirm this exact target")
    )
    confirmation.set_value(True).run(timeout=30)

    assert not list(app.exception)
    assert not list(app.error)
    assert app.button(key="mw_build_quantized").disabled is False
    preview_specs = app.session_state["_fixture_preview_plan"]["replacement_specs"]
    assert set(preview_specs) == {"blocks.0.projection"}
    spec = preview_specs["blocks.0.projection"]
    assert spec == {
        "target_id": "qbench.quant_linear",
        "constructor_args": [],
        "constructor_kwargs": {},
        "state_mapping": {"weight": "weight", "bias": "bias"},
        "state_initializers": {"scale": "target_default"},
        "confirmed": True,
    }
    assert "QuantLinear" in _type_map_html(app)

    app.button(key="mw_build_quantized").click().run(timeout=30)

    assert not list(app.exception)
    fingerprint = json.loads(app.session_state["_mw_build_fingerprint"])
    assert fingerprint["replacement_specs"] == preview_specs
    assert any(
        element.get("data", {}).get("kind") == "user_replacement"
        for element in _type_map_elements(app)
        if "conversion-edge" in str(element.get("classes", ""))
    )
    assert app.session_state["_fixture_built_plan"]["replacement_specs"] == preview_specs
    assert app.session_state["_mw_conversion_result"]["recipe"][
        "replacement_specs"
    ] == preview_specs
    app.button(key="mw_prepare_export").click().run(timeout=30)
    exported = torch.load(
        io.BytesIO(app.session_state["_mw_export_bundle"]),
        map_location="cpu",
        weights_only=False,
    )
    assert exported["conversion_recipe"]["replacement_specs"] == preview_specs


def test_replacement_confirmation_is_invalidated_by_constructor_edit():
    app = _run_app(explicit_replacement_count=1)
    app.button(key="mw_analyze").click().run(timeout=30)
    _widget_with_label(app.selectbox, "Safe replacement target").select(
        "qbench.quant_linear"
    ).run(timeout=30)
    next(
        checkbox
        for checkbox in app.checkbox
        if str(checkbox.label).startswith("I explicitly confirm this exact target")
    ).set_value(True).run(timeout=30)
    assert app.button(key="mw_build_quantized").disabled is False
    original_preview_specs = copy.deepcopy(
        app.session_state["_fixture_preview_plan"]["replacement_specs"]
    )

    _widget_with_label(
        app.text_area,
        "Constructor keyword arguments (JSON object)",
    ).set_value('{"bias": false}').run(timeout=30)

    assert not list(app.exception)
    confirmation = next(
        checkbox
        for checkbox in app.checkbox
        if str(checkbox.label).startswith("I explicitly confirm this exact target")
    )
    assert confirmation.value is False
    assert app.button(key="mw_build_quantized").disabled is True
    assert app.session_state["_fixture_preview_plan"]["replacement_specs"] == (
        original_preview_specs
    )
    assert any(
        "Explicit confirmation is required for this exact replacement recipe" in value
        for value in _values(app.error)
    )


def test_backend_rejection_keeps_replacement_build_locked():
    app = _run_app(
        explicit_replacement_count=1,
        replacement_validation_fails=True,
    )
    app.button(key="mw_analyze").click().run(timeout=30)
    _widget_with_label(app.selectbox, "Safe replacement target").select(
        "qbench.quant_linear"
    ).run(timeout=30)
    next(
        checkbox
        for checkbox in app.checkbox
        if str(checkbox.label).startswith("I explicitly confirm this exact target")
    ).set_value(True).run(timeout=30)

    assert not list(app.exception)
    assert app.button(key="mw_build_quantized").disabled is True
    assert any(
        "fixture backend rejected incompatible replacement" in value
        for value in _values(app.error)
    )
    assert "_mw_conversion_result" not in app.session_state


def test_repeated_custom_type_uses_grouped_bulk_recipe_and_validates_each_path():
    app = _run_app(explicit_replacement_count=3)
    app.button(key="mw_analyze").click().run(timeout=30)

    group_labels = [
        str(expander.label)
        for expander in app.expander
        if str(expander.label).startswith("MyLinear ·")
    ]
    assert group_labels == ["MyLinear · 3 concrete paths"]
    active_path = _widget_with_label(app.selectbox, "Edit concrete source path")
    assert list(active_path.options) == [
        "blocks.0.projection",
        "blocks.1.projection",
        "blocks.2.projection",
    ]

    _widget_with_label(app.selectbox, "Safe replacement target").select(
        "qbench.quant_linear"
    ).run(timeout=30)
    next(
        checkbox
        for checkbox in app.checkbox
        if str(checkbox.label).startswith("I explicitly confirm this exact target")
    ).set_value(True).run(timeout=30)
    bulk_confirmation = next(
        checkbox
        for checkbox in app.checkbox
        if str(checkbox.label).startswith("I explicitly confirm applying this exact recipe")
    )
    bulk_confirmation.set_value(True).run(timeout=30)
    next(
        button
        for button in app.button
        if button.label == "Apply recipe to selected paths"
    ).click().run(timeout=30)

    assert not list(app.exception)
    assert not list(app.error)
    assert app.button(key="mw_build_quantized").disabled is False
    expected_paths = {
        "blocks.0.projection",
        "blocks.1.projection",
        "blocks.2.projection",
    }
    preview_specs = app.session_state["_fixture_preview_plan"]["replacement_specs"]
    assert set(preview_specs) == expected_paths
    assert {
        call["source_path"]
        for call in app.session_state["_fixture_validated_replacements"]
    } == expected_paths
    assert all(spec["confirmed"] is True for spec in preview_specs.values())


def test_failed_validation_persists_and_gates_state_export():
    app = _run_app(validation_fails=True)
    app.button(key="mw_analyze").click().run(timeout=30)
    app.button(key="mw_build_quantized").click().run(timeout=30)

    expected = (
        "Converted-model validation failed. State export is disabled. "
        "Automatic sample inference raised RuntimeError: fixture inference boom"
    )
    assert not list(app.exception)
    assert any(expected in value for value in _values(app.error))
    assert app.session_state["_mw_validation_state"]["status"] == "failed"
    assert app.button(key="mw_prepare_export").disabled is True
    assert [button.label for button in app.get("download_button")] == [
        "Download conversion recipe"
    ]
    assert "_mw_export_bundle" not in app.session_state

    # A later Streamlit rerun must not erase the failed status or expose export.
    app.run(timeout=30)
    assert not list(app.exception)
    assert any(expected in value for value in _values(app.error))
    assert app.session_state["_mw_validation_state"]["status"] == "failed"
    assert app.button(key="mw_prepare_export").disabled is True
    assert "_mw_export_bundle" not in app.session_state


def test_successful_validation_enables_and_prepares_state_export():
    app = _run_app()
    app.button(key="mw_analyze").click().run(timeout=30)
    app.button(key="mw_build_quantized").click().run(timeout=30)

    assert not list(app.exception)
    assert app.session_state["_mw_validation_state"]["status"] == "passed"
    assert app.button(key="mw_prepare_export").disabled is False
    assert _metric_value(app, "Realized QBench modules") == "1"
    assert _metric_value(app, "Executed QBench modules") == "1"
    assert _metric_value(app, "QBench runtime calls") == "1"
    build_text = " ".join(
        _values(app.markdown)
        + _values(app.caption)
        + _values(app.info)
        + _values(app.success)
    ).lower()
    assert "separate cloned model" in build_text
    assert "reference object remains unchanged" in build_text
    assert "temporary execution-audit hooks" in build_text

    app.button(key="mw_prepare_export").click().run(timeout=30)

    assert not list(app.exception)
    assert app.session_state["_mw_export_bundle"]
    assert [button.label for button in app.get("download_button")] == [
        "Download conversion recipe",
        "Download converted state bundle",
    ]


def test_image_folder_subset_compares_accuracy_and_is_embedded_in_export():
    app = _run_app()
    app.button(key="mw_analyze").click().run(timeout=30)
    app.button(key="mw_build_quantized").click().run(timeout=30)

    assert app.button(key="mw_run_dataset_benchmark").disabled is False
    app.button(key="mw_run_dataset_benchmark").click().run(timeout=30)

    assert not list(app.exception)
    assert not list(app.error)
    assert _metric_value(app, "Reference top-1") == "75.00%"
    assert _metric_value(app, "Converted top-1") == "72.50%"
    assert _metric_value(app, "Reference top-5") == "95.00%"
    assert _metric_value(app, "Converted top-5") == "92.50%"
    assert _metric_value(app, "Prediction agreement") == "90.00%"
    assert _metric_value(app, "Evaluated samples") == "40"
    converted_top1 = next(
        metric for metric in app.metric if metric.label == "Converted top-1"
    )
    assert str(converted_top1.delta) == "-2.50 pp"

    loader_kwargs = app.session_state["_fixture_dataset_loader_kwargs"]
    assert loader_kwargs["dataset_kind"] == "image_folder"
    assert loader_kwargs["path"] == "/data/imagenet"
    assert loader_kwargs["split"] == "val"
    assert loader_kwargs["max_samples"] == 128
    assert loader_kwargs["batch_size"] == 8
    assert loader_kwargs["seed"] == 42
    assert loader_kwargs["num_workers"] == 0
    assert app.session_state["_fixture_benchmark_kwargs"] == {
        "max_samples": 128,
        "device": "auto",
        "loader_length": 1,
    }
    assert "_mw_dataset_benchmark" in app.session_state
    assert [button.label for button in app.get("download_button")] == [
        "Download conversion recipe",
        "Download validation report",
    ]

    app.button(key="mw_prepare_export").click().run(timeout=30)
    payload = torch.load(
        io.BytesIO(app.session_state["_mw_export_bundle"]),
        map_location="cpu",
        weights_only=False,
    )
    exported_benchmark = payload["validation"]["dataset_benchmark"]
    assert payload["validation"]["sample_status"]["status"] == "passed"
    assert exported_benchmark["config"]["max_samples"] == 128
    assert exported_benchmark["loader_metadata"]["transform_source"] == (
        "fixture weights"
    )
    assert exported_benchmark["result"]["delta"][
        "top1_accuracy_percentage_points"
    ] == -2.5


@pytest.mark.parametrize(
    ("failure_flag", "message"),
    [
        ("dataset_loader_fails", "FileNotFoundError: fixture dataset missing"),
        ("dataset_benchmark_fails", "RuntimeError: fixture benchmark boom"),
    ],
)
def test_dataset_benchmark_failure_does_not_invalidate_sample_export(
    failure_flag: str,
    message: str,
):
    app = _run_app(**{failure_flag: True})
    app.button(key="mw_analyze").click().run(timeout=30)
    app.button(key="mw_build_quantized").click().run(timeout=30)
    app.button(key="mw_run_dataset_benchmark").click().run(timeout=30)

    assert not list(app.exception)
    assert any(message in value for value in _values(app.error))
    assert app.session_state["_mw_validation_state"]["status"] == "passed"
    assert app.button(key="mw_prepare_export").disabled is False
    assert "_mw_dataset_benchmark" not in app.session_state

    app.run(timeout=30)
    assert any(message in value for value in _values(app.error))
    assert app.button(key="mw_prepare_export").disabled is False


def test_custom_dataset_factory_kwargs_and_stale_benchmark_invalidation():
    app = _run_app()
    app.button(key="mw_analyze").click().run(timeout=30)
    app.button(key="mw_build_quantized").click().run(timeout=30)
    app.selectbox(key="mw_dataset_kind").select(
        "Custom dataset factory"
    ).run(timeout=30)
    app.text_input(key="mw_dataset_factory").set_value(
        "fixture.datasets:validation"
    ).run(timeout=30)
    app.text_area(key="mw_dataset_factory_kwargs").set_value(
        '{"partition": "test"}'
    ).run(timeout=30)
    app.button(key="mw_run_dataset_benchmark").click().run(timeout=30)

    loader_kwargs = app.session_state["_fixture_dataset_loader_kwargs"]
    assert loader_kwargs["dataset_kind"] == "custom_factory"
    assert loader_kwargs["path"] is None
    assert loader_kwargs["custom_factory"] == "fixture.datasets:validation"
    assert loader_kwargs["factory_kwargs"] == {"partition": "test"}
    assert "_mw_dataset_benchmark" in app.session_state

    app.button(key="mw_prepare_export").click().run(timeout=30)
    assert "_mw_export_bundle" in app.session_state
    app.number_input(key="mw_dataset_samples").set_value(64).run(timeout=30)

    assert "_mw_dataset_benchmark" not in app.session_state
    assert "_mw_export_bundle" not in app.session_state
    assert not [
        metric for metric in app.metric if metric.label == "Converted top-1"
    ]
    assert any(
        "dataset, subset, or execution settings changed" in value.lower()
        for value in _values(app.info)
    )

    app.text_area(key="mw_dataset_factory_kwargs").set_value("[]").run(timeout=30)
    assert app.button(key="mw_run_dataset_benchmark").disabled is True
    assert any("Factory arguments must be a JSON object" in value for value in _values(app.error))
