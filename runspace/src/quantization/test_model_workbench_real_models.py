"""Real-model integration coverage for the Model Quantization Workbench.

These tests intentionally exercise the same public workflow as the dashboard:
load -> analyze -> default plan -> convert -> sample inference.  Quantization
boundaries and weights remain disabled so the converted CPU graph must preserve
the reference result while still executing QBench operation wrappers.
"""

from collections import Counter

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from runspace.src.quantization.model_workbench import (  # noqa: E402
    ModelWorkbenchError,
    analyze_model,
    build_conversion_plan,
    convert_model,
    inspect_replacement_target,
    list_replacement_targets,
    load_model,
    run_sample_inference,
    validate_replacement_spec,
)
from runspace.src.registry.op_registry import OpRegistry  # noqa: E402


CPU_PREVIEW_OPTIONS = {
    "input_quantization": False,
    "weight_quantization": False,
    "output_quantization": False,
    "fold_layers": False,
    "fold_input_norm": False,
    "skip_calibration": True,
    "enable_fx_quantization": True,
    "quantized_ops": ["all"],
}


def _module_row(analysis, path):
    return next(row for row in analysis.module_rows if row["path"] == path)


def _operation_nodes(analysis):
    return [
        node
        for node in analysis.source_graph["nodes"]
        if node.get("kind") == "operation"
    ]


def _quantized_module_counts(model):
    return Counter(
        type(module).__name__
        for module in model.modules()
        if type(module).__name__.startswith(("Quant", "Decomposed"))
    )


def _attach_quantized_runtime_hooks(model):
    hits = Counter()
    handles = []

    def hook_for(module_type):
        def hook(_module, _inputs, _output):
            hits[module_type] += 1

        return hook

    for module in model.modules():
        module_type = type(module).__name__
        if module_type.startswith(("Quant", "Decomposed")):
            handles.append(module.register_forward_hook(hook_for(module_type)))
    return hits, handles


def _run_and_count(reference, converted, sample):
    hits, handles = _attach_quantized_runtime_hooks(converted.model)
    try:
        inference = run_sample_inference(reference, converted, sample)
    finally:
        for handle in handles:
            handle.remove()
    return inference, hits


def _assert_capture_succeeded(analysis):
    summary = analysis.summary
    capture_backend = (
        summary.get("capture_backend")
        or summary.get("graph_capture_backend")
        or getattr(analysis, "capture_backend", None)
        or getattr(analysis, "capture_kind", None)
    )
    capture_succeeded = summary.get(
        "capture_succeeded",
        getattr(
            analysis,
            "capture_succeeded",
            analysis.trace_succeeded or capture_backend in {"torch_export", "export"},
        ),
    )
    assert capture_succeeded is True, analysis.warnings
    if capture_backend is not None:
        assert capture_backend in {"fx", "torch_fx", "torch_export", "export"}
    assert summary["total_operations"] > 0


def _assert_output_equivalence(inference, expected_shape):
    comparison = inference["comparison"]
    assert comparison["structure_match"] is True
    assert comparison["allclose"] is True
    assert comparison["max_abs_error"] <= 1e-5
    assert inference["reference_summary"]["shape"] == list(expected_shape)
    assert inference["quantized_summary"]["shape"] == list(expected_shape)


def test_torchvision_vit_b_16_workbench_end_to_end():
    torch.manual_seed(101)
    reference = load_model(
        "torchvision", "vit_b_16", pretrained=False
    ).cpu().eval()
    # torchvision intentionally zero-initializes the classification head.
    # Randomize it so an incorrect internal conversion cannot pass merely by
    # producing the same all-zero logits.
    with torch.no_grad():
        reference.heads.head.weight.normal_(mean=0.0, std=0.02)
        reference.heads.head.bias.zero_()
    sample = torch.randn(1, 3, 224, 224)

    analysis = analyze_model(
        reference,
        model_name="vit_b_16",
        source="torchvision",
        sample_input=sample,
    )
    _assert_capture_succeeded(analysis)
    assert analysis.capture_kind == "fx"

    # Equality checks, integer patch-grid division and torch._assert are
    # shape/control operations.  They must not turn the encoder into an FP32
    # island or appear as unsupported numerical work.
    control_targets = {"_operator.eq", "torch._assert", "_operator.floordiv"}
    control_nodes = [
        node for node in _operation_nodes(analysis) if node.get("target") in control_targets
    ]
    assert len(control_nodes) == 32
    assert all(node["status"] == "structural_passthrough" for node in control_nodes)
    assert not [
        node for node in _operation_nodes(analysis) if node["status"] == "unsupported"
    ]

    assert _module_row(analysis, "encoder")["status"] == "custom_expanded"
    encoder_blocks = [
        row
        for row in analysis.module_rows
        if row["path"].startswith("encoder.layers.encoder_layer_")
        and row["path"].count(".") == 2
    ]
    assert len(encoder_blocks) == 12
    assert all(row["status"] == "custom_expanded" for row in encoder_blocks)
    assert not [
        row for row in analysis.module_rows if row["status"] == "fp32_fallback"
    ]

    plan = build_conversion_plan(
        analysis, quant_options=CPU_PREVIEW_OPTIONS
    )
    assert plan.decisions["encoder"] == "expand"
    assert all(plan.decisions[row["path"]] == "expand" for row in encoder_blocks)

    converted = convert_model(reference, plan)
    quantized_counts = _quantized_module_counts(converted.model)
    assert sum(quantized_counts.values()) >= 150
    assert quantized_counts["QuantLinear"] >= 70
    assert quantized_counts["QuantLayerNorm"] >= 20
    assert quantized_counts["QuantMatMul"] >= 20
    assert quantized_counts["QuantSoftmax"] >= 12

    inference, hits = _run_and_count(reference, converted, sample)
    _assert_output_equivalence(inference, (1, 1000))
    assert inference["reference_output"].abs().max().item() > 0
    assert torch.isfinite(inference["quantized_output"]).all()
    assert inference["runtime_audit"]["executed_quantized_modules"] >= 100
    assert sum(hits.values()) >= 100
    assert hits["QuantLinear"] >= 60
    assert hits["QuantMatMul"] >= 20
    assert hits["QuantSoftmax"] >= 12


def test_torchvision_vit_b_16_rejects_wrong_image_size_instead_of_hierarchy_fallback():
    reference = load_model(
        "torchvision", "vit_b_16", pretrained=False
    ).cpu().eval()

    with pytest.raises(ModelWorkbenchError, match="preferred input size"):
        analyze_model(
            reference,
            model_name="vit_b_16",
            source="torchvision",
            sample_input=torch.randn(1, 3, 256, 256),
        )


def test_timm_vit_base_mci_user_maps_layernorm_to_real_quant_layernorm():
    timm = pytest.importorskip("timm")
    if not timm.is_model("vit_base_mci_224"):
        pytest.skip("Installed timm does not provide vit_base_mci_224")

    from timm.layers.norm import LayerNorm as TimmLayerNorm

    torch.manual_seed(303)
    reference = load_model(
        "timm", "vit_base_mci_224", pretrained=False
    ).cpu().eval()
    layer_norm_paths = [
        path
        for path, module in reference.named_modules()
        if path and type(module) is TimmLayerNorm
    ]
    assert len(layer_norm_paths) == 25

    source_path = layer_norm_paths[0]
    source_layer = reference.get_submodule(source_path)
    source_type = type(source_layer)
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in source_layer.state_dict().items()
    }

    layer_norm_target = next(
        target
        for target in list_replacement_targets()
        if target["target_name"] == "QuantLayerNorm"
    )
    inspection = inspect_replacement_target(
        reference,
        source_path,
        layer_norm_target["id"],
        constructor_kwargs={
            "normalized_shape": list(source_layer.normalized_shape),
            "eps": float(source_layer.eps),
            "elementwise_affine": bool(source_layer.elementwise_affine),
            "bias": source_layer.bias is not None,
        },
    )
    assert inspection["source"]["type"] == "timm.layers.norm.LayerNorm"
    assert inspection["target"]["target_name"] == "QuantLayerNorm"
    assert inspection["suggested_state_mapping"] == {
        "weight": "weight",
        "bias": "bias",
    }

    replacement_spec = inspection["spec_template"]
    replacement_spec["confirmed"] = True
    replacement_spec = validate_replacement_spec(
        reference, source_path, replacement_spec
    )

    analysis = analyze_model(
        reference,
        model_name="vit_base_mci_224",
        source="timm",
        sample_input=torch.randn(1, 3, 224, 224),
    )
    assert analysis.capture_kind == "torch_export"
    assert _module_row(analysis, source_path)["status"] == "custom_expanded"
    plan = build_conversion_plan(
        analysis,
        decisions={source_path: "replace"},
        replacement_specs={source_path: replacement_spec},
        quant_options={
            **CPU_PREVIEW_OPTIONS,
            "enable_fx_quantization": False,
            # The explicit recipe must realize its selected target even when
            # the broad adapter filters would otherwise exclude LayerNorm.
            "quantized_ops": ["Linear"],
            "excluded_ops": ["LayerNorm"],
        },
    )

    converted = convert_model(reference, plan)
    quant_layer_norm_cls = OpRegistry.get_quantized_op(torch.nn.LayerNorm)
    realized_layer = converted.model.get_submodule(source_path)

    # Resolve through OpRegistry: src.* and runspace.src.* can otherwise expose
    # distinct Python class objects for the same registered implementation.
    assert type(realized_layer) is quant_layer_norm_cls
    assert sum(
        type(module) is quant_layer_norm_cls
        for module in converted.model.modules()
    ) == 1
    assert type(reference.get_submodule(source_path)) is source_type
    assert reference.get_submodule(source_path) is source_layer
    for name, original in source_state.items():
        assert torch.equal(source_layer.state_dict()[name], original)
        assert torch.equal(realized_layer.state_dict()[name], original)
        assert (
            realized_layer.state_dict()[name].data_ptr()
            != source_layer.state_dict()[name].data_ptr()
        )

    probe = torch.randn(2, 7, source_layer.normalized_shape[-1])
    with torch.no_grad():
        reference_output = source_layer(probe)
        converted_output = realized_layer(probe)
    torch.testing.assert_close(
        converted_output, reference_output, rtol=1e-5, atol=1e-6
    )
    assert converted.recipe["resolved"]["user_replacements"][source_path][
        "target_name"
    ] == "QuantLayerNorm"
    assert converted.recipe["resolved"]["user_replacements"][source_path][
        "state_mapping"
    ] == {"weight": "weight", "bias": "bias"}


@pytest.mark.parametrize(
    ("image_size", "expected_capture"),
    [(224, "torch_export"), (256, "fx")],
)
def test_timm_mobilevit_s_workbench_end_to_end(image_size, expected_capture):
    timm = pytest.importorskip("timm")
    if not timm.is_model("mobilevit_s"):
        pytest.skip("Installed timm does not provide mobilevit_s")

    torch.manual_seed(202)
    reference = load_model(
        "timm", "mobilevit_s", pretrained=False
    ).cpu().eval()
    sample = torch.randn(1, 3, image_size, image_size)

    analysis = analyze_model(
        reference,
        model_name="mobilevit_s",
        source="timm",
        sample_input=sample,
    )
    # MobileViT contains data-dependent/control-flow code that may require the
    # torch.export fallback when ordinary FX cannot capture the full model.
    _assert_capture_succeeded(analysis)
    assert analysis.capture_kind == expected_capture
    assert analysis.capture_details["sample_input_shape"] == [1, 3, image_size, image_size]
    assert analysis.summary["total_operations"] >= 100

    fused_bn_rows = [row for row in analysis.module_rows if row["type"] == "BatchNormAct2d"]
    attention_rows = [row for row in analysis.module_rows if row["type"] == "Attention"]
    assert len(fused_bn_rows) >= 20
    assert all(row["recommended"] == "QuantBatchNormAct2d" for row in fused_bn_rows)
    assert len(attention_rows) == 9
    assert all(row["recommended"] == "DecomposedQkvAttention" for row in attention_rows)
    assert not [
        node
        for node in _operation_nodes(analysis)
        if node["status"] == "unsupported"
        and node["label"] in {"batch_norm.default", "scaled_dot_product_attention.default"}
    ]

    plan = build_conversion_plan(
        analysis, quant_options=CPU_PREVIEW_OPTIONS
    )
    assert plan.capture_kind == expected_capture
    converted = convert_model(reference, plan)
    if expected_capture == "torch_export":
        assert not isinstance(converted.model, torch.fx.GraphModule)
        assert any("functional FX rewriting was disabled" in warning for warning in converted.warnings)
    quantized_counts = _quantized_module_counts(converted.model)
    assert sum(quantized_counts.values()) >= 40
    assert quantized_counts["QuantConv2d"] >= 10
    assert quantized_counts["QuantLinear"] >= 5
    if expected_capture == "fx":
        assert quantized_counts["QuantMatMul"] >= 10
        assert quantized_counts["QuantSoftmax"] >= 9

    inference, hits = _run_and_count(reference, converted, sample)
    _assert_output_equivalence(inference, (1, 1000))
    assert inference["reference_output"].abs().max().item() > 0
    assert torch.isfinite(inference["quantized_output"]).all()
    assert inference["runtime_audit"]["executed_quantized_modules"] >= 30
    assert sum(hits.values()) >= 30
    assert hits["QuantConv2d"] >= 10
    assert hits["QuantLinear"] >= 5
    if expected_capture == "fx":
        assert hits["QuantMatMul"] >= 10
        assert hits["QuantSoftmax"] >= 9
