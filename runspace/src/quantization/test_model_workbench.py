import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from runspace.src.quantization.model_workbench import (
    ConversionPlan,
    ConversionResult,
    ModelWorkbenchError,
    WORKBENCH_DATASET_BENCHMARK_API_VERSION,
    WORKBENCH_REPLACEMENT_API_VERSION,
    analyze_model,
    benchmark_classification_models,
    build_conversion_plan,
    build_classification_validation_loader,
    convert_model,
    inspect_replacement_target,
    list_replacement_targets,
    load_model,
    preview_conversion_plan,
    run_sample_inference,
    validate_replacement_spec,
)
from runspace.src.registry.op_registry import OpRegistry


class MyLinear(nn.Linear):
    """A Linear subclass whose extra math must not disappear."""

    def forward(self, inputs):
        projected = super().forward(inputs)
        activated = F.relu(projected)
        return activated * 1.5


class TransparentLinear(nn.Linear):
    pass


class UnsupportedWave(nn.Module):
    def forward(self, inputs):
        return torch.sin(inputs)


class MixedCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = MyLinear(4, 3)
        self.wave = UnsupportedWave()

    def forward(self, inputs):
        return self.wave(self.projection(inputs))


class ExpandedOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = MyLinear(4, 3)

    def forward(self, inputs):
        output = self.projection(inputs)
        return {"scores": output, "aux": (output.mean(dim=-1),)}


class TransparentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = TransparentLinear(4, 3)

    def forward(self, inputs):
        return self.projection(inputs)


class RenamedLayerNorm(nn.Module):
    """A custom LayerNorm implementation with deliberately renamed state."""

    def __init__(self, features=4):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.offset = nn.Parameter(torch.zeros(features))
        self.eps = 1e-5

    def forward(self, inputs):
        return F.layer_norm(inputs, (4,), self.gain, self.offset, self.eps)


class RenamedLayerNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RenamedLayerNorm(4)

    def forward(self, inputs):
        return self.norm(inputs)


class ReplacementIsolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RenamedLayerNorm(4)
        self.excluded_sibling = nn.LayerNorm(4)

    def forward(self, inputs):
        return self.norm(inputs) + self.excluded_sibling(inputs)


def _layer_norm_replacement_entry():
    return next(
        entry
        for entry in list_replacement_targets()
        if entry["target_name"] == "QuantLayerNorm"
        and entry["native_name"] == "LayerNorm"
    )


def _renamed_layer_norm_spec(model):
    entry = _layer_norm_replacement_entry()
    inspection = inspect_replacement_target(
        model,
        "norm",
        entry["id"],
        constructor_kwargs={
            "normalized_shape": [4],
            "eps": 1e-5,
            "elementwise_affine": True,
            "bias": True,
        },
    )
    spec = inspection["spec_template"]
    spec["state_mapping"] = {"weight": "gain", "bias": "offset"}
    spec["state_initializers"] = {}
    spec["confirmed"] = True
    return entry, inspection, spec


def _row(analysis, path):
    return next(row for row in analysis.module_rows if row["path"] == path)


def test_analysis_expands_overridden_subclass_and_preserves_ownership():
    analysis = analyze_model(
        MixedCustomModel().eval(),
        model_name="mixed_custom",
        sample_input=torch.randn(2, 4),
    )

    projection = _row(analysis, "projection")
    assert projection["status"] == "custom_expanded"
    assert projection["recommended"] == "expand"
    assert "alias:torch.nn.modules.linear.Linear" in projection["candidates"]

    owned_ops = [
        node
        for node in analysis.source_graph["nodes"]
        if node.get("kind") == "operation" and node.get("module_path") == "projection"
    ]
    proposed = {node.get("quantized_target") for node in owned_ops}
    assert {"QuantLinear", "QuantReLU", "QuantMul"} <= proposed

    group_mapping = next(
        mapping
        for mapping in analysis.mappings
        if mapping["source_node_ids"] == ["module:projection"]
        and mapping["kind"] == "decomposed"
        and len(mapping["target_node_ids"]) > 1
    )
    assert len(group_mapping["target_node_ids"]) >= 3

    unsupported = _row(analysis, "wave")
    assert unsupported["status"] == "fp32_fallback"
    assert unsupported["recommended"] == "fp32"
    assert any(
        node["status"] == "unsupported" and node.get("module_path") == "wave"
        for node in analysis.source_graph["nodes"]
    )

    # The dashboard payload and conversion recipe must remain JSON exportable.
    json.dumps(analysis.to_dict())


def test_transparent_subclass_is_distinct_from_overridden_subclass():
    analysis = analyze_model(TransparentModel())
    row = _row(analysis, "projection")

    assert row["status"] == "transparent_subclass"
    assert row["recommended"] == "QuantLinear"


def test_plan_defaults_are_cpu_safe_and_decisions_are_editable():
    analysis = analyze_model(MixedCustomModel())
    plan = build_conversion_plan(
        analysis,
        decisions={"module:wave": "fp32", "projection": "expand"},
        quant_options={"quantization_type": "fp8_e4m3"},
    )

    assert plan.decisions["projection"] == "expand"
    assert plan.decisions["wave"] == "fp32"
    assert plan.quant_options["weight_quantization"] is False
    assert plan.quant_options["input_quantization"] is False
    assert plan.quant_options["output_quantization"] is False
    assert plan.quant_options["fold_layers"] is False
    assert plan.quant_options["fold_input_norm"] is False
    json.dumps(plan.to_dict())


def test_plan_preview_marks_complete_fp32_subtree_without_mutating_analysis():
    analysis = analyze_model(ExpandedOnlyModel())
    original_target = analysis.to_dict()["target_graph"]
    plan = build_conversion_plan(analysis, decisions={"projection": "fp32"})

    target_graph, mappings = preview_conversion_plan(analysis, plan)

    projection_targets = [
        node
        for node in target_graph["nodes"]
        if node.get("module_path") == "projection"
        or str(node.get("module_path", "")).startswith("projection.")
    ]
    assert projection_targets
    assert all(node["status"] == "fp32_fallback" for node in projection_targets)
    projection_source_ids = {
        node["id"]
        for node in analysis.source_graph["nodes"]
        if node.get("module_path") == "projection"
        or str(node.get("module_path", "")).startswith("projection.")
    }
    assert all(
        mapping["kind"] == "fp32_fallback"
        for mapping in mappings
        if projection_source_ids & set(mapping["source_node_ids"])
    )
    assert analysis.target_graph == original_target
    _assert_preview_ids_are_valid(analysis, target_graph, mappings)


def test_plan_preview_collapses_explicit_alias_to_many_to_one_mapping():
    analysis = analyze_model(ExpandedOnlyModel())
    original_operation_targets = {
        node["id"]
        for node in analysis.target_graph["nodes"]
        if node.get("kind") == "operation" and node.get("module_path") == "projection"
    }
    alias = "alias:torch.nn.modules.linear.Linear"
    plan = build_conversion_plan(analysis, decisions={"projection": alias})

    target_graph, mappings = preview_conversion_plan(analysis, plan)

    alias_target_id = "target:module:projection"
    alias_node = next(node for node in target_graph["nodes"] if node["id"] == alias_target_id)
    assert alias_node["status"] == "native_alias"
    assert alias_node["target"] == "torch.nn.modules.linear.Linear"
    remaining_ids = {node["id"] for node in target_graph["nodes"]}
    assert original_operation_targets.isdisjoint(remaining_ids)

    alias_mapping = next(
        mapping for mapping in mappings if mapping["kind"] == "many_to_one_alias"
    )
    assert alias_mapping["target_node_ids"] == [alias_target_id]
    assert "module:projection" in alias_mapping["source_node_ids"]
    assert len(alias_mapping["source_node_ids"]) > 2
    assert analysis.target_graph["nodes"]  # the original graph remains expanded
    assert original_operation_targets <= {
        node["id"] for node in analysis.target_graph["nodes"]
    }
    _assert_preview_ids_are_valid(analysis, target_graph, mappings)


def _assert_preview_ids_are_valid(analysis, target_graph, mappings):
    source_ids = {node["id"] for node in analysis.source_graph["nodes"]}
    target_ids = {node["id"] for node in target_graph["nodes"]}
    assert all(
        edge["source"] in target_ids and edge["target"] in target_ids
        for edge in target_graph["edges"]
    )
    assert all(
        set(mapping["source_node_ids"]) <= source_ids
        and set(mapping["target_node_ids"]) <= target_ids
        for mapping in mappings
    )


def test_replacement_catalog_inspection_and_explicit_state_mapping_are_json_safe():
    model = RenamedLayerNormModel().eval()
    entry, inspection, spec = _renamed_layer_norm_spec(model)

    assert inspection["api_version"] == WORKBENCH_REPLACEMENT_API_VERSION
    assert inspection["target"]["id"] == entry["id"]
    assert inspection["target"]["target_name"] == "QuantLayerNorm"
    assert inspection["suggested_state_mapping"] == {}
    assert inspection["exact_name_shape_suggestions"] == []
    assert {field["local_key"] for field in inspection["source"]["state_fields"]} == {
        "gain",
        "offset",
    }
    assert {
        (field["local_key"], field["kind"])
        for field in inspection["target"]["state_fields"]
    } == {("weight", "parameter"), ("bias", "parameter")}
    assert all(
        field["qualified_key"].startswith("norm.")
        for field in inspection["source"]["state_fields"]
        + inspection["target"]["state_fields"]
    )

    normalized = validate_replacement_spec(model, "norm", spec)
    assert normalized == spec
    json.dumps(list_replacement_targets())
    json.dumps(inspection)
    json.dumps(normalized)


def test_user_replacement_preview_conversion_and_recipe_preserve_source_model():
    torch.manual_seed(29)
    reference = RenamedLayerNormModel().eval()
    reference.gain_snapshot = reference.norm.gain.detach().clone()
    reference.offset_snapshot = reference.norm.offset.detach().clone()
    sample = torch.randn(3, 4)
    entry, _inspection, spec = _renamed_layer_norm_spec(reference)
    analysis = analyze_model(reference, sample_input=sample)
    plan = build_conversion_plan(
        analysis,
        decisions={"norm": "replace"},
        replacement_specs={"norm": spec},
        quant_options={
            "enable_fx_quantization": False,
            "quantized_ops": [],
            "excluded_ops": ["LayerNorm", "QuantLayerNorm"],
        },
    )

    target_graph, mappings = preview_conversion_plan(analysis, plan)
    preview_node = next(
        node for node in target_graph["nodes"] if node.get("module_path") == "norm"
    )
    assert preview_node["status"] == "user_replacement"
    assert preview_node["target_id"] == entry["id"]
    assert preview_node["label"] == "QuantLayerNorm"
    assert any(mapping["kind"] == "user_replacement" for mapping in mappings)
    _assert_preview_ids_are_valid(analysis, target_graph, mappings)

    converted = convert_model(reference, plan)
    quantized_cls = OpRegistry.get_quantized_op(nn.LayerNorm)
    actual = converted.model.get_submodule("norm")
    assert type(actual) is quantized_cls
    assert isinstance(reference.norm, RenamedLayerNorm)
    assert torch.equal(reference.norm.gain, reference.gain_snapshot)
    assert torch.equal(reference.norm.offset, reference.offset_snapshot)
    assert actual.weight.data_ptr() != reference.norm.gain.data_ptr()
    assert actual.bias.data_ptr() != reference.norm.offset.data_ptr()
    torch.testing.assert_close(actual.weight, reference.norm.gain)
    torch.testing.assert_close(actual.bias, reference.norm.offset)
    torch.testing.assert_close(converted.model(sample), reference(sample))
    assert converted.recipe["replacement_specs"]["norm"] == spec
    assert converted.recipe["resolved"]["user_replacements"]["norm"]["target_id"] == entry["id"]
    assert any("scoped to selected paths only" in warning for warning in converted.warnings)
    json.dumps(converted.recipe)


def test_user_replacement_does_not_enable_excluded_same_type_sibling():
    reference = ReplacementIsolationModel().eval()
    sample = torch.randn(2, 4)
    entry, _inspection, spec = _renamed_layer_norm_spec(reference)
    plan = build_conversion_plan(
        analyze_model(reference, sample_input=sample),
        decisions={"norm": "replace"},
        replacement_specs={"norm": spec},
        quant_options={
            "enable_fx_quantization": False,
            "quantized_ops": [],
            "excluded_ops": ["LayerNorm", "QuantLayerNorm"],
        },
    )

    converted = convert_model(reference, plan)

    assert type(converted.model.norm) is OpRegistry.get_quantized_op(nn.LayerNorm)
    assert type(converted.model.excluded_sibling) is nn.LayerNorm
    torch.testing.assert_close(converted.model(sample), reference(sample))
    assert converted.recipe["resolved"]["adapter_options"]["quantized_ops"] == []
    assert converted.recipe["resolved"]["adapter_options"]["excluded_ops"] == [
        "LayerNorm",
        "QuantLayerNorm",
    ]


def test_replacement_validation_rejects_unconfirmed_incomplete_and_invalid_mappings():
    model = RenamedLayerNormModel()
    entry, _inspection, valid_spec = _renamed_layer_norm_spec(model)

    unconfirmed = dict(valid_spec, confirmed=False)
    with pytest.raises(ModelWorkbenchError, match="confirmed=true"):
        validate_replacement_spec(model, "norm", unconfirmed)

    incomplete = dict(valid_spec, state_mapping={"weight": "gain"})
    with pytest.raises(ModelWorkbenchError, match="missing: bias"):
        validate_replacement_spec(model, "norm", incomplete)

    unknown_source = dict(
        valid_spec,
        state_mapping={"weight": "missing", "bias": "offset"},
    )
    with pytest.raises(ModelWorkbenchError, match="unknown source field 'missing'"):
        validate_replacement_spec(model, "norm", unknown_source)

    wrong_shape = dict(
        valid_spec,
        constructor_kwargs={
            "normalized_shape": [3],
            "eps": 1e-5,
            "elementwise_affine": True,
            "bias": True,
        },
    )
    with pytest.raises(ModelWorkbenchError, match="shape mismatch"):
        validate_replacement_spec(model, "norm", wrong_shape)

    unknown_target = dict(valid_spec, target_id=entry["id"] + ":missing")
    with pytest.raises(ModelWorkbenchError, match="Unknown replacement target_id"):
        validate_replacement_spec(model, "norm", unknown_target)

    invalid_number = dict(
        valid_spec,
        constructor_kwargs={"normalized_shape": [4], "eps": float("nan")},
    )
    with pytest.raises(ModelWorkbenchError, match="NaN or infinity"):
        validate_replacement_spec(model, "norm", invalid_number)


def test_replacement_plan_serialization_rejects_conflicts_and_future_versions():
    model = RenamedLayerNormModel()
    _entry, _inspection, spec = _renamed_layer_norm_spec(model)
    payload = {
        "version": 2,
        "model_name": "nested",
        "source": "custom",
        "decisions": {"block": "expand", "block.norm": "replace"},
        "quant_options": {},
        "replacement_specs": {"block.norm": spec},
    }
    with pytest.raises(ModelWorkbenchError, match="hidden by ancestor decision"):
        ConversionPlan.from_dict(payload)

    overlapping = dict(
        payload,
        decisions={"block": "replace", "block.norm": "replace"},
        replacement_specs={"block": spec, "block.norm": spec},
    )
    with pytest.raises(ModelWorkbenchError, match="cannot overlap"):
        ConversionPlan.from_dict(overlapping)

    with pytest.raises(ModelWorkbenchError, match="Unsupported conversion plan version"):
        ConversionPlan.from_dict(dict(payload, version=99))

    legacy = ConversionPlan.from_dict(
        {
            "model_name": "legacy",
            "source": "custom",
            "decisions": {"norm": "fp32"},
            "quant_options": {},
        }
    )
    assert legacy.replacement_specs == {}
    assert legacy.to_dict()["version"] == 2


def test_conversion_preserves_custom_forward_and_compares_nested_outputs():
    torch.manual_seed(7)
    reference = ExpandedOnlyModel().eval()
    sample = torch.randn(3, 4)
    reference_module_types = {
        path: type(module) for path, module in reference.named_modules()
    }
    reference_state = {
        name: tensor.detach().clone() for name, tensor in reference.state_dict().items()
    }
    analysis = analyze_model(reference, sample_input=sample)
    plan = build_conversion_plan(analysis)

    converted = convert_model(reference, plan)
    assert isinstance(converted, ConversionResult)
    assert converted.model is not reference
    assert {
        path: type(module) for path, module in reference.named_modules()
    } == reference_module_types
    assert reference.state_dict().keys() == reference_state.keys()
    assert all(
        torch.equal(reference.state_dict()[name], original)
        for name, original in reference_state.items()
    )

    hooks_before_inference = sum(
        len(module._forward_hooks) for module in converted.model.modules()
    )
    assert hooks_before_inference == 0
    comparison = run_sample_inference(reference, converted, sample)
    hooks_after_inference = sum(
        len(module._forward_hooks) for module in converted.model.modules()
    )
    assert hooks_after_inference == 0

    assert comparison["comparison"]["structure_match"] is True
    assert comparison["comparison"]["allclose"] is True
    assert comparison["comparison"]["max_abs_error"] <= 1e-6
    assert comparison["runtime_audit"]["executed_quantized_modules"] >= 3
    assert comparison["runtime_audit"]["runtime_calls_by_type"]["QuantLinear"] >= 1
    converted_types = {type(module).__name__ for module in converted.model.modules()}
    # Every quantized target advertised for MyLinear's executable operations is
    # actually materialized by the converter.
    assert {"QuantLinear", "QuantReLU", "QuantMul"} <= converted_types
    assert converted.realization["total"] >= 3
    json.dumps(converted.to_dict())


def test_fp32_decision_is_an_opaque_island_for_generic_adapter_fx():
    torch.manual_seed(11)
    reference = ExpandedOnlyModel().eval()
    sample = torch.randn(2, 4)
    analysis = analyze_model(reference, sample_input=sample)
    plan = build_conversion_plan(analysis, decisions={"projection": "fp32"})

    converted = convert_model(reference, plan)
    converted_types = {type(module).__name__ for module in converted.model.modules()}

    assert "MyLinear" in converted_types
    assert not any(name in converted_types for name in ("QuantLinear", "QuantReLU", "QuantMul"))
    comparison = run_sample_inference(reference, converted, sample)
    assert comparison["comparison"]["allclose"] is True


class NonRewritableFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 2, 1, 1))

    def forward(self, inputs):
        convolved = F.conv2d(inputs, self.weight)
        activated = convolved.relu()
        return torch.softmax(activated, dim=1)


def test_preview_does_not_advertise_unimplemented_functional_rewrites():
    analysis = analyze_model(
        NonRewritableFunctionalModel(), sample_input=torch.randn(1, 2, 3, 3)
    )
    operation_nodes = [
        node for node in analysis.source_graph["nodes"] if node.get("kind") == "operation"
    ]
    by_label = {node["label"]: node for node in operation_nodes}

    assert by_label["conv2d"]["status"] == "unsupported"
    assert by_label["relu"]["status"] == "unsupported"  # call_method
    assert by_label["softmax"]["status"] == "unsupported"  # torch.softmax, not F.softmax


def test_custom_factory_errors_are_actionable():
    with pytest.raises(ModelWorkbenchError, match="package.module:callable"):
        load_model("custom", "anything", custom_factory="not-a-factory-path")


def _classification_tensor_factory(
    *, sample_count=12, class_count=6, **_factory_context
):
    labels = torch.arange(sample_count, dtype=torch.long) % class_count
    features = torch.full((sample_count, class_count), -10.0)
    features[torch.arange(sample_count), labels] = 10.0
    return TensorDataset(features, labels)


def test_validation_loader_custom_factory_subset_is_seeded_and_described():
    common = dict(
        dataset_kind="custom_factory",
        model=nn.Identity(),
        source="custom",
        model_name="identity",
        custom_factory=_classification_tensor_factory,
        factory_kwargs={"sample_count": 12, "class_count": 6},
        batch_size=2,
        max_samples=5,
        seed=17,
        num_workers=0,
        pin_memory=False,
    )
    first_loader, first_metadata = build_classification_validation_loader(**common)
    second_loader, second_metadata = build_classification_validation_loader(**common)

    first_features = torch.cat([features for features, _labels in first_loader])
    second_features = torch.cat([features for features, _labels in second_loader])

    assert torch.equal(first_features, second_features)
    assert len(first_loader.dataset) == 5
    assert first_metadata == second_metadata
    assert first_metadata["api_version"] == WORKBENCH_DATASET_BENCHMARK_API_VERSION
    assert first_metadata["dataset_size"] == 12
    assert first_metadata["subset_size"] == 5
    assert first_metadata["class_count"] == 6
    assert first_metadata["seed"] == 17
    assert first_loader.qbench_metadata == first_metadata
    json.dumps(first_metadata)


def test_validation_loader_accepts_factory_configured_dataloader():
    produced = {}

    def loader_factory(*, batch_size, **_context):
        loader = DataLoader(_classification_tensor_factory(sample_count=4), batch_size=batch_size)
        produced["loader"] = loader
        return loader

    loader, metadata = build_classification_validation_loader(
        dataset_kind="custom_factory",
        model=nn.Identity(),
        source="custom",
        model_name="identity",
        custom_factory=loader_factory,
        batch_size=3,
        max_samples=4,
        pin_memory=False,
    )

    assert loader is produced["loader"]
    assert metadata["factory_controls_loader"] is True
    assert metadata["dataset_size"] == 4
    assert metadata["batch_size"] == 3


def test_validation_loader_imagefolder_uses_default_transform_and_wnid_map(tmp_path):
    pil_image = pytest.importorskip("PIL.Image")
    torchvision = pytest.importorskip("torchvision")
    # n01443537 is canonical ImageNet class 1.  With only one local folder,
    # ImageFolder would otherwise assign target 0, making the remap observable.
    class_directory = tmp_path / "val" / "n01443537"
    class_directory.mkdir(parents=True)
    pil_image.new("RGB", (40, 32), color=(40, 90, 170)).save(
        class_directory / "sample.png"
    )

    loader, metadata = build_classification_validation_loader(
        dataset_kind="image_folder",
        model=torchvision.models.resnet18(weights=None),
        source="torchvision",
        model_name="resnet18",
        path=tmp_path,
        batch_size=1,
        max_samples=1,
        num_workers=0,
        pin_memory=False,
    )
    images, targets = next(iter(loader))

    assert images.shape == (1, 3, 224, 224)
    assert targets.tolist() == [1]
    assert metadata["transform_source"] == "torchvision:resnet18:DEFAULT"
    assert metadata["canonical_imagenet_labels_mapped"] == 1
    assert metadata["canonical_target_map"] == {"0": 1}
    assert metadata["dataset_size"] == 1
    assert metadata["subset_size"] == 1
    assert metadata["resolved_path"] == str(tmp_path / "val")


class _ReferenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        return inputs + self.anchor * 0.0


class _ClassOneQuantizedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        logits = inputs + self.anchor * 0.0
        logits = logits.clone()
        logits[:, 1] += 25.0
        return {"logits": logits}


def test_classification_benchmark_metrics_progress_and_runtime_state():
    loader, metadata = build_classification_validation_loader(
        dataset_kind="custom_factory",
        model=nn.Identity(),
        source="custom",
        model_name="identity",
        custom_factory=_classification_tensor_factory,
        factory_kwargs={"sample_count": 6, "class_count": 6},
        batch_size=2,
        max_samples=6,
        seed=3,
        num_workers=0,
        pin_memory=False,
    )
    reference = _ReferenceClassifier().train()
    quantized_model = _ClassOneQuantizedClassifier().eval()
    converted = ConversionResult(
        model=quantized_model,
        adapter=None,
        warnings=[],
        recipe={},
    )
    progress = []

    result = benchmark_classification_models(
        reference,
        converted,
        loader,
        device="cpu",
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )

    assert result["api_version"] == WORKBENCH_DATASET_BENCHMARK_API_VERSION
    assert result["device"] == "cpu"
    assert result["samples"] == 6
    assert result["batches"] == 3
    assert result["class_dimension"] == 6
    assert result["effective_top5_k"] == 5
    assert result["reference"]["top1_accuracy_percent"] == pytest.approx(100.0)
    assert result["reference"]["top5_accuracy_percent"] == pytest.approx(100.0)
    assert result["quantized"]["top1_accuracy_percent"] == pytest.approx(100.0 / 6.0)
    assert result["quantized"]["top5_accuracy_percent"] == pytest.approx(100.0)
    assert result["delta"]["top1_accuracy_percentage_points"] == pytest.approx(
        -500.0 / 6.0
    )
    assert result["delta"]["top5_accuracy_percentage_points"] == pytest.approx(0.0)
    assert result["prediction_agreement_count"] == 1
    assert result["prediction_agreement_percent"] == pytest.approx(100.0 / 6.0)
    assert result["loader_metadata"] == metadata
    assert progress == [(2, 6), (4, 6), (6, 6)]
    assert result["reference"]["elapsed_seconds"] >= 0.0
    assert result["quantized"]["throughput_samples_per_second"] > 0.0

    # Benchmarking is temporary: preserve both device and the caller's mode.
    assert reference.training is True
    assert quantized_model.training is False
    assert reference.anchor.device.type == "cpu"
    assert quantized_model.anchor.device.type == "cpu"
    json.dumps(result)


def test_classification_benchmark_rejects_unlabeled_or_empty_data():
    with pytest.raises(ModelWorkbenchError, match="labeled samples"):
        benchmark_classification_models(
            _ReferenceClassifier(),
            _ClassOneQuantizedClassifier(),
            DataLoader(TensorDataset(torch.empty(0, 6), torch.empty(0, dtype=torch.long))),
            device="cpu",
        )

    bad_loader = DataLoader(TensorDataset(torch.randn(2, 6)), batch_size=1)
    with pytest.raises(ModelWorkbenchError, match=r"at least \(inputs, targets\)"):
        benchmark_classification_models(
            _ReferenceClassifier(), _ClassOneQuantizedClassifier(), bad_loader, device="cpu"
        )
