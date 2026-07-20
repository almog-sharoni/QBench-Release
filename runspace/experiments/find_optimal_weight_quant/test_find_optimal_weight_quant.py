import argparse
import json

import pytest
import torch
import torch.nn as nn

from runspace.experiments.find_optimal_weight_quant import find_optimal_weight_quant as weight_quant


def test_candidate_formats_are_grouped_by_bit_width():
    assert weight_quant._candidate_formats_by_bit_width([
        "fp8_e1m6",
        "fp8_e2m5",
        "fp7_e1m5",
        "fp7_e2m4",
    ]) == {
        8: ["fp8_e1m6", "fp8_e2m5"],
        7: ["fp7_e1m5", "fp7_e2m4"],
    }


def test_candidate_formats_reject_names_without_a_bit_width():
    with pytest.raises(ValueError, match="must include a bit width"):
        weight_quant._candidate_formats_by_bit_width(["custom_format"])


def test_process_single_model_runs_each_width_with_an_isolated_experiment(monkeypatch):
    calls = []

    def record_call(args, device, metrics, base_root):
        calls.append({
            "formats": args.baseline_formats,
            "bit_width": args.optimization_bit_width,
            "experiment_type": args.optimized_experiment_type,
            "device": device,
            "metrics": metrics,
            "base_root": base_root,
        })

    monkeypatch.setattr(
        weight_quant,
        "_process_single_model_for_bit_width",
        record_call,
    )
    args = argparse.Namespace(
        baseline_formats="fp8_e1m6,fp8_e2m5,fp7_e1m5,fp7_e2m4",
        optimized_experiment_type="weight_quant_optimized",
    )

    weight_quant.process_single_model(args, "cuda", ["mse"], "/tmp/results")

    assert calls == [
        {
            "formats": "fp8_e1m6,fp8_e2m5",
            "bit_width": 8,
            "experiment_type": "weight_quant_optimized_8",
            "device": "cuda",
            "metrics": ["mse"],
            "base_root": "/tmp/results",
        },
        {
            "formats": "fp7_e1m5,fp7_e2m4",
            "bit_width": 7,
            "experiment_type": "weight_quant_optimized_7",
            "device": "cuda",
            "metrics": ["mse"],
            "base_root": "/tmp/results",
        },
    ]
    assert args.baseline_formats.startswith("fp8_e1m6")
    assert args.optimized_experiment_type == "weight_quant_optimized"


def test_ablation_parsers_validate_and_deduplicate_values():
    assert weight_quant._parse_ablation_layer_types(
        "Conv2d, Linear,conv2d,WeightedMatMul"
    ) == ["Conv2d", "Linear", "WeightedMatMul"]
    assert weight_quant._parse_bit_widths("8,3,8,4") == [8, 3, 4]

    with pytest.raises(ValueError, match="at least one layer type"):
        weight_quant._parse_ablation_layer_types(" , ")
    with pytest.raises(ValueError, match="positive"):
        weight_quant._parse_bit_widths("3,0")


def test_ablation_runs_layer_types_outermost_and_widths_ascending(monkeypatch):
    calls = []

    def record_call(args, device, metrics, base_root):
        calls.append({
            "layer_type": args.ablation_layer_type,
            "bit_width": args.optimization_bit_width,
            "formats": args.baseline_formats,
            "baseline_type": args.baseline_experiment_type,
            "optimized_type": args.optimized_experiment_type,
            "per_chunk": args.per_chunk_format,
            "run_reference": args.run_reference,
        })
        return True

    monkeypatch.setattr(
        weight_quant,
        "_process_single_model_for_bit_width",
        record_call,
    )
    args = argparse.Namespace(
        baseline_formats=(
            "fp4_e1m2,fp4_e2m1,fp4_e3m0,"
            "fp3_e1m1,fp3_e2m0"
        ),
        optimized_experiment_type="weight_quant_optimized",
        ablation_layer_types="Conv2d,Linear,WeightedMatMul",
        bit_widths="4,3",
        per_chunk_format=False,
        weight_chunk_size=128,
    )

    weight_quant.process_single_model(args, "cuda", ["mse"], "/tmp/results")

    assert [(call["layer_type"], call["bit_width"]) for call in calls] == [
        ("Conv2d", 3),
        ("Conv2d", 4),
        ("Linear", 3),
        ("Linear", 4),
        ("WeightedMatMul", 3),
        ("WeightedMatMul", 4),
    ]
    assert calls[0]["formats"] == "fp3_e1m1,fp3_e2m0"
    assert calls[1]["formats"] == "fp4_e1m2,fp4_e2m1,fp4_e3m0"
    assert calls[0]["baseline_type"] == "weight_quant_ablation_conv2d_3"
    assert calls[0]["optimized_type"] == (
        "weight_quant_optimized_ablation_conv2d_3"
    )
    assert all(call["per_chunk"] for call in calls)
    assert [call["run_reference"] for call in calls] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_missing_ablation_type_skips_remaining_widths(monkeypatch):
    calls = []

    def skip_empty(args, device, metrics, base_root):
        calls.append(args.optimization_bit_width)
        return weight_quant._SKIPPED_EMPTY_ABLATION_TYPE

    monkeypatch.setattr(
        weight_quant,
        "_process_single_model_for_bit_width",
        skip_empty,
    )
    args = argparse.Namespace(
        baseline_formats="fp4_e1m2,fp3_e1m1",
        optimized_experiment_type="weight_quant_optimized",
        ablation_layer_types="WeightedMatMul",
        bit_widths="3,4",
        per_chunk_format=False,
        weight_chunk_size=128,
    )

    weight_quant.process_single_model(args, "cuda", ["mse"], "/tmp/results")

    assert calls == [3]


def test_ablation_defaults_to_bit_widths_three_through_eight(monkeypatch):
    calls = []

    def record_width(args, device, metrics, base_root):
        calls.append(args.optimization_bit_width)
        return True

    monkeypatch.setattr(
        weight_quant,
        "_process_single_model_for_bit_width",
        record_width,
    )
    args = argparse.Namespace(
        baseline_formats=','.join(weight_quant.baseline_formats),
        optimized_experiment_type="weight_quant_optimized",
        ablation_layer_types="Conv2d",
        bit_widths=None,
        per_chunk_format=False,
        weight_chunk_size=128,
    )

    weight_quant.process_single_model(args, "cpu", ["mse"], "/tmp/results")

    assert calls == [3, 4, 5, 6, 7, 8]


def test_ablation_requires_positive_chunk_size():
    args = argparse.Namespace(
        baseline_formats="fp3_e1m1",
        optimized_experiment_type="weight_quant_optimized",
        ablation_layer_types="Conv2d",
        bit_widths="3",
        weight_chunk_size=0,
    )

    with pytest.raises(ValueError, match="positive --weight_chunk_size"):
        weight_quant.process_single_model(args, "cpu", ["mse"], "/tmp/results")


def test_list_layer_types_mode_short_circuits_the_sweep(monkeypatch):
    calls = []
    monkeypatch.setattr(
        weight_quant,
        "print_ablation_layer_types",
        lambda args, device, base_root: calls.append(
            (args.model_name, device, base_root)
        ),
    )
    args = argparse.Namespace(
        model_name="toy",
        list_ablation_layer_types=True,
    )

    weight_quant.process_single_model(args, "cpu", ["mse"], "/tmp/results")

    assert calls == [("toy", "cpu", "/tmp/results")]


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(4, 1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )


class WeightedMatMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, value):
        return torch.matmul(value, self.weight)


class StatelessMatMul(nn.Module):
    def forward(self, left, right):
        return torch.matmul(left, right)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)
        self.weighted_matmul = WeightedMatMul()
        self.stateless_matmul = StatelessMatMul()


class MobileVitBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_conv = nn.Conv2d(4, 4, 1)
        self.transformer = nn.Sequential(Block())
        self.conv_proj = nn.Conv2d(4, 4, 1)


class LayerGroupToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.stem_norm = nn.BatchNorm2d(4)
        self.encoder = EncoderBlock()
        self.head = nn.Linear(4, 2)
        self.mobile = MobileVitBlock()


def test_weight_layer_discovery_uses_concrete_types_inside_transformers():
    model = LayerGroupToyModel()

    layer_types = weight_quant.discover_weight_layer_types(model)

    assert layer_types["Conv2d"] == [
        "stem",
        "mobile.local_conv",
        "mobile.conv_proj",
    ]
    assert layer_types["BatchNorm2d"] == ["stem_norm"]
    assert "head" in layer_types["Linear"]
    assert "encoder.self_attention.q_proj" in layer_types["Linear"]
    assert "encoder.mlp.0" in layer_types["Linear"]
    assert "mobile.transformer.0.qkv" in layer_types["Linear"]
    assert "mobile.transformer.0.fc2" in layer_types["Linear"]
    assert layer_types["WeightedMatMul"] == [
        "mobile.transformer.0.weighted_matmul"
    ]
    assert "StatelessMatMul" not in layer_types
    assert "Transformer" not in layer_types


def test_ablation_type_resolution_is_case_insensitive_and_lists_choices():
    discovered = {"Conv2d": ["stem"], "Linear": ["head"]}

    assert (
        weight_quant._resolve_ablation_layer_type("linear", discovered)
        == "Linear"
    )
    with pytest.raises(ValueError, match="Available types: Conv2d, Linear"):
        weight_quant._resolve_ablation_layer_type("MatMul", discovered)


def test_print_ablation_layer_types_lists_only_weighted_types(
    monkeypatch,
    tmp_path,
    capsys,
):
    class FakeRunner:
        def __init__(self, device):
            self.device = device

        def prepare_model_with_materialized_weights(self, config, output_dir):
            return LayerGroupToyModel(), None, None

    monkeypatch.setattr(weight_quant, "Runner", FakeRunner)
    monkeypatch.setattr(
        weight_quant,
        "build_fp32_runtime_config",
        lambda args: {},
    )
    args = argparse.Namespace(model_name="toy")

    discovered = weight_quant.print_ablation_layer_types(
        args,
        "cpu",
        str(tmp_path),
    )

    output = capsys.readouterr().out
    assert "Ablatable weighted layer types for toy:" in output
    assert "Linear (" in output
    assert "WeightedMatMul (1)" in output
    assert "StatelessMatMul" not in discovered
    assert "stateless MatMul/BMM operations" in output
    manifest_path = (
        tmp_path / "toy" / "ablations" / "weighted_layer_types.json"
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["layer_types"]["WeightedMatMul"]["layers"] == [
        "mobile.transformer.0.weighted_matmul"
    ]


def test_weight_analysis_visits_only_selected_logical_layers(monkeypatch):
    visited = []

    def record_tensor(weight, layer_name, *unused_args):
        visited.append(layer_name)

    monkeypatch.setattr(weight_quant, "_analyze_weight_tensor", record_tensor)
    model = LayerGroupToyModel()
    selected = {
        "stem",
        "encoder.self_attention.q_proj",
        "mobile.transformer.0.weighted_matmul",
    }

    weight_quant.run_weight_quantization_analysis(
        argparse.Namespace(),
        model,
        ["mse"],
        ["fp4_e2m1"],
        {},
        (nn.Conv2d, nn.Linear, nn.MultiheadAttention),
        target_layer_names=selected,
    )

    assert visited == [
        "stem",
        "encoder.self_attention.q_proj",
        "mobile.transformer.0.weighted_matmul",
    ]


def test_target_only_uniform_materialization_preserves_other_weights(monkeypatch):
    monkeypatch.setattr(
        weight_quant,
        "get_quantized_tensor_sim",
        lambda tensor, *args, **kwargs: (tensor + 1.0, 1.0),
    )
    model = LayerGroupToyModel()
    reference = {key: value.clone() for key, value in model.state_dict().items()}
    linear_names = weight_quant.discover_weight_layer_types(model)["Linear"]
    layer_results = {name: {} for name in linear_names}

    candidate, quant_map = weight_quant.create_uniform_quantized_state_dict(
        model,
        layer_results,
        argparse.Namespace(weight_chunk_size=128),
        "fp4_e2m1",
    )

    assert not torch.equal(
        candidate["encoder.self_attention.in_proj_weight"],
        reference["encoder.self_attention.in_proj_weight"],
    )
    assert not torch.equal(
        candidate["mobile.transformer.0.fc1.weight"],
        reference["mobile.transformer.0.fc1.weight"],
    )
    assert not torch.equal(candidate["head.weight"], reference["head.weight"])
    assert torch.equal(candidate["stem.weight"], reference["stem.weight"])
    assert torch.equal(
        candidate["mobile.local_conv.weight"],
        reference["mobile.local_conv.weight"],
    )
    weight_quant.assert_non_target_state_dict_unchanged(
        reference,
        candidate,
        quant_map.keys(),
    )


def test_weighted_matmul_is_materialized_as_its_own_type(monkeypatch):
    monkeypatch.setattr(
        weight_quant,
        "get_quantized_tensor_sim",
        lambda tensor, *args, **kwargs: (tensor + 1.0, 1.0),
    )
    model = LayerGroupToyModel()
    reference = {key: value.clone() for key, value in model.state_dict().items()}
    matmul_names = weight_quant.discover_weight_layer_types(model)[
        "WeightedMatMul"
    ]

    candidate, quant_map = weight_quant.create_uniform_quantized_state_dict(
        model,
        {name: {} for name in matmul_names},
        argparse.Namespace(weight_chunk_size=128),
        "fp4_e2m1",
    )

    matmul_key = "mobile.transformer.0.weighted_matmul.weight"
    assert not torch.equal(candidate[matmul_key], reference[matmul_key])
    assert torch.equal(candidate["head.weight"], reference["head.weight"])
    assert torch.equal(candidate["stem.weight"], reference["stem.weight"])
    assert quant_map == {
        "mobile.transformer.0.weighted_matmul": "fp4_e2m1"
    }
    weight_quant.assert_non_target_state_dict_unchanged(
        reference,
        candidate,
        quant_map.keys(),
    )


def test_non_target_state_assertion_rejects_unselected_changes():
    model = LayerGroupToyModel()
    reference = {key: value.clone() for key, value in model.state_dict().items()}
    candidate = {key: value.clone() for key, value in reference.items()}
    candidate["head.weight"].add_(1.0)

    with pytest.raises(AssertionError, match="head.weight"):
        weight_quant.assert_non_target_state_dict_unchanged(
            reference,
            candidate,
            ["stem"],
        )
