import os
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.pseudo_mse.pseudo_mse import (  # noqa: E402
    BASELINE_METRIC_NAME,
    DEFAULT_BIT_WIDTHS,
    DEFAULT_EXP_BITS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RANDOM_SUBSET_SIZE,
    L1_METRIC_LABEL,
    L1_METRIC_NAME,
    METRIC_NAME,
    WEIGHT_DT,
    build_metric_comparison_specs,
    build_pseudo_mse_sweep_specs,
    build_pseudo_mse_input_quant_cfg,
    build_pseudo_mse_runtime_config,
    get_args,
    _plot_model_summary,
    _write_model_summary,
)


def test_pseudo_mse_configs_use_fp32_weights_dynamic_activations_and_subset():
    args = get_args([])
    specs = build_pseudo_mse_sweep_specs(args)

    assert args.bit_widths == DEFAULT_BIT_WIDTHS
    assert args.exp_bits == DEFAULT_EXP_BITS
    assert [spec.bit_width for spec in specs] == [8, 7, 6, 5, 4]
    assert [spec.activation_dt for spec in specs] == [
        "dyn_a8_e1e2",
        "dyn_a7_e1e2",
        "dyn_a6_e1e2",
        "dyn_a5_e1e2",
        "dyn_a4_e1e2",
    ]
    assert [spec.candidate_formats for spec in specs] == [
        ["fp8_e1m6", "fp8_e2m5"],
        ["fp7_e1m5", "fp7_e2m4"],
        ["fp6_e1m4", "fp6_e2m3"],
        ["fp5_e1m3", "fp5_e2m2"],
        ["fp4_e1m2", "fp4_e2m1"],
    ]

    comparison_specs = build_metric_comparison_specs(args)
    mse_spec = comparison_specs[0]
    l1_spec = comparison_specs[1]
    pseudo_spec = comparison_specs[2]

    runtime_cfg = build_pseudo_mse_runtime_config(args, pseudo_spec)
    input_quant_cfg = build_pseudo_mse_input_quant_cfg(args, pseudo_spec)

    assert runtime_cfg["adapter"]["weight_quantization"] is False
    assert runtime_cfg["adapter"]["input_quantization"] is True
    assert runtime_cfg["quantization"]["weight_source"] == "fp32"
    assert runtime_cfg["dataset"]["random_subset_size"] == DEFAULT_RANDOM_SUBSET_SIZE
    assert runtime_cfg["dataset"]["random_seed"] == DEFAULT_RANDOM_SEED
    assert runtime_cfg["experiment"]["metric"] == METRIC_NAME
    assert runtime_cfg["experiment"]["metric_label"] == METRIC_NAME
    assert runtime_cfg["experiment"]["weight_dt"] == WEIGHT_DT
    assert runtime_cfg["experiment"]["activation_dt"] == "dyn_a8_e1e2_pseudo_mse"
    assert runtime_cfg["experiment"]["bit_width"] == 8
    assert runtime_cfg["experiment"]["candidate_formats"] == ["fp8_e1m6", "fp8_e2m5"]

    assert input_quant_cfg["enabled"] is True
    assert input_quant_cfg["mode"] == "dynamic"
    assert input_quant_cfg["metric"] == METRIC_NAME
    assert input_quant_cfg["restrict_post_relu_ufp"] is False
    assert input_quant_cfg["candidate_formats"] == ["fp8_e1m6", "fp8_e2m5"]

    mse_cfg = build_pseudo_mse_runtime_config(args, mse_spec)
    mse_input_quant_cfg = build_pseudo_mse_input_quant_cfg(args, mse_spec)
    assert mse_cfg["experiment"]["metric"] == BASELINE_METRIC_NAME
    assert mse_cfg["experiment"]["metric_label"] == "MSE"
    assert mse_cfg["experiment"]["activation_dt"] == "dyn_a8_e1e2_mse"
    assert mse_input_quant_cfg["metric"] == BASELINE_METRIC_NAME

    l1_cfg = build_pseudo_mse_runtime_config(args, l1_spec)
    l1_input_quant_cfg = build_pseudo_mse_input_quant_cfg(args, l1_spec)
    assert l1_cfg["experiment"]["metric"] == L1_METRIC_NAME
    assert l1_cfg["experiment"]["metric_label"] == L1_METRIC_LABEL
    assert l1_cfg["experiment"]["activation_dt"] == "dyn_a8_e1e2_l1"
    assert l1_input_quant_cfg["metric"] == L1_METRIC_NAME


def test_pseudo_mse_builds_metric_comparison_specs():
    args = get_args([])

    specs = build_metric_comparison_specs(args)

    assert len(specs) == 15
    assert [(spec.bit_width, spec.metric_label, spec.activation_dt) for spec in specs] == [
        (8, "MSE", "dyn_a8_e1e2_mse"),
        (8, L1_METRIC_LABEL, "dyn_a8_e1e2_l1"),
        (8, METRIC_NAME, "dyn_a8_e1e2_pseudo_mse"),
        (7, "MSE", "dyn_a7_e1e2_mse"),
        (7, L1_METRIC_LABEL, "dyn_a7_e1e2_l1"),
        (7, METRIC_NAME, "dyn_a7_e1e2_pseudo_mse"),
        (6, "MSE", "dyn_a6_e1e2_mse"),
        (6, L1_METRIC_LABEL, "dyn_a6_e1e2_l1"),
        (6, METRIC_NAME, "dyn_a6_e1e2_pseudo_mse"),
        (5, "MSE", "dyn_a5_e1e2_mse"),
        (5, L1_METRIC_LABEL, "dyn_a5_e1e2_l1"),
        (5, METRIC_NAME, "dyn_a5_e1e2_pseudo_mse"),
        (4, "MSE", "dyn_a4_e1e2_mse"),
        (4, L1_METRIC_LABEL, "dyn_a4_e1e2_l1"),
        (4, METRIC_NAME, "dyn_a4_e1e2_pseudo_mse"),
    ]


def test_summary_and_plots_include_dataset_size(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [
        {
            "model": "resnet50",
            "metric": "MSE",
            "bit_width": 8,
            "activation_dt": "dyn_a8_e1e2_mse",
            "candidate_formats": "fp8_e1m6,fp8_e2m5",
            "dataset_size": 5000,
            "random_seed": 42,
            "limit_batches": -1,
            "status": "SUCCESS",
            "acc1": 75.0,
            "acc5": 92.0,
            "certainty": 0.8,
            "norm_mse": 0.01,
            "norm_l1": 0.02,
            "error": None,
        },
        {
            "model": "resnet50",
            "metric": L1_METRIC_LABEL,
            "bit_width": 8,
            "activation_dt": "dyn_a8_e1e2_l1",
            "candidate_formats": "fp8_e1m6,fp8_e2m5",
            "dataset_size": 5000,
            "random_seed": 42,
            "limit_batches": -1,
            "status": "SUCCESS",
            "acc1": 75.5,
            "acc5": 92.5,
            "certainty": 0.805,
            "norm_mse": 0.0095,
            "norm_l1": 0.0195,
            "error": None,
        },
        {
            "model": "resnet50",
            "metric": METRIC_NAME,
            "bit_width": 8,
            "activation_dt": "dyn_a8_e1e2_pseudo_mse",
            "candidate_formats": "fp8_e1m6,fp8_e2m5",
            "dataset_size": 5000,
            "random_seed": 42,
            "limit_batches": -1,
            "status": "SUCCESS",
            "acc1": 76.0,
            "acc5": 93.0,
            "certainty": 0.81,
            "norm_mse": 0.009,
            "norm_l1": 0.019,
            "error": None,
        },
    ]
    dataset_label = "ImageNet random subset=5000, seed=42; evaluated samples=5000"

    _csv_path, txt_path = _write_model_summary(tmp_path, "resnet50", rows, dataset_label)
    assert dataset_label in Path(txt_path).read_text()

    plot_paths = _plot_model_summary(tmp_path, "resnet50", rows, dataset_label)
    assert {os.path.basename(path) for path in plot_paths} == {
        "resnet50_accuracy_vs_bits.png",
        "resnet50_norm_mse_vs_bits.png",
        "resnet50_norm_l1_vs_bits.png",
    }
