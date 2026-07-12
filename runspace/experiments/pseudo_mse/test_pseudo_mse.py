import csv
import os
from pathlib import Path
import sys

import pytest
import torch

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
    _load_models,
    _models_from_args,
    _plot_model_summary,
    _write_model_summary,
)
from runspace.experiments.pseudo_mse.generate_hw_vectors import (  # noqa: E402
    BIT_WIDTHS,
    compare_pseudo_mse_with_metric,
    decision_for_bit_width,
    make_raw_chunks,
    scale_raw_chunks,
)
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    pseudo_mse_decode_emb_python,
    pseudo_mse_encode_emb_python,
    pseudo_mse_choose_exp2_from_diff,
    pseudo_mse_err2_minus_err1_from_scaled,
    pseudo_mse_shifted_e2_wins,
    pseudo_mse_win_counts_from_diff,
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


def test_pseudo_mse_uses_shifted_e2_win_count_not_summed_mse_diff():
    diff = torch.tensor(
        [
            [10.0] + [-1.0] * 8 + [0.0] * 7,
            [-10.0] + [1.0] * 2 + [0.0] * 13,
            [1.0] + [-1.0] * 5 + [0.0] * 10,
            [0.0] + [-1.0] * 3 + [0.0] * 12,
        ],
        dtype=torch.float32,
    )

    exp1_wins, exp2_wins = pseudo_mse_win_counts_from_diff(diff)
    assert exp1_wins.tolist() == [1, 2, 1, 0]
    assert exp2_wins.tolist() == [8, 1, 5, 3]
    torch.testing.assert_close(
        pseudo_mse_shifted_e2_wins(exp2_wins),
        torch.tensor([2.0, 0.25, 1.25, 0.75], dtype=torch.float32),
    )
    torch.testing.assert_close(
        pseudo_mse_shifted_e2_wins(exp2_wins, e2_win_divisor=2),
        torch.tensor([4.0, 0.5, 2.5, 1.5], dtype=torch.float32),
    )

    count_decision = pseudo_mse_choose_exp2_from_diff(diff)
    mse_sum_decision = diff.sum(dim=1) < 0
    assert count_decision.tolist() == [True, False, True, True]
    assert mse_sum_decision.tolist() == [False, True, True, True]

    mode_diff = torch.tensor([[1.0, 1.0] + [-1.0] * 5], dtype=torch.float32)
    assert pseudo_mse_choose_exp2_from_diff(mode_diff).tolist() == [False]
    assert pseudo_mse_choose_exp2_from_diff(mode_diff, e2_win_divisor=2).tolist() == [True]


def test_pseudo_mse_divisor2_can_disagree_with_l1_on_rounded_hw_vector_chunks():
    raw_chunks = make_raw_chunks(num_chunks=50, seed=42)
    _scales, scaled_chunks = scale_raw_chunks(raw_chunks)

    total_bad_non_tie = 0
    for bit_width in BIT_WIDTHS:
        (
            _err1,
            _err2,
            _chunk_diff,
            _expected_e1_wins,
            _expected_e2_wins,
            _expected_e2_wins_shifted,
            choose_exp2,
            _expected_error,
            _q1_bits,
            _q2_bits,
            err_exp1_pre_square,
            err_exp2_pre_square,
            _pseudo_diff,
        ) = decision_for_bit_width(
            scaled_chunks,
            bit_width,
            e2_win_divisor=2,
        )

        l1_exp1 = err_exp1_pre_square.abs().sum(dim=1)
        l1_exp2 = err_exp2_pre_square.abs().sum(dim=1)
        l1_choose_exp2 = l1_exp2 < l1_exp1
        non_tie = l1_exp1 != l1_exp2
        non_tie_indices = torch.nonzero(non_tie, as_tuple=False).flatten()
        bad_non_tie = non_tie_indices[choose_exp2[non_tie] != l1_choose_exp2[non_tie]]
        total_bad_non_tie += int(bad_non_tie.numel())

        selected_l1 = torch.where(choose_exp2, l1_exp2, l1_exp1)
        assert torch.any(selected_l1 != torch.minimum(l1_exp1, l1_exp2))

    assert total_bad_non_tie > 0


def test_pseudo_mse_compare_report_divisor2_l1_reports_metric_min_mismatches(tmp_path):
    csv_path = tmp_path / "l1_div2_mismatches.csv"

    totals = compare_pseudo_mse_with_metric(
        str(csv_path),
        compare_metric="l1",
        e2_win_divisor=2,
        compare_tie_policy="min-error",
        num_chunks=50,
        seed=42,
        max_mismatches=0,
    )

    assert totals["metric_min_mismatched_chunks"] > 0
    assert totals["reported_mismatched_chunks"] == totals["metric_min_mismatched_chunks"]
    assert totals["rows_written"] == totals["reported_mismatched_chunks"] * 128
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == totals["rows_written"]
    assert {row["mismatch_kind"] for row in rows} == {"metric_min"}


def test_pseudo_mse_compare_report_writes_mismatch_metadata_csv(tmp_path):
    csv_path = tmp_path / "l1_div2_exp1_tie_mismatches.csv"

    totals = compare_pseudo_mse_with_metric(
        str(csv_path),
        compare_metric="l1",
        e2_win_divisor=2,
        compare_tie_policy="exp1",
        num_chunks=10,
        seed=42,
        max_mismatches=0,
    )

    assert totals["metric_min_mismatched_chunks"] > 0
    assert totals["reported_mismatched_chunks"] > totals["metric_min_mismatched_chunks"]
    assert totals["rows_written"] == totals["reported_mismatched_chunks"] * 128

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == totals["rows_written"]
    first = rows[0]
    for field in [
        "compare_metric",
        "compare_tie_policy",
        "mismatch_kind",
        "metric_decision",
        "metric_exp1_error_dec",
        "metric_exp2_error_dec",
        "value_index",
        "raw_fp32_hex",
        "raw_fp32_dec",
        "scaled_fp32_hex",
        "scaled_fp32_dec",
        "q_exp1_bits",
        "q_exp1_exp_field",
        "q_exp1_mant_field",
        "q_exp1_mant_bits",
        "q_exp2_bits",
        "q_exp2_exp_field",
        "q_exp2_mant_field",
        "q_exp2_mant_bits",
        "err_exp1_pre_square_fp32_hex",
        "err_exp1_pre_square_fp32_dec",
        "err_exp2_pre_square_fp32_hex",
        "err_exp2_pre_square_fp32_dec",
        "pseudo_diff_exp2_minus_exp1_fp32_hex",
        "pseudo_diff_exp2_minus_exp1_fp32_dec",
    ]:
        assert field in first
    assert first["compare_metric"] == "l1"
    assert first["compare_tie_policy"] == "exp1"
    assert first["mismatch_kind"] in {"metric_min", "tie_decision"}
    assert {"metric_min", "tie_decision"} <= {row["mismatch_kind"] for row in rows}


def test_pseudo_mse_bit_level_err2_minus_err1_cases():
    m1 = 6
    values = torch.tensor(
        [
            1.0 + 2.0 ** -m1,             # e=0 -> +X_M
            1.0,                          # e=0 with X_M=0 -> 0
            1.5 * 2.0 ** -1,              # e=1 -> 0
            (1.0 + 2.0 ** -5) * 2.0 ** -2,  # e=2 -> -X_(M-1)
            2.0 ** -(m1 + 1),             # e=M+1 -> hidden leading 1 -> -1
            2.0 ** -(m1 + 2),             # e>M+1 -> 0
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    diff = pseudo_mse_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
    )

    assert diff.tolist() == [[1.0, 0.0, 0.0, -1.0, -1.0, 0.0]]


def test_pseudo_mse_encode_rounds_mantissa_bits():
    m1 = 6
    value = torch.tensor([1.0 + 0.75 * (2.0 ** -m1)], dtype=torch.float32)

    packed = pseudo_mse_encode_emb_python(value, exp_bits=1, mantissa_bits=m1, is_signed=True)
    decoded = pseudo_mse_decode_emb_python(packed, exp_bits=1, mantissa_bits=m1, is_signed=True)

    assert int(packed.item()) & ((1 << m1) - 1) == 1
    assert decoded.item() == 1.0 + 2.0 ** -m1


def test_pseudo_mse_loads_models_file(tmp_path):
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "\n".join(
            [
                "- name: resnet50",
                "  weights: DEFAULT",
                "  source: torchvision",
                "- mobilevit_s",
                "- model_name: vit_b_16",
                "  model_source: torchvision",
            ]
        )
    )

    assert _load_models(str(models_file)) == [
        {"name": "resnet50", "weights": "DEFAULT", "source": "torchvision"},
        {"name": "mobilevit_s", "weights": "DEFAULT", "source": None},
        {"name": "vit_b_16", "weights": "DEFAULT", "source": "torchvision"},
    ]

    args = get_args(["--models_file", str(models_file), "--model_name", "ignored"])
    models, resolved_path = _models_from_args(args)

    assert resolved_path == str(models_file)
    assert [model["name"] for model in models] == ["resnet50", "mobilevit_s", "vit_b_16"]


def test_pseudo_mse_accepts_yaml_path_as_model_name(tmp_path):
    models_file = tmp_path / "models.yaml"
    models_file.write_text("models:\n  - name: efficientnet_b0\n    weights: DEFAULT\n")

    args = get_args(["--model_name", str(models_file)])
    models, resolved_path = _models_from_args(args)

    assert resolved_path == str(models_file)
    assert models == [{"name": "efficientnet_b0", "weights": "DEFAULT", "source": None}]


def test_pseudo_mse_rejects_empty_models_file(tmp_path):
    models_file = tmp_path / "models.yaml"
    models_file.write_text("")

    args = get_args(["--models_file", str(models_file)])
    with pytest.raises(ValueError, match="No model entries"):
        _models_from_args(args)


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
