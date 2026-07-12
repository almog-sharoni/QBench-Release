import csv
import os
from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.pseudo_mse2.pseudo_mse import (  # noqa: E402
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
from runspace.experiments.pseudo_mse2.generate_hw_vectors import (  # noqa: E402
    BIT_WIDTHS,
    _debug_chunk_sums,
    _value_metadata_row,
    compare_pseudo_mse_with_metric,
    decision_for_bit_width,
    make_raw_chunks,
    normalize_mantissa_window_bits,
    scale_raw_chunks,
)
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    pseudo_mse_decode_emb_python,
    pseudo_mse_encode_emb_python,
    pseudo_mse_choose_exp2_from_diff,
    pseudo_mse2_err2_minus_err1_from_scaled,
    pseudo_mse_shifted_e2_wins,
    pseudo_mse_weighted_win_counts_from_diff,
)


def _pseudo_mse2_diff_vector(values, *, m1=6, mantissa_window_bits=None):
    return pseudo_mse2_err2_minus_err1_from_scaled(
        torch.tensor([values], dtype=torch.float32),
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
        mantissa_window_bits=mantissa_window_bits,
    )


def _assert_pseudo_mse2_diff_tensor_ranges(diff):
    assert diff.dtype == torch.int32

    one = 1 << 24
    three = 3 << 24
    quarter = 1 << 22
    three_quarters = 3 << 22

    is_zero = diff == 0
    positive_ok = (diff >= one) & (diff < three)
    negative_ok = (diff <= -quarter) & (diff > -three_quarters)
    bad = ~(is_zero | positive_ok | negative_ok)
    assert not bool(bad.any()), diff[bad]
    return diff


def _assert_pseudo_mse2_diff_ranges(values, *, m1=6, mantissa_window_bits=None):
    diff = _pseudo_mse2_diff_vector(
        values,
        m1=m1,
        mantissa_window_bits=mantissa_window_bits,
    )
    return _assert_pseudo_mse2_diff_tensor_ranges(diff)


def test_pseudo_mse2_diff_range_assertion_rejects_invalid_values():
    _assert_pseudo_mse2_diff_tensor_ranges(
        torch.tensor([[0, 1 << 24, (3 << 24) - 1, -(1 << 22), -((3 << 22) - 1)]], dtype=torch.int32)
    )

    invalid_values = [
        1 << 23,     # positive but below 1.0
        3 << 24,     # positive upper bound is exclusive
        -(1 << 21),  # negative but above -1/4
        -(3 << 22),  # negative lower bound is exclusive
    ]
    for invalid_value in invalid_values:
        with pytest.raises(AssertionError):
            _assert_pseudo_mse2_diff_tensor_ranges(
                torch.tensor([[invalid_value]], dtype=torch.int32)
            )


def test_pseudo_mse_configs_use_fp32_weights_dynamic_activations_and_subset():
    args = get_args(["--mantissa-window-bits", "3"])
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
    assert runtime_cfg["experiment"]["activation_dt"] == "dyn_a8_e1e2_pseudo_mse2"
    assert runtime_cfg["experiment"]["bit_width"] == 8
    assert runtime_cfg["experiment"]["candidate_formats"] == ["fp8_e1m6", "fp8_e2m5"]
    assert runtime_cfg["experiment"]["mantissa_window_bits"] == 3

    assert input_quant_cfg["enabled"] is True
    assert input_quant_cfg["mode"] == "dynamic"
    assert input_quant_cfg["metric"] == METRIC_NAME
    assert input_quant_cfg["restrict_post_relu_ufp"] is False
    assert input_quant_cfg["candidate_formats"] == ["fp8_e1m6", "fp8_e2m5"]
    assert input_quant_cfg["pseudo_mse2_mantissa_window_bits"] == 3

    mse_cfg = build_pseudo_mse_runtime_config(args, mse_spec)
    mse_input_quant_cfg = build_pseudo_mse_input_quant_cfg(args, mse_spec)
    assert mse_cfg["experiment"]["metric"] == BASELINE_METRIC_NAME
    assert mse_cfg["experiment"]["metric_label"] == "MSE"
    assert mse_cfg["experiment"]["activation_dt"] == "dyn_a8_e1e2_mse"
    assert mse_input_quant_cfg["metric"] == BASELINE_METRIC_NAME
    assert mse_input_quant_cfg["pseudo_mse2_mantissa_window_bits"] == 0

    l1_cfg = build_pseudo_mse_runtime_config(args, l1_spec)
    l1_input_quant_cfg = build_pseudo_mse_input_quant_cfg(args, l1_spec)
    assert l1_cfg["experiment"]["metric"] == L1_METRIC_NAME
    assert l1_cfg["experiment"]["metric_label"] == L1_METRIC_LABEL
    assert l1_cfg["experiment"]["activation_dt"] == "dyn_a8_e1e2_l1"
    assert l1_input_quant_cfg["metric"] == L1_METRIC_NAME
    assert l1_input_quant_cfg["pseudo_mse2_mantissa_window_bits"] == 0


def test_pseudo_mse_builds_metric_comparison_specs():
    args = get_args([])

    specs = build_metric_comparison_specs(args)

    assert len(specs) == 15
    assert [(spec.bit_width, spec.metric_label, spec.activation_dt) for spec in specs] == [
        (8, "MSE", "dyn_a8_e1e2_mse"),
        (8, L1_METRIC_LABEL, "dyn_a8_e1e2_l1"),
        (8, METRIC_NAME, "dyn_a8_e1e2_pseudo_mse2"),
        (7, "MSE", "dyn_a7_e1e2_mse"),
        (7, L1_METRIC_LABEL, "dyn_a7_e1e2_l1"),
        (7, METRIC_NAME, "dyn_a7_e1e2_pseudo_mse2"),
        (6, "MSE", "dyn_a6_e1e2_mse"),
        (6, L1_METRIC_LABEL, "dyn_a6_e1e2_l1"),
        (6, METRIC_NAME, "dyn_a6_e1e2_pseudo_mse2"),
        (5, "MSE", "dyn_a5_e1e2_mse"),
        (5, L1_METRIC_LABEL, "dyn_a5_e1e2_l1"),
        (5, METRIC_NAME, "dyn_a5_e1e2_pseudo_mse2"),
        (4, "MSE", "dyn_a4_e1e2_mse"),
        (4, L1_METRIC_LABEL, "dyn_a4_e1e2_l1"),
        (4, METRIC_NAME, "dyn_a4_e1e2_pseudo_mse2"),
    ]


def test_pseudo_mse2_uses_weighted_shifted_e2_win_count():
    diff = torch.tensor(
        [
            [2.0, 1.0, -3.0, -1.0, 0.0],
            [1.0, -2.0, -2.0, 0.0, 0.0],
            [1.0, -3.0, -3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    exp1_wins, exp2_wins = pseudo_mse_weighted_win_counts_from_diff(diff)
    assert exp1_wins.tolist() == [3, 1, 1, 0]
    assert exp2_wins.tolist() == [4, 4, 6, 0]
    assert pseudo_mse_shifted_e2_wins(exp2_wins).tolist() == [1.0, 1.0, 1.5, 0.0]
    assert pseudo_mse_shifted_e2_wins(exp2_wins, e2_win_divisor=2).tolist() == [2, 2, 3, 0]

    default_decision = pseudo_mse_choose_exp2_from_diff(diff, weighted=True)
    divisor2_decision = pseudo_mse_choose_exp2_from_diff(diff, e2_win_divisor=2, weighted=True)
    assert default_decision.tolist() == [True, True, True, False]
    assert divisor2_decision.tolist() == [True, True, True, False]


def test_pseudo_mse2_hw_vector_chunks_have_weighted_diff_range():
    raw_chunks = make_raw_chunks(num_chunks=50, seed=42)
    _scales, scaled_chunks = scale_raw_chunks(raw_chunks)

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
            pseudo_diff,
        ) = decision_for_bit_width(
            scaled_chunks,
            bit_width,
            e2_win_divisor=2,
        )

        _assert_pseudo_mse2_diff_tensor_ranges(pseudo_diff)
        assert (pseudo_diff.abs() > 1).any()
        exp1_wins, exp2_wins = pseudo_mse_weighted_win_counts_from_diff(pseudo_diff)
        assert torch.equal(exp1_wins, _expected_e1_wins)
        assert torch.equal(exp2_wins, _expected_e2_wins)
        assert choose_exp2.dtype == torch.bool


def test_pseudo_mse2_compare_report_writes_l1_metric_min_mismatches(tmp_path):
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
    assert rows[0]["mismatch_kind"] == "metric_min"


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

    assert totals["reported_mismatched_chunks"] > 0
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
        "chunk_sum_pseudo_diff_exp2_minus_exp1_neg_div4_fp32_dec",
        "chunk_sum_square_err_exp2_minus_square_err_exp1_pre_square_div_2_neg_m_dec",
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
        "pseudo_diff_exp2_minus_exp1_neg_div4_fp32_dec",
        "square_err_exp2_minus_square_err_exp1_pre_square_div_2_neg_m_dec",
    ]:
        assert field in first
    assert first["compare_metric"] == "l1"
    assert first["compare_tie_policy"] == "exp1"
    assert first["mismatch_kind"] in {"metric_min", "tie_decision", "decision"}


def test_pseudo_mse2_value_metadata_writes_shifted_and_scaled_debug_terms():
    row = _value_metadata_row(
        "scaled",
        bit_width=8,
        m1=6,
        m2=5,
        raw_chunks=torch.tensor([[0.0]], dtype=torch.float32),
        scaled_chunks=torch.tensor([[1.0]], dtype=torch.float32),
        q1_bits=torch.tensor([[0]], dtype=torch.int64),
        q2_bits=torch.tensor([[0]], dtype=torch.int64),
        err_exp1_pre_square=torch.tensor([[-0.125]], dtype=torch.float32),
        err_exp2_pre_square=torch.tensor([[0.5]], dtype=torch.float32),
        pseudo_diff=torch.tensor([[-3.0]], dtype=torch.float32),
        chunk_idx=0,
        value_idx=0,
    )

    assert row["pseudo_diff_exp2_minus_exp1_neg_div4_fp32_dec"] == "-0.75"
    assert row["square_err_exp2_minus_square_err_exp1_pre_square_div_2_neg_m_dec"] == "960"


def test_pseudo_mse2_debug_chunk_sums_are_separate_columns():
    pseudo_sum, square_sum = _debug_chunk_sums(
        err_exp1_pre_square=torch.tensor([[-0.125, 0.25]], dtype=torch.float32),
        err_exp2_pre_square=torch.tensor([[0.5, 0.125]], dtype=torch.float32),
        pseudo_diff=torch.tensor([[-3.0, 2.0]], dtype=torch.float32),
        m1=6,
    )

    assert pseudo_sum.tolist() == [1.25]
    assert square_sum.tolist() == [768.0]


def test_pseudo_mse2_diff_vector_assertion_starter():
    m1 = 6
    cases = [
        (
            "e0_xm_xm1_xm2",
            1.0 + 2.0 ** -m1 + 2.0 ** -(m1 + 1) + 2.0 ** -(m1 + 2),
            1,
        ),
        (
            "e1_zero",
            1.5 * 2.0 ** -1,
            0,
        ),
        (
            "e2_xk_xk1_xk2_shifted",
            (1.0 + 2.0 ** -5 + 2.0 ** -6 + 2.0 ** -7) * 2.0 ** -2,
            -1,
        ),
        (
            "hidden_x1_x2_shifted",
            (1.0 + 2.0 ** -1 + 2.0 ** -2) * 2.0 ** -(m1 + 1),
            -1,
        ),
        (
            "too_small_zero",
            2.0 ** -(m1 + 2),
            0,
        ),
    ]

    labels = [label for label, _value, _expected in cases]
    assert len(set(labels)) == len(labels)
    diff = _assert_pseudo_mse2_diff_ranges(
        [value for _label, value, _expected_sign in cases],
        m1=m1,
    )
    assert torch.sign(diff.squeeze(0)).tolist() == [
        expected_sign for _label, _value, expected_sign in cases
    ]


def test_pseudo_mse2_bit_level_err2_minus_err1_cases():
    m1 = 6
    full_mantissa = 1.0 + sum(2.0 ** -i for i in range(1, 24))
    values = torch.tensor(
        [
            1.0 + 2.0 ** -m1 + 2.0 ** -(m1 + 1) + 2.0 ** -(m1 + 2),
            1.0 + 2.0 ** -m1 + 2.0 ** -(m1 + 2),
            1.0 + 2.0 ** -m1,
            1.0,                          # e=0 with X_M=0 -> 0
            1.5 * 2.0 ** -1,              # e=1 -> 0
            (1.0 + 2.0 ** -5 + 2.0 ** -6 + 2.0 ** -7) * 2.0 ** -2,
            (1.0 + 2.0 ** -5 + 2.0 ** -7) * 2.0 ** -2,
            (1.0 + 2.0 ** -5) * 2.0 ** -2,
            2.0 ** -(m1 + 1),
            (1.0 + 2.0 ** -1 + 2.0 ** -2) * 2.0 ** -(m1 + 1),
            full_mantissa * 2.0 ** -(m1 + 1),
            2.0 ** -(m1 + 2),             # e>M+1 -> 0
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    diff = pseudo_mse2_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
    )

    expected = torch.tensor([[
        (1 << 24) + (1 << 24) + (1 << 23),
        (1 << 24) + (1 << 23),
        1 << 24,
        0.0,
        0.0,
        -((1 << 22) + (1 << 22) + (1 << 21)),
        -((1 << 22) + (1 << 21)),
        -(1 << 22),
        -(1 << 22),
        -((1 << 22) + (1 << 22) + (1 << 21)),
        -((3 << 22) - 1),
        0.0,
    ]], dtype=torch.int32)
    torch.testing.assert_close(diff, expected)


def test_pseudo_mse2_bit_level_err2_minus_err1_limited_window():
    m1 = 6
    value_with_all_bits_from_m = 1.0 + sum(2.0 ** -i for i in range(m1, 24))
    value_with_all_bits_from_k = (1.0 + sum(2.0 ** -i for i in range(5, 24))) * 2.0 ** -2
    full_mantissa = 1.0 + sum(2.0 ** -i for i in range(1, 24))
    values = torch.tensor(
        [
            value_with_all_bits_from_m,
            value_with_all_bits_from_k,
            full_mantissa * 2.0 ** -(m1 + 1),
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    full_diff = pseudo_mse2_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
    )
    limited_diff = pseudo_mse2_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
        mantissa_window_bits=3,
    )

    full_expected = torch.tensor([[
        (3 << 24) - (1 << 8),
        -((3 << 22) - (1 << 5)),
        -((3 << 22) - 1),
    ]], dtype=torch.int32)
    limited_expected = torch.tensor([[
        (1 << 24) + (1 << 24) + (1 << 23),
        -((1 << 22) + (1 << 22) + (1 << 21)),
        -((1 << 22) + (1 << 22) + (1 << 21)),
    ]], dtype=torch.int32)
    torch.testing.assert_close(full_diff, full_expected)
    torch.testing.assert_close(limited_diff, limited_expected)


def test_pseudo_mse2_mantissa_window_24_matches_full_window():
    m1 = 6
    values = torch.tensor(
        [
            1.0 + sum(2.0 ** -i for i in range(m1, 24)),
            (1.0 + sum(2.0 ** -i for i in range(5, 24))) * 2.0 ** -2,
            (1.0 + sum(2.0 ** -i for i in range(1, 24))) * 2.0 ** -(m1 + 1),
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    full_diff = pseudo_mse2_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
    )
    window_24_diff = pseudo_mse2_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
        mantissa_window_bits=24,
    )

    torch.testing.assert_close(window_24_diff, full_diff)
    with pytest.raises(ValueError, match="at most 24"):
        normalize_mantissa_window_bits(25)


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
            "activation_dt": "dyn_a8_e1e2_pseudo_mse2",
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
