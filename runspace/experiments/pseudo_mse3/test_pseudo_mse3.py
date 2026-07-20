import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.pseudo_mse3.pseudo_mse import (  # noqa: E402
    BASELINE_METRIC_NAME,
    L1_METRIC_NAME,
    METRIC_NAME,
    _plot_model_summary,
    _write_combined_summary,
    _write_model_summary,
    build_metric_comparison_specs,
    build_pseudo_mse_input_quant_cfg,
    build_pseudo_mse_runtime_config,
    get_args,
)
from runspace.experiments.pseudo_mse3.generate_hw_vectors import (  # noqa: E402
    compare_pseudo_mse3_with_metric,
    decision_for_bit_width,
    make_raw_chunks,
    scale_raw_chunks,
    verify_python_vectors,
    write_vectors,
)
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    _assert_pseudo_mse3_scaled_diff_ranges,
    dynamic_input_metric_code,
    get_dynamic_input_metric_spec,
    pseudo_mse3_choose_exp2_from_diff,
    pseudo_mse3_err2_minus_err1_from_scaled,
    pseudo_mse3_fixed_point_from_diff,
    pseudo_mse_decode_emb_python,
    pseudo_mse_reconstruct_scaled_python,
    pseudo_mse_reconstruct_scaled_trunc_python,
)


def test_pseudo_mse3_metric_is_pairwise_cuda_metric():
    spec = get_dynamic_input_metric_spec("pseudo_MSE3")

    assert spec.name == "pseudo_mse3"
    assert spec.display_name == METRIC_NAME
    assert spec.implemented
    assert spec.cuda_code == 9
    assert dynamic_input_metric_code("pseudo_mse3") == 9


def test_pseudo_mse3_zero_bits_uses_fixed_point_scale_one():
    m1 = 6
    values = torch.tensor(
        [
            0.0,
            17.0 / 4096.0,
            1.0 + 33.0 / 4096.0,
            1.75,
            2.0 ** -(m1 + 2),
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    diff = pseudo_mse3_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
    )

    q1_scaled = pseudo_mse_reconstruct_scaled_trunc_python(values, 1, m1, True)
    q2_scaled = pseudo_mse_reconstruct_scaled_trunc_python(values, 2, m1 - 1, True)
    expected = ((values - q2_scaled).pow(2) - (values - q1_scaled).pow(2)) * float(
        2.0 ** (2 * m1)
    )
    _assert_pseudo_mse3_scaled_diff_ranges(expected)
    assert torch.equal(diff, torch.floor(expected).to(torch.int32))
    assert diff.dtype == torch.int32


def test_pseudo_mse3_bits_to_take_returns_fixed_point_diff():
    m1 = 6
    values = torch.tensor(
        [
            17.0 / 4096.0,
            1.0 + 33.0 / 4096.0,
            1.75,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    q1_scaled = pseudo_mse_reconstruct_scaled_trunc_python(values, 1, m1, True)
    q2_scaled = pseudo_mse_reconstruct_scaled_trunc_python(values, 2, m1 - 1, True)
    normalized_diff = (
        (values - q2_scaled).pow(2) - (values - q1_scaled).pow(2)
    ) * float(2.0 ** (2 * m1))
    fixed = pseudo_mse3_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
        bits_to_take=20,
    )

    assert fixed.dtype == torch.int32
    torch.testing.assert_close(
        fixed.to(torch.float32),
        torch.floor(normalized_diff * float(2.0**20)),
    )

    fixed_from_float = pseudo_mse3_err2_minus_err1_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m1 - 1,
        is_signed=True,
        bits_to_take=20.0,
    )
    torch.testing.assert_close(fixed_from_float, fixed)

    with pytest.raises(ValueError):
        pseudo_mse3_err2_minus_err1_from_scaled(
            values,
            exp1_mantissa_width=m1,
            exp2_mantissa_width=m1 - 1,
            is_signed=True,
            bits_to_take=20.5,
        )

    direct_diff = torch.tensor(
        [[0.24, 0.25, 0.26, -0.24, -0.25, -0.26]],
        dtype=torch.float32,
    )
    nearest = pseudo_mse3_fixed_point_from_diff(
        direct_diff,
        bits_to_take=1,
        fixed_rounding="nearest",
    )
    assert nearest.tolist() == [[0, 4, 4, -2, -2, -2]]
    zero_bits = pseudo_mse3_fixed_point_from_diff(
        direct_diff,
        bits_to_take=0,
        fixed_rounding="nearest",
    )
    assert zero_bits.tolist() == [[0, 0, 0, -1, -1, -1]]
    assert zero_bits.dtype == torch.int32


def test_pseudo_mse3_assertion_accepts_random_scaled_chunks_for_supported_widths():
    generator = torch.Generator().manual_seed(42)
    values = torch.rand((50, 128), generator=generator, dtype=torch.float32) * 4.0 - 2.0

    for bit_width in (8, 7, 6, 5, 4):
        m1 = bit_width - 2
        diff = pseudo_mse3_err2_minus_err1_from_scaled(
            values,
            exp1_mantissa_width=m1,
            exp2_mantissa_width=m1 - 1,
            is_signed=True,
        )
        assert diff.shape == values.shape


def test_pseudo_mse3_selection_truncates_but_output_codec_remains_rtn():
    midpoint = torch.tensor([[1.0 + 2.0**-7]], dtype=torch.float32)

    selection_value = pseudo_mse_reconstruct_scaled_trunc_python(
        midpoint,
        exp_bits=1,
        mantissa_bits=6,
        is_signed=True,
    )
    output_value = pseudo_mse_reconstruct_scaled_python(
        midpoint,
        exp_bits=1,
        mantissa_bits=6,
        is_signed=True,
    )

    assert selection_value.item() == 1.0
    assert output_value.item() == 1.0 + 2.0**-6


def test_pseudo_mse3_scaled_diff_assertion_rejects_invalid_values():
    _assert_pseudo_mse3_scaled_diff_ranges(
        torch.tensor([[0.0, 0.5, 2.999, -0.75, -0.125]], dtype=torch.float32)
    )

    for invalid_value in (3.0, 3.25, -0.7501, -1.0):
        with pytest.raises(AssertionError):
            _assert_pseudo_mse3_scaled_diff_ranges(
                torch.tensor([[invalid_value]], dtype=torch.float32)
            )


def test_pseudo_mse3_decision_uses_exact_summed_diff():
    diff = torch.tensor(
        [
            [2.5, -0.625, -0.625],
            [0.0, -0.625, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, -0.625, 0.0],
        ],
        dtype=torch.float32,
    )

    assert pseudo_mse3_choose_exp2_from_diff(diff).tolist() == [False, True, False, True]
    assert pseudo_mse3_choose_exp2_from_diff(
        diff,
        tie_break="exp2",
    ).tolist() == [False, True, True, True]
    assert pseudo_mse3_choose_exp2_from_diff(
        diff,
        fixed_rounding="nearest",
    ).tolist() == [False, True, False, True]
    assert pseudo_mse3_choose_exp2_from_diff(
        diff,
        tie_break="exp2",
        fixed_rounding="nearest",
    ).tolist() == [False, True, True, True]


def test_pseudo_mse3_hw_vector_zero_bits_uses_fixed_point_decision():
    raw_chunks = make_raw_chunks(num_chunks=20, seed=42)
    _scales, scaled_chunks = scale_raw_chunks(raw_chunks)

    for bit_width in (8, 7, 6, 5, 4):
        (
            err1,
            err2,
            chunk_diff,
            choose_exp2,
            expected_error,
            q1_bits,
            q2_bits,
            err_exp1_pre_square,
            err_exp2_pre_square,
            pseudo_diff,
        ) = decision_for_bit_width(scaled_chunks, bit_width)

        normalization = float(2.0 ** (2 * (bit_width - 2)))
        normalized_diff = (
            err_exp2_pre_square.pow(2) - err_exp1_pre_square.pow(2)
        ) * normalization
        _assert_pseudo_mse3_scaled_diff_ranges(normalized_diff)
        expected_contributions = torch.floor(normalized_diff).to(torch.int32)
        q1_from_fields = pseudo_mse_decode_emb_python(
            q1_bits,
            exp_bits=1,
            mantissa_bits=bit_width - 2,
            is_signed=True,
        )
        q2_from_fields = pseudo_mse_decode_emb_python(
            q2_bits,
            exp_bits=2,
            mantissa_bits=bit_width - 3,
            is_signed=True,
        )
        torch.testing.assert_close(q1_from_fields, scaled_chunks - err_exp1_pre_square)
        torch.testing.assert_close(q2_from_fields, scaled_chunks - err_exp2_pre_square)
        assert torch.equal(pseudo_diff, expected_contributions)
        assert torch.equal(
            chunk_diff,
            expected_contributions.sum(dim=1, dtype=torch.int64),
        )
        assert torch.equal(choose_exp2, chunk_diff < 0)
        torch.testing.assert_close(expected_error, torch.where(choose_exp2, err2, err1))


def test_pseudo_mse3_generate_hw_vectors_outputs_pytorch_reference(tmp_path):
    output = tmp_path / "pseudo_mse3_hw_vectors.txt"

    write_vectors(str(output), num_chunks=2, seed=42)

    text = output.read_text()
    assert "pseudo_MSE3 PyTorch hardware test vectors" in text
    assert (
        "implementation: PyTorch reference; optional CUDA verification uses the same quantization path"
        in text
    )
    assert "decision rule: choose_exp2 if sum(err2^2 - err1^2) < 0 else choose_exp1" in text
    assert "mantissa mode: round-to-nearest" in text
    assert "format-selection candidate mode: truncate" in text
    assert "selected dynamic-quantizer output mode: round-to-nearest" in text
    assert "q_exp*_bits are truncated candidate fields used by format selection" in text
    assert "BEGIN_BIT_WIDTH 8" in text
    assert "pseudo_diff_times_2_2m" in text


def test_pseudo_mse3_zero_bits_reports_fixed_point_l2_mismatches(tmp_path):
    csv_path = tmp_path / "pseudo_mse3_l2_mismatches.csv"

    totals = compare_pseudo_mse3_with_metric(
        str(csv_path),
        compare_metric="mse",
        num_chunks=20,
        seed=42,
        max_mismatches=5,
    )

    assert totals["metric_min_mismatched_chunks"] > 0
    assert (
        totals["reported_mismatched_chunks"]
        == totals["metric_min_mismatched_chunks"]
        == totals["decision_disagreements"]
    )
    assert csv_path.exists()


def test_pseudo_mse3_verify_python_vectors_matches_reference():
    assert verify_python_vectors(num_chunks=10, seed=42, max_mismatches=5) == 0


def test_pseudo_mse3_experiment_configs_use_bits_to_take_metric_param():
    args = get_args(
        [
            "--bits-to-take",
            "20",
            "--fixed-rounding",
            "nearest",
            "--tie-break",
            "exp2",
        ]
    )
    specs = build_metric_comparison_specs(args)

    assert args.bits_to_take == 20
    assert args.fixed_rounding == "nearest"
    assert args.tie_break == "exp2"
    assert len(specs) == 15
    assert [(spec.bit_width, spec.metric_label, spec.activation_dt) for spec in specs[:3]] == [
        (8, "MSE", "dyn_a8_e1e2_mse"),
        (8, "L1", "dyn_a8_e1e2_l1"),
        (8, METRIC_NAME, "dyn_a8_e1e2_pseudo_mse3"),
    ]

    pseudo_spec = specs[2]
    runtime_cfg = build_pseudo_mse_runtime_config(args, pseudo_spec)
    input_quant_cfg = build_pseudo_mse_input_quant_cfg(args, pseudo_spec)

    assert runtime_cfg["experiment"]["type"] == "pseudo_mse3"
    assert runtime_cfg["experiment"]["metric"] == METRIC_NAME
    assert runtime_cfg["experiment"]["activation_dt"] == "dyn_a8_e1e2_pseudo_mse3"
    assert "mantissa_window_bits" not in runtime_cfg["experiment"]
    assert runtime_cfg["experiment"]["bits_to_take"] == 20
    assert runtime_cfg["experiment"]["fixed_rounding"] == "nearest"
    assert runtime_cfg["experiment"]["tie_break"] == "exp2"
    assert input_quant_cfg["metric"] == METRIC_NAME
    assert input_quant_cfg["metric_param"] == 20.0
    assert input_quant_cfg["pseudo_mse3_fixed_rounding"] == "nearest"
    assert input_quant_cfg["pseudo_mse3_tie_break"] == "exp2"
    assert input_quant_cfg["pseudo_mse2_mantissa_window_bits"] == 0

    mse_cfg = build_pseudo_mse_input_quant_cfg(args, specs[0])
    l1_cfg = build_pseudo_mse_input_quant_cfg(args, specs[1])
    assert mse_cfg["metric"] == BASELINE_METRIC_NAME
    assert l1_cfg["metric"] == L1_METRIC_NAME
    assert "metric_param" not in mse_cfg
    assert "metric_param" not in l1_cfg


def test_pseudo_mse3_bits_to_take_rejects_negative_values():
    with pytest.raises(ValueError):
        get_args(["--bits-to-take", "-1"])


def test_pseudo_mse3_summary_and_plots_include_bits_to_take(tmp_path):
    rows = [
        {
            "model": "resnet50",
            "metric": METRIC_NAME,
            "bit_width": 8,
            "activation_dt": "dyn_a8_e1e2_pseudo_mse3",
            "candidate_formats": "fp8_e1m6,fp8_e2m5",
            "bits_to_take": 20,
            "dataset_size": 100,
            "random_seed": 42,
            "limit_batches": 1,
            "status": "SUCCESS",
            "acc1": 80.25,
            "acc5": 95.0,
            "certainty": 0.5,
            "norm_mse": 0.001,
            "norm_l1": 0.01,
            "error": None,
        }
    ]
    dataset_label = "ImageNet random subset=100, seed=42; evaluated samples=100"

    csv_path, txt_path = _write_model_summary(
        str(tmp_path),
        "resnet50",
        rows,
        dataset_label,
        bits_to_take=20,
    )
    combined_path = _write_combined_summary(str(tmp_path), rows, bits_to_take=20)
    plot_paths = _plot_model_summary(
        str(tmp_path),
        "resnet50",
        rows,
        dataset_label,
        bits_to_take=20,
    )

    assert os.path.basename(csv_path) == "resnet50_bits_to_take20_summary.csv"
    assert os.path.basename(txt_path) == "resnet50_bits_to_take20_summary.txt"
    assert os.path.basename(combined_path) == "summary_bits_to_take20.csv"
    assert "bits_to_take" in open(csv_path).readline()
    assert "bits_to_take: 20" in open(txt_path).read()
    assert plot_paths
    assert all("bits_to_take20" in os.path.basename(path) for path in plot_paths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for dispatch check")
def test_pseudo_mse3_bits_to_take_uses_cuda_search_path(monkeypatch):
    from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

    quantizer = object.__new__(DynamicInputQuantizer)
    quantizer.metric = "pseudo_mse3"
    quantizer.metric_param = 20.0
    quantizer.pseudo_mse3_bits_to_take = 20
    quantizer.chunk_size = 128
    quantizer._candidate_param_cache = {}
    quantizer.collect_format_stats = False
    quantizer.running_error = 0.0
    quantizer.total_chunks = 0

    called = {}

    def fake_cuda_search(ref_chunks, cands_e, cands_m, cands_sgn, capture):
        called["used_cuda"] = True
        n_chunks = ref_chunks.shape[0]
        return (
            torch.zeros(n_chunks, dtype=torch.long, device=ref_chunks.device),
            torch.ones(n_chunks, dtype=torch.float32, device=ref_chunks.device),
            ref_chunks.reshape(-1).contiguous(),
            torch.empty(0, dtype=torch.float32, device=ref_chunks.device),
        )

    monkeypatch.setattr(quantizer, "_search_best_chunk_format_cuda", fake_cuda_search)
    tensor = torch.zeros((1, 128), dtype=torch.float32, device="cuda")

    quantizer._select_best_format(tensor, "layer", ["fp8_e1m6", "fp8_e2m5"])

    assert called["used_cuda"]
