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
    build_metric_comparison_specs,
    build_pseudo_mse_input_quant_cfg,
    build_pseudo_mse_runtime_config,
    get_args,
)
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    _assert_pseudo_mse3_scaled_diff_ranges,
    dynamic_input_metric_code,
    get_dynamic_input_metric_spec,
    pseudo_mse3_choose_exp2_from_diff,
    pseudo_mse3_err2_minus_err1_from_scaled,
)


def test_pseudo_mse3_metric_is_python_pairwise_metric():
    spec = get_dynamic_input_metric_spec("pseudo_MSE3")

    assert spec.name == "pseudo_mse3"
    assert spec.display_name == METRIC_NAME
    assert spec.implemented
    assert spec.cuda_code is None
    with pytest.raises(NotImplementedError, match="CUDA search code"):
        dynamic_input_metric_code("pseudo_mse3")


def test_pseudo_mse3_exact_diff_matches_expected_scaled_ranges():
    m1 = 6
    values = torch.tensor(
        [
            1.0 + 2.0 ** -m1 + 2.0 ** -(m1 + 1) + 2.0 ** -(m1 + 2),
            1.5 * 2.0 ** -1,
            (1.0 + 2.0 ** -5 + 2.0 ** -6 + 2.0 ** -7) * 2.0 ** -2,
            (1.0 + 2.0 ** -1 + 2.0 ** -2) * 2.0 ** -(m1 + 1),
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

    scaled_diff = diff * float(2.0 ** (2 * m1))
    expected = torch.tensor([[2.5, 0.0, -0.625, -0.625, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(scaled_diff, expected)


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


def test_pseudo_mse3_scaled_diff_assertion_rejects_invalid_values():
    _assert_pseudo_mse3_scaled_diff_ranges(
        torch.tensor([[0.0, 1.0, 2.999, -0.25, -0.749]], dtype=torch.float32)
    )

    for invalid_value in (0.5, 3.0, -0.125, -0.75):
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
        ],
        dtype=torch.float32,
    )

    assert pseudo_mse3_choose_exp2_from_diff(diff).tolist() == [False, True, False]


def test_pseudo_mse3_experiment_configs_use_metric_without_window_bits():
    args = get_args([])
    specs = build_metric_comparison_specs(args)

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
    assert input_quant_cfg["metric"] == METRIC_NAME
    assert input_quant_cfg["pseudo_mse2_mantissa_window_bits"] == 0

    mse_cfg = build_pseudo_mse_input_quant_cfg(args, specs[0])
    l1_cfg = build_pseudo_mse_input_quant_cfg(args, specs[1])
    assert mse_cfg["metric"] == BASELINE_METRIC_NAME
    assert l1_cfg["metric"] == L1_METRIC_NAME
