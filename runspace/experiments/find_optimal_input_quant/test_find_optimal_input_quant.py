from copy import deepcopy

from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (
    _build_dynamic_input_quant_cfg,
    _dynamic_activation_dt,
    _filter_candidate_formats_by_activation_exponents,
    _prepare_dynamic_candidate_groups,
    candidate_formats,
    get_args,
)


def test_default_quantizes_every_supported_op():
    assert get_args([]).excluded_ops == []


def test_pseudo_mse3_cli_forces_e1e2_without_reusing_mse_identity():
    args = get_args(["--metric", "pseudo_mse3", "--only_dynamic"])

    assert args.metrics == ["pseudo_mse3"]
    assert args.activation_exponents == "e1e2"
    assert _dynamic_activation_dt("mse", args) == "dyn_input_mse_e1e2"
    assert _dynamic_activation_dt("pseudo_mse3", args) == (
        "dyn_input_pseudo_mse3_e1e2_btt0_floor_exp1_c128"
    )


def test_l1_cli_uses_distinct_identity_and_runtime_metric():
    args = get_args(["--metric", "l1", "--only_dynamic"])

    assert args.metrics == ["l1"]
    assert _dynamic_activation_dt("mse", args) == "dyn_input_mse"
    assert _dynamic_activation_dt("l1", args) == "dyn_input_l1"

    cfg = _build_dynamic_input_quant_cfg(
        args,
        "l1",
        ["fp8_e3m4", "fp8_e4m3"],
        "resnet50",
    )
    assert cfg["metric"] == "l1"
    assert cfg["candidate_formats"] == ["fp8_e3m4", "fp8_e4m3"]


def test_l1_aliases_are_normalized_without_duplicates():
    args = get_args(["--metric", "mae,l1,sad"])

    assert args.metrics == ["l1"]


def test_e1e2_policy_builds_one_pseudo_mse3_pair_per_supported_width():
    filtered = _filter_candidate_formats_by_activation_exponents(
        candidate_formats,
        "e1e2",
    )
    groups, skipped_widths = _prepare_dynamic_candidate_groups(
        filtered,
        "e1e2",
        ["pseudo_mse3"],
    )

    assert list(groups) == [8, 7, 6, 5, 4, 3]
    assert groups[8] == ["fp8_e1m6", "fp8_e2m5"]
    assert groups[4] == ["fp4_e1m2", "fp4_e2m1"]
    assert groups[3] == ["fp3_e1m1", "fp3_e2m0"]
    assert skipped_widths == [2]


def test_pseudo_mse3_runtime_config_carries_hardware_controls():
    args = get_args(
        [
            "--metric",
            "pseudo_mse3",
            "--bits-to-take",
            "7",
            "--fixed-rounding",
            "nearest",
            "--tie-break",
            "exp2",
        ]
    )
    cfg = _build_dynamic_input_quant_cfg(
        args,
        "pseudo_mse3",
        ["fp5_e1m3", "fp5_e2m2"],
        "resnet50",
    )

    assert cfg["metric"] == "pseudo_mse3"
    assert cfg["metric_param"] == 7
    assert cfg["candidate_formats"] == ["fp5_e1m3", "fp5_e2m2"]
    assert cfg["activation_exponents"] == "e1e2"
    assert cfg["pseudo_mse3_fixed_rounding"] == "nearest"
    assert cfg["pseudo_mse3_tie_break"] == "exp2"

    changed = deepcopy(args)
    changed.bits_to_take = 8
    assert _dynamic_activation_dt("pseudo_mse3", changed) != _dynamic_activation_dt(
        "pseudo_mse3",
        args,
    )
