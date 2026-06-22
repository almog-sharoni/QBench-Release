import os
import sys
from types import SimpleNamespace


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.activation_candidate_sweep.activation_candidate_sweep import (
    DEFAULT_EXP_CAPS,
    _build_sweep_input_quant_cfg,
    _build_w32_dynamic_runtime_config,
    _config_json_for_run,
    _format_choice_counts_by_exp_cap,
    _format_counts_from_input_map,
    _run_config_matches,
    build_sweep_specs,
    candidate_formats_for_bit_width,
)
from runspace.experiments.activation_candidate_sweep.plot_format_choices import (
    available_exp_caps_by_category,
    format_choice_counts_by_exp_cap as plot_format_choice_counts_by_exp_cap,
)


def _args(limit_batches=1, chunk_size=64):
    return SimpleNamespace(
        model_name="resnet18",
        weights="DEFAULT",
        model_source="auto",
        dataset_name="imagenet",
        dataset_path="/tmp/imagenet",
        batch_size=1,
        num_workers=0,
        limit_batches=limit_batches,
        chunk_size=chunk_size,
        fold_input_norm=True,
        force_rerun=False,
        unsigned_input_sources=["relu", "relu6", "softmax"],
    )


def test_candidate_formats_for_bit_width_drop_zero_mantissa_cases():
    assert candidate_formats_for_bit_width(8) == [
        "fp8_e1m6",
        "fp8_e2m5",
        "fp8_e3m4",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e6m1",
    ]
    assert candidate_formats_for_bit_width(8, 4) == [
        "fp8_e1m6",
        "fp8_e2m5",
        "fp8_e3m4",
        "fp8_e4m3",
    ]
    assert candidate_formats_for_bit_width(8, 3) == [
        "fp8_e1m6",
        "fp8_e2m5",
        "fp8_e3m4",
    ]
    assert candidate_formats_for_bit_width(8, 2) == [
        "fp8_e1m6",
        "fp8_e2m5",
    ]
    assert candidate_formats_for_bit_width(8, 1) == [
        "fp8_e1m6",
    ]
    assert candidate_formats_for_bit_width(7) == [
        "fp7_e1m5",
        "fp7_e2m4",
        "fp7_e3m3",
        "fp7_e4m2",
        "fp7_e5m1",
    ]
    assert candidate_formats_for_bit_width(6, 4) == [
        "fp6_e1m4",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp6_e4m1",
    ]
    assert candidate_formats_for_bit_width(5, 5) == [
        "fp5_e1m3",
        "fp5_e2m2",
        "fp5_e3m1",
    ]
    assert candidate_formats_for_bit_width(4, 3) == [
        "fp4_e1m2",
        "fp4_e2m1",
    ]

    for bit_width in range(4, 9):
        for exp_cap in (None, 3, 4, 5):
            assert all(
                not fmt.endswith("m0")
                for fmt in candidate_formats_for_bit_width(bit_width, exp_cap)
            )


def test_build_sweep_specs_skips_duplicate_pools_per_bit_width():
    specs = build_sweep_specs([8, 7, 6, 5, 4], DEFAULT_EXP_CAPS)

    assert [spec.activation_dt for spec in specs] == [
        "dyn_a8_all_l2",
        "dyn_a8_exp4_l2",
        "dyn_a8_exp3_l2",
        "dyn_a8_exp2_l2",
        "dyn_a8_exp1_l2",
        "dyn_a7_all_l2",
        "dyn_a7_exp4_l2",
        "dyn_a7_exp3_l2",
        "dyn_a7_exp2_l2",
        "dyn_a7_exp1_l2",
        "dyn_a6_all_l2",
        "dyn_a6_exp3_l2",
        "dyn_a6_exp2_l2",
        "dyn_a6_exp1_l2",
        "dyn_a5_all_l2",
        "dyn_a5_exp2_l2",
        "dyn_a5_exp1_l2",
        "dyn_a4_all_l2",
        "dyn_a4_exp1_l2",
    ]

    assert specs[10].candidate_formats == [
        "fp6_e1m4",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp6_e4m1",
    ]
    assert specs[11].candidate_formats == [
        "fp6_e1m4",
        "fp6_e2m3",
        "fp6_e3m2",
    ]
    assert specs[12].candidate_formats == [
        "fp6_e1m4",
        "fp6_e2m3",
    ]
    assert specs[13].candidate_formats == [
        "fp6_e1m4",
    ]


def test_format_counts_from_input_map_aggregates_chunk_counts():
    input_map = {
        "conv1": {
            "format_counts": {
                "fp8_e1m6": 3,
                "fp8_e2m5": "4",
            }
        },
        "relu": {
            "format": ["ufp8_e1m7", "ufp8_e1m7"],
        },
        "fc": {
            "format": "fp8_e1m6",
        },
    }

    assert _format_counts_from_input_map(input_map) == {
        "fp8_e1m6": 4,
        "fp8_e2m5": 4,
        "ufp8_e1m7": 2,
    }


def test_format_choice_counts_by_exp_cap_groups_summary_rows():
    rows = [
        {
            "exp_cap": "all",
            "format_counts_json": '{"fp8_e1m6": 3, "fp8_e2m5": 2}',
        },
        {
            "exp_cap": "all",
            "format_counts_json": '{"fp8_e1m6": 5}',
        },
        {
            "exp_cap": "exp1",
            "format_counts_json": '{"fp8_e1m6": 7}',
        },
    ]

    assert _format_choice_counts_by_exp_cap(rows) == {
        "all": {"fp8_e1m6": 8, "fp8_e2m5": 2},
        "exp1": {"fp8_e1m6": 7},
    }


def test_plot_counts_merge_fp_and_ufp_by_exponent_and_available_caps():
    rows = [
        {
            "bit_width": "4",
            "exp_cap": "all",
            "candidate_formats": "fp4_e1m2,fp4_e2m1",
            "format_counts_json": '{"fp4_e1m2": 10, "ufp4_e1m3": 5, "fp4_e2m1": 3}',
        },
        {
            "bit_width": "4",
            "exp_cap": "exp1",
            "candidate_formats": "fp4_e1m2",
            "format_counts_json": '{"ufp4_e1m3": 7}',
        },
    ]

    assert plot_format_choice_counts_by_exp_cap(rows) == {
        "all": {"b4_e1": 15, "b4_e2": 3},
        "exp1": {"b4_e1": 7},
    }
    assert available_exp_caps_by_category(rows) == {
        "b4_e1": ["all", "exp1"],
        "b4_e2": ["all"],
    }


def test_run_config_matching_requires_limit_chunk_and_candidates():
    args = _args(limit_batches=2, chunk_size=32)
    spec = build_sweep_specs([8], [None])[0]
    config = _build_w32_dynamic_runtime_config(
        args,
        model_name=args.model_name,
        weights=args.weights,
        candidate_formats=spec.candidate_formats,
    )
    input_quant_cfg = _build_sweep_input_quant_cfg(args, spec, args.model_name)
    config_json = _config_json_for_run(config, input_quant_cfg, args, spec)

    assert _run_config_matches(
        config_json,
        spec=spec,
        limit_batches=2,
        chunk_size=32,
    )
    assert not _run_config_matches(
        config_json,
        spec=spec,
        limit_batches=1,
        chunk_size=32,
    )
    assert not _run_config_matches(
        config_json,
        spec=spec,
        limit_batches=2,
        chunk_size=64,
    )

    different_candidates = build_sweep_specs([8], [3])[0]
    assert not _run_config_matches(
        config_json,
        spec=different_candidates,
        limit_batches=2,
        chunk_size=32,
    )
