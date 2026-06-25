import json
import os
import sys
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.activation_candidate_sweep.activation_candidate_sweep import (
    DEFAULT_EXP_CAPS,
    DEFAULT_SINGLETON_EXP_BITS,
    DEFAULT_SINGLETON_IMPORT_EXPERIMENT_TYPES,
    _candidate_pool_label,
    _build_sweep_input_quant_cfg,
    _build_w32_dynamic_runtime_config,
    _config_json_for_run,
    _format_choice_counts_by_exp_cap,
    _format_counts_from_input_map,
    _run_config_matches,
    _single_format_source_config_matches,
    _verify_singleton_import_source,
    CandidateSweepSpec,
    build_sweep_specs,
    candidate_formats_for_bit_width,
    candidate_formats_for_single_exp_bits,
)
from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (
    _build_input_quant_config,
)
from runspace.experiments.activation_candidate_sweep.plot_format_choices import (
    _candidate_pool_label as plot_candidate_pool_label,
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
        import_singleton_baselines=True,
        singleton_import_experiment_types=DEFAULT_SINGLETON_IMPORT_EXPERIMENT_TYPES,
        singleton_import_verify_acc1_tolerance=0.05,
    )


class _FakeDb:
    def __init__(self, rows):
        self._rows = pd.DataFrame(rows)

    def get_runs(self):
        return self._rows.copy()


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


def test_candidate_formats_for_single_exp_bits():
    assert candidate_formats_for_single_exp_bits(8, 1) == ["fp8_e1m6"]
    assert candidate_formats_for_single_exp_bits(8, 2) == ["fp8_e2m5"]
    assert candidate_formats_for_single_exp_bits(4, 2) == ["fp4_e2m1"]
    assert candidate_formats_for_single_exp_bits(3, 2) == []


def test_build_sweep_specs_skips_duplicate_pools_per_bit_width():
    specs = build_sweep_specs([8, 7, 6, 5, 4], DEFAULT_EXP_CAPS)

    assert [spec.activation_dt for spec in specs] == [
        "dyn_a8_all_l2",
        "dyn_a8_exp4_l2",
        "dyn_a8_exp3_l2",
        "dyn_a8_exp2_l2",
        "dyn_a8_exp1_l2",
        "dyn_a8_only_e2_l2",
        "dyn_a7_all_l2",
        "dyn_a7_exp4_l2",
        "dyn_a7_exp3_l2",
        "dyn_a7_exp2_l2",
        "dyn_a7_exp1_l2",
        "dyn_a7_only_e2_l2",
        "dyn_a6_all_l2",
        "dyn_a6_exp3_l2",
        "dyn_a6_exp2_l2",
        "dyn_a6_exp1_l2",
        "dyn_a6_only_e2_l2",
        "dyn_a5_all_l2",
        "dyn_a5_exp2_l2",
        "dyn_a5_exp1_l2",
        "dyn_a5_only_e2_l2",
        "dyn_a4_all_l2",
        "dyn_a4_exp1_l2",
        "dyn_a4_only_e2_l2",
    ]

    assert DEFAULT_SINGLETON_EXP_BITS == [2]
    by_activation_dt = {spec.activation_dt: spec for spec in specs}
    assert by_activation_dt["dyn_a8_only_e2_l2"].candidate_formats == ["fp8_e2m5"]
    assert by_activation_dt["dyn_a4_only_e2_l2"].candidate_formats == ["fp4_e2m1"]

    assert by_activation_dt["dyn_a6_all_l2"].candidate_formats == [
        "fp6_e1m4",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp6_e4m1",
    ]
    assert by_activation_dt["dyn_a6_exp3_l2"].candidate_formats == [
        "fp6_e1m4",
        "fp6_e2m3",
        "fp6_e3m2",
    ]
    assert by_activation_dt["dyn_a6_exp2_l2"].candidate_formats == [
        "fp6_e1m4",
        "fp6_e2m3",
    ]
    assert by_activation_dt["dyn_a6_exp1_l2"].candidate_formats == [
        "fp6_e1m4",
    ]


def test_candidate_pool_labels_are_readable():
    assert _candidate_pool_label("all") == "full pool"
    assert _candidate_pool_label("exp4") == "e <= 4"
    assert _candidate_pool_label("exp1") == "e1 only"
    assert _candidate_pool_label("only_e2") == "e2 only"
    assert plot_candidate_pool_label("only_e2") == "e2 only"


def test_sweep_runtime_config_matches_input_only_baseline_graph_surface():
    args = _args(limit_batches=2, chunk_size=32)
    args.excluded_ops = []
    args.input_size = 224

    sweep_config = _build_w32_dynamic_runtime_config(
        args,
        model_name=args.model_name,
        weights=args.weights,
        candidate_formats=["fp8_e1m6"],
    )
    baseline_config = _build_input_quant_config(
        args,
        args.model_name,
        args.weights,
        default_format="fp8_e1m6",
    )

    assert sweep_config["adapter"]["quantized_ops"] == baseline_config["adapter"]["quantized_ops"]
    assert sweep_config["adapter"]["input_quantization"] is True
    assert baseline_config["adapter"]["input_quantization"] is True
    assert sweep_config["adapter"]["weight_quantization"] is False
    assert baseline_config["adapter"]["weight_quantization"] is False
    assert sweep_config["adapter"]["build_quantized"] is True
    assert sweep_config["quantization"]["mode"] == baseline_config["quantization"]["mode"] == "chunk"
    assert sweep_config["quantization"]["weight_source"] == baseline_config["quantization"]["weight_source"] == "fp32"


def test_single_format_source_config_matching_checks_limit_chunk_and_format():
    config_json = json.dumps(
        {
            "dataset": {"limit_batches": 2},
            "quantization": {"chunk_size": 32, "input_format": "fp8_e2m5"},
        }
    )

    assert _single_format_source_config_matches(
        config_json,
        activation_format="fp8_e2m5",
        limit_batches=2,
        chunk_size=32,
    )
    assert not _single_format_source_config_matches(
        config_json,
        activation_format="fp8_e2m5",
        limit_batches=1,
        chunk_size=32,
    )
    assert not _single_format_source_config_matches(
        config_json,
        activation_format="fp8_e2m5",
        limit_batches=2,
        chunk_size=64,
    )
    assert not _single_format_source_config_matches(
        config_json,
        activation_format="fp8_e1m6",
        limit_batches=2,
        chunk_size=32,
    )


def test_singleton_import_verification_uses_matching_e1_rows():
    args = _args(limit_batches=2, chunk_size=32)
    e1_spec = build_sweep_specs([8], [1], singleton_exp_bits=[])[0]
    e2_spec = CandidateSweepSpec(
        bit_width=8,
        exp_cap=2,
        exp_cap_label="only_e2",
        candidate_formats=["fp8_e2m5"],
        activation_dt="dyn_a8_only_e2_l2",
    )
    config = _build_w32_dynamic_runtime_config(
        args,
        model_name=args.model_name,
        weights=args.weights,
        candidate_formats=e1_spec.candidate_formats,
    )
    input_quant_cfg = _build_sweep_input_quant_cfg(args, e1_spec, args.model_name)
    e1_sweep_config_json = _config_json_for_run(config, input_quant_cfg, args, e1_spec)
    e1_source_config_json = json.dumps(
        {
            "dataset": {"limit_batches": 2},
            "quantization": {"chunk_size": 32, "input_format": "fp8_e1m6"},
        }
    )
    rows = [
        {
            "id": 2,
            "model_name": args.model_name,
            "experiment_type": "activation_candidate_sweep",
            "weight_dt": "fp32",
            "activation_dt": "dyn_a8_exp1_l2",
            "status": "SUCCESS",
            "acc1": 80.02,
            "config_json": e1_sweep_config_json,
        },
        {
            "id": 1,
            "model_name": args.model_name,
            "experiment_type": "input_quant_baseline_4_8",
            "weight_dt": "fp32",
            "activation_dt": "fp8_e1m6",
            "status": "SUCCESS",
            "acc1": 80.0,
            "config_json": e1_source_config_json,
        },
    ]

    ok, message = _verify_singleton_import_source(
        _FakeDb(rows),
        args.model_name,
        e2_spec,
        ["input_quant_baseline_4_8"],
        limit_batches=2,
        chunk_size=32,
        acc1_tolerance=0.05,
    )
    assert ok
    assert "delta=0.020" in message

    rows[1]["acc1"] = 79.0
    ok, message = _verify_singleton_import_source(
        _FakeDb(rows),
        args.model_name,
        e2_spec,
        ["input_quant_baseline_4_8"],
        limit_batches=2,
        chunk_size=32,
        acc1_tolerance=0.05,
    )
    assert not ok
    assert "delta=1.020" in message


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

    stale_config = json.loads(config_json)
    stale_config["adapter"]["input_quantization"] = False
    assert not _run_config_matches(
        json.dumps(stale_config),
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
