import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd


class _CacheDataStub:
    def __call__(self, *args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func


class _StreamlitStub:
    cache_data = _CacheDataStub()


HELPERS = runpy.run_path(
    Path(__file__).with_name("data_helpers.py"),
    init_globals={
        "st": _StreamlitStub(),
        "pd": pd,
        "np": np,
        "json": json,
        "os": os,
        "RunDatabase": object,
        "DB_PATH": "runs.db",
        "FM_DB_PATH": "fm_runs.db",
    },
)
build_hybrid_accuracy_plot_df = HELPERS["build_hybrid_accuracy_plot_df"]
build_hybrid_directional_plot_df = HELPERS["build_hybrid_directional_plot_df"]
hybrid_best_dynamic_run_ids = HELPERS["hybrid_best_dynamic_run_ids"]
filter_common_hybrid_width_rows = HELPERS["filter_common_hybrid_width_rows"]
build_baseline_accuracy_plot_df = HELPERS["build_baseline_accuracy_plot_df"]


def test_baseline_accuracy_plot_df_groups_formats_by_exponent_family():
    rows = pd.DataFrame([
        {
            "id": 1,
            "model_name": "resnet50",
            "experiment_type": "input_quant_baseline",
            "weight_dt": "fp32",
            "activation_dt": "fp8_e1m6",
            "acc1": 80.0,
            "status": "SUCCESS",
            "run_date": "2026-07-13 10:00:01",
        },
        {
            "id": 2,
            "model_name": "resnet50",
            "experiment_type": "input_quant_baseline",
            "weight_dt": "fp32",
            "activation_dt": "fp7_e1m5",
            "acc1": 79.0,
            "status": "SUCCESS",
            "run_date": "2026-07-13 10:00:02",
        },
        {
            "id": 3,
            "model_name": "resnet50",
            "experiment_type": "input_quant_baseline",
            "weight_dt": "fp32",
            "activation_dt": "fp7_e2m4",
            "acc1": 78.0,
            "status": "SUCCESS",
            "run_date": "2026-07-13 10:00:03",
        },
    ])

    plot_df = build_baseline_accuracy_plot_df(rows)

    assert plot_df.set_index("Run ID")["Exponent Family"].to_dict() == {
        1: "e1",
        2: "e1",
        3: "e2",
    }


def test_baseline_accuracy_plot_df_reads_per_width_weight_experiment_type():
    plot_df = build_baseline_accuracy_plot_df(pd.DataFrame([{
        "id": 10,
        "model_name": "resnet50",
        "experiment_type": "weight_quant_optimized_7",
        "weight_dt": "opt_layer_mse",
        "activation_dt": "fp32",
        "acc1": 79.5,
        "status": "SUCCESS",
        "run_date": "2026-07-14 10:00:00",
    }]))

    assert plot_df.iloc[0]["Bits"] == 7
    assert str(plot_df.iloc[0]["Series"]) == "Weight opt"


def test_baseline_accuracy_plot_df_separates_dynamic_mse_and_l1_series():
    rows = pd.DataFrame([
        {
            "id": 20,
            "model_name": "resnet50",
            "experiment_type": "input_quant_dynamic_8",
            "weight_dt": "fp32",
            "activation_dt": "dyn_input_mse",
            "acc1": 79.0,
            "status": "SUCCESS",
            "run_date": "2026-07-14 10:00:00",
        },
        {
            "id": 21,
            "model_name": "resnet50",
            "experiment_type": "input_quant_dynamic_8",
            "weight_dt": "fp32",
            "activation_dt": "dyn_input_l1",
            "acc1": 78.5,
            "status": "SUCCESS",
            "run_date": "2026-07-14 10:00:01",
        },
    ])

    plot_df = build_baseline_accuracy_plot_df(rows).set_index("Run ID")

    assert str(plot_df.loc[20, "Series"]) == "Input dynamic MSE"
    assert str(plot_df.loc[21, "Series"]) == "Input dynamic L1"
    assert plot_df.loc[20, "Bits"] == 8
    assert plot_df.loc[21, "Bits"] == 8


def _hybrid_config(
    mode,
    *,
    bit_width=None,
    candidates=None,
    max_batches=-1,
    include_selection=True,
):
    config = {
        "experiment": {"name": f"hybrid_quant_fp8/{mode}"},
        "evaluation": {
            "input_quant": {
                "mode": "dynamic" if mode == "dynamic" else "uniform",
                "candidate_formats": list(candidates or []),
            },
        },
    }
    if max_batches is not None:
        config["evaluation"]["max_batches"] = max_batches
    if include_selection:
        config["meta"] = {
            "selection": {
                "weight": {"mode": "best_baseline"},
                "input": {
                    "mode": "dynamic" if mode == "dynamic" else "best_baseline",
                    "bit_width": bit_width,
                    "candidate_formats": list(candidates or []),
                },
            },
        }
    return json.dumps(config)


def _run(
    run_id,
    activation_dt,
    acc1,
    config_json,
    *,
    experiment_type="hybrid_quant_optimal",
    status="SUCCESS",
):
    return {
        "id": run_id,
        "model_name": "resnet50",
        "experiment_type": experiment_type,
        "weight_dt": "fp8_e2m5",
        "activation_dt": activation_dt,
        "acc1": acc1,
        "status": status,
        "run_date": f"2026-07-13 10:00:{run_id:02d}",
        "config_json": config_json,
        "input_map_json": None,
        "mse": 1e-4,
        "certainty": 0.4,
    }


def test_hybrid_accuracy_plot_df_resolves_sweep_and_legacy_rows():
    rows = [
        _run(
            1,
            "fp7_e1m5",
            80.1,
            _hybrid_config("fixed", bit_width=7),
        ),
        _run(
            2,
            "dyn_input_mse_7bit",
            80.0,
            _hybrid_config(
                "dynamic",
                bit_width=7,
                candidates=["fp7_e1m5", "fp7_e2m4"],
            ),
        ),
        _run(
            3,
            "dyn_input_mse",
            79.0,
            _hybrid_config(
                "dynamic",
                candidates=["fp6_e1m4", "fp6_e2m3"],
                include_selection=False,
            ),
        ),
        _run(
            4,
            "dyn_input_mse_5bit",
            78.0,
            _hybrid_config("dynamic", bit_width=5, max_batches=1),
        ),
        _run(5, "fp32", 81.0, _hybrid_config("fixed", bit_width=None)),
        _run(
            6,
            "fp8_e1m6",
            82.0,
            json.dumps({"experiment": {"name": "find_optimal_input_quant"}}),
            experiment_type="input_quant_baseline",
        ),
        _run(
            7,
            "fp8_e1m6",
            82.0,
            _hybrid_config("fixed", bit_width=8),
            status="ERROR",
        ),
        _run(
            8,
            "dyn_input_mse_4bit",
            float("nan"),
            "not-json",
        ),
    ]

    plot_df = build_hybrid_accuracy_plot_df(pd.DataFrame(rows))

    assert plot_df["Run ID"].tolist() == [4, 3, 1, 2]
    assert plot_df.set_index("Run ID")["Bits"].to_dict() == {
        1: 7,
        2: 7,
        3: 6,
        4: 5,
    }
    assert str(plot_df.set_index("Run ID").loc[1, "Series"]) == "Hybrid fixed"
    assert str(plot_df.set_index("Run ID").loc[2, "Series"]) == "Hybrid dynamic"
    assert bool(plot_df.set_index("Run ID").loc[1, "Sweep Entry"])
    assert not bool(plot_df.set_index("Run ID").loc[3, "Sweep Entry"])
    assert bool(plot_df.set_index("Run ID").loc[3, "Full Evaluation"])
    assert not bool(plot_df.set_index("Run ID").loc[4, "Full Evaluation"])
    assert plot_df.set_index("Run ID").loc[4, "Evaluation Scope"] == "1 batch"


def test_hybrid_accuracy_plot_df_handles_unknown_scope_zero_and_nullable_text():
    rows = [
        _run(
            20,
            "fp4_e1m2",
            70.0,
            _hybrid_config("fixed", bit_width=4, max_batches=None),
        ),
        _run(
            21,
            "fp4_e2m1",
            71.0,
            _hybrid_config("fixed", bit_width=4, max_batches=0),
        ),
        _run(
            22,
            pd.NA,
            72.0,
            _hybrid_config("dynamic", bit_width=3),
            experiment_type=pd.NA,
        ),
    ]

    plot_df = build_hybrid_accuracy_plot_df(pd.DataFrame(rows)).set_index("Run ID")

    assert not bool(plot_df.loc[20, "Full Evaluation"])
    assert not bool(plot_df.loc[20, "Evaluation Scope Known"])
    assert plot_df.loc[20, "Evaluation Scope"] == "Unknown"
    assert bool(plot_df.loc[21, "Full Evaluation"])
    assert plot_df.loc[21, "Evaluation Scope"] == "Full dataset"
    assert str(plot_df.loc[22, "Series"]) == "Hybrid dynamic"
    assert plot_df.loc[22, "Bits"] == 3
    assert plot_df.loc[20, "Setup Label"] != plot_df.loc[21, "Setup Label"]


def test_filter_common_hybrid_width_rows_removes_mixed_width_points():
    plot_df = pd.DataFrame([
        {"Run ID": 1, "Weight Bits": 4, "Input Bits": 4},
        {"Run ID": 2, "Weight Bits": 8, "Input Bits": 4},
        {"Run ID": 3, "Weight Bits": None, "Input Bits": 4},
    ])

    filtered = filter_common_hybrid_width_rows(plot_df)

    assert filtered["Run ID"].tolist() == [1]


def test_hybrid_accuracy_plot_df_recognizes_bidirectional_sweep_entries():
    config = json.loads(_hybrid_config("fixed", bit_width=8))
    config["meta"]["selection"] = {
        "direction": "input_fixed",
        "bit_width": 8,
        "weight": {"mode": "optimized"},
        "input": {"mode": "baseline", "bit_width": 8},
    }
    plot_df = build_hybrid_accuracy_plot_df(pd.DataFrame([
        _run(30, "fp8_e2m5", 80.0, json.dumps(config))
    ]))

    assert bool(plot_df.iloc[0]["Sweep Entry"])


def test_hybrid_directional_plot_restores_shared_best_best_point():
    def bidirectional_run(run_id, direction, weight_dt, activation_dt, acc1):
        config = json.loads(_hybrid_config("fixed", bit_width=8))
        config["meta"]["selection"] = {
            "direction": direction,
            "bit_width": 8,
            "weight": {"mode": "baseline", "format": weight_dt},
            "input": {
                "mode": "baseline",
                "format": activation_dt,
                "bit_width": 8,
            },
        }
        row = _run(run_id, activation_dt, acc1, json.dumps(config))
        row["weight_dt"] = weight_dt
        return row

    plot_df = build_hybrid_accuracy_plot_df(pd.DataFrame([
        bidirectional_run(40, "weight_fixed", "fp8_e2m5", "fp8_e1m6", 81.0),
        bidirectional_run(41, "weight_fixed", "fp8_e2m5", "fp8_e2m5", 82.0),
        bidirectional_run(42, "input_fixed", "fp8_e1m6", "fp8_e1m6", 80.0),
        _run(
            43,
            "dyn_input_mse_8bit",
            80.5,
            _hybrid_config(
                "dynamic",
                bit_width=8,
                candidates=["fp8_e1m6", "fp8_e2m5"],
            ),
        ),
    ]))

    directional_df = build_hybrid_directional_plot_df(plot_df)
    weight_sweep = directional_df[
        directional_df["Sweep Direction"].eq("weight_fixed")
    ]
    input_sweep = directional_df[
        directional_df["Sweep Direction"].eq("input_fixed")
    ]

    assert set(weight_sweep["Run ID"]) == {40, 41, 43}
    assert set(input_sweep["Run ID"]) == {40, 42}
    assert set(weight_sweep.loc[weight_sweep["Winner"], "Run ID"]) == {41}
    assert set(input_sweep.loc[input_sweep["Winner"], "Run ID"]) == {40}
    assert set(weight_sweep["Candidate Label"]) == {
        "fp8_e1m6", "fp8_e2m5", "dyn_input_mse_8bit"
    }
    assert set(input_sweep["Candidate Label"]) == {"fp8_e1m6", "fp8_e2m5"}


def test_hybrid_best_dynamic_run_ids_selects_best_at_each_width():
    plot_df = pd.DataFrame([
        {
            "Run ID": 1, "Model": "resnet50", "Bits": 7,
            "Series": "Hybrid dynamic", "Accuracy (%)": 79.5,
        },
        {
            "Run ID": 2, "Model": "resnet50", "Bits": 7,
            "Series": "Hybrid dynamic", "Accuracy (%)": 79.6,
        },
        {
            "Run ID": 3, "Model": "resnet50", "Bits": 8,
            "Series": "Hybrid dynamic", "Accuracy (%)": 80.1,
        },
        {
            "Run ID": 4, "Model": "resnet50", "Bits": 8,
            "Series": "Hybrid fixed", "Accuracy (%)": 81.0,
        },
    ])

    assert hybrid_best_dynamic_run_ids(plot_df) == {2, 3}
