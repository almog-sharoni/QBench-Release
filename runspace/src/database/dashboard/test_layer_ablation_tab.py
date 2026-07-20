import json
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


HELPERS = runpy.run_path(Path(__file__).with_name("layer_ablation_tab.py"))
build_layer_ablation_plot_df = HELPERS["build_layer_ablation_plot_df"]
best_per_strategy = HELPERS["_layer_ablation_best_per_strategy"]


class _CacheDataStub:
    def __call__(self, *args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func


class _StreamlitStub:
    cache_data = _CacheDataStub()


DATA_HELPERS = runpy.run_path(
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
build_baseline_accuracy_plot_df = DATA_HELPERS[
    "build_baseline_accuracy_plot_df"
]


def _ablation_config(
    layer_type,
    bits,
    *,
    target_layer_count=3,
    nested=False,
    max_batches=None,
    weights=None,
    dataset=None,
):
    config = {
        "quantization": {
            "target_layer_type": layer_type,
            "target_layer_count": target_layer_count,
            "non_target_weight_format": "fp32",
            "optimization_bit_width": bits,
        }
    }
    if max_batches is not None:
        config["evaluation"] = {"max_batches": max_batches}
    if weights is not None:
        config["model"] = {"weights": weights}
    if dataset is not None:
        config["dataset"] = {"name": dataset}
    if nested:
        config = {
            "experiment": {
                "ablation_policy": "target_only",
                "config_json": json.dumps(config),
            }
        }
    return json.dumps(config)


def _run(
    run_id,
    *,
    experiment_type,
    weight_dt,
    acc1,
    config_json=None,
    model_name="mobilevit_s",
    status="SUCCESS",
    ref_acc1=None,
    mse=None,
    l1=None,
):
    return {
        "id": run_id,
        "model_name": model_name,
        "experiment_type": experiment_type,
        "weight_dt": weight_dt,
        "activation_dt": "fp32",
        "acc1": acc1,
        "ref_acc1": ref_acc1,
        "status": status,
        "mse": mse,
        "l1": l1,
        "run_date": f"2026-07-20 10:00:{run_id:02d}",
        "config_json": config_json,
    }


def test_build_layer_ablation_plot_df_extracts_all_sweep_strategies():
    rows = pd.DataFrame(
        [
            _run(
                1,
                experiment_type="fp32_ref",
                weight_dt="fp32",
                acc1=80.0,
            ),
            _run(
                2,
                experiment_type="weight_quant_ablation_conv2d_4",
                weight_dt="fp4_e1m2",
                acc1=76.0,
                config_json=_ablation_config("Conv2d", 4, target_layer_count=35),
                mse=1e-3,
            ),
            _run(
                3,
                experiment_type="weight_quant_ablation_conv2d_4",
                weight_dt="fp4_e2m1",
                acc1=77.0,
                config_json=_ablation_config("Conv2d", 4, target_layer_count=35),
                mse=8e-4,
            ),
            _run(
                4,
                experiment_type="weight_quant_optimized_ablation_conv2d_4",
                weight_dt="opt_layer_mse",
                acc1=78.5,
                config_json=_ablation_config("Conv2d", 4, target_layer_count=35),
                mse=5e-4,
            ),
            _run(
                5,
                experiment_type="weight_quant_optimized_ablation_conv2d_4",
                weight_dt="opt_chunk_l1",
                acc1=79.0,
                config_json=_ablation_config("Conv2d", 4, target_layer_count=35),
                l1=2e-3,
            ),
            _run(
                6,
                experiment_type="weight_quant_optimized_4",
                weight_dt="opt_chunk_mse",
                acc1=79.5,
                config_json=json.dumps(
                    {"quantization": {"optimization_bit_width": 4}}
                ),
            ),
        ]
    )

    plot_df = build_layer_ablation_plot_df(rows).set_index("Run ID")

    assert plot_df.index.tolist() == [2, 3, 4, 5]
    assert plot_df["Layer Type"].unique().tolist() == ["Conv2d"]
    assert plot_df["Bits"].unique().tolist() == [4]
    assert plot_df["Strategy"].astype(str).to_dict() == {
        2: "Uniform",
        3: "Uniform",
        4: "Layer optimal",
        5: "Chunk optimal",
    }
    assert plot_df.loc[4, "Optimization Metric"] == "MSE"
    assert plot_df.loc[5, "Optimization Metric"] == "L1"
    assert plot_df.loc[5, "Reference Accuracy (%)"] == 80.0
    assert plot_df.loc[5, "Accuracy Drop (pp)"] == 1.0
    assert plot_df.loc[5, "Target Layer Count"] == 35


def test_build_layer_ablation_plot_df_falls_back_to_experiment_type():
    rows = pd.DataFrame(
        [
            _run(
                10,
                experiment_type="fp32_ref",
                weight_dt="fp32",
                acc1=75.0,
                model_name="vit_b_16",
            ),
            _run(
                11,
                experiment_type=(
                    "weight_quant_optimized_ablation_weighted_matmul_7"
                ),
                weight_dt="opt_chunk_mse",
                acc1=72.5,
                model_name="vit_b_16",
                config_json="not-json",
            ),
        ]
    )

    plot_df = build_layer_ablation_plot_df(rows)

    assert plot_df["Run ID"].tolist() == [11]
    assert plot_df.iloc[0]["Layer Type"] == "weighted_matmul"
    assert plot_df.iloc[0]["Bits"] == 7
    assert str(plot_df.iloc[0]["Strategy"]) == "Chunk optimal"
    assert plot_df.iloc[0]["Reference Accuracy (%)"] == 75.0


def test_build_layer_ablation_plot_df_accepts_nested_target_only_metadata():
    renamed_run = _run(
        20,
        experiment_type="manually_renamed_experiment",
        weight_dt="fp6_e2m3",
        acc1=71.0,
        ref_acc1=74.0,
        config_json=_ablation_config(
            "Linear", 6, target_layer_count=73, nested=True
        ),
    )

    plot_df = build_layer_ablation_plot_df(pd.DataFrame([renamed_run]))

    assert plot_df["Run ID"].tolist() == [20]
    assert plot_df.iloc[0]["Layer Type"] == "Linear"
    assert plot_df.iloc[0]["Bits"] == 6
    assert plot_df.iloc[0]["Accuracy Drop (pp)"] == 3.0


def test_build_layer_ablation_plot_df_ignores_failed_and_malformed_rows():
    rows = pd.DataFrame(
        [
            _run(
                30,
                experiment_type="weight_quant_ablation_linear_5",
                weight_dt="fp5_e2m2",
                acc1=70.0,
                status="ERROR",
                config_json=_ablation_config("Linear", 5),
            ),
            _run(
                31,
                experiment_type=pd.NA,
                weight_dt="fp5_e2m2",
                acc1=70.0,
                config_json=pd.NA,
            ),
            _run(
                32,
                experiment_type="weight_quant_ablation_linear_missing_width",
                weight_dt="fp5_e2m2",
                acc1=70.0,
                config_json="{}",
            ),
        ]
    )

    assert build_layer_ablation_plot_df(rows).empty


def test_build_layer_ablation_plot_df_rejects_operand_and_activation_ablations():
    rows = pd.DataFrame(
        [
            _run(
                33,
                experiment_type="activation_ablation_matmul_4",
                weight_dt="fp4_e2m1",
                acc1=70.0,
                config_json="{}",
            ),
            {
                **_run(
                    34,
                    experiment_type="weight_quant_ablation_linear_4",
                    weight_dt="fp4_e2m1",
                    acc1=70.0,
                    config_json=_ablation_config("Linear", 4),
                ),
                "activation_dt": "fp4_e2m1",
            },
        ]
    )

    assert build_layer_ablation_plot_df(rows).empty


def test_layer_ablation_reference_prefers_explicit_fp32_reference():
    rows = pd.DataFrame(
        [
            {
                **_run(
                    40,
                    experiment_type="fp32_ref",
                    weight_dt="fp32",
                    acc1=81.0,
                ),
                "run_date": "2026-07-20 09:00:00",
            },
            _run(
                41,
                experiment_type="unrelated_fp32",
                weight_dt="fp32",
                acc1=99.0,
            ),
            _run(
                42,
                experiment_type="weight_quant_ablation_linear_8",
                weight_dt="fp8_e3m4",
                acc1=80.0,
                config_json=_ablation_config("Linear", 8),
            ),
        ]
    )

    plot_df = build_layer_ablation_plot_df(rows)

    assert plot_df.iloc[0]["Reference Accuracy (%)"] == 81.0
    assert plot_df.iloc[0]["Accuracy Drop (pp)"] == 1.0


def test_layer_ablation_reference_rejects_failed_explicit_reference():
    rows = pd.DataFrame(
        [
            _run(
                43,
                experiment_type="fp32_ref",
                weight_dt="fp32",
                acc1=95.0,
                status="ERROR",
            ),
            _run(
                44,
                experiment_type="legacy_fp32",
                weight_dt="fp32",
                acc1=81.0,
            ),
            _run(
                45,
                experiment_type="weight_quant_ablation_linear_8",
                weight_dt="fp8_e3m4",
                acc1=80.0,
                config_json=_ablation_config("Linear", 8),
            ),
        ]
    )

    plot_df = build_layer_ablation_plot_df(rows)

    assert plot_df.iloc[0]["Reference Accuracy (%)"] == 81.0


def test_best_per_strategy_keeps_best_uniform_format_at_each_width():
    rows = pd.DataFrame(
        [
            _run(
                50,
                experiment_type="weight_quant_ablation_linear_4",
                weight_dt="fp4_e1m2",
                acc1=60.0,
                ref_acc1=75.0,
                config_json=_ablation_config("Linear", 4),
            ),
            _run(
                51,
                experiment_type="weight_quant_ablation_linear_4",
                weight_dt="fp4_e2m1",
                acc1=65.0,
                ref_acc1=75.0,
                config_json=_ablation_config("Linear", 4),
            ),
            _run(
                52,
                experiment_type="weight_quant_optimized_ablation_linear_4",
                weight_dt="opt_layer_mse",
                acc1=67.0,
                ref_acc1=75.0,
                config_json=_ablation_config("Linear", 4),
            ),
        ]
    )

    best_df = best_per_strategy(build_layer_ablation_plot_df(rows))

    assert best_df["Run ID"].tolist() == [51, 52]


def test_best_per_strategy_does_not_mix_evaluation_setups():
    rows = pd.DataFrame(
        [
            _run(
                53,
                experiment_type="weight_quant_ablation_linear_4",
                weight_dt="fp4_e1m2",
                acc1=65.0,
                ref_acc1=75.0,
                config_json=_ablation_config(
                    "Linear",
                    4,
                    max_batches=1,
                    weights="DEFAULT",
                    dataset="imagenet",
                ),
            ),
            _run(
                54,
                experiment_type="weight_quant_ablation_linear_4",
                weight_dt="fp4_e2m1",
                acc1=64.0,
                ref_acc1=75.0,
                config_json=_ablation_config(
                    "Linear",
                    4,
                    max_batches=-1,
                    weights="DEFAULT",
                    dataset="imagenet",
                ),
            ),
        ]
    )

    plot_df = build_layer_ablation_plot_df(rows)
    best_df = best_per_strategy(plot_df)

    assert set(plot_df["Evaluation Scope"]) == {"1 batch", "Full dataset"}
    assert set(best_df["Run ID"]) == {53, 54}


def test_generic_baseline_plot_excludes_layer_ablation_runs():
    ablation_run = _run(
        60,
        experiment_type="weight_quant_optimized_ablation_linear_6",
        weight_dt="opt_chunk_mse",
        acc1=73.0,
        ref_acc1=75.0,
        config_json=_ablation_config("Linear", 6),
    )

    assert build_baseline_accuracy_plot_df(
        pd.DataFrame([ablation_run])
    ).empty


def test_layer_ablation_dashboard_renders_populated_results():
    tab_path = str(Path(__file__).with_name("layer_ablation_tab.py"))
    app_source = f"""
import json
import runpy

import pandas as pd

tab = runpy.run_path({tab_path!r})
config = json.dumps({{
    "quantization": {{
        "target_layer_type": "Conv2d",
        "target_layer_count": 20,
        "non_target_weight_format": "fp32",
        "optimization_bit_width": 4,
    }}
}})
rows = pd.DataFrame([
    {{
        "id": 1,
        "model_name": "resnet18",
        "experiment_type": "fp32_ref",
        "weight_dt": "fp32",
        "activation_dt": "fp32",
        "acc1": 70.0,
        "ref_acc1": 70.0,
        "status": "SUCCESS",
        "run_date": "2026-07-20 10:00:00",
    }},
    {{
        "id": 2,
        "model_name": "resnet18",
        "experiment_type": "weight_quant_ablation_conv2d_4",
        "weight_dt": "fp4_e2m1",
        "activation_dt": "fp32",
        "acc1": 68.0,
        "ref_acc1": 70.0,
        "status": "SUCCESS",
        "mse": 0.001,
        "run_date": "2026-07-20 10:00:01",
        "config_json": config,
    }},
    {{
        "id": 3,
        "model_name": "resnet18",
        "experiment_type": "weight_quant_optimized_ablation_conv2d_4",
        "weight_dt": "opt_layer_mse",
        "activation_dt": "fp32",
        "acc1": 69.0,
        "ref_acc1": 70.0,
        "status": "SUCCESS",
        "mse": 0.0005,
        "run_date": "2026-07-20 10:00:02",
        "config_json": config,
    }},
])
tab["_render_layer_ablation_dashboard"](rows)
"""

    app = AppTest.from_string(app_source).run(timeout=15)

    assert not list(app.exception)
    assert len(app.get("vega_lite_chart")) == 1
    assert len(app.dataframe) == 1
    assert len(app.get("download_button")) == 1
