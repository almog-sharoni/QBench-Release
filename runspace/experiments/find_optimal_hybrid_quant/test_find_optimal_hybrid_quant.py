import json
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from runspace.experiments.find_optimal_hybrid_quant.find_optimal_hybrid_quant import (
    _build_best_db_sweep_plan,
    _build_bidirectional_db_sweep_plan,
    _build_hybrid_log_config,
    _build_weight_materialization_source_config,
    _candidate_formats_by_bit_width,
    get_args,
    _hybrid_run_exists,
    _materialize_weight_buffers_from_map,
    _pending_hybrid_entries,
)
from runspace.core.runner import Runner


def test_db_sweep_default_quantizes_every_supported_op(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["find_optimal_hybrid_quant.py", "--weight_mode", "best", "--input_mode", "sweep"],
    )

    assert get_args().excluded_ops == []


def _run(
    run_id,
    *,
    model="model_a",
    experiment_type,
    weight_dt,
    activation_dt,
    acc1,
    acc5=0.0,
    certainty=0.0,
    status="SUCCESS",
    transport="encoded",
    run_identity="identity-a",
    config_json=None,
    quant_map_json=None,
    cli_command="baseline.py",
):
    return {
        "id": run_id,
        "model_name": model,
        "experiment_type": experiment_type,
        "weight_dt": weight_dt,
        "activation_dt": activation_dt,
        "acc1": acc1,
        "acc5": acc5,
        "certainty": certainty,
        "status": status,
        "config_json": config_json or (
            '{"evaluation":{"input_quant":{"transport":"'
            + transport
            + '"}}}'
        ),
        "quant_map_json": quant_map_json,
        "cli_command": cli_command,
        "run_identity": run_identity,
    }


def _source_config(
    *,
    kind,
    fmt,
    chunk_size=128,
    limit_batches=-1,
    excluded_ops=None,
):
    config = {
        "model": {"name": "model_a", "weights": "DEFAULT"},
        "adapter": {
            "fold_input_norm": True,
            "excluded_ops": list(
                ["LayerNorm"] if excluded_ops is None else excluded_ops
            ),
            "input_size": 224,
        },
        "dataset": {
            "name": "imagenet",
            "path": "/data/imagenet/val",
            "limit_batches": limit_batches,
        },
        "quantization": {},
        "evaluation": {},
    }
    if kind == "weight":
        config["quantization"] = {
            "weight_format": fmt,
            "weight_mode": "chunk",
            "weight_chunk_size": chunk_size,
            "weight_source": "prequantized_state_dict",
        }
    else:
        config["evaluation"]["input_quant"] = {
            "enabled": True,
            "mode": "uniform",
            "transport": "encoded",
            "format": fmt,
            "chunk_size": chunk_size,
            "unsigned_input_sources": [
                "relu",
                "softmax",
                "quantrelu",
                "quantsoftmax",
            ],
            "uniform_unsigned_input_candidates": True,
        }
    return json.dumps(config)


SOURCE_REQUIREMENTS = {
    "model_weights": "DEFAULT",
    "dataset_name": "imagenet",
    "dataset_path": "/data/imagenet/val",
    "weight_chunk_size": 128,
    "input_chunk_size": 128,
    "fold_input_norm": True,
    "input_size": 224,
    "excluded_ops": ["LayerNorm"],
    "unsigned_input_sources": [
        "relu",
        "softmax",
        "quantrelu",
        "quantsoftmax",
    ],
    "uniform_unsigned_input_candidates": True,
    "require_full_evaluation": True,
}


def test_best_db_sweep_selects_one_weight_and_best_input_per_width():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp32",
            acc1=99.0,
        ),
        _run(
            2,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e1m6",
            activation_dt="fp32",
            acc1=80.0,
            acc5=90.0,
        ),
        _run(
            3,
            experiment_type="weight_quant_baseline",
            weight_dt="fp7_e2m4",
            activation_dt="fp32",
            acc1=80.0,
            acc5=91.0,
        ),
        _run(
            4,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e2m5",
            activation_dt="fp32",
            acc1=100.0,
            status="ERROR",
        ),
        _run(
            10,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e1m6",
            acc1=78.0,
        ),
        _run(
            11,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e2m5",
            acc1=79.0,
        ),
        _run(
            12,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp7_e1m5",
            acc1=77.0,
        ),
        _run(
            13,
            model="other_model",
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp7_e2m4",
            acc1=100.0,
        ),
    ])

    plan = _build_best_db_sweep_plan(
        runs,
        "model_a",
        ["fp8_e1m6", "fp8_e2m5", "fp7_e1m5", "fp7_e2m4"],
    )

    assert plan["weight_format"] == "fp7_e2m4"
    assert plan["weight_source_run_id"] == 3
    assert plan["entries"] == [
        {
            "mode": "fixed",
            "bit_width": 8,
            "format": "fp8_e2m5",
            "source_acc1": 79.0,
            "source_run_id": 11,
        },
        {
            "mode": "dynamic",
            "bit_width": 8,
            "candidate_formats": ["fp8_e1m6", "fp8_e2m5"],
        },
        {
            "mode": "fixed",
            "bit_width": 7,
            "format": "fp7_e1m5",
            "source_acc1": 77.0,
            "source_run_id": 12,
        },
        {
            "mode": "dynamic",
            "bit_width": 7,
            "candidate_formats": ["fp7_e1m5", "fp7_e2m4"],
        },
    ]


def test_bidirectional_sweep_includes_optimal_dynamic_and_deduplicates_best_pair():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e1m6",
            activation_dt="fp32",
            acc1=79.0,
        ),
        _run(
            2,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e2m5",
            activation_dt="fp32",
            acc1=80.0,
        ),
        _run(
            3,
            experiment_type="weight_quant_optimized_8",
            weight_dt="opt_layer_mse",
            activation_dt="fp32",
            acc1=82.0,
        ),
        _run(
            10,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e1m6",
            acc1=78.0,
        ),
        _run(
            11,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e2m5",
            acc1=80.0,
        ),
        _run(
            12,
            experiment_type="input_quant_dynamic_8",
            weight_dt="fp32",
            activation_dt="dyn_input_mse",
            acc1=81.0,
        ),
    ])

    plan = _build_bidirectional_db_sweep_plan(
        runs,
        "model_a",
        ["fp8_e1m6", "fp8_e2m5"],
    )

    width = plan["widths"][0]
    assert width["best_weight"]["source_run_id"] == 3
    assert width["best_weight"]["mode"] == "optimized"
    assert width["best_input"]["source_run_id"] == 12
    assert width["best_input"]["mode"] == "dynamic"
    assert len(width["input_options"]) == 3
    assert len(width["weight_options"]) == 3
    assert len(width["entries"]) == 5
    assert {
        (entry["weight"]["source_run_id"], entry["input"]["source_run_id"])
        for entry in width["entries"]
    } == {
        (3, 10),
        (3, 11),
        (3, 12),
        (1, 12),
        (2, 12),
    }


def test_bidirectional_sweep_accepts_legacy_unsuffixed_optimized_weight_run():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e2m5",
            activation_dt="fp32",
            acc1=80.0,
        ),
        _run(
            2,
            experiment_type="weight_quant_optimized",
            weight_dt="opt_chunk_mse",
            activation_dt="fp32",
            acc1=82.0,
            quant_map_json=json.dumps({
                "layer": {"format": ["fp8_e2m5", "fp8_e3m4"]},
            }),
        ),
        _run(
            3,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e2m5",
            acc1=79.0,
        ),
        _run(
            4,
            experiment_type="input_quant_dynamic_8",
            weight_dt="fp32",
            activation_dt="dyn_input_mse",
            acc1=81.0,
        ),
    ])

    plan = _build_bidirectional_db_sweep_plan(
        runs,
        "model_a",
        ["fp8_e2m5", "fp8_e3m4"],
    )

    assert plan["widths"][0]["best_weight"]["source_run_id"] == 2
    assert plan["widths"][0]["best_weight"]["mode"] == "optimized"


def test_bidirectional_sweep_uses_baselines_when_optimized_weight_is_missing():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp7_e2m4",
            activation_dt="fp32",
            acc1=80.0,
        ),
        _run(
            2,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp7_e2m4",
            acc1=79.0,
        ),
        _run(
            3,
            experiment_type="input_quant_dynamic_7",
            weight_dt="fp32",
            activation_dt="dyn_input_mse",
            acc1=81.0,
        ),
    ])

    plan = _build_bidirectional_db_sweep_plan(
        runs,
        "model_a",
        ["fp7_e2m4"],
    )

    width = plan["widths"][0]
    assert width["best_weight"]["source_run_id"] == 1
    assert width["best_weight"]["mode"] == "baseline"
    assert len(width["weight_options"]) == 1


def test_weight_materialization_source_does_not_treat_optimized_label_as_format():
    weight_config = {
        "adapter": {
            "quantized_ops": ["all"],
            "weight_quantization": True,
        },
        "quantization": {
            "format": "opt_chunk_mse",
            "weight_mode": "chunk",
            "weight_source": "prequantized_state_dict",
        },
    }

    source_config = _build_weight_materialization_source_config(weight_config)

    assert source_config["adapter"]["quantized_ops"] == ["all"]
    assert source_config["adapter"]["weight_quantization"] is False
    assert "format" not in source_config["quantization"]
    assert source_config["quantization"]["weight_source"] == "fp32"
    assert weight_config["quantization"]["format"] == "opt_chunk_mse"
    assert weight_config["adapter"]["weight_quantization"] is True


def test_best_db_sweep_requires_a_baseline_for_every_requested_width():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e2m5",
            activation_dt="fp32",
            acc1=80.0,
        ),
        _run(
            2,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e2m5",
            acc1=79.0,
        ),
    ])

    with pytest.raises(ValueError, match="Missing successful input baselines.*7"):
        _build_best_db_sweep_plan(
            runs,
            "model_a",
            ["fp8_e2m5", "fp7_e2m4"],
            requested_bit_widths=[8, 7],
        )


def test_best_db_sweep_requires_input_rows_without_pandas_keyerror():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e2m5",
            activation_dt="fp32",
            acc1=80.0,
        ),
    ])

    with pytest.raises(ValueError, match="Missing successful input baselines"):
        _build_best_db_sweep_plan(
            runs,
            "model_a",
            ["fp8_e2m5"],
        )


def test_best_db_sweep_filters_limited_and_incompatible_source_rows():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e1m6",
            activation_dt="fp32",
            acc1=99.0,
            config_json=_source_config(
                kind="weight",
                fmt="fp8_e1m6",
                limit_batches=1,
            ),
            cli_command="baseline.py --limit_batches 1",
        ),
        _run(
            2,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e2m5",
            activation_dt="fp32",
            acc1=80.0,
            config_json=_source_config(kind="weight", fmt="fp8_e2m5"),
        ),
        _run(
            3,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e1m6",
            acc1=99.0,
            config_json=_source_config(
                kind="input",
                fmt="fp8_e1m6",
                excluded_ops=[],
            ),
        ),
        _run(
            4,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e2m5",
            acc1=79.0,
            config_json=_source_config(kind="input", fmt="fp8_e2m5"),
        ),
    ])

    plan = _build_best_db_sweep_plan(
        runs,
        "model_a",
        ["fp8_e1m6", "fp8_e2m5"],
        source_requirements=SOURCE_REQUIREMENTS,
    )

    assert plan["weight_format"] == "fp8_e2m5"
    assert plan["weight_source_run_id"] == 2
    assert plan["entries"][0]["format"] == "fp8_e2m5"
    assert plan["entries"][0]["source_run_id"] == 4


def test_best_db_sweep_ignores_reference_transport_input_rows():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="weight_quant_baseline",
            weight_dt="fp8_e2m5",
            activation_dt="fp32",
            acc1=80.0,
        ),
        _run(
            2,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e1m6",
            acc1=99.0,
            transport="reference",
        ),
        _run(
            3,
            experiment_type="input_quant_baseline",
            weight_dt="fp32",
            activation_dt="fp8_e2m5",
            acc1=79.0,
            transport="encoded",
        ),
    ])

    plan = _build_best_db_sweep_plan(
        runs,
        "model_a",
        ["fp8_e1m6", "fp8_e2m5"],
    )

    assert plan["entries"][0]["format"] == "fp8_e2m5"
    assert plan["entries"][0]["source_run_id"] == 3


def test_candidate_group_validation_and_hybrid_deduplication():
    assert _candidate_formats_by_bit_width(
        ["fp4_e1m2", "ufp4_e2m1", "fp3_e1m1"]
    ) == {
        4: ["fp4_e1m2", "ufp4_e2m1"],
        3: ["fp3_e1m1"],
    }
    with pytest.raises(ValueError, match="must include a bit width"):
        _candidate_formats_by_bit_width(["not_a_format"])
    with pytest.raises(ValueError, match="FP32 cannot be a dynamic input candidate"):
        _candidate_formats_by_bit_width(["fp32"])

    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="hybrid_quant_optimal",
            weight_dt="fp8_e2m5",
            activation_dt="dyn_input_mse_4bit",
            acc1=70.0,
        ),
    ])
    assert _hybrid_run_exists(
        runs,
        "model_a",
        "hybrid_quant_optimal",
        "fp8_e2m5",
        "dyn_input_mse_4bit",
    )
    assert _hybrid_run_exists(
        runs,
        "model_a",
        "hybrid_quant_optimal",
        "fp8_e2m5",
        "dyn_input_mse_4bit",
        run_identity="identity-a",
    )
    assert not _hybrid_run_exists(
        runs,
        "model_a",
        "hybrid_quant_optimal",
        "fp8_e2m5",
        "dyn_input_mse_4bit",
        run_identity="identity-b",
    )
    assert not _hybrid_run_exists(
        runs,
        "model_a",
        "hybrid_quant_optimal",
        "fp8_e2m5",
        "dyn_input_mse_5bit",
    )


def test_pending_hybrid_entries_skip_successful_prior_combinations():
    runs = pd.DataFrame([
        _run(
            1,
            experiment_type="hybrid_quant_optimal",
            weight_dt="fp8_e2m5",
            activation_dt="fp8_e1m6",
            acc1=70.0,
        ),
    ])
    entries = [
        {"weight_dt": "fp8_e2m5", "activation_dt": "fp8_e1m6"},
        {"weight_dt": "fp8_e2m5", "activation_dt": "dyn_input_mse_8bit"},
    ]

    assert _pending_hybrid_entries(
        entries,
        runs,
        "model_a",
        "hybrid_quant_optimal",
    ) == [entries[1]]
    assert _pending_hybrid_entries(
        entries,
        runs,
        "model_a",
        "hybrid_quant_optimal",
        force_rerun=True,
    ) == entries

    identity_entries = [
        {
            "weight_dt": "fp8_e2m5",
            "activation_dt": "fp8_e1m6",
            "run_identity": "identity-a",
        },
        {
            "weight_dt": "fp8_e2m5",
            "activation_dt": "fp8_e1m6",
            "run_identity": "identity-b",
        },
    ]
    assert _pending_hybrid_entries(
        identity_entries,
        runs,
        "model_a",
        "hybrid_quant_optimal",
    ) == [identity_entries[1]]


def test_materialized_checkpoint_keeps_quantized_weight_and_fresh_buffers():
    class FakeQuantLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.register_buffer("weight_fp8", torch.tensor([-1.0]))
            self.register_buffer("weight_scale", torch.tensor([-1.0]))
            self.register_buffer("weight_scale_packed", torch.tensor([-1.0]))

        def calibrate_weights(self):
            # Make it observable whether calibration saw original or prequantized
            # weights; the production method performs the real format quantization.
            self.weight_fp8 = self.weight.detach() * 2
            self.weight_scale = self.weight.detach() * 3
            self.weight_scale_packed = self.weight.detach() * 4

    model = torch.nn.Module()
    model.layer = FakeQuantLayer()
    q_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    q_state["layer.weight"] = torch.tensor([0.75])
    q_state["layer.weight_fp8"] = torch.tensor([99.0])
    q_state["layer.weight_scale"] = torch.tensor([99.0])
    q_state["layer.weight_scale_packed"] = torch.tensor([99.0])

    materialized = _materialize_weight_buffers_from_map(
        model,
        q_state,
        {"layer": "fp7_e2m4"},
        SimpleNamespace(weight_chunk_size=128),
    )

    assert materialized["layer.weight"].item() == pytest.approx(0.75)
    assert materialized["layer.weight_fp8"].item() == pytest.approx(2.0)
    assert materialized["layer.weight_scale"].item() == pytest.approx(3.0)
    assert materialized["layer.weight_scale_packed"].item() == pytest.approx(4.0)
    assert model.layer.weight_mode == "chunk"
    assert model.layer.chunk_formats == ["fp7_e2m4"]


def test_hybrid_identity_tracks_runtime_semantics_not_source_row_provenance():
    base = {
        "model": {"name": "model_a", "weights": "DEFAULT"},
        "adapter": {"type": "generic"},
        "quantization": {
            "format": "fp8_e2m5",
            "weight_mode": "chunk",
            "weight_chunk_size": 128,
        },
        "dataset": {"name": "imagenet", "batch_size": 32},
        "evaluation": {"max_batches": -1},
    }

    def identity(candidates, source_run_id):
        cfg = _build_hybrid_log_config(
            base,
            experiment_name="hybrid_quant_fp8/fp8_dynamic",
            experiment_type="hybrid_quant_optimal",
            weight_dt="fp8_e2m5",
            activation_dt="dyn_input_mse_8bit",
            input_quant_cfg={
                "enabled": True,
                "mode": "dynamic",
                "transport": "encoded",
                "metric": "mse",
                "chunk_size": 128,
                "candidate_formats": candidates,
                "use_cache_sim_db": False,
            },
            selection_metadata={"weight": {"source_run_id": source_run_id}},
        )
        return Runner._run_identity(cfg)

    assert identity(["fp8_e1m6", "fp8_e2m5"], 10) == identity(
        ["fp8_e1m6", "fp8_e2m5"],
        11,
    )
    assert identity(["fp8_e1m6", "fp8_e2m5"], 10) != identity(
        ["fp8_e1m6"],
        10,
    )
