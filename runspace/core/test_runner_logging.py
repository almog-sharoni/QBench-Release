import os
import sys
import tempfile
import json

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner
from src.ops.quant_dropout import QuantDropout
from src.ops.quant_matmul import QuantMatMul
from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer


def test_runner_logs_fp32_weight_dt_when_weight_quantization_disabled():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        runner = Runner(device="cpu")
        config = {
            "model": {"name": "resnet18"},
            "adapter": {
                "type": "generic",
                "weight_quantization": False,
                "input_quantization": True,
            },
            "quantization": {
                "format": "fp8_e4m3",
                "input_format": "fp8_e4m3",
            },
            "experiment": {
                "type": "runner_eval",
                "resolve_ref_from_db": False,
            },
        }
        result = {
            "model_name": "resnet18",
            "quant_format": "fp8_e4m3",
            "acc1": 81.25,
            "acc5": 95.0,
            "status": "SUCCESS",
            "weight_quant_map": {"conv1": {"format": "fp8_e4m3"}},
        }

        payload = runner.log_experiment_result(config, result, db_path=db_path)

        assert payload["weight_dt"] == "fp32"
        assert payload["activation_dt"] == "fp8_e4m3"
        assert payload["quant_map_json"] is None
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_materialized_cache_detects_weight_quant_buffers():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        weight_path = f.name

    try:
        torch.save(
            {
                "conv.weight": torch.ones(1),
                "conv.weight_fp8": torch.ones(1),
                "conv.weight_scale": torch.ones(1),
            },
            weight_path,
        )

        assert Runner._materialized_file_has_weight_quant_buffers(weight_path)
    finally:
        if os.path.exists(weight_path):
            os.remove(weight_path)


def test_runner_synthesizes_uniform_input_quant_config():
    config = {
        "adapter": {"input_quantization": True},
        "quantization": {"format": "fp8_e4m3", "input_format": "fp8_e4m3", "chunk_size": 128},
    }

    input_quant_cfg = Runner._implicit_uniform_input_quant_cfg(config)

    assert input_quant_cfg == {
        "enabled": True,
        "mode": "uniform",
        "format": "fp8_e4m3",
        "chunk_size": 128,
        "quant_mode": "chunk",
    }


def test_runner_logs_dynamic_input_map_from_processed_layer_stats():
    class FakeQuantizer:
        def __init__(self):
            self.model = nn.Sequential(nn.Conv2d(3, 4, 1))

        def get_final_stats(self):
            return {
                "norm_l1": 0.25,
                "norm_mse": 0.125,
                "total_l1": 10.0,
                "total_mse": 5.0,
                "layer_stats": {
                    "0": {
                        "format_counts": {"fp8_e4m3": 3, "fp4_e1m2": 1},
                        "total_chunks": 4,
                    }
                },
            }

    stats = Runner._collect_layer_input_quant_stats(
        FakeQuantizer(),
        {"enabled": True, "mode": "dynamic", "metric": "mse", "chunk_size": 128},
    )
    input_map_json = Runner._build_input_map_json(stats["layer_stats"])
    input_map = json.loads(input_map_json)

    assert stats["layer_stats"]["0"]["type"] == "Conv2d"
    assert input_map["0"]["format_counts"] == {"fp8_e4m3": 3, "fp4_e1m2": 1}
    assert input_map["0"]["total_chunks"] == 4
    assert input_map["0"]["dominant_format"] == "fp8_e4m3"


def test_dynamic_input_quantizer_uses_unsigned_candidates_after_source_dropout():
    model = nn.Sequential(
        nn.ReLU(),
        QuantDropout(p=0.0),
        nn.Linear(4, 4),
    )
    model[1].input_q_type = "ufp4_e1m3"
    quantizer = DynamicInputQuantizer(
        model,
        candidate_formats=["fp4_e1m2", "fp4_e2m1"],
        unsigned_input_sources=["relu"],
    )

    assert "2" in quantizer.post_unsigned_layers
    assert "1" not in quantizer.post_unsigned_layers
    assert "1" in quantizer.unsigned_passthrough_layers
    assert quantizer._candidates_for_layer("1", model[1]) == ["ufp4_e1m3", "ufp4_e2m2"]
    assert quantizer._candidates_for_layer("2", model[2]) == ["ufp4_e1m3", "ufp4_e2m2"]

    disabled_quantizer = DynamicInputQuantizer(
        model,
        candidate_formats=["fp4_e1m2", "fp4_e2m1"],
        unsigned_input_sources=["relu"],
        use_unsigned_input_candidates=False,
    )
    assert "2" not in disabled_quantizer.post_unsigned_layers
    assert disabled_quantizer._candidates_for_layer("2", model[2]) == ["fp4_e1m2", "fp4_e2m1"]


def test_dynamic_input_quantizer_targets_actual_consumer_after_softmax_dropout():
    class AttentionLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=-1)
            self.dropout_layer = QuantDropout(p=0.0)
            self.out_proj = nn.Linear(4, 4)
            self.quantmatmul = QuantMatMul()

        def forward(self, scores, values):
            attn = self.softmax(scores)
            attn = self.dropout_layer(attn)
            out = self.quantmatmul(attn, values)
            return self.out_proj(out)

    model = AttentionLike()
    quantizer = DynamicInputQuantizer(
        model,
        candidate_formats=["fp4_e1m2", "fp4_e2m1"],
        unsigned_input_sources=["softmax"],
    )

    assert "quantmatmul" in quantizer.post_unsigned_layers
    assert "dropout_layer" not in quantizer.post_unsigned_layers
    assert "dropout_layer" in quantizer.unsigned_passthrough_layers
    assert "out_proj" not in quantizer.post_unsigned_layers
    assert quantizer._candidates_for_layer("dropout_layer", model.dropout_layer) == ["ufp4_e1m3", "ufp4_e2m2"]
    assert quantizer._candidates_for_layer("quantmatmul", model.quantmatmul) == ["ufp4_e1m3", "ufp4_e2m2"]
    assert quantizer._candidates_for_layer("out_proj", model.out_proj) == ["fp4_e1m2", "fp4_e2m1"]


if __name__ == "__main__":
    test_runner_logs_fp32_weight_dt_when_weight_quantization_disabled()
    test_materialized_cache_detects_weight_quant_buffers()
    test_runner_synthesizes_uniform_input_quant_config()
    test_runner_logs_dynamic_input_map_from_processed_layer_stats()
    test_dynamic_input_quantizer_uses_unsigned_candidates_after_source_dropout()
    test_dynamic_input_quantizer_targets_actual_consumer_after_softmax_dropout()
