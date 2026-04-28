import os
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner


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
    }


if __name__ == "__main__":
    test_runner_logs_fp32_weight_dt_when_weight_quantization_disabled()
    test_materialized_cache_detects_weight_quant_buffers()
    test_runner_synthesizes_uniform_input_quant_config()
