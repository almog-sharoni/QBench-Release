"""Compatibility matrix for activation transport across configured image models.

The default tests are deliberately offline and lightweight. Set
``QBENCH_RUN_MODEL_LOADING_TESTS=1`` to build every architecture without
pretrained weights, ``QBENCH_RUN_ACTIVATION_TRANSPORT_CUDA_TESTS=1`` for CUDA
packet parity, or ``QBENCH_RUN_ACTIVATION_TRANSPORT_IMAGENET_TESTS=1`` to run
the full CUDA/ImageNet transport smoke matrix.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml


RUNSPACE_ROOT = Path(__file__).resolve().parents[2]
MODELS_PATH = RUNSPACE_ROOT / "inputs" / "models.yaml"
IMAGENET_PATH = Path("/data/imagenet/val")

MODEL_LOADING_ENV = "QBENCH_RUN_MODEL_LOADING_TESTS"
MODEL_RUNTIME_ENV = "QBENCH_RUN_MODEL_RUNTIME_TESTS"
CUDA_TRANSPORT_ENV = "QBENCH_RUN_ACTIVATION_TRANSPORT_CUDA_TESTS"
IMAGENET_SMOKE_ENV = "QBENCH_RUN_ACTIVATION_TRANSPORT_IMAGENET_TESTS"


@dataclass(frozen=True)
class ArchitectureExpectation:
    source: str
    required_module_groups: tuple[tuple[str, ...], ...]
    probe_producer: str
    probe_is_unsigned: bool = False
    requires_attention: bool = False
    requires_post_softmax_boundary: bool = False


@dataclass(frozen=True)
class ModelCase:
    name: str
    weights: Any
    source: str
    expectation: ArchitectureExpectation | None


# A tuple within required_module_groups lists acceptable implementation names.
# This keeps the live build probe stable across torchvision/timm releases while
# still checking the architecture features relevant to boundary planning.
ARCHITECTURE_EXPECTATIONS = {
    "resnet50": ArchitectureExpectation(
        source="torchvision",
        required_module_groups=(("conv2d",), ("relu",)),
        probe_producer="relu",
        probe_is_unsigned=True,
    ),
    "mobilenet_v3_large": ArchitectureExpectation(
        source="torchvision",
        required_module_groups=(
            ("conv2d",),
            ("hardswish",),
            ("squeezeexcitation",),
        ),
        probe_producer="hardswish",
    ),
    "efficientnet_b0": ArchitectureExpectation(
        source="torchvision",
        required_module_groups=(
            ("conv2d",),
            ("silu",),
            ("squeezeexcitation",),
        ),
        probe_producer="silu",
    ),
    "mobilevit_s": ArchitectureExpectation(
        source="timm",
        required_module_groups=(
            ("conv2d",),
            ("silu",),
            ("attention", "multiheadattention"),
        ),
        probe_producer="softmax",
        probe_is_unsigned=True,
        requires_attention=True,
        requires_post_softmax_boundary=True,
    ),
    "vit_b_16": ArchitectureExpectation(
        source="torchvision",
        required_module_groups=(
            ("conv2d",),
            ("gelu",),
            ("multiheadattention", "attention"),
        ),
        probe_producer="softmax",
        probe_is_unsigned=True,
        requires_attention=True,
        requires_post_softmax_boundary=True,
    ),
}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def load_model_cases(path: Path = MODELS_PATH) -> tuple[ModelCase, ...]:
    """Load the test matrix directly from the user-facing models file."""
    with path.open() as handle:
        raw = yaml.safe_load(handle)

    if isinstance(raw, dict):
        raw = raw.get("models")
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a model list")

    cases = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict) or not str(entry.get("name", "")).strip():
            raise ValueError(f"Invalid model entry in {path}: {entry!r}")

        name = str(entry["name"]).strip()
        expectation = ARCHITECTURE_EXPECTATIONS.get(name)
        source = str(
            entry.get("source")
            or (expectation.source if expectation is not None else "auto")
        )
        cases.append(
            ModelCase(
                name=name,
                weights=entry.get("weights", "DEFAULT"),
                source=source,
                expectation=expectation,
            )
        )
    return tuple(cases)


MODEL_CASES = load_model_cases()


def offline_adapter_config(case: ModelCase) -> dict[str, Any]:
    """Create a normal adapter config that cannot download model weights."""
    return {
        "model": {
            "name": case.name,
            "source": case.source,
            "weights": None,
        },
        "adapter": {
            "type": "generic",
            "build_quantized": False,
            "quantized_ops": [],
            "quantize_first_layer": False,
            "input_quantization": False,
            "weight_quantization": False,
            "fold_layers": False,
            "fold_input_norm": False,
        },
        "quantization": {"weight_source": "fp32"},
    }


def offline_transport_adapter_config(case: ModelCase) -> dict[str, Any]:
    """Build the exact transport-ready graph requested by Runner."""
    from runspace.core.runner import Runner

    config = offline_adapter_config(case)
    config["evaluation"] = {
        "input_quant": transport_quant_config("uniform", "reference")
    }
    config = Runner._adapter_build_config(config)
    config["adapter"].update(
        skip_calibration=True,
        enable_fx_quantization=True,
    )
    return config


def imagenet_adapter_config(case: ModelCase) -> dict[str, Any]:
    """Create the standard one-batch pretrained ImageNet smoke config."""
    config = offline_transport_adapter_config(case)
    config["model"]["weights"] = case.weights
    config["dataset"] = {
        "name": "imagenet",
        "path": str(IMAGENET_PATH),
        "batch_size": 1,
        "num_workers": 0,
    }
    config["experiment"] = {"materialize_weights": {"force_rebuild": True}}
    return config


def transport_quant_config(selection_mode: str, transport: str) -> dict[str, Any]:
    """Return an explicit transport config for Runner smoke tests."""
    common = {
        "enabled": True,
        "mode": selection_mode,
        "transport": transport,
        "chunk_size": 128,
        "unsigned_input_sources": ["relu", "relu6", "softmax"],
        "collect_error_stats": False,
    }
    if selection_mode == "dynamic":
        common.update(
            metric="mse",
            candidate_formats=["fp8_e1m6", "fp8_e2m5"],
            dynamic_unsigned_input_candidates=True,
        )
    elif selection_mode == "uniform":
        common.update(
            format="fp8_e1m6",
            quant_mode="chunk",
            uniform_unsigned_input_candidates=True,
        )
    else:
        raise ValueError(f"Unsupported selection mode: {selection_mode}")
    return common


def _probability_probe() -> torch.Tensor:
    logits = torch.linspace(-2.0, 2.0, 256, dtype=torch.float32).reshape(2, 128)
    return torch.softmax(logits, dim=-1)


def _activation_probe() -> torch.Tensor:
    return torch.linspace(-1.875, 1.875, 256, dtype=torch.float32).reshape(2, 128)


def _module_type_names(model: torch.nn.Module) -> set[str]:
    return {type(module).__name__.lower() for module in model.modules()}


def test_models_yaml_has_complete_unique_transport_matrix():
    names = [case.name for case in MODEL_CASES]

    assert len(names) == len(set(names)), "models.yaml contains duplicate model names"
    assert set(names) == set(ARCHITECTURE_EXPECTATIONS)


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.name)
def test_model_case_has_decision_complete_expectations(case: ModelCase):
    expectation = case.expectation

    assert expectation is not None, (
        f"{case.name} was added to models.yaml without an activation transport expectation"
    )
    assert case.source == expectation.source
    assert expectation.required_module_groups
    assert expectation.probe_producer
    assert expectation.requires_post_softmax_boundary == expectation.requires_attention
    if expectation.requires_post_softmax_boundary:
        assert expectation.probe_is_unsigned


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.name)
def test_offline_config_never_requests_weights(case: ModelCase):
    config = offline_adapter_config(case)

    assert config["model"]["weights"] is None
    assert config["model"]["source"] in {"torchvision", "timm"}
    assert config["adapter"]["build_quantized"] is False
    assert config["adapter"]["input_quantization"] is False

    transport_config = offline_transport_adapter_config(case)
    assert transport_config["model"]["weights"] is None
    assert transport_config["adapter"]["build_quantized"] is True
    assert transport_config["adapter"]["input_quantization"] is False
    assert transport_config["adapter"]["weight_quantization"] is False
    assert "Linear" in transport_config["adapter"]["quantized_ops"]


@pytest.mark.parametrize("selection_mode", ("uniform", "dynamic"))
@pytest.mark.parametrize("transport", ("reference", "encoded"))
def test_runner_smoke_config_covers_selection_and_transport_matrix(
    selection_mode: str,
    transport: str,
):
    config = transport_quant_config(selection_mode, transport)

    assert config["enabled"] is True
    assert config["mode"] == selection_mode
    assert config["transport"] == transport
    assert config["chunk_size"] == 128
    if selection_mode == "dynamic":
        assert config["candidate_formats"] == ["fp8_e1m6", "fp8_e2m5"]
    else:
        assert config["format"] == "fp8_e1m6"


@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.name)
def test_reference_transport_handles_expected_producer(case: ModelCase):
    from runspace.src.quantization.activation_transport import ActivationTransport

    expectation = case.expectation
    assert expectation is not None
    values = (
        _probability_probe()
        if expectation.requires_post_softmax_boundary
        else _activation_probe()
    )
    producer_id = f"{case.name}:{expectation.probe_producer}"
    q_type = "ufp8_e1m7" if expectation.probe_is_unsigned else "fp8_e1m6"

    reference = ActivationTransport(mode="reference", chunk_size=128)
    reference_value = reference.transmit_uniform(
        values,
        q_type,
        producer_id=producer_id,
    )

    assert isinstance(reference_value, torch.Tensor)
    assert reference_value.shape == values.shape
    if expectation.probe_is_unsigned:
        assert torch.all(reference_value >= 0)


@pytest.mark.skipif(
    not _env_enabled(CUDA_TRANSPORT_ENV) or not torch.cuda.is_available(),
    reason=f"Set {CUDA_TRANSPORT_ENV}=1 with CUDA to run packet parity tests",
)
@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.name)
def test_encoded_and_reference_transport_match_for_producer(case: ModelCase):
    from runspace.src.quantization.activation_transport import (
        ActivationPacket,
        ActivationTransport,
    )

    expectation = case.expectation
    assert expectation is not None
    values = (
        _probability_probe()
        if expectation.requires_post_softmax_boundary
        else _activation_probe()
    ).cuda()
    producer_id = f"{case.name}:{expectation.probe_producer}"
    q_type = "ufp8_e1m7" if expectation.probe_is_unsigned else "fp8_e1m6"

    reference = ActivationTransport(mode="reference", chunk_size=128)
    encoded = ActivationTransport(mode="encoded", chunk_size=128)
    reference_value = reference.transmit_uniform(
        values,
        q_type,
        producer_id=producer_id,
    )
    encoded_value = encoded.transmit_uniform(
        values,
        q_type,
        producer_id=producer_id,
    )

    assert isinstance(reference_value, torch.Tensor)
    assert isinstance(encoded_value, ActivationPacket)
    assert encoded_value.producer_id == producer_id
    torch.testing.assert_close(
        encoded.decode(encoded_value),
        reference.decode(reference_value),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "case",
    [case for case in MODEL_CASES if case.expectation and case.expectation.requires_attention],
    ids=lambda case: case.name,
)
def test_attention_reference_uses_unsigned_post_softmax_probe(case: ModelCase):
    from runspace.src.quantization.activation_transport import ActivationTransport

    values = _probability_probe()
    format_ids = torch.tensor([0, 1], dtype=torch.long)
    candidates = ("ufp8_e1m7", "ufp8_e2m6")
    producer_id = f"{case.name}:softmax"

    reference = ActivationTransport(mode="reference", chunk_size=128)
    reference_value = reference.transmit_dynamic(
        values,
        format_ids,
        candidates,
        producer_id=producer_id,
    )
    assert reference_value.shape == values.shape
    assert torch.all(reference.decode(reference_value) >= 0)


@pytest.mark.skipif(
    not _env_enabled(CUDA_TRANSPORT_ENV) or not torch.cuda.is_available(),
    reason=f"Set {CUDA_TRANSPORT_ENV}=1 with CUDA to run packet parity tests",
)
@pytest.mark.parametrize(
    "case",
    [case for case in MODEL_CASES if case.expectation and case.expectation.requires_attention],
    ids=lambda case: case.name,
)
def test_attention_encoded_matches_post_softmax_reference(case: ModelCase):
    from runspace.src.quantization.activation_transport import ActivationTransport

    values = _probability_probe().cuda()
    format_ids = torch.tensor([0, 1], dtype=torch.long, device=values.device)
    candidates = ("ufp8_e1m7", "ufp8_e2m6")
    producer_id = f"{case.name}:softmax"

    reference = ActivationTransport(mode="reference", chunk_size=128)
    encoded = ActivationTransport(mode="encoded", chunk_size=128)
    reference_value = reference.transmit_dynamic(
        values,
        format_ids,
        candidates,
        producer_id=producer_id,
    )
    encoded_value = encoded.transmit_dynamic(
        values,
        format_ids,
        candidates,
        producer_id=producer_id,
    )

    torch.testing.assert_close(
        encoded.decode(encoded_value),
        reference.decode(reference_value),
        rtol=0,
        atol=0,
    )
    assert torch.all(encoded.decode(encoded_value) >= 0)


@pytest.mark.skipif(
    not _env_enabled(MODEL_LOADING_ENV),
    reason=f"Set {MODEL_LOADING_ENV}=1 to build every configured model offline",
)
@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.name)
def test_offline_model_load_has_expected_structure(case: ModelCase):
    from runspace.src.adapters.adapter_factory import create_adapter
    from runspace.src.quantization.activation_stage_planner import (
        plan_activation_stages,
    )

    expectation = case.expectation
    assert expectation is not None
    adapter = create_adapter(offline_transport_adapter_config(case))
    model = adapter.model.eval()
    module_types = _module_type_names(model)

    for alternatives in expectation.required_module_groups:
        assert any(
            any(token in module_type for module_type in module_types)
            for token in alternatives
        ), f"{case.name} is missing expected module group {alternatives}"

    plan = plan_activation_stages(model)
    assert plan.stages
    assert plan.boundary_nodes
    assert plan.model_output_sources
    if expectation.probe_is_unsigned:
        assert any(
            stage.is_unsigned
            and stage.unsigned_source == expectation.probe_producer
            for stage in plan.stages
        ), (
            f"{case.name} has no unsigned boundary for "
            f"{expectation.probe_producer}"
        )
    if expectation.requires_post_softmax_boundary:
        assert any(
            stage.is_unsigned and stage.unsigned_source == "softmax"
            for stage in plan.stages
        ), f"{case.name} attention has no post-Softmax transport boundary"


@pytest.mark.skipif(
    not _env_enabled(MODEL_RUNTIME_ENV) or not torch.cuda.is_available(),
    reason=f"Set {MODEL_RUNTIME_ENV}=1 with CUDA to run model reference transport",
)
@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("selection_mode", ("uniform", "dynamic"))
def test_offline_model_reference_transport_forward(
    case: ModelCase,
    selection_mode: str,
):
    from runspace.src.adapters.adapter_factory import create_adapter
    from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer
    from runspace.src.quantization.uniform_input_quantizer import UniformInputQuantizer

    expectation = case.expectation
    assert expectation is not None
    adapter = create_adapter(offline_transport_adapter_config(case))
    model = adapter.model.cuda().eval()
    default_cfg = getattr(model, "default_cfg", {}) or {}
    input_size = tuple(default_cfg.get("input_size", (3, 224, 224)))
    inputs = torch.randn((1, *input_size), device="cuda")
    if selection_mode == "uniform":
        quantizer = UniformInputQuantizer(
            model,
            fmt="fp8_e1m6",
            chunk_size=128,
            transport="reference",
            collect_error_stats=False,
        )
    else:
        quantizer = DynamicInputQuantizer(
            model,
            metric="mse",
            candidate_formats=["fp8_e1m6", "fp8_e2m5"],
            chunk_size=128,
            transport="reference",
            collect_error_stats=False,
        )

    try:
        quantizer.register_hooks()
        outputs = model(inputs)
        stats = quantizer.get_final_stats()
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape[0] == 1
        assert stats["transmission_count"] > 0
        assert stats["decode_reads"] >= stats["transmission_count"]
        assert stats["stage_count"] > 0
        if expectation.requires_post_softmax_boundary:
            assert any(
                entry.get("unsigned_source") == "softmax"
                and bool(entry.get("format_counts"))
                and set(entry.get("format_counts", {})).issubset(
                    {"ufp8_e1m7", "ufp8_e2m6"}
                )
                for entry in stats["layer_stats"].values()
            )
    finally:
        quantizer.cleanup()
        del inputs, model, adapter
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not _env_enabled(IMAGENET_SMOKE_ENV)
    or not torch.cuda.is_available()
    or not IMAGENET_PATH.is_dir(),
    reason=(
        f"Set {IMAGENET_SMOKE_ENV}=1 with CUDA and ImageNet mounted at "
        f"{IMAGENET_PATH}"
    ),
)
@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.name)
def test_cuda_imagenet_transport_matrix(case: ModelCase, tmp_path: Path):
    """Run dynamic/uniform selection through reference and encoded transport."""
    from runspace.core.runner import Runner

    runner = Runner(device=torch.device("cuda"))
    config = imagenet_adapter_config(case)
    output_dir = tmp_path / case.name
    model = adapter = loader = None

    try:
        model, adapter, _weights_path = runner.prepare_model_with_materialized_weights(
            config,
            output_dir=str(output_dir),
        )
        loader = runner.setup_data_loader(config)

        for selection_mode in ("uniform", "dynamic"):
            for transport in ("reference", "encoded"):
                result = runner.evaluate_model(
                    model=model,
                    data_loader=loader,
                    adapter=adapter,
                    max_batches=1,
                    desc=f"{case.name} {selection_mode}/{transport}",
                    input_quant_cfg=transport_quant_config(selection_mode, transport),
                )
                assert result.get("input_quant") is not None
                assert result["input_quant"]["mode"] == selection_mode
                assert torch.isfinite(torch.tensor(result.get("certainty", 0.0)))
    finally:
        del loader, model, adapter
        gc.collect()
        torch.cuda.empty_cache()
