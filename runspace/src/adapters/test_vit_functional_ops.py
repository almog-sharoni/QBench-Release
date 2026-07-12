#!/usr/bin/env python3
"""
One-batch functional-op quantization smoke test for torchvision ViT-B/16.

This script checks two things:
1. Baseline build: the transport-ready ViT graph runs one batch.
2. Dynamic input build: producer-stage transport reaches decomposed attention.

It is intentionally small and model-specific so we can debug functional-op handling
without running a full benchmark.
"""

import os
import sys
from collections import Counter

import torch
import torch.fx

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
RUNSPACE_ROOT = os.path.join(PROJECT_ROOT, "runspace")
if RUNSPACE_ROOT not in sys.path:
    sys.path.insert(0, RUNSPACE_ROOT)

from runspace.src.adapters.generic_adapter import GenericAdapter  # noqa: E402
from runspace.src.ops.quant_arithmetic import QuantAdd, QuantMul  # noqa: E402
from runspace.src.ops.quant_matmul import QuantMatMul  # noqa: E402
from runspace.src.quantization.dynamic_input_quantizer import (  # noqa: E402
    DynamicInputQuantizer,
)
from runspace.src.utils.model_input_utils import resolve_model_input_size  # noqa: E402


TARGET_LABELS = {
    QuantMatMul: "QuantMatMul",
    QuantMul: "QuantMul",
    QuantAdd: "QuantAdd",
}


def build_vit_functional_quant_model():
    adapter = GenericAdapter(
        model_name="vit_b_16",
        model_source="torchvision",
        quantized_ops=["Linear", "QuantMatMul", "QuantMul", "QuantAdd"],
        input_quantization=False,
        weight_quantization=False,
        skip_calibration=True,
        enable_fx_quantization=True,
    )
    return adapter.model.eval()


def inspect_graph(model):
    call_module_counts = Counter()
    raw_functional_counts = Counter()

    for module in model.modules():
        for module_type, label in TARGET_LABELS.items():
            if isinstance(module, module_type):
                call_module_counts[label] += 1
                break

    graph_modules = {
        id(module): module
        for module in model.modules()
        if isinstance(module, torch.fx.GraphModule)
    }
    for graph_module in graph_modules.values():
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                target_name = getattr(node.target, "__name__", str(node.target))
                if target_name in {"matmul", "mul", "add"}:
                    raw_functional_counts[target_name] += 1

    return call_module_counts, raw_functional_counts


def attach_runtime_counters(model):
    hit_counts = Counter()
    handles = []

    def make_hook(label):
        def hook(_module, _inputs, _output):
            hit_counts[label] += 1
        return hook

    for _name, module in model.named_modules():
        for module_type, label in TARGET_LABELS.items():
            if isinstance(module, module_type):
                handles.append(module.register_forward_hook(make_hook(label)))
                break

    return hit_counts, handles


def active_transport_stages(stats):
    configured = {}
    for stage_id, stage in stats.get("layer_stats", {}).items():
        chunks = int(stage.get("total_chunks", 0))
        if chunks:
            configured[stage_id] = {
                "type": stage.get("type", "unknown"),
                "num_chunks": chunks,
                "formats": sorted(stage.get("format_counts", {})),
            }
    return configured


def run_one_batch(model):
    _, c, h, w = resolve_model_input_size(model)
    device = next(model.parameters()).device
    x = torch.randn(1, c, h, w, device=device)
    with torch.no_grad():
        out = model(x)
    if not isinstance(out, torch.Tensor):
        raise AssertionError(f"Expected tensor output, got {type(out).__name__}")
    return tuple(out.shape)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    print_section("ViT-B/16 Functional-Op Quantization Test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transport = "encoded" if device.type == "cuda" else "reference"
    print(f"\nDevice: {device}; activation transport: {transport}")

    print("\nBuilding baseline model...")
    baseline_model = build_vit_functional_quant_model().to(device)
    baseline_graph_counts, baseline_raw_counts = inspect_graph(baseline_model)
    baseline_hits, baseline_handles = attach_runtime_counters(baseline_model)
    baseline_shape = run_one_batch(baseline_model)
    for handle in baseline_handles:
        handle.remove()

    print("\nBaseline graph replacement counts:")
    for label in ("QuantMatMul", "QuantMul", "QuantAdd"):
        print(f"  {label:12s}: {baseline_graph_counts.get(label, 0)}")

    print("\nBaseline leftover raw functional nodes:")
    for label in ("matmul", "mul", "add"):
        print(f"  {label:12s}: {baseline_raw_counts.get(label, 0)}")

    print("\nBaseline runtime hits for one batch:")
    for label in ("QuantMatMul", "QuantMul", "QuantAdd"):
        print(f"  {label:12s}: {baseline_hits.get(label, 0)}")
    print(f"  Output shape : {baseline_shape}")

    print("\nBuilding dynamic-input model...")
    dynamic_model = build_vit_functional_quant_model().to(device)
    dynamic_graph_counts, dynamic_raw_counts = inspect_graph(dynamic_model)
    dynamic_hits, dynamic_handles = attach_runtime_counters(dynamic_model)
    dynamic_quantizer = DynamicInputQuantizer(
        dynamic_model,
        metric="mse",
        chunk_size=128,
        transport=transport,
    )
    dynamic_quantizer.register_hooks()
    dynamic_shape = run_one_batch(dynamic_model)
    dynamic_stats = dynamic_quantizer.get_final_stats()
    configured_transport_stages = active_transport_stages(dynamic_stats)
    dynamic_quantizer.cleanup()
    for handle in dynamic_handles:
        handle.remove()

    print("\nDynamic graph replacement counts:")
    for label in ("QuantMatMul", "QuantMul", "QuantAdd"):
        print(f"  {label:12s}: {dynamic_graph_counts.get(label, 0)}")

    print("\nDynamic leftover raw functional nodes:")
    for label in ("matmul", "mul", "add"):
        print(f"  {label:12s}: {dynamic_raw_counts.get(label, 0)}")

    print("\nDynamic runtime hits for one batch:")
    for label in ("QuantMatMul", "QuantMul", "QuantAdd"):
        print(f"  {label:12s}: {dynamic_hits.get(label, 0)}")
    print(f"  Output shape : {dynamic_shape}")

    print("\nDynamic-input activation transport stages (first 12):")
    if configured_transport_stages:
        displayed = sorted(configured_transport_stages.items())[:12]
        for name, info in displayed:
            print(
                f"  {name}: {info['type']} "
                f"(chunks={info['num_chunks']}, formats={info['formats']})"
            )
        remaining = len(configured_transport_stages) - len(displayed)
        if remaining:
            print(f"  ... {remaining} more active stages")
    else:
        print("  None")

    baseline_ok = baseline_shape == (1, 1000)

    dynamic_ok = (
        dynamic_stats.get("transmission_count", 0) > 0
        and dynamic_stats.get("decode_reads", 0) > 0
        and bool(configured_transport_stages)
        and any(
            stage.get("unsigned_source") == "softmax"
            and stage.get("transmissions", 0) > 0
            for stage in dynamic_stats.get("activation_plan", {}).values()
        )
    )

    print("\nSummary:")
    print(f"  Baseline transport-ready forward : {'PASS' if baseline_ok else 'FAIL'}")
    print(f"  Attention stage transport        : {'PASS' if dynamic_ok else 'FAIL'}")

    if not baseline_ok or not dynamic_ok:
        print("\nResult: ViT activation transport is not fully covered yet.")
        return 1

    print("\nResult: ViT baseline and activation transport both passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
