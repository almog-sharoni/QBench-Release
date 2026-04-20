#!/usr/bin/env python3
"""
One-batch functional-op quantization smoke test for torchvision ViT-B/16.

This script checks two things:
1. Baseline build: FX replacement converts functional ops into quantized modules.
2. Dynamic input build: runtime dynamic-input hooks also reach those functional modules.

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

from runspace.src.adapters.generic_adapter import GenericAdapter
from runspace.src.ops.quant_arithmetic import QuantAdd, QuantMul
from runspace.src.ops.quant_matmul import QuantMatMul
from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer
from runspace.src.utils.model_input_utils import resolve_model_input_size


TARGET_TYPES = (QuantMatMul, QuantMul, QuantAdd)
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
    model = adapter.model.eval()
    if not isinstance(model, torch.fx.GraphModule):
        raise AssertionError(
            f"Expected GraphModule after FX quantization, got {type(model).__name__}"
        )
    return model


def inspect_graph(model):
    call_module_counts = Counter()
    raw_functional_counts = Counter()

    for node in model.graph.nodes:
        if node.op == "call_module":
            module = model.get_submodule(node.target)
            for module_type, label in TARGET_LABELS.items():
                if isinstance(module, module_type):
                    call_module_counts[label] += 1
                    break
        elif node.op == "call_function":
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


def functional_modules_with_dynamic_state(model):
    configured = {}
    for name, module in model.named_modules():
        if isinstance(module, TARGET_TYPES):
            chunk_formats = getattr(module, "input_chunk_formats", None)
            if chunk_formats:
                configured[name] = {
                    "type": type(module).__name__,
                    "input_mode": getattr(module, "input_mode", None),
                    "num_chunk_formats": len(chunk_formats),
                }
    return configured


def run_one_batch(model):
    _, c, h, w = resolve_model_input_size(model)
    x = torch.randn(1, c, h, w)
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

    print("\nBuilding baseline model...")
    baseline_model = build_vit_functional_quant_model()
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
    dynamic_model = build_vit_functional_quant_model()
    dynamic_graph_counts, dynamic_raw_counts = inspect_graph(dynamic_model)
    dynamic_hits, dynamic_handles = attach_runtime_counters(dynamic_model)
    dynamic_quantizer = DynamicInputQuantizer(dynamic_model, metric="mse", chunk_size=128)
    dynamic_quantizer.register_hooks()
    dynamic_shape = run_one_batch(dynamic_model)
    configured_functional_modules = functional_modules_with_dynamic_state(dynamic_model)
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

    print("\nDynamic-input configured functional modules:")
    if configured_functional_modules:
        for name, info in sorted(configured_functional_modules.items()):
            print(
                f"  {name}: {info['type']} "
                f"(mode={info['input_mode']}, chunks={info['num_chunk_formats']})"
            )
    else:
        print("  None")

    baseline_ok = (
        baseline_graph_counts.get("QuantMatMul", 0) > 0
        and baseline_graph_counts.get("QuantAdd", 0) > 0
        and baseline_hits.get("QuantMatMul", 0) > 0
        and baseline_hits.get("QuantAdd", 0) > 0
        and baseline_raw_counts.get("matmul", 0) == 0
        and baseline_raw_counts.get("add", 0) == 0
    )

    dynamic_ok = any(
        info["type"] in {"QuantMatMul", "QuantAdd"}
        for info in configured_functional_modules.values()
    )

    print("\nSummary:")
    print(f"  Baseline functional-op quantization: {'PASS' if baseline_ok else 'FAIL'}")
    print(f"  Dynamic functional-op coverage    : {'PASS' if dynamic_ok else 'FAIL'}")

    if not baseline_ok or not dynamic_ok:
        print("\nResult: functional-op quantization is not fully covered yet.")
        return 1

    print("\nResult: baseline and dynamic functional-op quantization both passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
