import argparse
import copy
import gc
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner
from runspace.experiments.utils.common import (
    build_dynamic_input_quant_cfg,
    build_loader,
    build_runtime_config,
    build_uniform_input_quant_cfg,
)


FP8_FORMATS = [
    "fp8_e1m6",
    "fp8_e2m5",
    "fp8_e3m4",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e6m1",
    "fp8_e7m0",
]

FP8_UFP_FORMATS = [
    "fp8_e1m6",
    "fp8_e2m5",
    "fp8_e3m4",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e6m1",
    "fp8_e7m0",
    "ufp8_e1m7",
    "ufp8_e2m6",
    "ufp8_e3m5",
    "ufp8_e4m4",
    "ufp8_e5m3",
    "ufp8_e6m2",
    "ufp8_e7m1",
    "ufp8_e8m0",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Small-batch ablation for MobileNetV3 hybrid quant accuracy drop. "
            "Reuses cached optimized weight artifacts; does not recompute weights."
        )
    )
    parser.add_argument("--model_name", default="mobilenet_v3_large")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--dataset_name", default="imagenet")
    parser.add_argument("--dataset_path", default="/data/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit_batches", type=int, default=10)
    parser.add_argument("--input_chunk_size", type=int, default=128)
    parser.add_argument("--weight_file", default=None)
    parser.add_argument(
        "--output_json",
        default="runspace/scratch/mobilenet_hybrid_ablation_results.json",
    )
    parser.add_argument("--no_fold_input_norm", action="store_false", dest="fold_input_norm")
    parser.set_defaults(fold_input_norm=True)
    return parser.parse_args()


def quantized_wrapper_config(args, *, input_quantization=False, weight_quantization=False):
    cfg = build_runtime_config(args, model_name=args.model_name, weights=args.weights)
    cfg.setdefault("adapter", {})
    cfg["adapter"].update(
        {
            "quantized_ops": ["all"],
            "build_quantized": True,
            "input_quantization": bool(input_quantization),
            "weight_quantization": bool(weight_quantization),
            "fold_input_norm": bool(args.fold_input_norm),
            "quantize_first_layer": bool(args.fold_input_norm),
            "skip_calibration": True,
        }
    )
    cfg["quantization"] = {
        "format": "fp8_e1m6",
        "input_format": "fp8_e1m6",
        "mode": "chunk",
        "chunk_size": int(args.input_chunk_size),
        "weight_mode": "chunk",
        "weight_chunk_size": 128,
    }
    return cfg


def default_weight_file(args):
    return os.path.join(
        "runspace",
        "experiments",
        "find_optimal_hybrid_quant",
        "results",
        args.model_name,
        "weights_mse",
        "quantized_weights.pt",
    )


def load_model_for_variant(runner, args, variant):
    uses_quantized_weights = variant.get("weights") == "optimized_cached"
    cfg = quantized_wrapper_config(
        args,
        input_quantization=False,
        weight_quantization=uses_quantized_weights,
    )

    if uses_quantized_weights:
        weight_file = args.weight_file or default_weight_file(args)
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Optimized weight file not found: {weight_file}")
        return runner.load_model_from_weight_file(
            config=cfg,
            weight_file_path=weight_file,
            skip_calibration=True,
        )

    out_dir = os.path.join(
        "/tmp",
        "qbench_mobilenet_hybrid_ablation",
        args.model_name,
        variant["name"],
    )
    return runner.prepare_model_with_materialized_weights(config=cfg, output_dir=out_dir)[:2]


def input_cfg_for_variant(args, variant):
    mode = variant.get("input")
    if mode in (None, "none"):
        return None
    if mode == "uniform_fp8_e1m6":
        return build_uniform_input_quant_cfg("fp8_e1m6", args.input_chunk_size)
    if mode == "dynamic_fp8_only":
        return build_dynamic_input_quant_cfg(
            metric="mse",
            chunk_size=args.input_chunk_size,
            candidate_formats=FP8_FORMATS,
            unsigned_input_sources=[],
            dynamic_unsigned_input_candidates=False,
            model_name=args.model_name,
        )
    if mode == "dynamic_all":
        return build_dynamic_input_quant_cfg(
            metric="mse",
            chunk_size=args.input_chunk_size,
            candidate_formats=FP8_UFP_FORMATS,
            unsigned_input_sources=["relu", "softmax", "quantrelu", "quantsoftmax"],
            dynamic_unsigned_input_candidates=True,
            model_name=args.model_name,
        )
    if mode == "dynamic_all_skip_depthwise":
        return build_dynamic_input_quant_cfg(
            metric="mse",
            chunk_size=args.input_chunk_size,
            candidate_formats=FP8_UFP_FORMATS,
            unsigned_input_sources=["relu", "softmax", "quantrelu", "quantsoftmax"],
            dynamic_unsigned_input_candidates=True,
            model_name=args.model_name,
            skip_depthwise_input_quant=True,
        )
    raise ValueError(f"Unknown input ablation mode: {mode}")


def summarize_input_stats(input_stats):
    if not input_stats:
        return {}
    layer_stats = input_stats.get("layer_stats", {}) or {}
    format_counts = {}
    total_layers = 0
    for stats in layer_stats.values():
        counts = stats.get("format_counts", {}) or {}
        if counts:
            total_layers += 1
        for fmt, count in counts.items():
            format_counts[fmt] = format_counts.get(fmt, 0) + int(count)
    return {
        "norm_mse": input_stats.get("norm_mse"),
        "total_mse": input_stats.get("total_mse"),
        "layers_with_input_stats": total_layers,
        "format_counts": dict(sorted(format_counts.items(), key=lambda kv: -kv[1])),
    }


def run_variant(runner, loader, args, variant):
    model, adapter = load_model_for_variant(runner, args, variant)
    input_quant_cfg = input_cfg_for_variant(args, variant)
    try:
        eval_results = runner.evaluate_model(
            model=model,
            data_loader=loader,
            adapter=adapter,
            max_batches=args.limit_batches,
            desc=variant["name"],
            input_quant_cfg=input_quant_cfg,
        )
        input_stats = eval_results.get("input_quant") or eval_results.get("dynamic_input_quant")
        return {
            "name": variant["name"],
            "weights": variant.get("weights", "fp32"),
            "input": variant.get("input", "none"),
            "acc1": float(eval_results.get("acc1", 0.0)),
            "acc5": float(eval_results.get("acc5", 0.0)),
            "certainty": float(eval_results.get("certainty", 0.0)),
            "input_stats": summarize_input_stats(input_stats),
        }
    finally:
        del model, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Runner(device)

    loader_cfg = quantized_wrapper_config(args)
    loader = build_loader(args, device, runner, config_builder=lambda _: loader_cfg)

    variants = [
        {"name": "fp32_wrapped", "weights": "fp32", "input": "none"},
        {"name": "weight_only_optimized_cached", "weights": "optimized_cached", "input": "none"},
        {"name": "input_only_uniform_fp8_e1m6", "weights": "fp32", "input": "uniform_fp8_e1m6"},
        {"name": "input_only_dynamic_fp8_only", "weights": "fp32", "input": "dynamic_fp8_only"},
        {"name": "input_only_dynamic_all_with_ufp", "weights": "fp32", "input": "dynamic_all"},
        {"name": "hybrid_uniform_fp8_e1m6", "weights": "optimized_cached", "input": "uniform_fp8_e1m6"},
        {"name": "hybrid_dynamic_fp8_only", "weights": "optimized_cached", "input": "dynamic_fp8_only"},
        {"name": "hybrid_dynamic_all_with_ufp", "weights": "optimized_cached", "input": "dynamic_all"},
        {
            "name": "hybrid_dynamic_all_skip_depthwise_input",
            "weights": "optimized_cached",
            "input": "dynamic_all_skip_depthwise",
        },
    ]

    results = []
    for variant in variants:
        print(f"\n=== ABLATION: {variant['name']} ===")
        result = run_variant(runner, loader, args, variant)
        results.append(result)
        print(
            f"{result['name']}: acc1={result['acc1']:.4f} "
            f"acc5={result['acc5']:.4f} certainty={result['certainty']:.6f} "
            f"norm_mse={result['input_stats'].get('norm_mse')}"
        )

    ref_acc = next((r["acc1"] for r in results if r["name"] == "fp32_wrapped"), None)
    if ref_acc is not None:
        for result in results:
            result["drop_vs_fp32_wrapped"] = ref_acc - result["acc1"]

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "variants": results,
            },
            f,
            indent=2,
        )

    print("\n=== SUMMARY ===")
    for result in results:
        print(
            f"{result['name']:<40} "
            f"acc1={result['acc1']:8.4f} "
            f"drop={result.get('drop_vs_fp32_wrapped', 0.0):8.4f} "
            f"cert={result['certainty']:.6f} "
            f"norm_mse={result['input_stats'].get('norm_mse')}"
        )
    print(f"\nSaved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
