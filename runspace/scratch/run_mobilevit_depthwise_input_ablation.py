import argparse
import os

import torch

from runspace.core.runner import Runner
from runspace.experiments.utils.common import (
    build_dynamic_input_quant_cfg,
    build_loader,
    build_runtime_config,
    run_inference,
)


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mobilevit_s")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--dataset_name", default="imagenet")
    parser.add_argument("--dataset_path", default="/data/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit_batches", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        default="runspace/experiments/find_optimal_hybrid_quant/results",
    )
    parser.add_argument(
        "--weight_file",
        default=(
            "runspace/experiments/find_optimal_hybrid_quant/results/"
            "mobilevit_s/weights_mse/quantized_weights.pt"
        ),
    )
    parser.add_argument("--skip_depthwise_input_quant", action="store_true")
    parser.add_argument("--skip_depthwise_weight_quant", action="store_true")
    parser.add_argument("--no_fold_input_norm", action="store_false", dest="fold_input_norm")
    parser.set_defaults(fold_input_norm=True)
    return parser.parse_args()


def _is_depthwise_conv(module):
    return (
        isinstance(module, torch.nn.Conv2d)
        and getattr(module, "groups", 1) > 1
        and getattr(module, "groups", 1) == getattr(module, "in_channels", None)
        and getattr(module, "groups", 1) == getattr(module, "out_channels", None)
    )


def _skip_depthwise_weight_quant(model):
    skipped = []
    for name, module in model.named_modules():
        if not _is_depthwise_conv(module):
            continue
        if not (hasattr(module, "weight_fp8") or hasattr(module, "weight_scale")):
            continue

        module.weight_quantization = False
        if hasattr(module, "weight_fp8"):
            module.weight_fp8 = None
        if hasattr(module, "weight_scale"):
            module.weight_scale = None
        if hasattr(module, "weight_scale_packed"):
            module.weight_scale_packed = None
        skipped.append(name)

    print(
        "Depthwise weight-quant ablation: skipped weight quantization for "
        f"{len(skipped)} depthwise Conv2d layers."
    )
    return skipped


def main():
    args = _args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = build_runtime_config(args, model_name=args.model_name, weights=args.weights)
    base_config.setdefault("adapter", {})
    base_config["adapter"].update(
        {
            "quantized_ops": ["all"],
            "input_quantization": True,
            "weight_quantization": True,
            "fold_input_norm": args.fold_input_norm,
            "quantize_first_layer": args.fold_input_norm,
        }
    )

    runner = Runner(device)
    model, adapter = runner.load_model_from_weight_file(
        config=base_config,
        weight_file_path=args.weight_file,
        skip_calibration=True,
    )
    if args.skip_depthwise_weight_quant:
        _skip_depthwise_weight_quant(model)

    loader = build_loader(args, device, runner, config_builder=lambda _: base_config)

    fp8_formats = [
        "fp8_e1m6",
        "fp8_e2m5",
        "fp8_e3m4",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e6m1",
        "fp8_e7m0",
    ]
    input_quant_cfg = build_dynamic_input_quant_cfg(
        metric="mse",
        chunk_size=128,
        candidate_formats=fp8_formats,
        unsigned_input_sources=["relu", "softmax", "quantrelu", "quantsoftmax"],
        dynamic_unsigned_input_candidates=True,
        skip_depthwise_input_quant=args.skip_depthwise_input_quant,
        model_name=args.model_name,
    )

    labels = []
    if args.skip_depthwise_input_quant:
        labels.append("skip_depthwise_input")
    if args.skip_depthwise_weight_quant:
        labels.append("skip_depthwise_weight")
    label = "+".join(labels) if labels else "control"
    acc1, acc5, certainty, input_stats = run_inference(
        runner,
        model,
        adapter,
        loader,
        args,
        input_quant_cfg=input_quant_cfg,
        desc=f"MobileViT depthwise ablation ({label})",
    )

    norm_l1 = input_stats.get("norm_l1") if input_stats else None
    norm_mse = input_stats.get("norm_mse") if input_stats else None
    print(
        f"ABLATION {label}: Top1={acc1:.4f} Top5={acc5:.4f} "
        f"Certainty={certainty:.6f} NormL1={norm_l1:.6e} NormMSE={norm_mse:.6e}"
    )


if __name__ == "__main__":
    main()
