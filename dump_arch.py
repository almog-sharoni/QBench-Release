#!/usr/bin/env python3
"""Dump a quantized QBench model architecture to JSON.

Examples:
    python dump_arch.py --model resnet18 --source torchvision --output resnet18_arch.json
    python dump_arch.py --config runspace/src/configs/mobilevit_fp8.yaml --output mobilevit_arch.json --skip-calibration
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import sys
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import yaml


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")

from runspace.src.adapters.adapter_factory import create_adapter  # noqa: E402

try:
    from runspace.src.ops.quant_base import QuantizedLayerMixin  # noqa: E402
except Exception:  # pragma: no cover - keeps the dumper usable during import issues.
    QuantizedLayerMixin = ()  # type: ignore[assignment]


QUANT_ATTRS = (
    "q_type",
    "quantization_bias",
    "input_q_type",
    "input_quantization",
    "weight_quantization",
    "output_q_type",
    "output_quantization",
    "input_mode",
    "input_chunk_size",
    "weight_mode",
    "weight_chunk_size",
    "output_mode",
    "output_chunk_size",
    "quant_mode",
    "chunk_size",
    "act_mode",
    "act_chunk_size",
    "rounding",
    "is_first_layer",
    "quantize_first_layer",
    "simulate_tf32_accum",
    "layer_name",
    "run_id",
)


def parse_input_size(value: str | None) -> int | tuple[int, ...] | None:
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    if text.isdigit():
        return int(text)

    if "," in text and not text.startswith(("(", "[")):
        text = f"({text})"

    parsed = ast.literal_eval(text)
    if isinstance(parsed, int):
        return parsed
    if isinstance(parsed, (tuple, list)) and all(isinstance(item, int) for item in parsed):
        return tuple(parsed)

    raise argparse.ArgumentTypeError(
        "--input-size must be an int or tuple/list of ints, e.g. 224 or '(224, 224)'"
    )


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config {path!r} must contain a YAML mapping.")
    return config


def set_if_present(config: dict[str, Any], section: str, key: str, value: Any) -> None:
    if value is None:
        return
    config.setdefault(section, {})[key] = value


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    q_type = args.q_type or "fp8_e4m3"

    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "model": {
                "name": args.model,
                "source": args.source,
            },
            "adapter": {
                "type": "generic",
                "quantized_ops": args.quantized_ops or ["all"],
                "quantize_first_layer": bool(args.quantize_first_layer),
                "input_quantization": (
                    True if args.input_quantization is None else args.input_quantization
                ),
                "weight_quantization": (
                    True if args.weight_quantization is None else args.weight_quantization
                ),
                "output_quantization": (
                    False if args.output_quantization is None else args.output_quantization
                ),
            },
            "quantization": {
                "format": q_type,
            },
        }

    config = deepcopy(config)

    set_if_present(config, "model", "name", args.model_override)
    set_if_present(config, "model", "source", args.source_override)
    set_if_present(config, "model", "weights", args.weights)

    set_if_present(config, "adapter", "quantized_ops", args.quantized_ops)
    set_if_present(config, "adapter", "excluded_ops", args.excluded_ops)
    set_if_present(config, "adapter", "quantize_first_layer", args.quantize_first_layer)
    set_if_present(config, "adapter", "input_quantization", args.input_quantization)
    set_if_present(config, "adapter", "weight_quantization", args.weight_quantization)
    set_if_present(config, "adapter", "output_quantization", args.output_quantization)
    set_if_present(config, "adapter", "fold_layers", args.fold_layers)
    set_if_present(config, "adapter", "fold_input_norm", args.fold_input_norm)
    set_if_present(config, "adapter", "skip_calibration", args.skip_calibration)
    set_if_present(config, "adapter", "enable_fx_quantization", args.enable_fx_quantization)
    set_if_present(config, "adapter", "input_size", args.input_size)

    q_type_override = args.q_type_override if args.q_type_override is not None else args.q_type
    set_if_present(config, "quantization", "format", q_type_override)
    set_if_present(config, "quantization", "bias", args.bias)
    set_if_present(config, "quantization", "mode", args.mode)
    set_if_present(config, "quantization", "chunk_size", args.chunk_size)
    set_if_present(config, "quantization", "weight_mode", args.weight_mode)
    set_if_present(config, "quantization", "weight_chunk_size", args.weight_chunk_size)
    set_if_present(config, "quantization", "act_mode", args.act_mode)
    set_if_present(config, "quantization", "act_chunk_size", args.act_chunk_size)
    set_if_present(config, "quantization", "output_format", args.output_format)
    set_if_present(config, "quantization", "output_mode", args.output_mode)
    set_if_present(config, "quantization", "output_chunk_size", args.output_chunk_size)

    config.setdefault("adapter", {})["build_quantized"] = True
    return config


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": value.numel(),
        }
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return repr(value)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.numel() == 0:
        return {}

    try:
        data = tensor.detach()
        if data.is_complex():
            data = data.abs()
        if not data.is_floating_point():
            data = data.float()
        data = data.cpu()
        return {
            "min": data.min().item(),
            "max": data.max().item(),
            "mean": data.mean().item(),
            "std": data.std(unbiased=False).item() if data.numel() > 1 else 0.0,
        }
    except Exception as exc:
        return {"stats_error": f"{type(exc).__name__}: {exc}"}


def tensor_summary(
    tensor: torch.Tensor | None,
    *,
    include_stats: bool,
    include_values: bool,
    max_value_elements: int,
) -> dict[str, Any] | None:
    if tensor is None:
        return None

    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }

    if include_stats:
        summary["stats"] = tensor_stats(tensor)

    if include_values:
        if tensor.numel() <= max_value_elements:
            summary["values"] = tensor.detach().cpu().tolist()
        else:
            summary["values_truncated"] = True
            summary["max_value_elements"] = max_value_elements

    return summary


def direct_parameters(
    module: nn.Module,
    *,
    include_stats: bool,
    include_values: bool,
    max_value_elements: int,
) -> dict[str, Any]:
    return {
        name: {
            **(
                tensor_summary(
                    param,
                    include_stats=include_stats,
                    include_values=include_values,
                    max_value_elements=max_value_elements,
                )
                or {}
            ),
            "requires_grad": bool(param.requires_grad),
        }
        for name, param in module.named_parameters(recurse=False)
    }


def direct_buffers(
    module: nn.Module,
    *,
    include_stats: bool,
    include_values: bool,
    max_value_elements: int,
) -> dict[str, Any]:
    non_persistent = getattr(module, "_non_persistent_buffers_set", set())
    result = {}
    for name, buffer in module.named_buffers(recurse=False):
        summary = tensor_summary(
            buffer,
            include_stats=include_stats,
            include_values=include_values,
            max_value_elements=max_value_elements,
        )
        if summary is None:
            summary = {"value": None}
        summary["persistent"] = name not in non_persistent
        result[name] = summary
    return result


def module_extra_repr(module: nn.Module) -> str:
    try:
        return module.extra_repr()
    except Exception:
        text = repr(module)
        return text if "\n" not in text else ""


def forward_signature(module: nn.Module) -> str | None:
    try:
        return str(inspect.signature(module.forward))
    except Exception:
        return None


def is_quantized_module(module: nn.Module) -> bool:
    class_name = module.__class__.__name__
    if class_name.startswith(("Quant", "Observed")):
        return True
    try:
        return isinstance(module, QuantizedLayerMixin)
    except TypeError:
        return False


def quantization_attrs(module: nn.Module) -> dict[str, Any]:
    attrs = {}
    for attr in QUANT_ATTRS:
        if hasattr(module, attr):
            attrs[attr] = to_jsonable(getattr(module, attr))
    return attrs


def dump_architecture(
    model: nn.Module,
    config: dict[str, Any],
    *,
    include_stats: bool,
    include_values: bool,
    max_value_elements: int,
) -> dict[str, Any]:
    modules = []
    named_modules = list(model.named_modules())

    for name, module in named_modules:
        display_name = name or "<root>"
        parent = None if not name else ".".join(name.split(".")[:-1]) or "<root>"
        children = [
            f"{name}.{child_name}" if name else child_name
            for child_name, _ in module.named_children()
        ]
        direct_param_count = sum(p.numel() for p in module.parameters(recurse=False))
        recursive_param_count = sum(p.numel() for p in module.parameters(recurse=True))

        modules.append(
            {
                "name": display_name,
                "parent": parent,
                "children": children,
                "depth": 0 if not name else name.count(".") + 1,
                "class_name": module.__class__.__name__,
                "class_module": module.__class__.__module__,
                "is_leaf": len(children) == 0,
                "is_quantized": is_quantized_module(module),
                "module_args": module_extra_repr(module),
                "forward_signature": forward_signature(module),
                "direct_parameter_count": direct_param_count,
                "recursive_parameter_count": recursive_param_count,
                "parameters": direct_parameters(
                    module,
                    include_stats=include_stats,
                    include_values=include_values,
                    max_value_elements=max_value_elements,
                ),
                "buffers": direct_buffers(
                    module,
                    include_stats=include_stats,
                    include_values=include_values,
                    max_value_elements=max_value_elements,
                ),
                "quantization": quantization_attrs(module),
            }
        )

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return {
        "metadata": {
            "model_name": config.get("model", {}).get("name"),
            "model_source": config.get("model", {}).get("source"),
            "weights": config.get("model", {}).get("weights"),
            "module_count": len(modules),
            "quantized_module_count": sum(1 for item in modules if item["is_quantized"]),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "include_stats": include_stats,
            "include_values": include_values,
        },
        "config": config,
        "modules": modules,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a QBench quantized model and write its architecture to JSON."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="YAML QBench config to load.")
    source.add_argument("--model", default="resnet18", help="Model name when --config is not used.")

    parser.add_argument("--source", default="auto", help="Model source when --config is not used.")
    parser.add_argument("--weights", help="Weights token/path override. Use none by omitting this flag.")
    parser.add_argument("--output", default="quantized_architecture.json", help="Output JSON path.")

    parser.add_argument("--model-override", help="Override config model.name.")
    parser.add_argument("--source-override", help="Override config model.source.")

    parser.add_argument("--q-type", help="Quantization format. Defaults to fp8_e4m3 without --config.")
    parser.add_argument("--q-type-override", help="Deprecated alias for --q-type.")
    parser.add_argument("--bias", type=int, help="Override config quantization.bias.")
    parser.add_argument("--mode", choices=("tensor", "chunk", "channel"), help="Activation/input quant mode.")
    parser.add_argument("--chunk-size", type=int, help="Activation/input chunk size.")
    parser.add_argument("--weight-mode", choices=("tensor", "chunk", "channel"), help="Weight quant mode.")
    parser.add_argument("--weight-chunk-size", type=int, help="Weight chunk size.")
    parser.add_argument("--act-mode", choices=("tensor", "chunk", "channel"), help="Activation quant mode.")
    parser.add_argument("--act-chunk-size", type=int, help="Activation chunk size.")
    parser.add_argument("--output-format", help="Output quantization format.")
    parser.add_argument("--output-mode", choices=("tensor", "chunk", "channel"), help="Output quant mode.")
    parser.add_argument("--output-chunk-size", type=int, help="Output chunk size.")

    parser.add_argument("--quantized-ops", nargs="+", help="Ops to quantize, e.g. all Conv2d Linear.")
    parser.add_argument("--excluded-ops", nargs="+", help="Ops to leave unquantized.")
    parser.add_argument("--input-size", type=parse_input_size, help="Model input size override for timm models.")

    parser.add_argument("--quantize-first-layer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--input-quantization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--weight-quantization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-quantization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fold-layers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fold-input-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-calibration", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-fx-quantization", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--include-stats", action="store_true", help="Include min/max/mean/std for tensors.")
    parser.add_argument("--include-values", action="store_true", help="Include full tensor values for small tensors.")
    parser.add_argument(
        "--max-value-elements",
        type=int,
        default=128,
        help="Largest tensor to serialize with --include-values.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config = build_config(args)

    adapter = create_adapter(config)
    model = adapter.model
    model.eval()

    architecture = dump_architecture(
        model,
        config,
        include_stats=args.include_stats,
        include_values=args.include_values,
        max_value_elements=args.max_value_elements,
    )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(architecture, f, indent=2, sort_keys=True)
        f.write("\n")

    meta = architecture["metadata"]
    print(
        f"Wrote {output_path} "
        f"({meta['module_count']} modules, {meta['quantized_module_count']} quantized, "
        f"{meta['total_parameters']} params)."
    )


if __name__ == "__main__":
    main()
