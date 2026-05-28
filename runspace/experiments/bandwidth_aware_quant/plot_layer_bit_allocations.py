#!/usr/bin/env python3
"""Plot per-layer bit allocations for bandwidth-aware quantization.

This script runs the cache analysis and bit-width optimizer only. It does not
materialize quantized weights or run dataset inference.
"""

import argparse
import os
import sys

# Keep matplotlib writable inside containers and force a non-interactive backend.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.bandwidth_aware_quant.bandwidth_aware_quant import (
    compute_model_runtime,
    run_cache_simulation,
)


DEFAULT_CACHE_SIZES = [0.0, 2.0, 4.0]
DEFAULT_THRESHOLDS = list(range(2, 9))
BIT_KIND_TO_LABEL = {
    "weight": "Weight bits",
    "input": "Input bits",
    "output": "Output bits",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-layer bit-width assignments for bandwidth-aware "
            "quantization thresholds and cache sizes."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Model name or path to a YAML file with a list of models.",
    )
    parser.add_argument(
        "--cache_sizes",
        type=float,
        nargs="+",
        default=DEFAULT_CACHE_SIZES,
        help="Cache sizes in millions of elements.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Minimum bit thresholds to plot.",
    )
    parser.add_argument(
        "--bit_kind",
        choices=sorted(BIT_KIND_TO_LABEL),
        default="weight",
        help="Which per-layer bit assignment to plot.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to bandwidth_aware_quant/results/<model>.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used for cache analysis shape tracing.",
    )
    parser.add_argument("--num_banks", type=int, default=16)
    parser.add_argument("--metadata_bits", type=int, default=0)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--file_format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Plot file format.",
    )
    return parser.parse_args()


def resolve_models(model_name):
    if not (model_name.endswith(".yaml") or model_name.endswith(".yml")):
        return [model_name]

    with open(model_name, "r") as f:
        yaml_content = yaml.safe_load(f)

    if isinstance(yaml_content, list):
        models = []
        for item in yaml_content:
            if isinstance(item, dict):
                models.append(item.get("name"))
            else:
                models.append(item)
        return [m for m in models if m]

    if isinstance(yaml_content, dict):
        name = yaml_content.get("name")
        return [name] if name else []

    raise ValueError(f"Could not resolve model list from {model_name}")


def sanitize_for_filename(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return "".join(c if c.isalnum() else "_" for c in text)


def cache_label(cache_size):
    return f"{cache_size:g}M elements"


def model_output_dir(base_output_dir, model_name, model_count):
    if base_output_dir:
        return (
            os.path.join(base_output_dir, model_name)
            if model_count > 1
            else base_output_dir
        )
    return os.path.join(
        PROJECT_ROOT,
        "runspace/experiments/bandwidth_aware_quant/results",
        model_name,
    )


def get_bits_map(bit_kind, input_bits, weight_bits, output_bits):
    if bit_kind == "input":
        return input_bits
    if bit_kind == "output":
        return output_bits
    return weight_bits


def compute_threshold_allocations(sim_layers, thresholds, bit_kind, bandwidth):
    allocations = {}
    for threshold in sorted(set(thresholds)):
        (
            cycles,
            layer_input_bits,
            layer_weight_bits,
            layer_output_bits,
            _layer_residual_input_bits,
        ) = compute_model_runtime(sim_layers, threshold, bandwidth=bandwidth)

        bits_map = get_bits_map(
            bit_kind,
            layer_input_bits,
            layer_weight_bits,
            layer_output_bits,
        )
        allocations[threshold] = {
            "cycles": cycles,
            "bits": [bits_map[layer["name"]] for layer in sim_layers],
        }
    return allocations


def plot_cache_allocations(
    model_name,
    cache_size,
    sim_layers,
    allocations,
    bit_kind,
    output_dir,
    file_format,
    dpi,
):
    thresholds = sorted(allocations)
    if not thresholds:
        return None

    lowest_threshold = thresholds[0]
    x_values = list(range(1, len(sim_layers) + 1))
    plot_order = [t for t in thresholds if t != lowest_threshold] + [lowest_threshold]
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(13, 6.8))
    for idx, threshold in enumerate(plot_order):
        is_front = threshold == lowest_threshold
        color = "black" if is_front else colors(idx % 10)
        ax.plot(
            x_values,
            allocations[threshold]["bits"],
            label=f"threshold {threshold}",
            color=color,
            linestyle="-" if is_front else "--",
            linewidth=2.8 if is_front else 1.6,
            marker="o" if is_front else None,
            markersize=3.2 if is_front else 0,
            alpha=1.0 if is_front else 0.72,
            zorder=20 if is_front else 5,
        )

    ax.set_title(
        f"{model_name}: {BIT_KIND_TO_LABEL[bit_kind]} by layer "
        f"({cache_label(cache_size)})"
    )
    ax.set_xlabel("Layer number")
    ax.set_ylabel(BIT_KIND_TO_LABEL[bit_kind])
    ax.set_xlim(0.5, max(1.5, len(sim_layers) + 0.5))
    ax.set_ylim(1.5, 8.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=18))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Minimum bit threshold", ncol=2)
    fig.tight_layout()

    filename = (
        f"layer_{bit_kind}_bits_cache_{sanitize_for_filename(cache_size)}."
        f"{file_format}"
    )
    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)
    return plot_path


def maybe_fallback_device(device):
    if device != "cuda":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return device
    except Exception:
        pass
    print("CUDA is requested but not available. Falling back to cpu.")
    return "cpu"


def main():
    args = parse_args()
    args.device = maybe_fallback_device(args.device)
    models = resolve_models(args.model_name)
    if not models:
        raise ValueError(f"No models resolved from {args.model_name}")

    thresholds = sorted(set(args.thresholds))
    if any(t < 2 or t > 8 for t in thresholds):
        raise ValueError("Thresholds must be in the supported 2..8 bit range.")

    for model_name in models:
        out_dir = model_output_dir(args.output_dir, model_name, len(models))
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nGenerating layer bit plots for {model_name}")
        print(f"Output directory: {out_dir}")

        for cache_size in args.cache_sizes:
            print(f"  Cache {cache_label(cache_size)}: running cache analysis")
            sim_layers, _cache_sim_map = run_cache_simulation(
                model_name,
                cache_size,
                batch_size=args.batch_size,
                num_banks=args.num_banks,
                metadata_bits=args.metadata_bits,
                device=args.device,
            )
            allocations = compute_threshold_allocations(
                sim_layers,
                thresholds,
                args.bit_kind,
                args.bandwidth,
            )
            plot_path = plot_cache_allocations(
                model_name,
                cache_size,
                sim_layers,
                allocations,
                args.bit_kind,
                out_dir,
                args.file_format,
                args.dpi,
            )
            print(f"  Saved {plot_path}")


if __name__ == "__main__":
    main()
