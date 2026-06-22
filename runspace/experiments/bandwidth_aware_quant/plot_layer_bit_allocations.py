#!/usr/bin/env python3
"""Plot per-layer bit allocations for bandwidth-aware quantization.

This script runs the cache analysis and bit-width optimizer only. It does not
materialize quantized weights or run dataset inference.
"""

import argparse
import json
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
            _layer_need_input_transfer,
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


RESULTS_JSON_NAME = "bandwidth_aware_quant_results.json"


def load_descent_block(out_dir, model_name):
    """Locate a bandwidth_aware results JSON that carries a `descent` block.

    Checks the resolved output dir first, then the default results_descent dir.
    Returns (descent_dict, results_dir) or (None, None) if none found.
    """
    candidates = [out_dir]
    descent_default = os.path.join(
        PROJECT_ROOT,
        "runspace/experiments/bandwidth_aware_quant/results_descent",
        model_name,
    )
    if descent_default not in candidates:
        candidates.append(descent_default)

    for results_dir in candidates:
        json_path = os.path.join(results_dir, RESULTS_JSON_NAME)
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  Could not read {json_path}: {exc}")
            continue
        descent = data.get("descent")
        if descent:
            return descent, results_dir
    return None, None


def plot_cache_chosen_formats(
    model_name,
    cache_size,
    sim_layers,
    descent_cs,
    output_dir,
    file_format,
    dpi,
):
    """Companion plot: per-layer descent-chosen weight format for one cache size.

    Uses the deepest descent level (lowest b) whose `layer_formats` is the full
    per-layer assignment. Each layer is plotted at its allocated bit-width (y),
    coloured by its chosen format (categorical legend); faint context lines show
    the bit-width across all descent levels.
    """
    per_level = descent_cs.get("per_level", {})
    if not per_level:
        return None

    level_keys = sorted(per_level, key=lambda k: int(k))
    deepest = level_keys[0]
    layer_formats = per_level[deepest].get("layer_formats", {})
    if not layer_formats:
        return None

    names = [layer["name"] for layer in sim_layers]
    x_values = list(range(1, len(names) + 1))

    # Categorical colour map over the formats actually chosen.
    present_formats = sorted({layer_formats.get(n) for n in names if layer_formats.get(n)})
    cmap = plt.get_cmap("tab20")
    fmt_to_color = {fmt: cmap(i % 20) for i, fmt in enumerate(present_formats)}

    fig, ax = plt.subplots(figsize=(13, 6.8))

    # Faint context: bit-width across all descent levels.
    context_cmap = plt.get_cmap("Greys")
    for li, lvl in enumerate(level_keys):
        lwb = per_level[lvl].get("layer_weight_bits", {})
        bits_line = [lwb.get(n) for n in names]
        if any(b is None for b in bits_line):
            continue
        ax.plot(
            x_values, bits_line,
            color=context_cmap(0.25 + 0.45 * (li / max(1, len(level_keys) - 1))),
            linestyle="--", linewidth=1.0, alpha=0.5, zorder=2,
        )

    # Chosen-format markers at the deepest level's bit-width.
    deepest_bits = per_level[deepest].get("layer_weight_bits", {})
    for fmt in present_formats:
        xs = [x_values[i] for i, n in enumerate(names) if layer_formats.get(n) == fmt]
        ys = [deepest_bits.get(n, 0) for n in names if layer_formats.get(n) == fmt]
        ax.scatter(xs, ys, color=fmt_to_color[fmt], label=fmt, s=42,
                   edgecolors="black", linewidths=0.6, zorder=10)

    ax.set_title(
        f"{model_name}: descent-chosen weight format by layer "
        f"({cache_label(cache_size)}, deepest level b={deepest})"
    )
    ax.set_xlabel("Layer number")
    ax.set_ylabel("Weight bits (deepest level)")
    ax.set_xlim(0.5, max(1.5, len(names) + 0.5))
    ax.set_ylim(1.5, 8.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=18))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Chosen format", ncol=2, fontsize=8)
    fig.tight_layout()

    filename = (
        f"layer_chosen_formats_cache_{sanitize_for_filename(cache_size)}."
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

        # Optional descent companion plots — only if a results JSON with a
        # `descent` block exists (produced by bandwidth_aware_quant.py --descent).
        descent, descent_dir = load_descent_block(out_dir, model_name)
        if descent is None:
            print("  No descent block found; skipping chosen-format companion plots.")

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

            if descent is not None:
                descent_cs = descent.get(f"{cache_size:g}") or descent.get(str(cache_size))
                if descent_cs:
                    chosen_path = plot_cache_chosen_formats(
                        model_name,
                        cache_size,
                        sim_layers,
                        descent_cs,
                        descent_dir,
                        args.file_format,
                        args.dpi,
                    )
                    if chosen_path:
                        print(f"  Saved {chosen_path}")
                else:
                    print(f"  No descent data for cache {cache_label(cache_size)}; skipping companion plot.")


if __name__ == "__main__":
    main()
