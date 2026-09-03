#!/usr/bin/env python3
"""Plot per-layer bit allocations for bandwidth-aware quantization.

This script runs the cache analysis and bit-width optimizer only. It does not
materialize quantized weights or run dataset inference.
"""

import argparse
import json
import math
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
from runspace.experiments.asic_cache_simulation.simulate_cache import (
    _compute_layer_cycles,
)


DEFAULT_CACHE_SIZES = [0.0, 2.0, 4.0]
DEFAULT_THRESHOLDS = list(range(2, 9))
BIT_KIND_TO_LABEL = {
    "weight": "Weight bits",
    "input": "Input bits",
    "output": "Output bits",
}
ALL_BIT_KINDS = ("input", "output", "weight")


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
        choices=("all", *ALL_BIT_KINDS),
        default="weight",
        help=(
            "Which per-layer bit assignment to plot. Use 'all' to generate "
            "input, output, and weight plots."
        ),
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
    parser.add_argument(
        "--line_mode",
        choices=("linear", "step"),
        default="linear",
        help=(
            "How to connect adjacent per-layer bit allocations. "
            "'step' holds each value until the next layer."
        ),
    )
    parser.add_argument(
        "--x_axis",
        choices=("layer", "cycles"),
        default="layer",
        help=(
            "Use layer number or cumulative 128-MAC compute cycles on the "
            "x-axis. In cycles mode, each layer's width is proportional to "
            "its runtime."
        ),
    )
    parser.add_argument(
        "--y_axis",
        choices=("bits", "elements_per_second"),
        default="bits",
        help=(
            "Plot bit width or the equivalent element throughput at the "
            "configured bandwidth."
        ),
    )
    parser.add_argument(
        "--list_elements_per_second_above",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "Also save a metadata table of layers whose transfer-budget "
            "equivalent throughput exceeds RATE elements/s."
        ),
    )
    parser.add_argument(
        "--combined_plot",
        action="store_true",
        help=(
            "With --bit_kind all, also save one graph of the canonical shared "
            "per-layer bandwidth-aware bit width. Higher thresholds are nested "
            "inside the lowest-threshold allocation wells."
        ),
    )
    plot_mode = parser.add_mutually_exclusive_group()
    plot_mode.add_argument(
        "--bold_only",
        action="store_true",
        help="Plot only the bold lowest-threshold allocation line.",
    )
    plot_mode.add_argument(
        "--transfer_budget",
        action="store_true",
        help=(
            "Plot the continuous per-layer compute/transfer break-even width "
            "without integer rounding or bit-width bounds. With "
            "--combined_plot, save one budget-only graph."
        ),
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


def plot_x_axis(sim_layers, x_axis):
    """Return per-layer centers, interval edges, label, and plot limits."""
    if x_axis == "cycles":
        edges = [0.0]
        for layer in sim_layers:
            layer_cycles = _compute_layer_cycles(layer)
            edges.append(edges[-1] + layer_cycles)
        centers = [
            (left + right) / 2.0
            for left, right in zip(edges[:-1], edges[1:])
        ]
        upper = edges[-1] if edges[-1] > 0 else 1.0
        return centers, edges, "Cumulative compute cycles", (0.0, upper)

    centers = list(range(1, len(sim_layers) + 1))
    edges = [idx + 0.5 for idx in range(len(sim_layers) + 1)]
    upper = max(1.5, len(sim_layers) + 0.5)
    return centers, edges, "Layer number", (0.5, upper)


def plot_layer_values(
    ax,
    x_values,
    x_edges,
    y_values,
    x_axis,
    line_mode,
    *,
    marker=None,
    markersize=0,
    **kwargs,
):
    """Plot values while preserving true per-layer widths in cycles mode."""
    if x_axis == "cycles" and line_mode == "step":
        ax.stairs(y_values, x_edges, baseline=None, **kwargs)
        if marker is not None:
            ax.plot(
                x_values,
                y_values,
                linestyle="None",
                marker=marker,
                markersize=markersize,
                color=kwargs.get("color"),
                zorder=kwargs.get("zorder"),
                label="_nolegend_",
            )
        return

    ax.plot(
        x_values,
        y_values,
        marker=marker,
        markersize=markersize,
        drawstyle="steps-post" if line_mode == "step" else "default",
        **kwargs,
    )


def convert_bit_widths(values, y_axis, bandwidth):
    """Convert bits/element to billions of elements/s."""
    if y_axis == "bits":
        return values
    return [
        bandwidth * 8.0 / value
        if value is not None and math.isfinite(value) and value > 0
        else math.nan
        for value in values
    ]


def configure_y_axis(ax, y_axis, bits_label):
    if y_axis == "elements_per_second":
        ax.set_ylabel("Elements per second (×10⁹)")
    else:
        ax.set_ylabel(bits_label)


def truncate_element_rates(values, y_axis, bandwidth):
    """Clip sub-1-bit equivalent rates and return their true scaled values."""
    if y_axis != "elements_per_second":
        return values, []
    cap = bandwidth * 8.0
    outliers = [
        (idx, value)
        for idx, value in enumerate(values)
        if math.isfinite(value) and value > cap
    ]
    return [min(value, cap) if math.isfinite(value) else value for value in values], outliers


def annotate_rate_outliers(ax, x_values, outliers, bandwidth):
    """Label clipped values in ×10^9 elements/s units."""
    cap = bandwidth * 8.0
    for idx, actual_value in outliers:
        ax.annotate(
            f"{actual_value:g}",
            xy=(x_values[idx], cap),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=90,
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.8,
                  "ec": "none"},
            zorder=30,
        )


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


def compute_transfer_budget_bits(sim_layers, bandwidth):
    """Return each layer's continuous compute/transfer break-even bit width."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be greater than zero")

    budgets = {}
    prev_stay_on_chip = False
    for idx, layer in enumerate(sim_layers):
        stay_on_chip = layer.get("stay_on_chip", False)
        need_input_transfer = (
            idx == 0
            or not prev_stay_on_chip
            or not layer.get("xin_from_cache", True)
        )
        need_output_transfer = not stay_on_chip

        transferred_elems = [layer.get("weight_elems", 0)]
        if need_input_transfer:
            transferred_elems.append(layer.get("input_elems", 0))
        if need_output_transfer:
            transferred_elems.append(layer.get("output_elems", 0))

        residual_output_elems = layer.get("residual_output_elems", 0)
        residual_uses_main_output = (
            residual_output_elems > 0
            and need_output_transfer
            and residual_output_elems == layer.get("output_elems", 0)
        )
        if residual_output_elems > 0 and not residual_uses_main_output:
            transferred_elems.append(residual_output_elems)

        residual_input_elems = layer.get("residual_input_stream_elems", 0)
        if residual_input_elems > 0:
            transferred_elems.append(residual_input_elems)

        chunks_per_bit = sum(
            math.ceil(elems / 128) for elems in transferred_elems if elems > 0
        )
        budgets[layer["name"]] = (
            _compute_layer_cycles(layer) * bandwidth / (16.0 * chunks_per_bit)
            if chunks_per_bit
            else math.nan
        )
        prev_stay_on_chip = stay_on_chip

    return budgets


def plot_layer_rate_metadata(
    model_name,
    cache_size,
    sim_layers,
    transfer_budget,
    rate_threshold,
    bandwidth,
    output_dir,
    file_format,
    dpi,
):
    """Save an ordered metadata table for layers above an element-rate limit."""
    rows = []
    bits_per_second = bandwidth * 8.0e9
    for layer_idx, layer in enumerate(sim_layers, start=1):
        budget_bits = transfer_budget[layer["name"]]
        if not math.isfinite(budget_bits) or budget_bits <= 0:
            continue
        elements_per_second = bits_per_second / budget_bits
        if elements_per_second <= rate_threshold:
            continue
        rows.append(
            [
                str(layer_idx),
                layer["name"],
                layer.get("type", ""),
                f"{elements_per_second:.3e}",
                f"{budget_bits:.5g}",
                f"{_compute_layer_cycles(layer):,.0f}",
                f"{layer.get('input_elems', 0):,}",
                f"{layer.get('weight_elems', 0):,}",
                f"{layer.get('output_elems', 0):,}",
                "yes" if layer.get("stay_on_chip", False) else "no",
            ]
        )

    columns = [
        "Layer",
        "Name",
        "Type",
        "Elements/s",
        "Budget\n(bits/element)",
        "Compute\ncycles",
        "Input\nelements",
        "Weight\nelements",
        "Output\nelements",
        "On chip",
    ]
    figure_height = max(3.0, 1.8 + 0.42 * max(1, len(rows)))
    fig, ax = plt.subplots(figsize=(18, figure_height))
    ax.axis("off")
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)
        for column_idx in range(len(columns)):
            table[(0, column_idx)].set_text_props(weight="bold")
    else:
        ax.text(
            0.5,
            0.5,
            "No layers exceed the requested element-rate threshold.",
            ha="center",
            va="center",
            fontsize=12,
        )
    ax.set_title(
        f"{model_name}: Layers above {rate_threshold:.3e} elements/s "
        f"({cache_label(cache_size)})\n"
        f"128 MACs @ 1 GHz, {bandwidth:g} GB/s bandwidth — "
        f"{len(rows)} layer{'s' if len(rows) != 1 else ''}",
        pad=18,
    )
    fig.tight_layout()

    threshold_suffix = sanitize_for_filename(f"{rate_threshold:.3g}")
    filename = (
        f"layer_element_rate_above_{threshold_suffix}_cache_"
        f"{sanitize_for_filename(cache_size)}_list.{file_format}"
    )
    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_cache_allocations(
    model_name,
    cache_size,
    sim_layers,
    allocations,
    bit_kind,
    output_dir,
    file_format,
    dpi,
    line_mode="linear",
    x_axis="layer",
    y_axis="bits",
    bandwidth=1.0,
    bold_only=False,
    transfer_budget=None,
):
    thresholds = sorted(allocations)
    if not thresholds:
        return None

    lowest_threshold = thresholds[0]
    x_values, x_edges, x_label, x_limits = plot_x_axis(sim_layers, x_axis)
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(13, 6.8))
    if transfer_budget is not None:
        budget_label = (
            "break-even element throughput"
            if y_axis == "elements_per_second"
            else "continuous transfer budget"
        )
        budget_values = convert_bit_widths(
            [transfer_budget[layer["name"]] for layer in sim_layers],
            y_axis,
            bandwidth,
        )
        budget_values, rate_outliers = truncate_element_rates(
            budget_values, y_axis, bandwidth
        )
        plot_layer_values(
            ax,
            x_values,
            x_edges,
            budget_values,
            x_axis,
            line_mode,
            label=budget_label,
            color="black",
            linestyle="-",
            linewidth=2.8,
            marker="o",
            markersize=3.2,
            zorder=20,
        )
        annotate_rate_outliers(ax, x_values, rate_outliers, bandwidth)
    else:
        plot_order = (
            [lowest_threshold]
            if bold_only
            else [t for t in thresholds if t != lowest_threshold] + [lowest_threshold]
        )
        for idx, threshold in enumerate(plot_order):
            is_front = threshold == lowest_threshold
            color = "black" if is_front else colors(idx % 10)
            plotted_bits = convert_bit_widths(
                allocations[threshold]["bits"], y_axis, bandwidth
            )
            plot_layer_values(
                ax,
                x_values,
                x_edges,
                plotted_bits,
                x_axis,
                line_mode,
                label=f"threshold {threshold}",
                color=color,
                linestyle="-" if is_front else "--",
                linewidth=2.8 if is_front else 1.6,
                marker="o" if is_front else None,
                markersize=3.2 if is_front else 0,
                alpha=1.0 if is_front else 0.72,
                zorder=20 if is_front else 5,
            )

    if transfer_budget is not None:
        metric_title = (
            "Break-even element throughput"
            if y_axis == "elements_per_second"
            else "Continuous transfer budget"
        )
        ax.set_title(
            f"{model_name}: {metric_title} by layer "
            f"({cache_label(cache_size)})"
        )
        configure_y_axis(ax, y_axis, "Transfer budget (bits per element)")
    else:
        metric_title = (
            "Equivalent element throughput"
            if y_axis == "elements_per_second"
            else BIT_KIND_TO_LABEL[bit_kind]
        )
        ax.set_title(
            f"{model_name}: {metric_title} by layer "
            f"({cache_label(cache_size)})"
        )
        configure_y_axis(ax, y_axis, BIT_KIND_TO_LABEL[bit_kind])
    ax.set_xlabel(x_label)
    ax.set_xlim(*x_limits)
    if transfer_budget is None and y_axis == "bits":
        ax.set_ylim(1.5, 8.5)
    elif transfer_budget is not None and y_axis == "elements_per_second":
        ax.set_ylim(0, bandwidth * 8.5)
    elif transfer_budget is not None:
        ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=x_axis == "layer", nbins=18)
    )
    if transfer_budget is None and y_axis == "bits":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    if not bold_only or transfer_budget is not None:
        ax.legend(
            title=None if transfer_budget is not None else "Minimum bit threshold",
            ncol=2,
        )
    fig.tight_layout()

    if transfer_budget is not None:
        mode_suffix = "_transfer_budget"
    else:
        mode_suffix = "_bold_only" if bold_only else ""
    filename = (
        f"layer_{bit_kind}_bits_cache_{sanitize_for_filename(cache_size)}"
        f"{mode_suffix}{'_cycles' if x_axis == 'cycles' else ''}"
        f"{'_elements_per_second' if y_axis == 'elements_per_second' else ''}"
        f".{file_format}"
    )
    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)
    return plot_path


def plot_combined_allocations(
    model_name,
    cache_size,
    sim_layers,
    allocations_by_kind,
    output_dir,
    file_format,
    dpi,
    line_mode="linear",
    x_axis="layer",
    y_axis="bits",
    bold_only=False,
    bandwidth=1.0,
    transfer_budget=None,
):
    """Plot one combined allocation graph, or a budget-only graph.

    Bandwidth-aware quantization materializes the model from the per-layer
    weight-width map. Input/output maps are transfer diagnostics and can remain
    at 8 bits when that component is not transferred, so overlaying them would
    incorrectly imply multiple competing allocations for a layer.
    """
    x_values, x_edges, x_label, x_limits = plot_x_axis(sim_layers, x_axis)
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    if transfer_budget is not None:
        budget_label = (
            "break-even element throughput"
            if y_axis == "elements_per_second"
            else "continuous transfer budget"
        )
        budget_values = convert_bit_widths(
            [transfer_budget[layer["name"]] for layer in sim_layers],
            y_axis,
            bandwidth,
        )
        budget_values, rate_outliers = truncate_element_rates(
            budget_values, y_axis, bandwidth
        )
        plot_layer_values(
            ax,
            x_values,
            x_edges,
            budget_values,
            x_axis,
            line_mode,
            label=budget_label,
            color="black",
            linestyle="-",
            linewidth=2.8,
            marker="o",
            markersize=3.2,
            zorder=20,
        )
        annotate_rate_outliers(ax, x_values, rate_outliers, bandwidth)
        metric_title = (
            "Break-even element throughput"
            if y_axis == "elements_per_second"
            else "Continuous transfer budget"
        )
        ax.set_title(
            f"{model_name}: {metric_title} by layer "
            f"({cache_label(cache_size)})\n"
            f"128 MACs @ 1 GHz, {bandwidth:g} GB/s bandwidth"
        )
        ax.set_xlabel(x_label)
        configure_y_axis(ax, y_axis, "Continuous transfer budget (bits)")
        ax.set_xlim(*x_limits)
        if y_axis == "elements_per_second":
            ax.set_ylim(0, bandwidth * 8.5)
        else:
            ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(
            MaxNLocator(integer=x_axis == "layer", nbins=18)
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()

        filename = (
            f"layer_transfer_budget_cache_{sanitize_for_filename(cache_size)}"
            f"_combined{'_cycles' if x_axis == 'cycles' else ''}"
            f"{'_elements_per_second' if y_axis == 'elements_per_second' else ''}"
            f".{file_format}"
        )
        plot_path = os.path.join(output_dir, filename)
        fig.savefig(plot_path, dpi=dpi)
        plt.close(fig)
        return plot_path

    allocations = allocations_by_kind["weight"]
    thresholds = sorted(allocations)
    lowest_threshold = thresholds[0]
    base_bits = allocations[lowest_threshold]["bits"]
    if not bold_only:
        for idx, threshold in enumerate(thresholds[1:]):
            threshold_bits = [max(bits, threshold) for bits in base_bits]
            plotted_threshold = convert_bit_widths(
                threshold_bits, y_axis, bandwidth
            )
            plot_layer_values(
                ax,
                x_values,
                x_edges,
                plotted_threshold,
                x_axis,
                line_mode,
                label=f"threshold {threshold}",
                color=colors(idx % 10),
                linestyle="--",
                linewidth=1.6,
                alpha=0.72,
                zorder=5,
            )
    plotted_base = convert_bit_widths(base_bits, y_axis, bandwidth)
    plot_layer_values(
        ax,
        x_values,
        x_edges,
        plotted_base,
        x_axis,
        line_mode,
        label=f"threshold {lowest_threshold}",
        color="black",
        linestyle="-",
        linewidth=2.8,
        marker="o",
        markersize=3.2,
        zorder=20,
    )
    if not bold_only:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            title="Minimum bit threshold",
            ncol=2,
        )

    metric_title = (
        "Shared bandwidth-equivalent element throughput"
        if y_axis == "elements_per_second"
        else "Shared bandwidth-aware bit width"
    )
    ax.set_title(
        f"{model_name}: {metric_title} by layer "
        f"({cache_label(cache_size)})\n"
        f"128 MACs @ 1 GHz, {bandwidth:g} GB/s bandwidth"
    )
    ax.set_xlabel(x_label)
    configure_y_axis(ax, y_axis, "Shared bit width")
    ax.set_xlim(*x_limits)
    if y_axis == "bits":
        ax.set_ylim(1.5, 8.5)
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=x_axis == "layer", nbins=18)
    )
    if y_axis == "bits":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    mode_suffix = "_bold_only" if bold_only else ""
    filename = (
        f"layer_shared_bits_cache_{sanitize_for_filename(cache_size)}"
        f"{mode_suffix}{'_cycles' if x_axis == 'cycles' else ''}"
        f"{'_elements_per_second' if y_axis == 'elements_per_second' else ''}"
        f".{file_format}"
    )
    plot_path = os.path.join(output_dir, filename)
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)
    return plot_path


def export_combined_allocations_json(
    model_name,
    cache_size,
    sim_layers,
    allocations_by_kind,
    output_dir,
    bandwidth=1.0,
):
    """Save the exact shared bit-width curves used by the combined plot."""
    allocations = allocations_by_kind["weight"]
    thresholds = sorted(allocations)
    if not thresholds:
        return None

    base_bits = allocations[thresholds[0]]["bits"]
    plotted_bits = {
        str(threshold): [max(bits, threshold) for bits in base_bits]
        for threshold in thresholds
    }
    layers = []
    for layer_index, layer in enumerate(sim_layers, start=1):
        layers.append(
            {
                "index": layer_index,
                "name": layer["name"],
                "type": layer.get("type", ""),
                "shared_bits": {
                    str(threshold): plotted_bits[str(threshold)][layer_index - 1]
                    for threshold in thresholds
                },
            }
        )

    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "cache_size_million_elements": cache_size,
        "bandwidth_bytes_per_cycle": bandwidth,
        "thresholds": thresholds,
        "cycles_by_threshold": {
            str(threshold): allocations[threshold]["cycles"]
            for threshold in thresholds
        },
        "layers": layers,
    }
    filename = (
        f"layer_shared_bits_cache_{sanitize_for_filename(cache_size)}.json"
    )
    json_path = os.path.join(output_dir, filename)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return json_path


def export_transfer_budget_json(
    model_name,
    cache_size,
    sim_layers,
    transfer_budget,
    output_dir,
    bandwidth=1.0,
):
    """Save the continuous transfer-budget values used by the combined plot."""
    layers = []
    for layer_index, layer in enumerate(sim_layers, start=1):
        budget_bits = transfer_budget[layer["name"]]
        finite_budget = (
            budget_bits
            if math.isfinite(budget_bits)
            else None
        )
        layers.append(
            {
                "index": layer_index,
                "name": layer["name"],
                "type": layer.get("type", ""),
                "continuous_transfer_budget_bits_per_element": finite_budget,
                "equivalent_elements_per_second": (
                    bandwidth * 8.0e9 / finite_budget
                    if finite_budget is not None and finite_budget > 0
                    else None
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "cache_size_million_elements": cache_size,
        "bandwidth_bytes_per_cycle": bandwidth,
        "layers": layers,
    }
    filename = (
        f"layer_transfer_budget_cache_{sanitize_for_filename(cache_size)}"
        "_combined.json"
    )
    json_path = os.path.join(output_dir, filename)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return json_path


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
    x_axis="layer",
    y_axis="bits",
    bandwidth=1.0,
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
    x_values, _x_edges, x_label, x_limits = plot_x_axis(sim_layers, x_axis)

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
        plotted_line = convert_bit_widths(bits_line, y_axis, bandwidth)
        ax.plot(
            x_values, plotted_line,
            color=context_cmap(0.25 + 0.45 * (li / max(1, len(level_keys) - 1))),
            linestyle="--", linewidth=1.0, alpha=0.5, zorder=2,
        )

    # Chosen-format markers at the deepest level's bit-width.
    deepest_bits = per_level[deepest].get("layer_weight_bits", {})
    for fmt in present_formats:
        xs = [x_values[i] for i, n in enumerate(names) if layer_formats.get(n) == fmt]
        ys = convert_bit_widths(
            [deepest_bits.get(n, 0) for n in names if layer_formats.get(n) == fmt],
            y_axis,
            bandwidth,
        )
        ax.scatter(xs, ys, color=fmt_to_color[fmt], label=fmt, s=42,
                   edgecolors="black", linewidths=0.6, zorder=10)

    ax.set_title(
        f"{model_name}: descent-chosen weight format by layer "
        f"({cache_label(cache_size)}, deepest level b={deepest})"
    )
    ax.set_xlabel(x_label)
    configure_y_axis(ax, y_axis, "Weight bits (deepest level)")
    ax.set_xlim(*x_limits)
    if y_axis == "bits":
        ax.set_ylim(1.5, 8.5)
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=x_axis == "layer", nbins=18)
    )
    if y_axis == "bits":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Chosen format", ncol=2, fontsize=8)
    fig.tight_layout()

    filename = (
        f"layer_chosen_formats_cache_{sanitize_for_filename(cache_size)}"
        f"{'_cycles' if x_axis == 'cycles' else ''}"
        f"{'_elements_per_second' if y_axis == 'elements_per_second' else ''}"
        f".{file_format}"
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
    if args.combined_plot and args.bit_kind != "all":
        raise ValueError("--combined_plot requires --bit_kind all.")
    if (
        args.list_elements_per_second_above is not None
        and args.list_elements_per_second_above <= 0
    ):
        raise ValueError("--list_elements_per_second_above must be positive.")
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
            transfer_budget = (
                compute_transfer_budget_bits(sim_layers, args.bandwidth)
                if (
                    args.transfer_budget
                    or args.list_elements_per_second_above is not None
                )
                else None
            )
            bit_kinds = (
                ALL_BIT_KINDS if args.bit_kind == "all" else (args.bit_kind,)
            )
            allocations_by_kind = {}
            combined_budget_only = args.combined_plot and args.transfer_budget
            if not combined_budget_only:
                for bit_kind in bit_kinds:
                    allocations = compute_threshold_allocations(
                        sim_layers,
                        thresholds,
                        bit_kind,
                        args.bandwidth,
                    )
                    allocations_by_kind[bit_kind] = allocations
                    plot_path = plot_cache_allocations(
                        model_name,
                        cache_size,
                        sim_layers,
                        allocations,
                        bit_kind,
                        out_dir,
                        args.file_format,
                        args.dpi,
                        line_mode=args.line_mode,
                        x_axis=args.x_axis,
                        y_axis=args.y_axis,
                        bandwidth=args.bandwidth,
                        bold_only=args.bold_only,
                        transfer_budget=(
                            transfer_budget if args.transfer_budget else None
                        ),
                    )
                    print(f"  Saved {plot_path}")

            if args.combined_plot:
                combined_path = plot_combined_allocations(
                    model_name,
                    cache_size,
                    sim_layers,
                    allocations_by_kind,
                    out_dir,
                    args.file_format,
                    args.dpi,
                    line_mode=args.line_mode,
                    x_axis=args.x_axis,
                    y_axis=args.y_axis,
                    bold_only=args.bold_only,
                    bandwidth=args.bandwidth,
                    transfer_budget=(
                        transfer_budget if args.transfer_budget else None
                    ),
                )
                print(f"  Saved {combined_path}")
                if args.transfer_budget:
                    transfer_budget_json_path = export_transfer_budget_json(
                        model_name,
                        cache_size,
                        sim_layers,
                        transfer_budget,
                        out_dir,
                        bandwidth=args.bandwidth,
                    )
                    print(f"  Saved {transfer_budget_json_path}")
                else:
                    combined_json_path = export_combined_allocations_json(
                        model_name,
                        cache_size,
                        sim_layers,
                        allocations_by_kind,
                        out_dir,
                        bandwidth=args.bandwidth,
                    )
                    print(f"  Saved {combined_json_path}")

            if args.list_elements_per_second_above is not None:
                list_path = plot_layer_rate_metadata(
                    model_name,
                    cache_size,
                    sim_layers,
                    transfer_budget,
                    args.list_elements_per_second_above,
                    args.bandwidth,
                    out_dir,
                    args.file_format,
                    args.dpi,
                )
                print(f"  Saved {list_path}")

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
                        x_axis=args.x_axis,
                        y_axis=args.y_axis,
                        bandwidth=args.bandwidth,
                    )
                    if chosen_path:
                        print(f"  Saved {chosen_path}")
                else:
                    print(f"  No descent data for cache {cache_label(cache_size)}; skipping companion plot.")


if __name__ == "__main__":
    main()
