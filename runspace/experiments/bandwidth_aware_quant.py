
import os

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import sys
import csv
import json
import math
import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.core.report_aggregator import ReportAggregator

# Import weight quantization helpers
from runspace.experiments.find_optimal_weight_quant import (
    get_quantized_tensor_sim,
    calculate_error,
    create_quantized_state_dict,
)
from runspace.experiments.utils.plotting import (
    plot_error_histograms,
    plot_error_boxplot,
    plot_layer_error_comparison,
    plot_accuracy_comparison,
    get_format_bits,
    sort_formats,
    get_chunked_tensor,
)


# ============================================================
# RULES — Fill this section with your classification rules
# ============================================================
# Each rule is a callable: (layer_stats: dict) -> bool
# If ANY rule returns True, the layer is a "lower precision candidate"
# and will be tested with all LOWER_PRECISION_FORMATS.
#
# Available fields in layer_stats:
#   layer_name       : str  — full dotted name (e.g. "layer4.0.conv1")
#   layer_type       : str  — fused type string (e.g. "Conv2d + BatchNorm2d + ReLU")
#   num_weights      : str  — weight count (may be "X + Y" if fused)
#   num_inputs       : int  — total input elements
#   num_outputs      : int  — total output elements
#   num_macs         : int  — multiply-accumulate operations
#   cycles           : int  — compute cycles (ceil(MACs / 128))
#   bw_weights_epc   : float — weight bandwidth (elements/cycle)
#   bw_inputs_epc    : float — input bandwidth (elements/cycle)
#   bw_outputs_epc   : float — output bandwidth (elements/cycle)
#   bw_total_elements_per_cycle : float — total bandwidth (elements/cycle)
#   is_bandwidth_limited : bool — True if data transfer time > compute time @ 1 B/C
#
# Example rules:
#   lambda s: s['is_bandwidth_limited']
#   lambda s: s['bw_total_elements_per_cycle'] > 2.0
#   lambda s: s['bw_weights_epc'] > 1.5

RULES = [
    # Rule 1: Weight transfer dominates — weights are more than half of total data time
    #         2 * t_weights > t_data
    lambda s: 2 * s['t_weights'] > s['t_data'] if s['t_compute'] < s['t_data'] else False,

    # Rule 2: Weight transfer alone exceeds the compute slack
    #         t_weights > (t_compute - t_data)  [only meaningful when compute > data]
    lambda s: s['t_weights'] > 2*((s['t_data'] - s['t_compute'])) if s['t_compute'] < s['t_data'] else False,
]


# Lower precision formats to evaluate on candidate layers
LOWER_PRECISION_FORMATS = [
    'fp6_e1m4', 'fp6_e2m3', 'fp6_e3m2', 'fp6_e4m1', 'fp6_e5m0',
]

# Default format for non-candidate layers
DEFAULT_FORMAT = 'fp8_e1m6'


# ============================================================
# Helpers
# ============================================================

def parse_val(v):
    """Parse a value that might be a string expression 'A + B + C' into a float."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return sum(float(x) for x in v.split('+') if x.strip())
    return 0.0


def get_args():
    parser = argparse.ArgumentParser(
        description="Bandwidth-aware mixed-precision quantization experiment"
    )
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: runspace/experiments/bandwidth_aware_quant)")
    parser.add_argument("--bandwidth_dir", type=str, default=None,
                        help="Path to existing bandwidth_analysis output dir (default: runspace/experiments/bandwidth_analysis)")
    parser.add_argument("--models_config", type=str, default=None,
                        help="Path to YAML file with list of models")

    # Dataset / Eval
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=64, help="Num workers")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit batches (-1 = all)")
    parser.add_argument("--run_eval", action="store_true", help="Run full evaluation")
    parser.add_argument("--weight_chunk_size", type=int, default=128, help="Weight chunk size")
    parser.add_argument("--per_chunk_format", action="store_true",
                        help="Enable per-chunk format selection for lower precision layers")

    # Metrics
    parser.add_argument("--metrics", type=str, default="l1,mse",
                        help="Comma-separated metrics: l1, mse, sqnr, cosine")

    # Force re-run bandwidth analysis even if CSV exists
    parser.add_argument("--force_bandwidth", action="store_true",
                        help="Force re-run bandwidth analysis")

    return parser.parse_args()


# ============================================================
# Step 1: Load / Run Bandwidth Analysis
# ============================================================

def load_bandwidth_stats(csv_path):
    """Load bandwidth stats from existing CSV file."""
    stats = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            # Strip whitespace/control chars from keys (handles \r in last column header)
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None}

            # Convert numeric fields
            for key in ['num_inputs', 'num_outputs', 'num_macs', 'cycles',
                        'input_data_elements', 'output_data_elements',
                        'bw_total_elements_per_cycle', 'batch_size']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass

            # Parse component bandwidths if present
            for key in ['bw_weights_epc', 'bw_inputs_epc', 'bw_outputs_epc']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = 0.0
                else:
                    row[key] = 0.0

            stats.append(row)
    return stats


def enrich_bandwidth_stats(stats):
    """Add derived fields like is_bandwidth_limited to stats."""
    TARGET_BW = 1.0  # Bytes/Cycle
    BYTES_PER_ELEM = 1.0

    for layer in stats:
        w_count = parse_val(layer.get('num_weights', 0))
        i_count = float(layer.get('num_inputs', 0))
        o_count = float(layer.get('num_outputs', 0))

        t_w = (w_count * BYTES_PER_ELEM) / TARGET_BW
        t_in = (i_count * BYTES_PER_ELEM) / TARGET_BW
        t_out = (o_count * BYTES_PER_ELEM) / TARGET_BW
        t_data = t_in + t_w + t_out
        t_comp = float(layer.get('cycles', 0))

        layer['is_bandwidth_limited'] = t_data > t_comp
        layer['t_data'] = t_data
        layer['t_compute'] = t_comp
        layer['t_weights'] = t_w
        layer['t_inputs'] = t_in
        layer['t_outputs'] = t_out

    return stats


def compute_adjusted_bandwidth(bandwidth_stats, classifications):
    """
    Recompute bandwidth stats accounting for the actual quantized format.
    fp6 layers have 6/8 = 0.75x weight data volume, fp8 layers remain 1x.
    Inputs/outputs stay at 8 bits (1 byte/elem).
    """
    BITS_FP6 = 6
    BITS_FP8 = 8

    for layer in bandwidth_stats:
        layer_name = layer.get('layer_name', '')
        classification = classifications.get(layer_name, 'default')

        if classification == 'lower':
            weight_scale = BITS_FP6 / BITS_FP8  # 0.75
        else:
            weight_scale = BITS_FP8 / BITS_FP8  # 1.0

        t_w_orig = layer.get('t_weights', 0)
        t_w_adjusted = t_w_orig * weight_scale
        t_in = layer.get('t_inputs', 0)
        t_out = layer.get('t_outputs', 0)
        t_data_adjusted = t_in + t_w_adjusted + t_out
        t_comp = layer.get('t_compute', 0)

        layer['weight_scale'] = weight_scale
        layer['t_weights_adjusted'] = t_w_adjusted
        layer['t_data_adjusted'] = t_data_adjusted
        layer['is_bw_limited_adjusted'] = t_data_adjusted > t_comp
        # Per-layer runtime = max(compute, data transfer)
        layer['t_runtime_orig'] = max(t_comp, layer.get('t_data', 0))
        layer['t_runtime_adjusted'] = max(t_comp, t_data_adjusted)

    return bandwidth_stats


def get_bandwidth_stats(model_name, args, device):
    """
    Get bandwidth stats for a model. Tries to load from existing CSV first,
    otherwise runs the bandwidth analysis.
    """
    bw_dir = args.bandwidth_dir or os.path.join(
        PROJECT_ROOT, "runspace/experiments/bandwidth_analysis"
    )
    csv_path = os.path.join(bw_dir, model_name, "layer_bandwidth_stats.csv")

    if os.path.exists(csv_path) and not args.force_bandwidth:
        print(f"Loading existing bandwidth stats from {csv_path}")
        stats = load_bandwidth_stats(csv_path)
        stats = enrich_bandwidth_stats(stats)
        return stats

    # Need to run bandwidth analysis
    print(f"No existing bandwidth stats found. Running bandwidth analysis for {model_name}...")
    import subprocess
    bw_script = os.path.join(PROJECT_ROOT, "runspace/experiments/layer_bandwidth_analysis.py")
    cmd = [
        sys.executable, bw_script,
        "--model_name", model_name,
        "--output_dir", bw_dir,
        "--dataset_name", args.dataset_name,
        "--dataset_path", args.dataset_path,
        "--limit_batches", "1",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Bandwidth analysis failed:\n{result.stderr}")
        raise RuntimeError("Bandwidth analysis failed")
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    stats = load_bandwidth_stats(csv_path)
    stats = enrich_bandwidth_stats(stats)
    return stats


# ============================================================
# Step 2: Classify Layers Using Rules
# ============================================================

def classify_layers(bandwidth_stats, rules):
    """
    Apply rules to bandwidth stats and classify each layer.

    Returns:
        classifications: dict mapping layer_name -> 'lower' | 'default'
        summary: dict with counts
    """
    classifications = {}
    reasons = {}

    for layer in bandwidth_stats:
        layer_name = layer.get('layer_name', '')
        is_candidate = False
        matched_rules = []

        for i, rule in enumerate(rules):
            try:
                if rule(layer):
                    is_candidate = True
                    matched_rules.append(f"rule_{i}")
            except Exception as e:
                print(f"Warning: Rule {i} failed on layer {layer_name}: {e}")

        if is_candidate:
            classifications[layer_name] = 'lower'
            reasons[layer_name] = matched_rules
        else:
            classifications[layer_name] = 'default'

    n_lower = sum(1 for v in classifications.values() if v == 'lower')
    n_default = sum(1 for v in classifications.values() if v == 'default')
    print(f"\nLayer Classification Summary:")
    print(f"  Lower precision candidates: {n_lower}")
    print(f"  Default ({DEFAULT_FORMAT}):  {n_default}")
    if reasons:
        print(f"\nLower precision layers:")
        for name, rules_hit in reasons.items():
            print(f"  {name} — matched: {', '.join(rules_hit)}")

    return classifications, {'n_lower': n_lower, 'n_default': n_default}


# ============================================================
# Step 3: Run Mixed-Precision Weight Analysis
# ============================================================

def build_layer_name_mapping(bandwidth_stats, model):
    """
    Build a mapping from bandwidth layer names (which may be fused like
    'layer1.0.conv1 + layer1.0.bn1 + layer1.0.relu') to model module names.

    Returns a dict: module_name -> classification
    """
    # Collect model module names that have weights
    supported_ops = tuple(OpRegistry.get_supported_ops().keys())
    weight_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, supported_ops) and hasattr(module, 'weight') and module.weight is not None:
            weight_modules[name] = module

    return weight_modules


def run_mixed_precision_analysis(model, classifications, bandwidth_stats, args, device, output_dir, metrics):
    """
    Run weight quantization analysis with mixed precision:
    - Lower precision layers: test all LOWER_PRECISION_FORMATS
    - Default layers: only test DEFAULT_FORMAT
    """
    supported_ops = tuple(OpRegistry.get_supported_ops().keys())

    # Build mapping from bandwidth fused names to actual module names
    # Bandwidth stats use fused names like "layer1.0.conv1 + layer1.0.bn1"
    # We need to map each model module to its classification
    module_classifications = {}
    for bw_layer in bandwidth_stats:
        bw_name = bw_layer.get('layer_name', '')
        classification = classifications.get(bw_name, 'default')

        # Extract individual module names from fused name
        parts = [p.strip() for p in bw_name.split('+')]
        for part in parts:
            module_classifications[part] = classification

    layer_results_map = {}

    print(f"\nRunning mixed-precision weight analysis...")
    print(f"  Lower precision formats: {LOWER_PRECISION_FORMATS}")
    print(f"  Default format: {DEFAULT_FORMAT}")

    for name, module in tqdm(model.named_modules(), desc="Analyzing Layers"):
        if not isinstance(module, supported_ops):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        w = module.weight.data
        layer_class = module_classifications.get(name, 'default')

        # Select formats to test for this layer
        if layer_class == 'lower':
            qt_options = LOWER_PRECISION_FORMATS.copy()
        else:
            qt_options = [DEFAULT_FORMAT]

        # Initialize record
        record = {
            'layer': name,
            'shape': str(tuple(w.shape)),
            'max_val': 0.0,
            'metrics': {},
            'classification': layer_class,
            'formats_tested': qt_options,
            'numel': w.numel(),
        }

        max_val_global = 0.0

        for m in metrics:
            record['metrics'][m] = {}

        for q_type in qt_options:
            try:
                w_deq, mv = get_quantized_tensor_sim(w, q_type, chunk_size=args.weight_chunk_size)
                max_val_global = max(max_val_global, mv)

                for m in metrics:
                    err = calculate_error(w, w_deq, m)
                    record['metrics'][m][q_type] = err
            except Exception as e:
                print(f"  Warning: {q_type} failed on {name}: {e}")
                for m in metrics:
                    record['metrics'][m][q_type] = float('inf')

        # Per-chunk format selection (only for lower-precision candidate layers)
        if layer_class == 'lower' and args.per_chunk_format and args.weight_chunk_size:
            chunk_winners = {}  # metric -> list of best format per chunk
            chunk_win_counts = {}  # metric -> {fmt: count}

            w_chunked, orig_shape, pad_len = get_chunked_tensor(w, args.weight_chunk_size)
            batch_dim, num_chunks, chunk_sz = w_chunked.shape
            w_flat = w_chunked.reshape(-1, chunk_sz)  # [total_chunks, chunk_size]
            total_chunks = w_flat.shape[0]

            # Compute per-chunk errors for each format and metric
            chunk_errors = {}  # q_type -> metric -> [total_chunks] tensor
            for q_type in qt_options:
                chunk_errors[q_type] = {}
                try:
                    dq_flat, _ = get_quantized_tensor_sim(w_flat, q_type)
                    for m_name in metrics:
                        if m_name == 'l1':
                            errs = (w_flat - dq_flat).abs().mean(dim=1)
                        elif m_name == 'mse':
                            errs = ((w_flat - dq_flat) ** 2).mean(dim=1)
                        elif m_name == 'cosine':
                            cos = torch.nn.functional.cosine_similarity(w_flat, dq_flat, dim=1)
                            errs = 1.0 - cos
                        elif m_name == 'sqnr':
                            noise = w_flat - dq_flat
                            sig_pow = (w_flat ** 2).mean(dim=1)
                            noi_pow = (noise ** 2).mean(dim=1).clamp(min=1e-20)
                            errs = -10 * torch.log10(sig_pow / noi_pow)  # negative = we minimize
                        else:
                            errs = (w_flat - dq_flat).abs().mean(dim=1)
                        chunk_errors[q_type][m_name] = errs.cpu()
                except Exception as e:
                    for m_name in metrics:
                        chunk_errors[q_type][m_name] = torch.full((total_chunks,), float('inf'))

            # Pick best format per chunk for each metric
            for m_name in metrics:
                winners = []
                win_counts = {fmt: 0 for fmt in qt_options}
                for c_idx in range(total_chunks):
                    best_fmt = qt_options[0]
                    best_err = float('inf')
                    for q_type in qt_options:
                        e = chunk_errors[q_type][m_name][c_idx].item()
                        if e < best_err:
                            best_err = e
                            best_fmt = q_type
                        elif e == best_err:
                            if get_format_bits(q_type) < get_format_bits(best_fmt):
                                best_fmt = q_type
                    winners.append(best_fmt)
                    win_counts[best_fmt] += 1
                chunk_winners[m_name] = winners
                chunk_win_counts[m_name] = win_counts

            record['chunk_winners'] = chunk_winners
            record['chunk_win_counts'] = chunk_win_counts

        record['max_val'] = max_val_global
        layer_results_map[name] = record

    return layer_results_map


def save_results(layer_results_map, metrics, output_dir, model):
    """Save per-metric CSVs and find best formats."""
    layer_order = []
    for name, module in model.named_modules():
        if name in layer_results_map:
            layer_order.append(name)

    layer_results = [layer_results_map[n] for n in layer_order]

    # Collect all unique formats tested
    all_formats = set()
    for record in layer_results:
        all_formats.update(record['formats_tested'])
    all_formats = sorted(all_formats)

    best_formats_per_metric = {}  # metric -> {layer_name: best_fmt}

    for m in metrics:
        metric_dir = os.path.join(output_dir, m)
        os.makedirs(metric_dir, exist_ok=True)

        csv_rows = []
        plot_data = []
        layer_winners = {}

        for record in layer_results:
            name = record['layer']
            errs = record['metrics'][m]
            classification = record['classification']

            best_error = float('inf')
            best_type = None

            for q_type, val in errs.items():
                if val < best_error:
                    best_error = val
                    best_type = q_type
                elif val == best_error and best_type:
                    if get_format_bits(q_type) < get_format_bits(best_type):
                        best_type = q_type

            if best_type:
                layer_winners[name] = best_type

            row = {
                'layer': name,
                'shape': record['shape'],
                'max_val': record['max_val'],
                'classification': classification,
                'best_type': best_type,
                'best_error': best_error,
            }
            for qt in record['formats_tested']:
                row[f"{qt}_error"] = errs.get(qt, "")
            csv_rows.append(row)
            plot_data.append({'errors': errs, 'best_error': best_error})

        best_formats_per_metric[m] = layer_winners

        # Save CSV
        csv_path = os.path.join(metric_dir, "layer_errors.csv")
        if csv_rows:
            csv_headers = list(csv_rows[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"Saved {m} results to {csv_path}")

        # Plots
        try:
            plot_error_histograms(plot_data, all_formats, metric_dir, m)
            plot_error_boxplot(plot_data, all_formats, metric_dir, m)
            plot_layer_error_comparison(layer_results, all_formats, metric_dir, m)
        except Exception as e:
            print(f"Warning: Plotting failed for {m}: {e}")

    return best_formats_per_metric


# ============================================================
# Step 4: Generate Runtime Bottleneck Plot
# ============================================================

def generate_runtime_bottleneck(bandwidth_stats, classifications, output_dir, model_name):
    """
    Generate a runtime bottleneck plot with layer classifications overlaid.
    Shows both original and adjusted weight transfer times.
    """
    TARGET_BW = 1.0

    t_inputs = []
    t_weights_orig = []
    t_weights_adj = []
    t_outputs = []
    t_computes = []
    layer_labels = []
    label_colors_orig = []
    label_colors_adj = []
    precision_markers = []

    for layer in bandwidth_stats:
        t_in = layer.get('t_inputs', 0)
        t_w_orig = layer.get('t_weights', 0)
        t_w_adj = layer.get('t_weights_adjusted', t_w_orig)
        t_out = layer.get('t_outputs', 0)
        t_comp = layer.get('t_compute', 0)

        t_inputs.append(t_in)
        t_weights_orig.append(t_w_orig)
        t_weights_adj.append(t_w_adj)
        t_outputs.append(t_out)
        t_computes.append(t_comp)

        layer_name = layer.get('layer_name', '')
        layer_type = layer.get('layer_type', '')
        layer_labels.append(layer_type)

        # Original bottleneck color
        t_data_orig = t_in + t_w_orig + t_out
        if t_data_orig > t_comp:
            label_colors_orig.append('red')
        else:
            label_colors_orig.append('green')

        # Adjusted bottleneck color
        t_data_adj = t_in + t_w_adj + t_out
        if t_data_adj > t_comp:
            label_colors_adj.append('red')
        else:
            label_colors_adj.append('green')

        layer_class = classifications.get(layer_name, 'default')
        precision_markers.append(layer_class)

    if not layer_labels:
        print("No layers to plot for runtime bottleneck.")
        return

    # --- Plot 1: Original vs Adjusted comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, max(8, len(layer_labels) * 0.4)),
                                    sharey=True)

    y_pos = np.arange(len(layer_labels))
    height = 0.4

    # Left: Original (fp8 for all weights)
    ax1.barh(y_pos + height / 2, t_inputs, height, label='Input Time', color='skyblue')
    ax1.barh(y_pos + height / 2, t_weights_orig, height, left=t_inputs,
             label='Weight Time (fp8)', color='orange')
    left_orig = [i + w for i, w in zip(t_inputs, t_weights_orig)]
    ax1.barh(y_pos + height / 2, t_outputs, height, left=left_orig,
             label='Output Time', color='lightgreen')
    ax1.barh(y_pos - height / 2, t_computes, height, label='Compute Time', color='gray', alpha=0.7)

    ax1.set_xlabel(f'Duration (Cycles) @ {TARGET_BW} B/Cycle')
    ax1.set_title(f'{model_name} — Original (all fp8)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([l[:30] for l in layer_labels], fontsize=8)
    for i, label in enumerate(ax1.get_yticklabels()):
        label.set_color(label_colors_orig[i])
        label.set_fontweight('bold')
    ax1.legend(fontsize=8)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    ax1.invert_yaxis()

    # Right: Adjusted (fp6 for lower-precision layers)
    ax2.barh(y_pos + height / 2, t_inputs, height, label='Input Time', color='skyblue')
    ax2.barh(y_pos + height / 2, t_weights_adj, height, left=t_inputs,
             label='Weight Time (adjusted)', color='darkorange')
    left_adj = [i + w for i, w in zip(t_inputs, t_weights_adj)]
    ax2.barh(y_pos + height / 2, t_outputs, height, left=left_adj,
             label='Output Time', color='lightgreen')
    ax2.barh(y_pos - height / 2, t_computes, height, label='Compute Time', color='gray', alpha=0.7)

    # Mark lower-precision layers
    for i, marker in enumerate(precision_markers):
        if marker == 'lower':
            ax2.plot(-max(t_computes) * 0.02, y_pos[i], marker='<', color='blue',
                     markersize=8, clip_on=False)

    ax2.set_xlabel(f'Duration (Cycles) @ {TARGET_BW} B/Cycle')
    ax2.set_title(f'{model_name} — Mixed Precision (fp6/fp8)')
    for i, label in enumerate(ax2.get_yticklabels()):
        label.set_color(label_colors_adj[i])
        label.set_fontweight('bold')

    # Precision annotations
    for i, marker in enumerate(precision_markers):
        fmt_label = "fp6" if marker == 'lower' else "fp8"
        color = 'blue' if marker == 'lower' else 'gray'
        total_time_adj = max(t_computes[i], t_inputs[i] + t_weights_adj[i] + t_outputs[i])
        ax2.text(total_time_adj * 1.02, y_pos[i], fmt_label,
                 va='center', fontsize=7, color=color, fontweight='bold')

    ax2.legend(fontsize=8)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    plt.suptitle(f'{model_name} Runtime Bottleneck: Original vs Mixed Precision\n'
                 f'(Red=BW Limited, Green=Compute Limited, <=fp6 Candidate)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_bottleneck.png'), dpi=150)
    plt.close()
    print(f"Saved runtime bottleneck plot to {output_dir}/runtime_bottleneck.png")


# ============================================================
# Step 5: Build Evaluation Configs & Run
# ============================================================

def run_evaluation(model, layer_results_map, best_formats, args, output_dir, metrics):
    """Build quantized models and run evaluation."""
    configs_to_run = []
    total_errors = {}

    dataset_base = {
        'name': args.dataset_name, 'path': args.dataset_path,
        'batch_size': args.batch_size, 'num_workers': args.num_workers,
    }
    eval_config = {'mode': 'evaluate'}
    if args.limit_batches > 0:
        eval_config['max_batches'] = args.limit_batches

    # 1. Reference FP32
    ref_config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'evaluation': eval_config,
        'dataset': dataset_base,
        'output_name': 'ref_fp32',
    }
    configs_to_run.append(ref_config)

    # 2. Baseline: all layers at DEFAULT_FORMAT
    baseline_config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {
            'type': 'generic',
            'quantized_ops': ['-1'],
            'input_quantization': False,
        },
        'quantization': {
            'format': DEFAULT_FORMAT,
            'weight_mode': 'chunk',
            'weight_chunk_size': args.weight_chunk_size,
        },
        'evaluation': eval_config,
        'dataset': dataset_base,
        'output_name': f'baseline_{DEFAULT_FORMAT}',
    }
    configs_to_run.append(baseline_config)

    # 2b. Baselines: one per LOWER_PRECISION_FORMAT (all layers uniform)
    for lp_fmt in LOWER_PRECISION_FORMATS:
        lp_config = {
            'model': {'name': args.model_name, 'weights': args.weights},
            'adapter': {
                'type': 'generic',
                'quantized_ops': ['-1'],
                'input_quantization': False,
            },
            'quantization': {
                'format': lp_fmt,
                'weight_mode': 'chunk',
                'weight_chunk_size': args.weight_chunk_size,
            },
            'evaluation': eval_config,
            'dataset': dataset_base,
            'output_name': f'baseline_{lp_fmt}',
        }
        configs_to_run.append(lp_config)

    base_adapter_config = {
        'type': 'generic',
        'input_quantization': False,
        'quantized_ops': [],
        'quantize_first_layer': False,
    }

    # 3. Mixed-precision configs (one per metric)
    for m in metrics:
        if m not in best_formats:
            continue

        metric_dir = os.path.join(output_dir, m)
        os.makedirs(metric_dir, exist_ok=True)

        # --- 3a. Layer-wise optimized ---
        q_state_dict, q_map = create_quantized_state_dict(
            model, layer_results_map, args, m, use_chunking=False
        )

        q_weights_path = os.path.join(metric_dir, "quantized_weights_layer.pt")
        torch.save(q_state_dict, q_weights_path)

        q_map_path = os.path.join(metric_dir, "quantization_map_layer.json")
        with open(q_map_path, 'w') as f:
            json.dump(q_map, f, indent=4)
        print(f"Saved layer-wise quantization map to {q_map_path}")

        layer_config = {
            'model': {'name': args.model_name, 'weights': os.path.abspath(q_weights_path)},
            'adapter': base_adapter_config,
            'quantization': {'format': DEFAULT_FORMAT, 'layers': {}},
            'evaluation': eval_config,
            'dataset': dataset_base,
            'output_name': f'optimized_layer_{m}',
        }
        configs_to_run.append(layer_config)

        # --- 3b. Per-chunk optimized (if enabled and chunk_winners exist) ---
        if args.per_chunk_format and args.weight_chunk_size:
            has_chunk_data = any(
                'chunk_winners' in rec and m in rec.get('chunk_winners', {})
                for rec in layer_results_map.values()
            )
            if has_chunk_data:
                q_state_dict_chunk, q_map_chunk = create_quantized_state_dict(
                    model, layer_results_map, args, m, use_chunking=True
                )

                q_weights_chunk_path = os.path.join(metric_dir, "quantized_weights_chunk.pt")
                torch.save(q_state_dict_chunk, q_weights_chunk_path)

                q_map_chunk_path = os.path.join(metric_dir, "quantization_map_chunk.json")
                # Serialize chunk_winners (list of formats per chunk) as JSON
                serializable_map = {}
                for k, v in q_map_chunk.items():
                    serializable_map[k] = v  # already str or list of str
                with open(q_map_chunk_path, 'w') as f:
                    json.dump(serializable_map, f, indent=4)
                print(f"Saved chunk-wise quantization map to {q_map_chunk_path}")

                chunk_config = {
                    'model': {'name': args.model_name, 'weights': os.path.abspath(q_weights_chunk_path)},
                    'adapter': base_adapter_config,
                    'quantization': {'format': DEFAULT_FORMAT, 'layers': {}},
                    'evaluation': eval_config,
                    'dataset': dataset_base,
                    'output_name': f'optimized_chunk_{m}',
                }
                configs_to_run.append(chunk_config)

    # Run evaluation
    print(f"\n--- Starting Evaluation ({len(configs_to_run)} configs) ---")
    runner = Runner()

    # Dedupe
    final_configs = []
    sigs = set()
    for c in configs_to_run:
        s = json.dumps(c, sort_keys=True)
        if s not in sigs:
            sigs.add(s)
            final_configs.append(c)

    results = runner.run_batch_parallel(final_configs, output_root=output_dir)

    # Aggregate
    aggregator = ReportAggregator()
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    aggregator.aggregate(results, summary_path)
    plot_accuracy_comparison(results, output_dir)
    print(f"Evaluation completed. Summary: {summary_path}")

    return results


# ============================================================
# Step 6: Save Classification Summary
# ============================================================

def save_classification_summary(classifications, bandwidth_stats, output_dir):
    """Save a summary CSV of the layer classification decisions."""
    csv_path = os.path.join(output_dir, "layer_classifications.csv")

    rows = []
    for layer in bandwidth_stats:
        name = layer.get('layer_name', '')
        rows.append({
            'layer_name': name,
            'layer_type': layer.get('layer_type', ''),
            'classification': classifications.get(name, 'default'),
            'precision': 'fp6 (search)' if classifications.get(name) == 'lower' else DEFAULT_FORMAT,
            'num_weights': layer.get('num_weights', 0),
            'num_macs': layer.get('num_macs', 0),
            'cycles': layer.get('cycles', 0),
            'bw_total_epc': layer.get('bw_total_elements_per_cycle', 0),
            'is_bw_limited': layer.get('is_bandwidth_limited', False),
            't_runtime_orig': f"{layer.get('t_runtime_orig', 0):.1f}",
            't_runtime_adjusted': f"{layer.get('t_runtime_adjusted', 0):.1f}",
            'weight_scale': f"{layer.get('weight_scale', 1.0):.3f}",
        })

    headers = ['layer_name', 'layer_type', 'classification', 'precision',
               'num_weights', 'num_macs', 'cycles', 'bw_total_epc', 'is_bw_limited',
               't_runtime_orig', 't_runtime_adjusted', 'weight_scale']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved classification summary to {csv_path}")


# ============================================================
# Main
# ============================================================

def process_single_model(args, device, metrics):
    """Process a single model end-to-end."""
    model_name = args.model_name

    # Output directory
    base_root = args.output_dir or os.path.join(
        PROJECT_ROOT, "runspace/experiments/bandwidth_aware_quant"
    )
    output_dir = os.path.join(base_root, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Bandwidth-Aware Mixed-Precision Quantization: {model_name}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")

    # Step 1: Get bandwidth stats
    print("\n--- Step 1: Bandwidth Analysis ---")
    bandwidth_stats = get_bandwidth_stats(model_name, args, device)
    print(f"Loaded {len(bandwidth_stats)} layer stats")

    # Step 2: Classify layers
    print("\n--- Step 2: Classify Layers ---")
    if not RULES:
        print("WARNING: No rules defined in RULES list!")
        print("All layers will use default format. Add rules to RULES list.")
    classifications, summary = classify_layers(bandwidth_stats, RULES)

    # Save classification summary (now includes adjusted fields)
    save_classification_summary(classifications, bandwidth_stats, output_dir)

    # Compute adjusted bandwidth stats (accounts for fp6 vs fp8 weight sizes)
    compute_adjusted_bandwidth(bandwidth_stats, classifications)

    # Print runtime comparison
    total_runtime_orig = sum(layer.get('t_runtime_orig', 0) for layer in bandwidth_stats)
    total_runtime_adj = sum(layer.get('t_runtime_adjusted', 0) for layer in bandwidth_stats)
    savings = total_runtime_orig - total_runtime_adj
    savings_pct = (savings / total_runtime_orig * 100) if total_runtime_orig > 0 else 0
    print(f"\n  Runtime Comparison:")
    print(f"    Original (all fp8):      {total_runtime_orig:,.0f} cycles")
    print(f"    Mixed (fp6/fp8):         {total_runtime_adj:,.0f} cycles")
    print(f"    Savings:                 {savings:,.0f} cycles ({savings_pct:.2f}%)")

    # Re-save classification summary with adjusted fields
    save_classification_summary(classifications, bandwidth_stats, output_dir)

    # Step 3: Load model and run analysis
    print("\n--- Step 3: Weight Quantization Analysis ---")
    config = {
        'model': {'name': model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }

    try:
        adapter = create_adapter(config)
        model = adapter.model
        model.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    layer_results_map = run_mixed_precision_analysis(
        model, classifications, bandwidth_stats, args, device, output_dir, metrics
    )

    # Save results and get best formats
    best_formats = save_results(layer_results_map, metrics, output_dir, model)

    # Step 4: Runtime bottleneck plot
    
    print("\n--- Step 4: Runtime Bottleneck Plot ---")
    generate_runtime_bottleneck(bandwidth_stats, classifications, output_dir, model_name)

    # Step 5: Evaluation (if requested)
    if args.run_eval:
        print("\n--- Step 5: Evaluation ---")
        run_evaluation(model, layer_results_map, best_formats, args, output_dir, metrics)
    else:
        print("\nSkipping evaluation (use --run_eval to enable)")

    print(f"\nDone! Results saved to {output_dir}")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse metrics
    available_metrics = ['l1', 'mse', 'sqnr', 'cosine']
    if args.metrics == 'all':
        metrics = available_metrics
    else:
        metrics = [m.strip().lower() for m in args.metrics.split(',')]
        metrics = [m for m in metrics if m in available_metrics]

    if not metrics:
        print("No valid metrics specified. Defaulting to l1.")
        metrics = ['l1']

    print(f"Metrics: {metrics}")

    # Multi-model support
    if args.models_config:
        with open(args.models_config, 'r') as f:
            config = yaml.safe_load(f)

        if isinstance(config, dict) and 'models' in config:
            models_list = config['models']
        elif isinstance(config, list):
            models_list = config
        else:
            print("Error: Invalid models config format.")
            return

        for model_cfg in models_list:
            if isinstance(model_cfg, str):
                args.model_name = model_cfg
                args.weights = 'DEFAULT'
            else:
                args.model_name = model_cfg.get('name')
                args.weights = model_cfg.get('weights', 'DEFAULT')

            try:
                process_single_model(args, device, metrics)
            except Exception as e:
                print(f"Error processing {args.model_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        process_single_model(args, device, metrics)


if __name__ == "__main__":
    main()
    # stats_orig = load_bandwidth_stats("runspace/experiments/bandwidth_analysis/resnet152/layer_bandwidth_stats.csv")
    # stats_bw_aware = load_bandwidth_stats("runspace/experiments/bandwidth_aware_quant/resnet152/layer_classifications.csv")

    # total_orig = sum([layer['bw_total_elements_per_cycle'] for layer in stats_orig])
    # total_bw_aware = sum(float(layer['bw_total_epc']) for layer in stats_bw_aware)

    # total_orig_percentage = total_orig 
    # total_bw_aware_percentage = total_bw_aware 

    # print(f"Total orig percentage: {total_orig_percentage}")
    # print(f"Total bw aware percentage: {total_bw_aware_percentage}")
    
   