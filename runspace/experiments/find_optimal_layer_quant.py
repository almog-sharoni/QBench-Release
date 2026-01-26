
import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.ops.quant_base import calculate_scale, quantize_tensor
from runspace.src.quantization.quantizer import quantize
from runspace.src.quantization.constants import get_quantization_bias
# Late import
from runspace.core.runner import Runner
from runspace.core.report_aggregator import ReportAggregator
from runspace.src.registry.op_registry import OpRegistry

baseline_formats = [
    'fp8_e4m3', 'fp8_e5m2', 'fp8_e3m4', 'fp8_e2m5', 'fp8_e1m6', 'fp8_e6m1', 'fp8_e7m0', 'fp8_e0m7',
    'fp7_e3m3', 'fp7_e4m2', 'fp7_e2m4', 'fp7_e5m1', 'fp7_e1m5', 'fp7_e6m0', 'fp7_e0m6',
]


def get_args():
    parser = argparse.ArgumentParser(description="Find optimal layer-wise quantization")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--include_fp32", action="store_true", help="Include FP32 in search")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: runspace/experiments/optimal_layer_quant)")
    parser.add_argument("--models_config", type=str, default=None, help="Path to a YAML file containing a list of models to run")
    
    # Validation / Metric Args
    parser.add_argument("--metrics", type=str, default="l1", help="Comma-separated metrics: l1, mse, sqnr, cosine OR 'all'")
    
    # Evaluation Args
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation on FP32, FP8, and Optimized models")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--weight_chunk_size", type=int, default=128, help="Weight chunk size (blocks). If set, enables chunked quantization.")
    parser.add_argument("--per_chunk_format", action="store_true", help="Enable per-chunk format selection (each chunk gets its own optimal format)")
    parser.add_argument("--plot_layers", action="store_true", help="Generate error bar plots for every single layer (warning: slow)")
    parser.add_argument("--max_batches", type=int, default=-1, help="Max batches to evaluate (default: -1 for all)")
    parser.add_argument("--force_recalc", action="store_true", help="Force recalculation of layer errors even if results exist")
    
    return parser.parse_args()

def get_quantized_weight(weight, q_type, bias=None, chunk_size=None, chunk_formats=None):
    """
    Returns the dequantized weight (simulated quantization).
    Returns (weight_dequant, max_val).
    """
    if q_type == 'fp32' and chunk_formats is None:
        return weight, weight.abs().max().item()
    
    if chunk_formats is not None and chunk_size is not None:
        # Mixed precision chunk quantization
        # Flatten and pad logic using get_chunked_tensor or similar?
        # Let's use get_chunked_tensor helper if available or reimplement locally as it's simple
        
        # Flatten
        if weight.dim() > 1:
            flat_weight = weight.flatten(1)
            batch_size = weight.shape[0]
        else:
            flat_weight = weight.flatten(0)
            batch_size = 1
        
        num_elements = flat_weight.shape[-1]
        pad_len = 0
        if num_elements % chunk_size != 0:
            pad_len = chunk_size - (num_elements % chunk_size)
            flat_weight = torch.nn.functional.pad(flat_weight, (0, pad_len))
        
        num_chunks = flat_weight.shape[-1] // chunk_size
        chunked = flat_weight.view(batch_size, num_chunks, chunk_size)
        
        # Expand formats
        total_chunks = batch_size * num_chunks
        final_formats = []
        
        if len(chunk_formats) == total_chunks:
            final_formats = chunk_formats
        elif len(chunk_formats) == num_chunks:
             # Broadcast
             for _ in range(batch_size):
                 final_formats.extend(chunk_formats)
        else:
             # Fallback
             final_formats = [chunk_formats[0]] * total_chunks
             
        # Flatten for processing
        chunked_flat = chunked.view(-1, chunk_size)
        quantized_flat = torch.zeros_like(chunked_flat)
        scale_flat = torch.zeros_like(chunked_flat)
        
        unique_fmts = set(final_formats)
        
        for fmt in unique_fmts:
            indices = [i for i, f in enumerate(final_formats) if f == fmt]
            if not indices: continue
            
            indices_tensor = torch.tensor(indices, device=weight.device)
            sub_chunks = chunked_flat[indices_tensor]
            
            # Quantize sub_chunks
            max_val = sub_chunks.abs().amax(dim=1, keepdim=True).clamp(min=1e-9)
            bias_val = get_quantization_bias(fmt)
            scale = calculate_scale(max_val, fmt, bias_val)
            
            scaled = sub_chunks / scale
            quant = quantize(scaled, q_type=fmt, bias=bias_val, rounding="nearest")
            
            # Dequantize immediately for simulation
            dequant = quant * scale
            
            quantized_flat[indices_tensor] = dequant
            
        # Reshape back
        w_dequant_padded = quantized_flat.view(batch_size, -1)
        if pad_len > 0:
            w_dequant_flat = w_dequant_padded[:, :num_elements]
        else:
            w_dequant_flat = w_dequant_padded
            
        w_dequant = w_dequant_flat.view_as(weight)
        return w_dequant, weight.abs().max().item() # Approx max val
    
    if chunk_size is not None:
        # Use quantize_tensor for chunking
        # quantize_tensor returns simulated dequantized tensor by default
        w_dequant, max_val = quantize_tensor(weight, q_type=q_type, bias=bias, mode='chunk', chunk_size=chunk_size, rounding="nearest")
        # quantize_tensor returns (result, max_val)
        return w_dequant, max_val.max().item()

    # Per-channel scale (dim 0)
    out_channels = weight.shape[0]
    weight_flat = weight.view(out_channels, -1)
    
    # Calculate scale per channel
    max_val_per_channel = weight_flat.abs().amax(dim=1, keepdim=True)
    max_val_per_channel = max_val_per_channel.clamp(min=1e-9)
    
    scale = calculate_scale(max_val_per_channel, q_type, bias)
    
    w_scaled = weight_flat / scale
    w_quant = quantize(w_scaled, q_type=q_type, bias=bias, rounding="nearest")
    w_dequant = w_quant * scale
    
    # Reshape back
    w_dequant = w_dequant.view_as(weight)
    
    return w_dequant, max_val_per_channel.max().item()

def calculate_error(original, dequantized, metric):
    """
    Calculate error between original and dequantized tensors.
    """
    diff = original - dequantized
    
    if metric == 'l1':
        return diff.abs().sum().item()
    
    elif metric == 'mse':
        return diff.pow(2).mean().item()
        
    elif metric == 'sqnr':
        # Signal to Quantization Noise Ratio
        # SQNR = 10 * log10( P_signal / P_noise )
        # maximize SQNR -> minimize -SQNR
        # We perform calculations in double for stability with small noise
        signal_pow = original.pow(2).sum().item()
        noise_pow = diff.pow(2).sum().item()
        
        if noise_pow < 1e-12:
            return -1000.0 # Effectively infinite SQNR (perfect match), so very small "error"
            
        sqnr = 10 * np.log10(signal_pow / noise_pow)
        return -sqnr # Return negative so we can minimize it
        
    elif metric == 'cosine':
        # Cosine Distance = 1 - CosineSimilarity
        # Flatten for vector comparison
        orig_flat = original.flatten()
        deq_flat = dequantized.flatten()
        
        # Avoid div by zero
        if orig_flat.norm() < 1e-9 or deq_flat.norm() < 1e-9:
            return 1.0 # Max distance if zero vector
            
        cos_sim = nn.functional.cosine_similarity(orig_flat.unsqueeze(0), deq_flat.unsqueeze(0)).item()
        return 1.0 - cos_sim
        
    else:
        raise ValueError(f"Unknown metric: {metric}")

def plot_error_histograms(data, formats, output_dir, metric):
    """Generate histograms of errors for each format."""
    print(f"Generating {metric} histograms...")
    
    # Collect errors per format
    format_errors = {fmt: [] for fmt in formats}
    for layer_data in data:
        for fmt in formats:
            if fmt in layer_data['errors']:
                err = layer_data['errors'][fmt]
                
                # Handling for plotting
                # SQNR is usually negative in our 'error' metric (minimized -SQNR)
                # But for plots, maybe show actual SQNR?
                # Or just show the distribution of the minimization objective
                
                # For L1/MSE/Cosine, values >= 0.
                if metric == 'sqnr':
                    # Convert back to actual SQNR for plotting? 
                    # error = -SQNR. So SQNR = -error
                    val = -err
                else:
                    val = err
                
                # Filter bad values
                if np.isfinite(val):
                    format_errors[fmt].append(val)

    # 1. Combined Histogram
    plt.figure(figsize=(12, 8))
    for fmt in formats:
        # Check if we have data
        if not format_errors[fmt]:
            continue
            
        vals = np.array(format_errors[fmt])
        
        if metric in ['l1', 'mse']:
             # Avoid log(0)
             vals = np.maximum(vals, 1e-12)
             plt.hist(np.log10(vals), bins=30, alpha=0.5, label=fmt)
             xlabel = f"Log10({metric.upper()} Error)"
        elif metric == 'cosine':
             # Cosine distance is 0 to 1 (usually small). Log is fine.
             vals = np.maximum(vals, 1e-12)
             plt.hist(np.log10(vals), bins=30, alpha=0.5, label=fmt)
             xlabel = f"Log10(Cosine Dist)"
        elif metric == 'sqnr':
             # SQNR in dB. 
             # Clip extremely low values for plotting visibility?
             # Or just let it be.
             plt.hist(vals, bins=30, alpha=0.5, label=fmt)
             xlabel = "SQNR (dB)"
             
    plt.title(f"{metric.upper()} Distribution by Format")
    plt.xlabel(xlabel)
    plt.ylabel("Count (Layers)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"error_histograms_{metric}.png"))
    plt.close()

def plot_error_boxplot(data, formats, output_dir, metric):
    """Generate boxplot of errors for each format including Optimized."""
    print(f"Generating {metric} boxplot...")
    
    # Collect errors per format
    # formats list usually contains 'fp8...', 'int8...'.
    # data is list of layer records.
    # We want to add 'Optimized' to the plot.
    
    plot_labels = formats + ['Optimized']
    plot_data = []
    
    for fmt in plot_labels:
        vals = []
        for layer_data in data:
            if fmt == 'Optimized':
                val = layer_data.get('best_error', float('inf'))
            elif fmt in layer_data['errors']:
                val = layer_data['errors'][fmt]
            else:
                continue
                
            # Handling for plotting (same as histogram)
            if metric == 'sqnr':
                val = -val # Back to dB
            
            if np.isfinite(val):
                vals.append(val)
        
        # Log transform for L1/MSE/Cosine if beneficial, or just plot raw/log scale axis
        # Boxplot handles outliers well, but log scale Y-axis might be better visualization
        plot_data.append(vals)

    plt.figure(figsize=(14, 8))
    
    # Check for empty data
    valid_data = [d for d in plot_data if len(d) > 0]
    valid_labels = [l for i, l in enumerate(plot_labels) if len(plot_data[i]) > 0]
    
    if not valid_data:
        print("No valid data for boxplot.")
        plt.close()
        return

    # Create boxplot
    bp = plt.boxplot(valid_data, labels=valid_labels, patch_artist=True)
    
    # Coloring
    colors = ['lightblue'] * len(valid_labels)
    if 'Optimized' in valid_labels:
        opt_idx = valid_labels.index('Optimized')
        colors[opt_idx] = 'lightgreen'
        
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title(f"{metric.upper()} Error Distribution Comparison")
    plt.ylabel(f"{metric.upper()} Error")
    plt.xlabel("Format")
    plt.xticks(rotation=45)
    
    if metric in ['l1', 'mse', 'cosine']:
        plt.yscale('log')
        
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"error_boxplot_{metric}.png"))
    plt.close()

def plot_layer_error_bar(layer_name, errors, formats, output_dir, metric):
    """Generate a bar chart of errors for a single layer."""
    plt.figure(figsize=(10, 6))
    
    # Extract values
    raw_values = [errors.get(fmt, 0) for fmt in formats]
    
    # Transform for plotting if needed
    if metric == 'sqnr':
        # Visualize actual SQNR (higher is better)
        # Our stored error is -SQNR.
        values = [-v for v in raw_values]
        ylabel = "SQNR (dB)"
        # For coloring, we look for MAX value
        best_val = max(values)
    else:
        values = raw_values
        ylabel = f"{metric.upper()} Error"
        best_val = min([v for v in values if v >= 0]) if any(v>=0 for v in values) else 0

    # Plot bars
    bars = plt.bar(formats, values, color='skyblue')
    
    # Highlight the best
    for i, bar in enumerate(bars):
        if values[i] == best_val:
            bar.set_color('orange')
            
    plt.title(f"{metric.upper()} - {layer_name}")
    plt.ylabel(ylabel)
    plt.xlabel("Format")
    
    if metric in ['l1', 'mse', 'cosine']:
        plt.yscale('log')
        
    plt.grid(axis='y', alpha=0.3)
    
    # Save
    safe_name = layer_name.replace('.', '_')
    plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
    plt.close()

def plot_accuracy_comparison(results, output_dir):
    """Generate a grouped bar chart for Acc1 and Acc5 comparison, sorted by Acc1."""
    print("Generating accuracy comparison plot...")
    
    # Sort results by Acc1 (descending)
    results = sorted(results, key=lambda x: x.get('acc1', 0.0), reverse=True)
    
    names = []
    acc1_vals = []
    acc5_vals = []
    error_vals = [] 
    
    for res in results:
        names.append(res.get('output_name', 'Unknown'))
        acc1_vals.append(res.get('acc1', 0.0))
        acc5_vals.append(res.get('acc5', 0.0))
        error_vals.append(res.get('total_error', 0.0)) # Default 0 if not set

    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Accuracy Bars
    rects1 = ax1.bar(x - width/2, acc1_vals, width, label='Top-1 Acc', color='royalblue')
    rects2 = ax1.bar(x + width/2, acc5_vals, width, label='Top-5 Acc', color='lightskyblue')
    
    ax1.set_ylabel('Accuracy (%)', color='blue')
    ax1.set_title('Accuracy & Total Error Comparison (Sorted by Top-1 Acc)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(axis='y', alpha=0.3)
    
    # Error Rate (Secondary Axis)
    # Check if we have 'errors' dictionary in results
    # Format: res['errors'] = {'l1': val, 'mse': val, ...}
    
    # Collect data series for each metric found
    error_series = {} 
    valid_metrics = set()
    
    # First pass to find all metrics present
    for res in results:
        errs = res.get('errors', {})
        if isinstance(errs, dict):
            for m, val in errs.items():
                valid_metrics.add(m)
    
    if valid_metrics:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Error (0=Best, 1=Worst)', color='black')
        
        # Color cycle for metrics
        cmap = plt.get_cmap('Dark2') 
        
        lines = []
        labels = []
        
        # Collect all data first
        metric_data = {}
        for m in valid_metrics:
            vals = []
            for res in results:
                err_dict = res.get('errors', {})
                val = err_dict.get(m, float('nan'))
                vals.append(val)
            metric_data[m] = np.array(vals)

        for i, m in enumerate(sorted(valid_metrics)):
            vals = metric_data[m]
            
            # Normalize to [0, 1]
            # Filter NaNs for min/max calculation
            valid_v = vals[np.isfinite(vals)]
            
            normalized_vals = vals.copy()
            
            if len(valid_v) > 0:
                v_min = valid_v.min()
                v_max = valid_v.max()
                
                # Check for zero range
                if v_max > v_min:
                    # Apply Min-Max scaling
                    # Note: We assume "Lower is Better" for the raw values (including -SQNR)
                    # So normalized 0.0 will correspond to Min Raw (Best)
                    # and 1.0 will correspond to Max Raw (Worst).
                    normalized_vals = (vals - v_min) / (v_max - v_min)
                else:
                    # Constant value across all models
                    # Plot as flat line at 0.0 (Best) or 0.5?
                    # If error is 0 everywhere (ref), it should be at 0.
                    # If error is massive everywhere, maybe 1?
                    # Let's put it at 0 to avoid visual clutter.
                    normalized_vals = np.zeros_like(vals)
            else:
                 # All NaNs
                 pass

            color = cmap(i % 8)
            label = f"{m.upper()}" # Short label since axis explains "Normalized Error"
            
            # Plot
            # We use only valid points for the line to avoid plotting gaps if possible, or leave gaps for NaNs
            line = ax2.plot(x, normalized_vals, color=color, marker='.', linestyle='-', linewidth=2, alpha=0.8, label=label)
            lines.extend(line)
            labels.append(label)

        # Set fixed range for normalized plot
        ax2.set_ylim(-0.1, 1.1)
        
        # Tick colors
        ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends and move outside
    lines1, labels1 = ax1.get_legend_handles_labels()
    if valid_metrics and lines:
        ax1.legend(lines1 + lines, labels1 + labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        ax1.legend(lines1, labels1, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.8)
    
    # Add labels to bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()

def get_chunked_tensor(tensor, chunk_size):
    """
    Reshapes tensor into [N, num_chunks, chunk_size].
    Returns chunked_tensor, original_shape, padding_length
    """
    shape = tensor.shape
    if tensor.dim() > 1:
        flat = tensor.flatten(1)
        batch = shape[0]
    else:
        flat = tensor.flatten(0)
        batch = 1
        
    num_elements = flat.shape[-1]
    pad_len = 0
    if num_elements % chunk_size != 0:
        pad_len = chunk_size - (num_elements % chunk_size)
        flat = torch.nn.functional.pad(flat, (0, pad_len))
        
    num_chunks = flat.shape[-1] // chunk_size
    chunked = flat.view(batch, num_chunks, chunk_size)
    
    return chunked, shape, pad_len

def plot_chunk_format_distribution(chunk_formats, formats, output_dir, metric):
    """
    Generate a heatmap showing format distribution across chunks for each layer.
    chunk_formats: {layer_name: [fmt_chunk0, fmt_chunk1, ...]}
    """
    print(f"Generating {metric} per-chunk format distribution heatmap...")
    
    if not chunk_formats:
        return
        
    # Sort formats to match other plots (consistent coloring)
    sorted_formats = sort_formats(formats)
        
    # Create format to index mapping
    fmt_to_idx = {fmt: i for i, fmt in enumerate(sorted_formats)}
    
    layers = list(chunk_formats.keys())
    max_chunks = max(len(chunk_formats[layer]) for layer in layers)
    
    # Create matrix: [num_layers, max_chunks]
    matrix = np.full((len(layers), max_chunks), -1, dtype=int)
    
    for i, layer in enumerate(layers):
        for j, fmt in enumerate(chunk_formats[layer]):
            if fmt in fmt_to_idx:
                matrix[i, j] = fmt_to_idx[fmt]
    
    plt.figure(figsize=(20, max(8, len(layers) * 0.3)))
    
    # Create custom colormap
    cmap = plt.get_cmap('tab20', len(sorted_formats))
    cmap.set_bad(color='lightgray') # Set color for masked values
    
    # Mask negative values (unused chunks)
    masked_matrix = np.ma.masked_where(matrix < 0, matrix)
    
    # Resize matrix if too large (visualization only)
    max_display_width = 1000
    if masked_matrix.shape[1] > max_display_width:
        print(f"Downsampling heatmap from {masked_matrix.shape[1]} to {max_display_width} columns...")
        factor = masked_matrix.shape[1] // max_display_width
        remainder = masked_matrix.shape[1] % max_display_width
        
        # Simple decimation or voting? Voting is better for categorical.
        # Reshape to [layers, new_width, factor]
        # We drop the remainder for simplicity in visualization
        trimmed_width = masked_matrix.shape[1] - remainder
        reshaped = masked_matrix[:, :trimmed_width].reshape(len(layers), max_display_width, factor)
        
        # Mode (majority vote)
        from scipy import stats
        # mstats.mode returns (mode, count)
        # We need to handle masked arrays
        # Scipy mstats mode is slow. Let's just take the first element (decimation) for speed
        # or max element?
        # Decimation is fastest and usually fine for dense patterns.
        masked_matrix = reshaped[:, :, 0]
    
    # Plot
    im = plt.imshow(masked_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=len(sorted_formats)-1)
    
    # Colorbar
    cbar = plt.colorbar(im, ticks=range(len(sorted_formats)))
    cbar.ax.set_yticklabels(sorted_formats)
    
    plt.title(f"{metric.upper()} Per-Chunk Format Selection")
    plt.ylabel("Layers")
    plt.xlabel("Chunk Index")
    plt.yticks(range(len(layers)), layers, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"chunk_format_heatmap_{metric}.png"), dpi=150)
    plt.close()

def get_format_bits(fmt):
    """
    Returns the bit width of a quantization format string.
    """
    if not fmt: return 32
    if fmt == 'fp32': return 32
    if fmt == 'int8': return 8
    if fmt == 'int4': return 4
    
    # Check for fpX_... or ufpX_...
    # The 'X' usually denotes total container bits in name, but we should parse carefully.
    # Logic same as quantizer: [u]fp[Total]_e[E]m[M]
    # For signed: S + E + M. For unsigned: E + M.
    try:
        is_signed = fmt.startswith('fp')
        if '_e' in fmt and 'm' in fmt:
             parts = fmt.split('_e')[1].split('m')
             exp = int(parts[0])
             mant = int(parts[1])
             
             bits = exp + mant
             if is_signed:
                 bits += 1
             return bits
    except:
        pass
        
    # Fallback to name parsing if strict loop failed or different format
    # fp8 -> 8, fp4 -> 4
    if fmt.startswith('fp'):
        try:
            return int(fmt.split('_')[0].replace('fp',''))
        except: pass
    if fmt.startswith('ufp'):
         try:
            return int(fmt.split('_')[0].replace('ufp',''))
         except: pass
         
    return 32 # Default high for unknown

def sort_formats(formats):
    """Sort formats by bit width (desc) then exponent size (desc)."""
    def parse_fmt(fmt):
        bits = get_format_bits(fmt)
        
        # Second key: exp bits for tie breaking? 
        exp = 0
        if '_e' in fmt:
             try:
                exp = int(fmt.split('_e')[1].split('m')[0])
             except: pass
        return (bits, exp)
        
    return sorted(formats, key=parse_fmt, reverse=True)

def plot_chunk_win_rate(win_counts, formats, output_dir, metric, layer_winners=None):
    """
    Generate a stacked bar chart of chunk 'wins' per layer.
    win_counts: {layer_name: {fmt: count, 'total': N}}
    layer_winners: {layer_name: winning_format} (optional)
    """
    print(f"Generating {metric} chunk win rate plot...")
    
    layers = list(win_counts.keys())
    if not layers:
        return
        
    # Sort formats
    sorted_formats = sort_formats(formats)

    # Prepare data
    # Matrix: [num_layers, num_formats]
    data = np.zeros((len(layers), len(sorted_formats)))
    
    for i, layer in enumerate(layers):
        total = win_counts[layer].get('total', 1)
        for j, fmt in enumerate(sorted_formats):
            count = win_counts[layer].get(fmt, 0)
            data[i, j] = count # Raw count
            
    plt.figure(figsize=(15, 8))
    
    # Plot Stacked Bars
    bottom = np.zeros(len(layers))
    
    # Use tab20 colors
    cmap = plt.get_cmap('tab20', len(sorted_formats))
    
    for j, fmt in enumerate(sorted_formats):
        plt.bar(layers, data[:, j], bottom=bottom, label=fmt, color=cmap(j))
        bottom += data[:, j]
        
    plt.ylabel('Number of Chunks')
    plt.title(f'Chunk Format Selection per Layer ({metric.upper()})')
    plt.xticks(rotation=90, fontsize=8)
    
    # Clean legend: only formats that appear at least once
    # Check if a column has any non-zero value
    active_indices = [j for j in range(len(sorted_formats)) if data[:, j].sum() > 0]
    
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter legend
    filtered_handles = [handles[j] for j in active_indices]
    filtered_labels = [labels[j] for j in active_indices]
    
    plt.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"chunk_win_rate_{metric}.png"))
    plt.close()

def plot_layer_error_comparison(layer_results, formats, output_dir, metric):
    """
    Generate a line plot comparing total error of each format across layers.
    layer_results: List of dicts containing 'layer' and 'metrics'
    """
    print(f"Generating {metric} layer-wise error comparison plot...")
    
    if not layer_results:
        return
        
    layers = [res['layer'] for res in layer_results]
    x = np.arange(len(layers))
    
    # Extract errors per format
    # format_series: {fmt: [err_layer1, err_layer2, ...]}
    format_series = {fmt: [] for fmt in formats}
    
    valid_formats = []
    
    for fmt in formats:
        has_data = False
        series = []
        for res in layer_results:
            err = res['metrics'][metric].get(fmt, float('nan'))
            
            # Handling SQNR negative values
            if metric == 'sqnr':
                err = -err
                
            series.append(err)
            if np.isfinite(err):
                has_data = True
        
        if has_data:
            format_series[fmt] = series
            valid_formats.append(fmt)
            
    if not valid_formats:
        return

    plt.figure(figsize=(14, 8))
    
    # Use consistent colormap with chunk plot
    cmap = plt.get_cmap('tab20')
    fmt_colors = {fmt: cmap(i % 20) for i, fmt in enumerate(formats)} # Use original formats list for consistent index

    for fmt in valid_formats:
        vals = np.array(format_series[fmt])
        # Mask NaNs/Infs for plotting
        mask = np.isfinite(vals)
        if metric in ['l1', 'mse', 'cosine']:
             # Use semilogy for error metrics
             plt.semilogy(x[mask], vals[mask], label=fmt, color=fmt_colors[fmt], marker='o', markersize=3, linewidth=1.5, alpha=0.7)
        else:
             plt.plot(x[mask], vals[mask], label=fmt, color=fmt_colors[fmt], marker='o', markersize=3, linewidth=1.5, alpha=0.7)

    plt.title(f"{metric.upper()} Total Error per Layer")
    plt.xlabel("Layers")
    plt.ylabel(f"{metric.upper()} Error" + (" (Log Scale)" if metric != 'sqnr' else " (dB)"))
    plt.xticks(x, layers, rotation=90, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"layer_error_comparison_{metric}.png"))
def calculate_model_error(model, config, metric, global_chunk_size=None):
    """
    Re-calculates the total model error for a given configuration.
    """
    print(f"Recalculating error using metric '{metric}'...")
    total_error = 0.0
    
    # Parse config
    global_quant = config.get('quantization', {})
    if isinstance(global_quant, dict):
        global_fmt = global_quant.get('format', 'fp32') # Default
        layer_configs = global_quant.get('layers', {})
        per_chunk = global_quant.get('per_chunk_format', False)
        # Check explicit chunk size in config, or use global arg
        chunk_size = global_quant.get('weight_chunk_size', global_chunk_size)
    else:
        # Maybe config is simple? Assuming structure from main
        global_fmt = 'fp32'
        layer_configs = {}
        per_chunk = False
        chunk_size = global_chunk_size

    # Iterate layers
    supported_ops = tuple(OpRegistry.get_supported_ops().keys())
    for name, module in tqdm(model.named_modules(), desc=f"Calc Error ({metric})", leave=False):
        if isinstance(module, supported_ops):
            if not hasattr(module, 'weight') or module.weight is None:
                continue
            
            w = module.weight.data
            
            # Determine format
            current_fmt = global_fmt
            chunk_formats = None
            
            if name in layer_configs:
                l_cfg = layer_configs[name]
                if isinstance(l_cfg, dict):
                    if 'format' in l_cfg:
                        current_fmt = l_cfg['format']
                    if 'chunk_formats' in l_cfg:
                        chunk_formats = l_cfg['chunk_formats']
                elif isinstance(l_cfg, str):
                    current_fmt = l_cfg
            
            if not per_chunk:
                chunk_formats = None # Ignore if disabled globally (safety)?
            
            try:
                w_deq, _ = get_quantized_weight(w, q_type=current_fmt, chunk_size=chunk_size, chunk_formats=chunk_formats)
                
                # Calc error
                err = calculate_error(w, w_deq, metric)
                
                # Handle SQNR summing
                # If metric is SQNR, calculate_error returns -SQNR.
                # Sum of -SQNRs is not mathematically meaningful for "Total SQNR", 
                # but as a minimization objective (Total Error), summing negative SQNR (pseudo-error) is fine.
                # It effectively penalizes low SQNR.
                
                total_error += err
                
            except Exception as e:
                print(f"Error calculating {name}: {e}")
                
    return total_error

def load_cached_results(csv_path, metric, format_col_suffix="_error"):
    """
    Load layer results from an existing CSV file.
    Returns:
       - results_list: List of dicts matching internal record structure (partial)
       - successful: Boolean indicating success
    """
    print(f"Loading cached results from {csv_path}...")
    try:
        results = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = row['layer']
                
                # Reconstruct metrics dict
                metrics_data = {}
                best_error = float(row.get('best_error', float('inf')))
                
                # We need to reconstruct the errors for all formats
                # The CSV has columns like "fp8_e4m3_error"
                for key, val in row.items():
                    if key.endswith(format_col_suffix):
                         fmt = key.replace(format_col_suffix, "")
                         try:
                             if val and val != "":
                                 metrics_data[fmt] = float(val)
                             else:
                                 metrics_data[fmt] = float('inf')
                         except:
                             metrics_data[fmt] = float('inf')
                             
                record = {
                    'layer': layer,
                    'shape': row.get('shape', ''),
                    'max_val': float(row.get('max_val', 0.0)),
                    'metrics': {metric: metrics_data},
                    'best_error': best_error
                }
                
                # We don't reconstruct chunk wins from CSV fully yet, 
                # but for 'Pre-calculate theoretical errors' we just need metrics.
                results.append(record)
        return results, True
    except Exception as e:
        print(f"Failed to load cached results: {e}")
        return [], False

def process_single_model(args, device, metrics, base_root):
    # Valid model path: base_root / model_name
    model_dir = os.path.join(base_root, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Root output directory: {model_dir}")
    
    # Load Model
    print(f"Loading model {args.model_name}...")
    config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []}
    }
    
    try:
        adapter = create_adapter(config)
        model = adapter.model
        model.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    qt_options = baseline_formats
    if args.include_fp32:
        qt_options.insert(0, 'fp32')

    print(f"Testing types: {qt_options}")
    
    # Storage for all data:
    # layer_results = [ { 'layer': name, 'shape': ..., 'metrics': { 'l1': {fmt: err}, 'mse': ... } } ]
    layer_results_map = {} # {layer_name: record_dict}
    
    # Check what needs calculation
    metrics_to_calc = []
    configs_to_run = []
    
    # Always add Ref and Baselines if we are evaling (but we manage duplicates later)
    # We construct the base configs list later, here we just check metrics cache.
    
    for m in metrics:
        metric_dir = os.path.join(model_dir, m)
        csv_path = os.path.join(metric_dir, "layer_errors.csv")
        config_path = os.path.join(metric_dir, "optimized_config.yaml")
        
        is_cached = False
        if not args.force_recalc and os.path.exists(csv_path) and os.path.exists(config_path):
            cached_results, success = load_cached_results(csv_path, m)
            if success:
                print(f"Metric {m} found in cache. Skipping calculation.")
                is_cached = True
                
                # Merge into layer_results_map
                for res in cached_results:
                    lname = res['layer']
                    if lname not in layer_results_map:
                        layer_results_map[lname] = {
                            'layer': lname,
                            'shape': res['shape'],
                            'max_val': res['max_val'],
                            'metrics': {},
                            'chunk_wins': {},
                            'chunk_winners': {}
                        }
                    # Merge metric data
                    layer_results_map[lname]['metrics'][m] = res['metrics'][m]
                
                # Load config to run
                try:
                    with open(config_path, 'r') as f:
                        opt_cfg = yaml.safe_load(f)
                    opt_cfg['output_name'] = f"optimized_{m}"
                    configs_to_run.append(opt_cfg)
                except Exception as e:
                     print(f"Warning: Failed to load config for cached metric {m}: {e}")
            else:
                metrics_to_calc.append(m)
        else:
            metrics_to_calc.append(m)
            
    if not metrics_to_calc:
        print("All metrics cached. Skipping layer analysis.")
    else:
        print(f"Analyzing layers for metrics: {metrics_to_calc}...")
        
        supported_ops = tuple(OpRegistry.get_supported_ops().keys())
        for name, module in tqdm(model.named_modules(), desc="Layers"):
            if isinstance(module, supported_ops):
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                    
                w = module.weight.data
                
                # Get or create record
                if name not in layer_results_map:
                    layer_results_map[name] = {
                        'layer': name,
                        'shape': str(tuple(w.shape)),
                        'max_val': 0.0,
                        'metrics': {},
                        'chunk_wins': {},
                        'chunk_winners': {}
                    }
                record = layer_results_map[name]
                
                # Init metric placeholders for NEW metrics
                for m in metrics_to_calc:
                    if m not in record['metrics']:
                        record['metrics'][m] = {}
                        record['chunk_wins'][m] = {}
                        record['chunk_winners'][m] = []
                
                # 1. Quantize and Calc Errors
                max_val_global = record['max_val']
                
                metric_chunk_errors = {m: {} for m in metrics_to_calc}
                
                for q_type in qt_options:
                    try:
                        # Dequantize once
                        w_deq, mv = get_quantized_weight(w, q_type, chunk_size=args.weight_chunk_size)
                        max_val_global = max(max_val_global, mv)
                        
                        # Calculate overall error for required metrics
                        for m in metrics_to_calc:
                            err = calculate_error(w, w_deq, m)
                            record['metrics'][m][q_type] = err
                            
                        # Calculate chunk errors if needed
                        if args.weight_chunk_size:
                            w_chunked, _, _ = get_chunked_tensor(w, args.weight_chunk_size)
                            w_deq_chunked, _, _ = get_chunked_tensor(w_deq, args.weight_chunk_size)
                            
                            # diff [N, C, size]
                            diff = w_chunked - w_deq_chunked
                            
                            for m in metrics_to_calc:
                                # Calculate per-chunk error
                                # Flatten last dim to compute error per chunk vector
                                if m == 'l1':
                                    chunk_errs = diff.abs().sum(dim=-1).view(-1)
                                elif m == 'mse':
                                    chunk_errs = diff.pow(2).mean(dim=-1).view(-1)
                                elif m == 'sqnr':
                                    # SQNR per chunk
                                    sig = w_chunked.pow(2).sum(dim=-1).view(-1)
                                    noise = diff.pow(2).sum(dim=-1).view(-1)
                                    # Avoid log0
                                    noise = torch.clamp(noise, min=1e-12)
                                    chunk_errs = -10 * torch.log10(sig / noise) # Negative SQNR for min
                                elif m == 'cosine':
                                    # Cosine per chunk
                                    # dim=-1 is vector dim
                                    sim = torch.nn.functional.cosine_similarity(w_chunked, w_deq_chunked, dim=-1)
                                    chunk_errs = (1.0 - sim).view(-1)
                                
                                metric_chunk_errors[m][q_type] = chunk_errs.cpu().numpy()
                            
                    except Exception as e:
                        # print(f"Error {q_type} on {name}: {e}")
                        for m in metrics_to_calc:
                            record['metrics'][m][q_type] = float('inf') # Error flag
                
                # Save metric_chunk_errors to record for later analysis
                if args.weight_chunk_size:
                    record['chunk_errors'] = metric_chunk_errors
                            
                # Process Chunk Wins
                if args.weight_chunk_size:
                    for m in metrics_to_calc:
                        # metric_chunk_errors[m] is {qt: np.array(num_chunks)}
                        # Create matrix [num_formats, num_chunks]
                        valid_fmts = [qt for qt in qt_options if qt in metric_chunk_errors[m]]
                        if not valid_fmts:
                            continue
                         
                        # Sort valid_fmts by bits ASC to allow argmin to pick lowest bits on tie
                        valid_fmts_sorted = sorted(valid_fmts, key=get_format_bits)
       
                        err_matrix = np.stack([metric_chunk_errors[m][qt] for qt in valid_fmts_sorted])
                        # Find min index axis 0 (returns first occurrence, which is lowest bits due to sort)
                        best_indices = np.argmin(err_matrix, axis=0)
                        
                        # Store per-chunk winners (format names)
                        chunk_winner_names = [valid_fmts_sorted[idx] for idx in best_indices]
                        record['chunk_winners'][m] = chunk_winner_names
                        
                        # Count wins
                        wins = {'total': len(best_indices)}
                        for idx, qt in enumerate(valid_fmts_sorted):
                            count = np.sum(best_indices == idx)
                            if count > 0:
                                wins[qt] = int(count)
                        
                        record['chunk_wins'][m] = wins
                            
                record['max_val'] = max_val_global
    
    # Flatten layer_results_map to list (preserve order if we can, but named_modules order matters)
    # We used named_modules order for insertion, so looping through map might be arbitrary?
    # Python 3.7+ dicts preserve insertion order.
    layer_results = list(layer_results_map.values())
    
    # 2. Process Results Per Metric
    # configs_to_run already populated with cached configs
    
    # Always add Ref and Baselines if we are evaling
    eval_config = {'mode': 'evaluate'}
    if args.max_batches > 0:
        eval_config['max_batches'] = args.max_batches
        
    if args.run_eval:
        dataset_base = {
            'name': args.dataset_name,
            'path': args.dataset_path,
            'batch_size': args.batch_size,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        }
        
        # eval_config already defined above
        
        # 1. Ref FP32
        ref_config = {
            'model': {'name': args.model_name, 'weights': args.weights},
            'adapter': {'type': 'generic', 'quantized_ops': []},
            'evaluation': eval_config,
            'dataset': dataset_base,
            'output_name': "ref_fp32"
        }
        configs_to_run.append(ref_config)
        
        # 2. Baselines for all formats (each format applied uniformly to all layers)
        if args.include_fp32:
            # FP32 baseline is the same as ref, skip
            pass
        
        for fmt in baseline_formats:
            baseline_config = {
                'model': {'name': args.model_name, 'weights': args.weights},
                'adapter': {
                    'type': 'generic', 'quantized_ops': ['-1'],
                    'input_quantization': False
                },
                'quantization': {
                    'format': fmt,
                    'weight_mode': 'chunk' if args.weight_chunk_size else 'channel',
                    'weight_chunk_size': args.weight_chunk_size
                },
                'evaluation': eval_config,
                'dataset': dataset_base,
                'output_name': f"baseline_{fmt}"
            }
            configs_to_run.append(baseline_config)

    # Only process metrics that we calculated in this run
    # For cached metrics, we already appended the config and loaded results
    for m in metrics_to_calc:
        print(f"\n--- Processing Metric: {m} ---")
        metric_dir = os.path.join(model_dir, m)
        os.makedirs(metric_dir, exist_ok=True)
        hist_dir = os.path.join(metric_dir, "histograms")
        os.makedirs(hist_dir, exist_ok=True)
        
        # Analysis for this metric
        layer_config_m = {}  # Layer-level config
        layer_config_per_chunk = {}  # Per-chunk config
        csv_rows = []
        
        plot_data = [] # For histogram function
        chunk_win_counts = {} # {layer: {fmt: count}}
        layer_config_m_simple = {} # {layer: winning_fmt} for plotting
        chunk_format_map = {}  # {layer: [fmt_per_chunk]} for per-chunk mode
        
        for record in layer_results:
            name = record['layer']
            errs = record['metrics'][m]
            
            best_error = float('inf')
            best_type = None
            
            # Find best
            for q_type, val in errs.items():
                 # Check if better error (strictly)
                 if val < best_error:
                     best_error = val
                     best_type = q_type
                 # Or if equal error, prefer fewer bits
                 elif val == best_error:
                    cur_bits = get_format_bits(best_type) if best_type else 999
                    new_bits = get_format_bits(q_type)
                    if new_bits < cur_bits:
                        best_error = val # Same
                        best_type = q_type
                    # If bits are same, maybe prefer lower exponent? (lower range, more precision?)
                    # For now just bits.
            
            if args.weight_chunk_size and 'chunk_wins' in record:
                chunk_win_counts[name] = record['chunk_wins'][m]
                
                # Store per-chunk winners for visualization
                if 'chunk_winners' in record and record['chunk_winners'][m]:
                    chunk_format_map[name] = record['chunk_winners'][m]
                    
            if best_type:
                layer_config_m[name] = {'format': best_type}
                layer_config_m_simple[name] = best_type
                
            # Per-chunk format config
            if args.per_chunk_format and args.weight_chunk_size:
                if 'chunk_winners' in record and record['chunk_winners'][m]:
                    chunk_winners = record['chunk_winners'][m]
                    # Count format distribution for this layer
                    fmt_counts = {}
                    for fmt in chunk_winners:
                        fmt_counts[fmt] = fmt_counts.get(fmt, 0) + 1
                    
                    layer_config_per_chunk[name] = {
                        'chunk_formats': chunk_winners,
                        'format_distribution': fmt_counts
                    }
            
            # Generate Layer Plot
            if args.plot_layers:
                try:
                    plot_layer_error_bar(name, errs, qt_options, hist_dir, m)
                except Exception as e:
                    pass # Squelch
                
            # CSV Row
            row = {
                'layer': name,
                'shape': record['shape'],
                'max_val': record['max_val'],
                'best_type': best_type,
                'best_error': best_error
            }
            
            # Add per-chunk format info if available
            if args.per_chunk_format and 'chunk_winners' in record and record['chunk_winners'][m]:
                chunk_winners = record['chunk_winners'][m]
                # Add format distribution summary
                fmt_dist = {}
                for fmt in chunk_winners:
                    fmt_dist[fmt] = fmt_dist.get(fmt, 0) + 1
                row['chunk_format_distribution'] = json.dumps(fmt_dist)
                row['num_chunks'] = len(chunk_winners)
            
            for qt in qt_options:
                row[f"{qt}_error"] = errs.get(qt, "")
            csv_rows.append(row)
            
            plot_data.append({'errors': errs, 'best_error': best_error})
            
            # Chunk Analysis
            if args.weight_chunk_size is not None and 'w_data' in record:
                # We need original w to be chunked.
                # We also need dequantized tensors per format to be recreated?
                # Storing them all in memory might be heavy.
                # Re-calculate on the fly in the loop above?
                # Wait, 'record' doesn't have the tensor data anymore.
                pass 
                
        # We need to do chunk analysis inside the layer loop where we have 'w'
        # But here we are iterating per metric.
        # So we need to compute ALL metric chunk errors in the first pass and store summary?
        # Storing summary {layer: {metric: {fmt: count}}} is light.
        # Let's Modify the FIRST loop (lines 402+).


        # Save CSV
        csv_path = os.path.join(metric_dir, "layer_errors.csv")
        csv_headers = ['layer', 'shape', 'max_val', 'best_type', 'best_error']
        if args.per_chunk_format:
            csv_headers.extend(['num_chunks', 'chunk_format_distribution'])
        csv_headers.extend([f"{qt}_error" for qt in qt_options])
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)
            
        # Global Plots
        plot_error_histograms(plot_data, qt_options, metric_dir, m)
        plot_error_boxplot(plot_data, qt_options, metric_dir, m)
        if args.weight_chunk_size:
            plot_chunk_win_rate(chunk_win_counts, qt_options, metric_dir, m, layer_winners=layer_config_m_simple)
            
            # Per-chunk format heatmap
            if args.per_chunk_format and chunk_format_map:
                plot_chunk_format_distribution(chunk_format_map, qt_options, metric_dir, m)
            
        # Layer-wise Error Comparison
        plot_layer_error_comparison(layer_results, qt_options, metric_dir, m)
        
        # Optimized Config
        quant_config = {
            'format': 'fp8_e4m3',
            'weight_mode': 'chunk' if args.weight_chunk_size else 'channel',
            'weight_chunk_size': args.weight_chunk_size
        }
        
        # Choose layer config based on mode
        if args.per_chunk_format and layer_config_per_chunk:
            quant_config['layers'] = layer_config_per_chunk
            quant_config['per_chunk_format'] = True
            print(f"Using per-chunk format selection for {len(layer_config_per_chunk)} layers")
        else:
            quant_config['layers'] = layer_config_m
        
        output_config = {
            'model': {'name': args.model_name, 'weights': args.weights},
            'adapter': {
                'type': 'generic', 
                'input_quantization': False,
                'quantized_ops': ['Conv2d', 'Linear'],
                'input_quantization_type': 'fp32'
            },
            'quantization': quant_config,
            'evaluation': eval_config,
            'dataset': dataset_base if args.run_eval else {
                 'name': args.dataset_name, 'path': args.dataset_path, 
                 'batch_size': args.batch_size, 'num_workers': args.num_workers
            }
        }
        
        config_path = os.path.join(metric_dir, "optimized_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(output_config, f, default_flow_style=False)
            
        print(f"Generated config for {m} at {config_path}")
        
        # Add to Evaluation
        if args.run_eval:
            run_config = output_config.copy()
            run_config['output_name'] = f"optimized_{m}"
            configs_to_run.append(run_config)

    # 3. Compute Theoretical Errors for all Configs (Pre-calc)
    # We do this from layer_results to avoid expensive re-calculation
    print("Pre-calculating theoretical errors from layer analysis...")
    precalc_errors = {} # output_name -> {metric: total_error}
    
    # helper to sum errors
    def get_total_error(fmt_map, metric_name):
        # fmt_map: {layer_name: format}
        total = 0.0
        for record in layer_results:
            lname = record['layer']
            fmt = fmt_map.get(lname)
            if fmt:
                val = record['metrics'][metric_name].get(fmt, 0.0)
                # Handle SQNR
                if metric_name == 'sqnr': val = -val
                total += val
        return total

    # 3.1 Ref & Baselines
    if args.run_eval:
        # Ref
        precalc_errors["ref_fp32"] = {m: 0.0 for m in metrics}
        
        # Baselines
        for fmt in baseline_formats:
            b_name = f"baseline_{fmt}"
            # map is just fmt for all layers
            fmt_map = {r['layer']: fmt for r in layer_results}
            
            b_errs = {}
            for m in metrics:
                b_errs[m] = get_total_error(fmt_map, m)
            precalc_errors[b_name] = b_errs

    # 3.2 Optimized Configs
    # We need to reconstruct the layer map for each optimized metric
    # We can do this by re-running the selection logic briefly
    for m_opt in metrics:
        opt_name = f"optimized_{m_opt}"
        
        # Determine selection for this metric
        sel_map = {}
        for record in layer_results:
            errs = record['metrics'][m_opt]
            # Find best
            best_val = float('inf')
            best_fmt = None
            for q_type, val in errs.items():
                 if val < best_val:
                     best_val = val
                     best_fmt = q_type
                 elif val == best_val:
                     # Tie-break (same as main loop)
                     if get_format_bits(q_type) < get_format_bits(best_fmt if best_fmt else 'fp32'):
                         best_val = val
                         best_fmt = q_type
            if best_fmt:
                sel_map[record['layer']] = best_fmt
                
        # Calculate errors for ALL metrics based on this selection
        # (e.g. what is the MSE of the model optimized for L1?)
        opt_errs = {}
        for m in metrics:
            opt_errs[m] = get_total_error(sel_map, m)
        precalc_errors[opt_name] = opt_errs

    # 4. Run Evaluation
    if args.run_eval and configs_to_run:
        print("\n--- Starting Evaluation Batch ---")
        
        # Deduplicate configs
        final_configs = []
        config_signatures = {} 
        skipped_map = {} 
        
        for cfg in configs_to_run:
            sig_dict = {
                'model': cfg.get('model'),
                'adapter': cfg.get('adapter'),
                'quantization': cfg.get('quantization')
            }
            signature = json.dumps(sig_dict, sort_keys=True)
            
            if signature in config_signatures:
                existing_name = config_signatures[signature]
                current_name = cfg['output_name']
                skipped_map[current_name] = existing_name
            else:
                config_signatures[signature] = cfg['output_name']
                final_configs.append(cfg)

        runner = Runner()
        print("Using Parallel Execution for Evaluation Phase...")
        results = runner.run_batch_parallel(final_configs, output_root=model_dir)

        # Re-hydrate skipped results
        if skipped_map:
            results_map = {r.get('output_name'): r for r in results}
            for skipped, original in skipped_map.items():
                if original in results_map:
                    res_clone = results_map[original].copy()
                    res_clone['output_name'] = skipped
                    results.append(res_clone)

        # Aggregate
        aggregator = ReportAggregator()
        summary_path = os.path.join(model_dir, "evaluation_summary.csv")
        aggregator.aggregate(results, summary_path)
        
        # Inject Pre-calculated Errors
        for res in results:
            name = res.get('output_name', '')
            if name in precalc_errors:
                res['errors'] = precalc_errors[name]
        
        # Plot Accuracy Comparison
        plot_accuracy_comparison(results, model_dir)
        
        # Markdown summary
        summary_md_path = os.path.join(model_dir, "evaluation_summary.md")
        aggregator.aggregate(results, summary_md_path)
        
        print(f"Evaluation completed. Summary saved to {summary_path}")

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
        for m in metrics:
            if m not in available_metrics:
                print(f"Warning: Skipping unknown metric '{m}'")
        metrics = [m for m in metrics if m in available_metrics]
    
    if not metrics:
        print("No valid metrics specified. Defaulting to l1.")
        metrics = ['l1']
        
    print(f"Metrics to analyze: {metrics}")
    
    # Base Output Directory
    if args.output_dir:
        base_root = args.output_dir
    else:
        base_root = os.path.join(PROJECT_ROOT, "runspace/experiments/optimal_layer_quant")

    if args.models_config:
        print(f"Loading models from config: {args.models_config}")
        with open(args.models_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Support both {'models': [...]} and direct list [...]
        if isinstance(config, dict) and 'models' in config:
            models_list = config['models']
        elif isinstance(config, list):
            models_list = config
        else:
            print("Error: Invalid models config format. Expected list or dict with 'models' key.")
            return

        print(f"Found {len(models_list)} models to process.")
        
        for model_cfg in models_list:
            # Extract name and weights
            if isinstance(model_cfg, str):
                name = model_cfg
                weights = "DEFAULT"
            else:
                name = model_cfg.get('name')
                weights = model_cfg.get('weights', 'DEFAULT')
            
            if not name:
                print("Skipping entry without name")
                continue

            # Update args for this run
            args.model_name = name
            args.weights = weights
            
            print(f"\n===========================================")
            print(f"Processing Model: {name}")
            print(f"===========================================")
            
            try:
                process_single_model(args, device, metrics, base_root)
            except Exception as e:
                print(f"Error processing model {name}: {e}")
                import traceback
                traceback.print_exc()
                
    else:
        # Single model mode
        process_single_model(args, device, metrics, base_root)

if __name__ == "__main__":
    main()
