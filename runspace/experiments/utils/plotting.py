import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_error_histograms(data, formats, output_dir, metric):
    """Generate histograms of errors for each format."""
    print(f"Generating {metric} histograms...")
    
    # Collect errors per format
    format_errors = {fmt: [] for fmt in formats}
    for layer_data in data:
        for fmt in formats:
            if fmt in layer_data['errors']:
                err = layer_data['errors'][fmt]
                
                if metric == 'sqnr':
                    val = -err
                else:
                    val = err
                
                # Filter bad values
                if np.isfinite(val):
                    format_errors[fmt].append(val)

    # 1. Combined Histogram
    xlabel = ""
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

def get_format_bits(fmt):
    """
    Returns the bit width of a quantization format string.
    """
    if not fmt: return 32
    if fmt == 'fp32': return 32
    if fmt == 'int8': return 8
    if fmt == 'int4': return 4
    
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
        
    if fmt.startswith('fp'):
        try:
            return int(fmt.split('_')[0].replace('fp',''))
        except: pass
    if fmt.startswith('ufp'):
         try:
            return int(fmt.split('_')[0].replace('ufp',''))
         except: pass
         
    return 32 

def plot_chunk_win_rate(win_counts, formats, output_dir, metric, layer_winners=None):
    """
    Generate a stacked bar chart of chunk 'wins' per layer.
    """
    print(f"Generating {metric} chunk win rate plot...")
    
    layers = list(win_counts.keys())
    if not layers:
        return
        
    # Sort formats
    sorted_formats = sort_formats(formats)

    # Prepare data
    data = np.zeros((len(layers), len(sorted_formats)))
    
    for i, layer in enumerate(layers):
        total = win_counts[layer].get('total', 1)
        for j, fmt in enumerate(sorted_formats):
            count = win_counts[layer].get(fmt, 0)
            data[i, j] = count 
            
    plt.figure(figsize=(15, 8))
    
    bottom = np.zeros(len(layers))
    cmap = plt.get_cmap('tab20', len(sorted_formats))
    
    for j, fmt in enumerate(sorted_formats):
        plt.bar(layers, data[:, j], bottom=bottom, label=fmt, color=cmap(j))
        bottom += data[:, j]
        
    plt.ylabel('Number of Chunks')
    plt.title(f'Chunk Format Selection per Layer ({metric.upper()})')
    plt.xticks(rotation=90, fontsize=8)
    
    active_indices = [j for j in range(len(sorted_formats)) if data[:, j].sum() > 0]
    
    handles, labels = plt.gca().get_legend_handles_labels()
    filtered_handles = [handles[j] for j in active_indices]
    filtered_labels = [labels[j] for j in active_indices]
    
    plt.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"chunk_win_rate_{metric}.png"))
    plt.close()
def compute_mean_pow16_error(tensor_a, tensor_b):
    delta = tensor_a - tensor_b
    delta_pow16 = delta.pow(16)
    return delta_pow16.mean().item()

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
    """
    print(f"Generating {metric} per-chunk format distribution heatmap...")
    
    if not chunk_formats:
        return
        
    sorted_formats = sort_formats(formats)
    fmt_to_idx = {fmt: i for i, fmt in enumerate(sorted_formats)}
    
    layers = list(chunk_formats.keys())
    max_chunks = max(len(chunk_formats[layer]) for layer in layers)
    
    matrix = np.full((len(layers), max_chunks), -1, dtype=int)
    
    for i, layer in enumerate(layers):
        for j, fmt in enumerate(chunk_formats[layer]):
            if fmt in fmt_to_idx:
                matrix[i, j] = fmt_to_idx[fmt]
    
    plt.figure(figsize=(20, max(8, len(layers) * 0.3)))
    
    cmap = plt.get_cmap('tab20', len(sorted_formats))
    cmap.set_bad(color='lightgray') 
    
    masked_matrix = np.ma.masked_where(matrix < 0, matrix)
    
    max_display_width = 1000
    if masked_matrix.shape[1] > max_display_width:
        print(f"Downsampling heatmap from {masked_matrix.shape[1]} to {max_display_width} columns...")
        factor = masked_matrix.shape[1] // max_display_width
        remainder = masked_matrix.shape[1] % max_display_width
        trimmed_width = masked_matrix.shape[1] - remainder
        reshaped = masked_matrix[:, :trimmed_width].reshape(len(layers), max_display_width, factor)
        masked_matrix = reshaped[:, :, 0]
    
    im = plt.imshow(masked_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=len(sorted_formats)-1)
    
    cbar = plt.colorbar(im, ticks=range(len(sorted_formats)))
    cbar.ax.set_yticklabels(sorted_formats)
    
    plt.title(f"{metric.upper()} Per-Chunk Format Selection")
    plt.ylabel("Layers")
    plt.xlabel("Chunk Index")
    plt.yticks(range(len(layers)), layers, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"chunk_format_heatmap_{metric}.png"), dpi=150)
    plt.close()

def plot_layer_error_comparison(layer_results, formats, output_dir, metric):
    """
    Generate a line plot comparing total error of each format across layers.
    """
    print(f"Generating {metric} layer-wise error comparison plot...")
    
    if not layer_results:
        return
        
    layers = [res['layer'] for res in layer_results]
    x = np.arange(len(layers))
    
    format_series = {fmt: [] for fmt in formats}
    valid_formats = []
    
    for fmt in formats:
        has_data = False
        series = []
        for res in layer_results:
            err = res['metrics'][metric].get(fmt, float('nan'))
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
    cmap = plt.get_cmap('tab20')
    fmt_colors = {fmt: cmap(i % 20) for i, fmt in enumerate(formats)}

    for fmt in valid_formats:
        vals = np.array(format_series[fmt])
        mask = np.isfinite(vals)
        if metric in ['l1', 'mse', 'cosine']:
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
        error_vals.append(res.get('total_error', 0.0)) 

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
    
    # Error Rate (Secondary Axes)
    error_series = {} 
    valid_metrics = set()
    
    for res in results:
        errs = res.get('errors', {})
        if isinstance(errs, dict):
            for m, val in errs.items():
                valid_metrics.add(m)
    
    if valid_metrics:
        # Sort metrics to ensure consistent ordering
        sorted_metrics = sorted(list(valid_metrics))
        
        # Collect all data first
        metric_data = {}
        for m in sorted_metrics:
            vals = []
            for res in results:
                err_dict = res.get('errors', {})
                val = err_dict.get(m, float('nan'))
                vals.append(val)
            metric_data[m] = np.array(vals)

        cmap = plt.get_cmap('Dark2') 
        extra_axes = []

        for i, m in enumerate(sorted_metrics):
            vals = metric_data[m]
            
            # Create new axis
            if i == 0:
                ax_new = ax1.twinx()
            else:
                ax_new = ax1.twinx()
                # Offset the right spine for subsequent axes
                offset = 60 * i 
                ax_new.spines["right"].set_position(("outward", offset))

            extra_axes.append(ax_new)
            
            color = cmap(i % 8)
            label = f"{m.upper()}" 
            
            ax_new.set_ylabel(f'{label} Value', color=color)
            ax_new.tick_params(axis='y', labelcolor=color)
            
            # Plot Raw Values
            mask = np.isfinite(vals)
            ax_new.plot(x[mask], vals[mask], color=color, marker='o', linestyle='-', linewidth=2, alpha=0.8, label=label)
            
            # Add value annotations
            for j, val in enumerate(vals):
                if np.isfinite(val):
                    ax_new.annotate(f'{val:.4g}', 
                                 xy=(x[j], val),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

        # Combine legends and move outside (Manual handling due to multiple axes)
        lines1, labels1 = ax1.get_legend_handles_labels()
        
        # Gather lines from extra axes
        all_lines = lines1[:]
        all_labels = labels1[:]
        
        # We need to explicitly get handles from extra axes because ax1 doesn't know about them
        for ax_ex in extra_axes:
            l, lb = ax_ex.get_legend_handles_labels()
            all_lines.extend(l)
            all_labels.extend(lb)

        ax1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1.05 + (len(sorted_metrics)-1)*0.1, 1))
        
        plt.subplots_adjust(right=0.75 - (len(sorted_metrics)-1)*0.1)

    # Add labels to bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()

def plot_oracle_heatmap(output_dir, stats, candidates, title_suffix=""):
    """
    Plots a stacked bar chart of format distribution per layer (Oracle Choices).
    """
    print(f"Generating Oracle Distribution Plot ({title_suffix})...")
    
    layers = sorted(list(stats.keys()))
    if not layers:
        return
        
    sorted_formats = sort_formats(candidates)
    fmt_to_idx = {fmt: i for i, fmt in enumerate(candidates)}
        
    data = np.zeros((len(layers), len(sorted_formats)))
    
    for i, layer in enumerate(layers):
        counts = stats[layer].get('total_counts', np.zeros(len(candidates)))
        for j, fmt in enumerate(sorted_formats):
            original_idx = fmt_to_idx.get(fmt)
            if original_idx is not None and original_idx < len(counts):
                data[i, j] = counts[original_idx]
            
    plt.figure(figsize=(15, 8))
    
    bottom = np.zeros(len(layers))
    cmap = plt.get_cmap('tab20', len(sorted_formats))
    
    for j, fmt in enumerate(sorted_formats):
        plt.bar(layers, data[:, j], bottom=bottom, label=fmt, color=cmap(j))
        bottom += data[:, j]
        
    plt.ylabel('Number of Chunks')
    plt.title(f"Dynamic Oracle Format Choices per Layer ({title_suffix})")
    plt.xticks(rotation=90, fontsize=8)
    
    active_indices = [j for j in range(len(sorted_formats)) if data[:, j].sum() > 0]
    
    if active_indices:
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles = [handles[j] for j in active_indices]
        filtered_labels = [labels[j] for j in active_indices]
        plt.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"oracle_distribution_{title_suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved oracle distribution to {save_path}")

