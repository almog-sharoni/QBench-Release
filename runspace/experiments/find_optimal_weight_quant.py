
import os

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
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
    'fp4_e3m0' , 'fp4_e2m1' , 'fp4_e1m2' , 'fp4_e0m3',
    'ufp4_e4m0', 'ufp4_e3m1', 'ufp4_e2m2', 'ufp4_e1m3', 'ufp4_e0m4'
    'efp4_e3m0' , 'efp4_e2m1' , 'efp4_e1m2' , 'efp4_e0m3',
    'fp8_e1m6' , 'fp8_e0m7', 'fp8_e4m3','fp8_e5m2','fp8_e3m4','fp8_e6m1','fp8_e7m0','fp8_e2m5'
]


def get_args():
    parser = argparse.ArgumentParser(description="Find optimal layer-wise quantization")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--include_fp32", action="store_true", help="Include FP32 in search")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: runspace/experiments/optimal_layer_quant)")
    parser.add_argument("--models_config", type=str, default=None, help="Path to a YAML file containing a list of models to run")
    
    # Validation / Metric Args
    parser.add_argument("--metrics", type=str, default="l1,mse", help="Comma-separated metrics: l1, mse, sqnr, cosine OR 'all'")
    
    # Experiment Target
    parser.add_argument("--target", type=str, default="weights", choices=['weights', 'inputs'], help="Target to optimize: 'weights' or 'inputs'")
    
    # Evaluation Args
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation on FP32, FP8, and Optimized models")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers")
    parser.add_argument("--weight_chunk_size", type=int, default=128, help="Weight/Input chunk size (blocks). If set, enables chunked quantization.")
    parser.add_argument("--per_chunk_format", action="store_true", help="Enable per-chunk format selection (each chunk gets its own optimal format)")
    parser.add_argument("--plot_layers", action="store_true", help="Generate error bar plots for every single layer (warning: slow)")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit number of batches to process (default: -1 for all)")
    parser.add_argument("--baseline_formats", type=str, default=','.join(baseline_formats), help="Comma-separated list of formats to run as baselines (full eval)")
    parser.add_argument("--skip_layer_wise", action="store_true", help="Skip the layer-wise optimization experiment (only run chunk/baselines)")
    parser.add_argument("--force_recalc", action="store_true", help="Force recalculation of layer errors even if results exist")
    parser.add_argument("--input_method", type=str, default=None, help="Input quantization method (e.g. 'dynamic_mse', 'dynamic_l1', 'fp8_e4m3')")
    
    return parser.parse_args()

def get_quantized_tensor_sim(tensor, q_type, chunk_size=None, chunk_formats=None, mode=None):
    """
    Returns the dequantized tensor (simulated quantization).
    Returns (tensor_dequant, max_val).
    """
    if chunk_size is not None:
        # Use simple manual chunking to avoid op registry mismatches
        w_chunked, original_shape, pad_len = get_chunked_tensor(tensor, chunk_size=chunk_size)
        
        # Max per chunk (Abs)
        max_vals = w_chunked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9)
        
        scale = calculate_scale(max_vals, q_type)
        
        scaled = w_chunked / scale
        quant = quantize(scaled, q_type=q_type, rounding="nearest", validate=False)
        dequant = quant * scale
        
        # Flatten back
        flat = dequant.view(w_chunked.shape[0], -1)
        if pad_len > 0:
            flat = flat[:, :-pad_len]
            
        w_dequant = flat.view(original_shape)
        
        return w_dequant, max_vals.max().item()

    # Standard "Per Channel" or "Per Tensor" logic
    
    # If the user passes 'tensor', we treat as global.
    if mode == 'tensor':
         max_val = tensor.abs().amax(dim=None).clamp(min=1e-9) # Scalar
         scale = calculate_scale(max_val, q_type)
         t_scaled = tensor / scale
         t_quant = quantize(t_scaled, q_type=q_type, rounding="nearest", validate=False)
         t_deq = t_quant * scale
         return t_deq, max_val.item()

    # Default: Assumes WEIGHT (Channel dim 0)
    out_channels = tensor.shape[0]
    flat = tensor.view(out_channels, -1)
    
    max_val_per_channel = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-9)
    scale = calculate_scale(max_val_per_channel, q_type)
    
    scaled = flat / scale
    quant = quantize(scaled, q_type=q_type, rounding="nearest", validate=False)
    dequant = quant * scale
    
    dequant = dequant.view_as(tensor)
    return dequant, max_val_per_channel.max().item()


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

from runspace.experiments.utils.plotting import (
    plot_error_histograms,
    plot_error_boxplot,
    plot_oracle_heatmap,
    plot_accuracy_comparison,
    plot_chunk_format_distribution,
    plot_chunk_win_rate,
    plot_layer_error_comparison,
    get_format_bits,
    sort_formats,
    get_chunked_tensor,
    compute_mean_pow16_error
)

def plot_layer_error_bar(layer_name, errors, formats, output_dir, metric):
    """Generate a bar chart of errors for a single layer."""
    plt.figure(figsize=(10, 6))
    
    # Extract values
    raw_values = [errors.get(fmt, 0) for fmt in formats]
    
    # Transform for plotting if needed
    if metric == 'sqnr':
        values = [-v for v in raw_values]
        ylabel = "SQNR (dB)"
        best_val = max(values)
    else:
        values = raw_values
        ylabel = f"{metric.upper()} Error"
        best_val = min([v for v in values if v >= 0]) if any(v>=0 for v in values) else 0

    bars = plt.bar(formats, values, color='skyblue')
    
    for i, bar in enumerate(bars):
        if values[i] == best_val:
            bar.set_color('orange')
            
    plt.title(f"{metric.upper()} - {layer_name}")
    plt.ylabel(ylabel)
    plt.xlabel("Format")
    
    if metric in ['l1', 'mse', 'cosine']:
        plt.yscale('log')
        
    plt.grid(axis='y', alpha=0.3)
    
    safe_name = layer_name.replace('.', '_')
    plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
    plt.close()


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
    if args.target == 'inputs':
        model_dir = model_dir + "_inputs"  # Suffix for inputs
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Root output directory: {model_dir}")
    print(f"Target: {args.target}")
    
    # Load Model
    print(f"Loading model {args.model_name}...")
    config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
             'name': args.dataset_name, 'path': args.dataset_path, 
             'batch_size': args.batch_size, 'num_workers': args.num_workers
        }
    }
    
    try:
        adapter = create_adapter(config)
        model = adapter.model
        model.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    qt_options = baseline_formats.copy()
    if args.target == 'inputs':
        # Add UFP variants for all FP/Baseline inputs
        ufp_fmts = []
        # for fmt in qt_options:
        #     if fmt.startswith('fp'):
        #          ufp_fmts.append('u' + fmt)
        # qt_options.extend(ufp_fmts)
        
    if args.include_fp32:
        qt_options.insert(0, 'fp32')

    print(f"Testing types: {qt_options}")
    
    # Storage for all data
    layer_results_map = {} 
    
    # Check what needs calculation
    metrics_to_calc = []
    configs_to_run = []
    
    for m in metrics:
        metric_dir = os.path.join(model_dir, m)
        csv_path = os.path.join(metric_dir, "layer_errors.csv")
        
        if not args.force_recalc and os.path.exists(csv_path):
            cached_results, success = load_cached_results(csv_path, m)
            if success:
                print(f"Metric {m} found in cache. Skipping calculation.")
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
                    layer_results_map[lname]['metrics'][m] = res['metrics'][m]
            else:
                metrics_to_calc.append(m)
        else:
            metrics_to_calc.append(m)

    if not metrics_to_calc:
        print("All metrics cached. Skipping analysis phase.")
    else:
        print(f"Analyzing layers for metrics: {metrics_to_calc}...")
        
        supported_ops = tuple(OpRegistry.get_supported_ops().keys())
        
        # --- INPUT QUANTIZATION MODE ---
        if args.target == 'inputs':
             run_input_quantization_analysis(args, model, adapter, metrics_to_calc, qt_options, layer_results_map, device, config)
             
        # --- WEIGHT QUANTIZATION MODE ---
        else:
             run_weight_quantization_analysis(args, model, metrics_to_calc, qt_options, layer_results_map, supported_ops)

    # ... Shared Post-Processing ...
    
    # Flatten layer_results_map to list
    # Use model order if possible, otherwise list
    layer_order = []
    for name, module in model.named_modules():
         if name in layer_results_map:
             layer_order.append(name)
             
    layer_results = [layer_results_map[n] for n in layer_order]
    
    # Add any remaining (e.g. from cache but not in current model?)
    for name, res in layer_results_map.items():
        if name not in layer_order:
            layer_results.append(res)
            
    # Process Results Per Metric & Plotting (Shared)
    metric_configs = {} # metric -> config
    total_errors_by_config = {} # metric -> {output_name: total_error}
    
    # Eval config
    eval_config = {'mode': 'evaluate'}
    
    # Use limit_batches for evaluation if specified
    if args.limit_batches > 0:
        eval_config['max_batches'] = args.limit_batches
         
    dataset_base = config['dataset']

    # Always add Ref and Baselines if we are evaling (and not cached)
    if args.run_eval:
        # Ref
        ref_config = {
            'model': {'name': args.model_name, 'weights': args.weights},
            'adapter': {'type': 'generic', 'quantized_ops': []},
            'evaluation': eval_config,
            'dataset': dataset_base,
            'output_name': "ref_fp32"
        }
        configs_to_run.append(ref_config)
        
        # 2. Baselines for user-specified formats
        if args.include_fp32:
            pass
        
        # Parse baseline formats and strip whitespace
        requested_baselines = [fmt.strip() for fmt in args.baseline_formats.split(',') if fmt.strip()]
        
        for fmt in requested_baselines:
            # Verify format is valid if needed, or just run it (runner will fail if invalid)
            # if fmt not in baseline_formats: print warning?
            
            b_cfg = {
                'model': {'name': args.model_name, 'weights': args.weights},
                'adapter': {
                    'type': 'generic', 
                    'quantized_ops': ['-1'], # Quantize all supported
                    'input_quantization': True, 
                    'input_quantization_type': args.input_method if args.input_method else (fmt if args.target == 'inputs' else 'fp32') 
                },
                'quantization': {
                    'format': fmt, 
                },
                'evaluation': eval_config,
                'dataset': dataset_base,
                'output_name': f"baseline_{fmt}"
            }
            
            if args.target == 'inputs':
                # Force weights to FP32, Inputs to fmt
                b_cfg['quantization']['format'] = 'fp32' 
                b_cfg['adapter']['quantized_ops'] = ['-1'] 
                b_cfg['adapter']['input_quantization'] = True
                b_cfg['adapter']['input_quantization_type'] = fmt
                
            configs_to_run.append(b_cfg)

    # Initialize total_errors_by_config for all metrics upfront
    for m in metrics:
        total_errors_by_config[m] = {"ref_fp32": 0.0}

    # Process Metrics
    for m in metrics:
        print(f"\n--- Processing Metric: {m} ---")
        metric_dir = os.path.join(model_dir, m)
        os.makedirs(metric_dir, exist_ok=True)
        hist_dir = os.path.join(metric_dir, "histograms")
        os.makedirs(hist_dir, exist_ok=True)
        
        csv_rows = []
        plot_data = [] 
        chunk_win_counts = {}
        chunk_format_map = {}
        layer_winners_map = {} # layer -> best_fmt
        
        # Layer Configs for Optimized Model
        layer_config_m = {} 
        layer_config_per_chunk = {}
        
        for record in layer_results:
            name = record['layer']
            errs = record['metrics'][m]
            
            best_error = float('inf')
            best_type = None
            
            # Find best
            for q_type, val in errs.items():
                 # Tracking baseline errors for this metric
                 out_name = f"baseline_{q_type}"
                 total_errors_by_config[m][out_name] = total_errors_by_config[m].get(out_name, 0.0) + (val if val == val else 0.0)

                 if val < best_error:
                     best_error = val
                     best_type = q_type
                 elif val == best_error:
                    # Tie break by bits
                    cur_bits = get_format_bits(best_type) if best_type else 999
                    new_bits = get_format_bits(q_type)
                    if new_bits < cur_bits:
                        best_error = val
                        best_type = q_type
            
            pass
            
            # Chunk wins handling
            if args.weight_chunk_size and 'chunk_wins' in record:
                if m in record['chunk_wins']:
                    chunk_win_counts[name] = record['chunk_wins'][m]
                if m in record['chunk_winners']:
                    chunk_format_map[name] = record['chunk_winners'][m]

            if best_type:
                layer_winners_map[name] = best_type
                
                if args.target == 'inputs':
                    # Configuration for inputs
                    layer_config_m[name] = {'input_format': best_type}
                else:
                    layer_config_m[name] = {'format': best_type}
                
                # Cross-Metric Error Calculation for Optimized Layer
                # We know the best format for metric 'm' is best_type.
                # We accumulate the error of this format for ALL metrics.
                out_name_layer = f"optimized_layer_{m}"
                for m_other in metrics:
                    err_other = record['metrics'][m_other].get(best_type, float('inf'))
                    if np.isfinite(err_other):
                        total_errors_by_config[m_other][out_name_layer] = total_errors_by_config[m_other].get(out_name_layer, 0.0) + err_other

            # Per Chunk Config
            if args.per_chunk_format and args.weight_chunk_size:
                # Same logic for inputs? "per_chunk_input_format"?
                # Currently GenericAdapter probably doesn't support 'chunk_formats' for inputs from config easily.
                # But we'll structure it.
                if m in record.get('chunk_winners', {}):
                    c_winners = record['chunk_winners'][m]
                    fmt_dist = {f: c_winners.count(f) for f in set(c_winners)}
                    
                    if args.target == 'inputs':
                        layer_config_per_chunk[name] = {'input_chunk_formats': c_winners}
                    else:
                        layer_config_per_chunk[name] = {'chunk_formats': c_winners}
                    
                    # Normalization Factor for Chunk Errors (to match Layer metrics)
                    scale_factor = 1.0
                    if m == 'mse':
                         scale_factor = args.weight_chunk_size / record['numel']
                    elif m == 'cosine':
                         scale_factor = 1.0 / record.get('num_chunks', 1)

                    # Cross-Metric Error Calculation for Optimized Chunk
                    if 'chunk_cross_errors' in record:
                         out_name_chunk = f"optimized_chunk_{m}"
                         for m_other in metrics:
                              if m in record['chunk_cross_errors'] and m_other in record['chunk_cross_errors'][m]:
                                   err_chunk = record['chunk_cross_errors'][m][m_other]
                                   
                                   # Determine scale factor for TARGET metric
                                   sf_other = 1.0
                                   if m_other == 'mse':
                                        sf_other = args.weight_chunk_size / record['numel']
                                   elif m_other == 'cosine':
                                        sf_other = 1.0 / record.get('num_chunks', 1)
                                   
                                   if np.isfinite(err_chunk):
                                        total_errors_by_config[m_other][out_name_chunk] = total_errors_by_config[m_other].get(out_name_chunk, 0.0) + (err_chunk * sf_other)
                
                # Best_error for optimized chunk is accumulated using self-cross-error (which is correct Chunk Opt Error)
                # We multiply by scale_factor to match layer metric scale
                self_chunk_err = 0.0
                if 'chunk_cross_errors' in record and m in record['chunk_cross_errors'] and m in record['chunk_cross_errors'][m]:
                     self_chunk_err = record['chunk_cross_errors'][m][m] * scale_factor
                elif np.isfinite(best_error):
                     # Fallback to Layer Opt Error (pessimistic but safe) if chunk error missing
                     self_chunk_err = best_error
                
                total_errors_by_config[m][f"optimized_chunk_{m}"] = total_errors_by_config[m].get(f"optimized_chunk_{m}", 0.0) + self_chunk_err
            
            # Plot Layer
            if args.plot_layers:
                 try:
                    plot_layer_error_bar(name, errs, qt_options, hist_dir, m)
                 except: pass

            row = {
                'layer': name,
                'shape': record['shape'],
                'max_val': record['max_val'],
                'best_type': best_type,
                'best_error': best_error
            }
            if args.per_chunk_format and m in record.get('chunk_winners', {}):
                 c_winners = record['chunk_winners'][m]
                 dist = {f: c_winners.count(f) for f in set(c_winners)}
                 row['chunk_format_distribution'] = json.dumps(dist)
                 row['num_chunks'] = len(c_winners)

            for qt in qt_options:
                row[f"{qt}_error"] = errs.get(qt, "")
            csv_rows.append(row)
            
            plot_data.append({'errors': errs, 'best_error': best_error})
            
        # CSV and Plots
        # ... (same as original code, assume helper functions exist) ...
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
            plot_chunk_win_rate(chunk_win_counts, qt_options, metric_dir, m, layer_winners=layer_winners_map)
            # Only plot distribution map if we have static winners (Weights only)
            if args.target != 'inputs' and args.per_chunk_format and chunk_format_map:
                plot_chunk_format_distribution(chunk_format_map, qt_options, metric_dir, m)
            
        # Layer-wise Error Comparison
        plot_layer_error_comparison(layer_results, qt_options, metric_dir, m)
        
        # --- Generate Configs ---
        
        # Common Config Parts
        base_adapter_config = {
            'type': 'generic', 
            'input_quantization': True,
            'quantized_ops': ['Conv2d', 'Linear'],
            'quantize_first_layer': False
        }
        
        # Apply Input Method Override
        if args.input_method:
            base_adapter_config['input_quantization_type'] = args.input_method

        if args.target == 'inputs':
            # For inputs, we might want to set base input type if supported, but typically 
            # we rely on 'layers' override.
            pass
        
        dataset_cfg = dataset_base if args.run_eval else {
             'name': args.dataset_name, 'path': args.dataset_path, 
             'batch_size': args.batch_size, 'num_workers': args.num_workers
        }

        # 1. OPTIMIZED LAYERS CONFIG (Generate ONLY if NOT skipped)
        if not args.skip_layer_wise:
            # Use layer_config_m (best format per layer)
            layer_opt_config = {
                'model': {'name': args.model_name, 'weights': args.weights},
                'adapter': base_adapter_config.copy(),
                'quantization': {
                    'format': 'fp8_e4m3', # Default
                    'weight_mode': 'chunk' if args.weight_chunk_size else 'channel',
                    'weight_chunk_size': args.weight_chunk_size,
                    'layers': layer_config_m
                },
                'evaluation': eval_config,
                'dataset': dataset_cfg
            }
            
            # if args.target == 'inputs':
            #     #  layer_opt_config['quantization']['format'] = 'fp32'

            layer_cfg_path = os.path.join(metric_dir, "optimized_layer_config.yaml")
            with open(layer_cfg_path, 'w') as f:
                yaml.dump(layer_opt_config, f, default_flow_style=False)
            print(f"Generated Layer-Opt config for {m} at {layer_cfg_path}")
            
            if args.run_eval:
                run_config = layer_opt_config.copy()
                run_config['output_name'] = f"optimized_layer_{m}"
                configs_to_run.append(run_config)
        
        
        # 2. OPTIMIZED CHUNK CONFIG (If Chunking Enabled)
        if args.weight_chunk_size and args.per_chunk_format and layer_config_per_chunk:
            chunk_opt_config = {
                'model': {'name': args.model_name, 'weights': args.weights},
                'adapter': base_adapter_config.copy(),
                'quantization': {
                    'format': 'fp8_e4m3',
                    'weight_mode': 'chunk',
                    'weight_chunk_size': args.weight_chunk_size,
                    'per_chunk_format': True,
                    'layers': layer_config_per_chunk
                },
                'evaluation': eval_config,
                'dataset': dataset_cfg
            }
            
            # if args.target == 'inputs':
            #      chunk_opt_config['quantization']['format'] = 'fp32'

            chunk_cfg_path = os.path.join(metric_dir, "optimized_chunk_config.yaml")
            with open(chunk_cfg_path, 'w') as f:
                yaml.dump(chunk_opt_config, f, default_flow_style=False)
            print(f"Generated Chunk-Opt config for {m} at {chunk_cfg_path}")
            
            if args.run_eval:
                run_config = chunk_opt_config.copy()
                run_config['output_name'] = f"optimized_chunk_{m}"
                configs_to_run.append(run_config)

        # 3. OPTIMIZED ORACLE CONFIG (Dynamic Per-Chunk Runtime)
        if args.weight_chunk_size and args.target == 'inputs':
            # Only valid for chunked inputs
            oracle_config = {
                'model': {'name': args.model_name, 'weights': args.weights},
                'adapter': base_adapter_config.copy(),
                'quantization': {
                    'format': 'fp32', # Weights FP32
                    # 'weight_mode': 'chunk', # Not needed for weights if fp32
                    # 'weight_chunk_size': args.weight_chunk_size, 
                    # 'layers': layer_config_m 
                },
                'evaluation': eval_config,
                'dataset': dataset_cfg
            }
            
            # Set input quantization to dynamic oracle
            oracle_config['adapter']['input_quantization_type'] = f"dynamic_oracle_{m}"
            oracle_config['adapter']['input_chunk_size'] = args.weight_chunk_size # reusing weight chunk size arg for inputs
            
            oracle_cfg_path = os.path.join(metric_dir, "optimized_oracle_config.yaml")
            with open(oracle_cfg_path, 'w') as f:
                yaml.dump(oracle_config, f, default_flow_style=False)
            print(f"Generated Oracle-Opt config for {m} at {oracle_cfg_path}")
            
            if args.run_eval:
                run_config = oracle_config.copy()
                run_config['output_name'] = f"optimized_oracle_{m}"
                configs_to_run.append(run_config)
            
    # Evaluation Logic (same as before)
    # Evaluation Logic (same as before)
    if args.run_eval and configs_to_run:
        # ... (Execution same as original) ...
        # (For brevity, relying on original execution block or we can copy it if we replaced it)
        # I removed the full execution block in replace, so I must re-include it.
        # ...
        print("\n--- Starting Evaluation Batch ---")
        runner = Runner()
        print("Using Parallel Execution...")
        
        # Dedupe logic
        final_configs = []
        sigs = set()
        for c in configs_to_run:
             s = json.dumps(c, sort_keys=True)
             if s not in sigs:
                 sigs.add(s)
                 final_configs.append(c)

        # Enable Oracle Tracker
        tracker = None
        try:
            from runspace.src.tracking.oracle_tracker import OracleTracker
            tracker = OracleTracker()
            tracker.reset()
            tracker.enable()
        except ImportError:
            print("Warning: Could not import OracleTracker. Oracle visualization disabled.")
            tracker = None

        results = runner.run_batch_parallel(final_configs, output_root=model_dir)
        
        # Plot Oracle Heatmaps
        if tracker:
            oracle_stats, candidates = tracker.get_stats()
            tracker.disable()
            
            # If candidates were never set (e.g. no oracle runs), we skip
            if candidates:
                for run_id, layer_stats in oracle_stats.items():
                     out_path = os.path.join(model_dir, run_id)
                     if os.path.exists(out_path):
                          plot_oracle_heatmap(out_path, layer_stats, candidates, title_suffix=run_id)
                     else:
                          # Fallback: try looking in subdirs (e.g. if run_id uses slashes - unlikely here)
                          pass

        # Aggregate
        aggregator = ReportAggregator()
        summary_path = os.path.join(model_dir, "evaluation_summary.csv")
        aggregator.aggregate(results, summary_path)
        # Attach errors to results for plotting
        for res in results:
            out_name = res.get('output_name', '')
            res['errors'] = {}
            for m in metrics:
                if out_name in total_errors_by_config.get(m, {}):
                    res['errors'][m] = total_errors_by_config[m][out_name]
                    
        plot_accuracy_comparison(results, model_dir)
        print(f"Evaluation completed. Summary: {summary_path}")

def run_weight_quantization_analysis(args, model, metrics_to_calc, qt_options, layer_results_map, supported_ops):
    for name, module in tqdm(model.named_modules(), desc="Analyzing Layers (Weights)"):
        if isinstance(module, supported_ops):
            if not hasattr(module, 'weight') or module.weight is None: continue
            
            w = module.weight.data
            
            if name not in layer_results_map:
                layer_results_map[name] = {
                    'layer': name, 'shape': str(tuple(w.shape)),
                    'max_val': 0.0, 'metrics': {},
                    'chunk_wins': {}, 'chunk_winners': {}
                }
            record = layer_results_map[name]
            
            # Init metrics
            for m in metrics_to_calc:
                if m not in record['metrics']: 
                    record['metrics'][m] = {}; record['chunk_wins'][m] = {}; record['chunk_winners'][m] = []

            # Store size info for normalization
            record['numel'] = w.numel()
            if args.weight_chunk_size:
                record['num_chunks'] = (w.numel() + args.weight_chunk_size - 1) // args.weight_chunk_size # Approx/Max chunks
                # Actual chunks might be slightly different due to padding? 
                # get_chunked_tensor handles padding. 
                # Let's count actual chunks from the tensor shape if possible, or just use calculation.
                # get_chunked_tensor returns [B, N_chunks, size].
                # We can just use the calculated one as it's consistent.
            
            max_val_global = record['max_val']
            metric_chunk_errors = {m: {} for m in metrics_to_calc}

            for q_type in qt_options:
                try:
                    w_deq, mv = get_quantized_tensor_sim(w, q_type, chunk_size=args.weight_chunk_size)
                    max_val_global = max(max_val_global, mv)
                    
                    for m in metrics_to_calc:
                        err = calculate_error(w, w_deq, m)
                        record['metrics'][m][q_type] = err
                    
                    if args.weight_chunk_size:
                        w_chunked, _, _ = get_chunked_tensor(w, args.weight_chunk_size)
                        w_deq_chunked, _, _ = get_chunked_tensor(w_deq, args.weight_chunk_size)
                        diff = w_chunked - w_deq_chunked
                        
                        for m in metrics_to_calc:
                            if m == 'l1': chunk_errs = diff.abs().sum(dim=-1).view(-1)
                            elif m == 'mse': chunk_errs = diff.pow(2).mean(dim=-1).view(-1)
                            elif m == 'sqnr': 
                                sig = w_chunked.pow(2).sum(dim=-1).view(-1)
                                noise = diff.pow(2).sum(dim=-1).view(-1).clamp(min=1e-12)
                                chunk_errs = -10 * torch.log10(sig / noise)
                            elif m == 'cosine': 
                                sim = torch.nn.functional.cosine_similarity(w_chunked, w_deq_chunked, dim=-1)
                                chunk_errs = (1.0 - sim).view(-1)
                            metric_chunk_errors[m][q_type] = chunk_errs.cpu().numpy()
                except:
                    for m in metrics_to_calc: record['metrics'][m][q_type] = float('inf')
            
            # Process Chunk Wins
            if args.weight_chunk_size:
                 for m in metrics_to_calc:
                      valid_fmts = [qt for qt in qt_options if qt in metric_chunk_errors[m]]
                      if not valid_fmts: continue
                      valid_fmts_sorted = sorted(valid_fmts, key=get_format_bits)
                      err_matrix = np.stack([metric_chunk_errors[m][qt] for qt in valid_fmts_sorted])
                      best_indices = np.argmin(err_matrix, axis=0) # [NumChunks]
                      
                      winners_fmts = [valid_fmts_sorted[i] for i in best_indices]
                      record['chunk_winners'][m] = winners_fmts
                      wins = {'total': len(best_indices)}
                      for idx, qt in enumerate(valid_fmts_sorted):
                          wins[qt] = int(np.sum(best_indices == idx))
                      record['chunk_wins'][m] = wins
                      
                 # Calculate Chunk Cross-Errors
                 # Now that we have winners for all metrics, calculate how they perform on OTHER metrics
                 if 'chunk_winners' in record:
                      record['chunk_cross_errors'] = {}
                      for src_m in metrics_to_calc:
                           if src_m not in record['chunk_winners']: continue
                           winners = record['chunk_winners'][src_m]
                           
                           record['chunk_cross_errors'][src_m] = {}
                           
                           for tgt_m in metrics_to_calc:
                                # Calculate error of 'src_m winners' using 'tgt_m' error data
                                total_err = 0.0
                                possible = True
                                for i, fmt in enumerate(winners):
                                     if fmt in metric_chunk_errors[tgt_m]:
                                          total_err += float(metric_chunk_errors[tgt_m][fmt][i])
                                     else:
                                          total_err = float('inf')
                                          possible = False
                                          break
                                
                                if possible:
                                     record['chunk_cross_errors'][src_m][tgt_m] = total_err
                                else:
                                     record['chunk_cross_errors'][src_m][tgt_m] = float('inf')
                      
            record['max_val'] = max_val_global


class OnlineParams:
    def __init__(self, metrics, formats, chunk_size=None):
        self.metrics = metrics
        self.formats = formats
        self.chunk_size = chunk_size
        self.layer_stats = {} # name -> {'count': N, 'metrics': {m: {fmt: cum_err}}, 'chunk_errors': {m: tensor}}

def run_input_quantization_analysis(args, model, adapter, metrics_to_calc, qt_options, layer_results_map, device, config):
    print(f"Starting Input Quantization Analysis with Limit Batches: {args.limit_batches}")
    
    # Setup data loader
    runner = Runner(device)
    loader = runner.setup_data_loader(config)
    
    # tracker
    tracker = OnlineParams(metrics_to_calc, qt_options, args.weight_chunk_size)
    
    hooks = []
    supported_ops = tuple(OpRegistry.get_supported_ops().keys())
    
    batch_limit = args.limit_batches if args.limit_batches > 0 else float('inf')
    
    # Shared progress bar container
    pbar_ctx = {'bar': None}
    
    def get_hook(layer_name):
        def hook(module, input, output):
            # Input is tuple
            x = input[0]
            if not isinstance(x, torch.Tensor): 
                 if pbar_ctx['bar']: pbar_ctx['bar'].update(1)
                 return
            
            # Update progress (less frequently to reduce overhead)
            if pbar_ctx['bar']:
                 pbar_ctx['bar'].update(1)

            # Initialize stats if needed
            if layer_name not in tracker.layer_stats:
                tracker.layer_stats[layer_name] = {
                    'count': 0, 'max_val': 0.0,
                    'metrics': {m: {fmt: 0.0 for fmt in tracker.formats} for m in tracker.metrics},
                    'oracle_errors': {m: 0.0 for m in tracker.metrics}, 
                    'oracle_wins': {m: {fmt: 0 for fmt in tracker.formats} for m in tracker.metrics},
                    'chunk_count': 0
                }
            stats = tracker.layer_stats[layer_name]
            stats['count'] += x.shape[0]
            
            # Pre-calculate chunked tensors if chunking enabled
            x_chunked = None
            if tracker.chunk_size:
                 x_flat = x.flatten(1)
                 num_el = x_flat.shape[-1]
                 if num_el % tracker.chunk_size != 0:
                     pad = tracker.chunk_size - (num_el % tracker.chunk_size)
                     x_flat = torch.nn.functional.pad(x_flat, (0, pad))
                     
                 n_chunks = x_flat.shape[-1] // tracker.chunk_size
                 x_chunked = x_flat.view(x.shape[0], n_chunks, tracker.chunk_size)
                 
                 if stats['chunk_count'] == 0:
                     stats['chunk_count'] = n_chunks
            
            # OPTIMIZATION: Cache quantized tensors to avoid redundant computation
            # This is the main performance bottleneck - we were quantizing twice per format
            format_cache = {}
            
            for fmt in tracker.formats:
                 # Quantize ONCE and cache the result
                 x_deq, max_v = get_quantized_tensor_sim(x, q_type=fmt, chunk_size=tracker.chunk_size, mode='chunk' if tracker.chunk_size else 'tensor')
                 format_cache[fmt] = (x_deq, max_v)
                 
                 # Update max val stats
                 stats['max_val'] = max(stats['max_val'], max_v)
                 
                 # Calculate all metrics using cached result
                 for m in tracker.metrics:
                     val = calculate_error(x, x_deq, metric=m)
                     
                     if m == 'mse':
                         stats['metrics'][m][fmt] += val * x.numel()
                     else:
                         stats['metrics'][m][fmt] += val
            
            # Chunk Error Accumulation - across all chunks and formats
            if tracker.chunk_size and x_chunked is not None:
                for m in tracker.metrics:
                     errs_list = []
                     for fmt in tracker.formats:
                         x_deq, _ = format_cache[fmt]
                         diff = (x - x_deq).flatten(1)
                         
                         if diff.shape[-1] != x_chunked.shape[1] * x_chunked.shape[2]:
                              pad = tracker.chunk_size - (diff.shape[-1] % tracker.chunk_size)
                              diff = torch.nn.functional.pad(diff, (0, pad))
                              
                         diff_chunks = diff.view(x_chunked.shape)
                         
                         if m == 'l1':
                             chunk_err = diff_chunks.abs().sum(dim=2)
                         elif m == 'mse':
                             chunk_err = diff_chunks.pow(2).mean(dim=2)
                         elif m == 'cosine':
                             x_deq_flat = x_deq.flatten(1)
                             if x_deq_flat.shape[-1] % tracker.chunk_size != 0:
                                 x_deq_flat = torch.nn.functional.pad(x_deq_flat, (0, tracker.chunk_size - (x_deq_flat.shape[-1] % tracker.chunk_size)))
                             x_deq_chunks = x_deq_flat.view(x_chunked.shape)
                             cos_sim = torch.nn.functional.cosine_similarity(x_chunked, x_deq_chunks, dim=2)
                             chunk_err = 1.0 - cos_sim
                         else:
                             chunk_err = diff_chunks.abs().sum(dim=2)
                         
                         # Keep error per chunk per sample: [Batch, NumChunks]
                         errs_list.append(chunk_err)
                     
                     # Stack: [NumFormats, Batch, NumChunks]
                     errs_stack = torch.stack(errs_list)
                     
                     # ORACLE: Find min error per chunk per sample
                     # [Batch, NumChunks]
                     min_vals, min_indices = torch.min(errs_stack, dim=0)
                     
                     stats['oracle_errors'][m] += min_vals.sum().item()
                     
                     # Track Wins (Dynamic Histogram)
                     for fmt_idx, fmt in enumerate(tracker.formats):
                         count = (min_indices == fmt_idx).sum().item()
                         stats['oracle_wins'][m][fmt] += count
            
            # Cleanup
            del x_chunked
            del format_cache
            
        return hook

    # Register hooks (recursive)
    handles = []
    registered_counts = 0
    for name, module in model.named_modules():
        # Hook on Conv2d / Linear?
        # If target=inputs, we hook on Quantized Layers (QuantConv, QuantLinear) 
        # OR standard layers if not quantized yet.
        # But we want to capture INPUTS to these layers.
        # If model is already quantized (with 'fp32' type or whatever), we can hook.
        # We hook into everything in supported_ops
        if isinstance(module, tuple(OpRegistry.get_supported_ops().values())):
             # Quantized Module
             handles.append(module.register_forward_hook(get_hook(name)))
             registered_counts += 1
        elif isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.linear.Linear):
             # Standard Module
             handles.append(module.register_forward_hook(get_hook(name)))
             registered_counts += 1
             
    print(f"Registered hooks on {registered_counts} layers.")

    # Run
    # Run
    print("Running inference to capture/analyze inputs...")
    
    # helper Runner not needed, we use raw model
    model.eval()
    
    try:
        count = 0
        with torch.no_grad():
            loader_bar = tqdm(loader, desc="Batches")
            for batch in loader_bar:
                if count >= batch_limit:
                    break
                    
                # Reset Layer Progress Bar for this batch
                pbar_ctx['bar'] = tqdm(total=registered_counts, desc="Layers", leave=False)
                
                # Check signature of runner? Runner doesn't have prepare_batch exposed maybe
                # But GenericAdapter does.
                # Assuming adapter available in scope? Yes `adapter` passed to function.
                if adapter is not None:
                     images, labels = adapter.prepare_batch(batch)
                else:
                     images, labels = batch
                
                images = images.to(device)
                
                # Forward
                model(images)
                
                # Close Layer Progress Bar
                pbar_ctx['bar'].close()
                pbar_ctx['bar'] = None
                
                count += 1
    finally:
        for h in handles: h.remove()
    
    # Process Stats into layer_results_map
    for name, stats in tracker.layer_stats.items():
        total_samples = stats['count']
        if total_samples == 0: continue
        
        # Init record
        if name not in layer_results_map:
             layer_results_map[name] = {
                    'layer': name, 'shape': "N/A", # Inputs dynamic
                    'max_val': stats['max_val'], 'metrics': {},
                    'chunk_wins': {}, 'chunk_winners': {}
             }
        record = layer_results_map[name]
        
        # Move tensors to CPU once
        # stats['metrics_t'] is dict of tensors [NumFormats]
        
        # Normalize Metrics
        for m_idx, m in enumerate(tracker.metrics):
            if m not in record['metrics']: record['metrics'][m] = {}
            for fmt in tracker.formats:
                val = stats['metrics'][m][fmt]
                record['metrics'][m][fmt] = val / total_samples
            
            # Oracle Error
            if tracker.chunk_size:
                 oracle_val = stats['oracle_errors'][m]
                 record['metrics'][m][f"dynamic_oracle_{m}"] = oracle_val / total_samples
                 
                 # Chunk Wins
                 record['chunk_wins'][m] = stats['oracle_wins'][m]
                 # No static chunk_winners for inputs
                 


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
        if args.target == 'inputs':
             base_root = os.path.join(PROJECT_ROOT, "runspace/experiments/optimal_weight_quant_inputs")
        else:
             base_root = os.path.join(PROJECT_ROOT, "runspace/experiments/optimal_weight_quant")

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
