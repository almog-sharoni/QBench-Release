
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.ops.quant_base import calculate_scale, quantize_tensor
from runspace.src.database.handler import RunDatabase
from runspace.src.quantization.quantizer import quantize 
# from runspace.src.quantization.quantizer import quantize_fp_generic_i32 as quantize_i32
from runspace.src.quantization.constants import get_quantization_bias
# Late import
from runspace.core.runner import Runner
from runspace.core.report_aggregator import ReportAggregator
from runspace.src.registry.op_registry import OpRegistry

baseline_formats = [
    # 'fp4_e1m2' , 'fp4_e2m1' ,'fp4_e3m0' ,
    # 'ufp4_e4m0', 'ufp4_e3m1', 'ufp4_e2m2', 'ufp4_e1m3',
    # 'efp4_e3m0' , 'efp4_e2m1' , 'efp4_e1m2' ,
    # 'fp8_e1m6' , 'fp8_e4m3','fp8_e5m2','fp8_e3m4','fp8_e6m1','fp8_e7m0','fp8_e2m5'
    #  'fp5_e1m3' , 'fp5_e2m2', 'fp5_e3m1', 'fp5_e4m0'
    'fp2_e1m0', 'fp3_e1m1', 'fp4_e1m2', 'fp5_e1m3', 'fp6_e1m4', 'fp7_e1m5', 'fp8_e1m6',
    'fp3_e2m0', 'fp4_e3m0', 'fp5_e4m0', 'fp6_e5m0', 'fp7_e6m0', 'fp8_e7m0'

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
    parser.add_argument("--target", type=str, default="weights", choices=['weights'], help="Target to optimize: 'weights'")
    
    # Evaluation Args
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation on FP32, FP8, and Optimized models")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--weight_chunk_size", type=int, default=128, help="Weight/Input chunk size (blocks). If set, enables chunked quantization.")
    parser.add_argument("--per_chunk_format", action="store_true", help="Enable per-chunk format selection (each chunk gets its own optimal format)")
    parser.add_argument("--plot_layers", action="store_true", help="Generate error bar plots for every single layer (warning: slow)")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit number of batches to process (default: -1 for all)")
    parser.add_argument("--baseline_formats", type=str, default=','.join(baseline_formats), help="Comma-separated list of formats to run as baselines (full eval)")
    parser.add_argument("--skip_layer_wise", action="store_true", help="Skip the layer-wise optimization experiment (only run chunk/baselines)")
    parser.add_argument("--force_recalc", action="store_true", help="Force recalculation of layer errors even if results exist")
    
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

def create_quantized_state_dict(model, layer_results_map, args, metric, use_chunking=False):
    """
    Creates a state dict with quantized weights based on the optimal formats found.

    MHA handling:
    The analysis splits nn.MultiheadAttention.in_proj_weight into virtual entries named
    '<mha>.q_proj', '<mha>.k_proj', '<mha>.v_proj' in layer_results_map.
    We detect these and re-pack them into the correct state dict key '<mha>.in_proj_weight'.
    The out_proj is handled via '<mha>.out_proj' → '<mha>.out_proj.weight' as usual.
    """
    print(f"Creating quantized state dict for metric {metric}...")
    state_dict = model.state_dict()
    quant_map = {}

    def _best_format(record):
        """Return (best_type, chunk_formats) for a layer record."""
        if use_chunking and args.per_chunk_format and args.weight_chunk_size and metric in record.get('chunk_winners', {}):
            return None, record['chunk_winners'][metric]
        errs = record['metrics'][metric]
        best_err = float('inf')
        best_type = None
        for qt, err in errs.items():
            if err < best_err:
                best_err = err
                best_type = qt
            elif err == best_err:
                if get_format_bits(qt) < (get_format_bits(best_type) if best_type else 999):
                    best_type = qt
        return best_type, None

    def _quantize_weight(w, best_type, chunk_formats):
        """Quantize a single weight tensor; returns dequantized tensor."""
        if chunk_formats:
            w_chunked, original_shape, pad_len = get_chunked_tensor(w, chunk_size=args.weight_chunk_size)
            batch_size, num_chunks, chunk_size_ = w_chunked.shape
            w_chunked_flat = w_chunked.reshape(-1, chunk_size_)
            total_chunks = w_chunked_flat.shape[0]
            w_dequant_flat = torch.zeros_like(w_chunked_flat)
            current_formats = chunk_formats[:total_chunks]
            fmt_to_indices = {}
            for idx, fmt in enumerate(current_formats):
                fmt_to_indices.setdefault(fmt, []).append(idx)
            for fmt, indices in fmt_to_indices.items():
                if not indices: continue
                idx_tensor = torch.tensor(indices, device=w.device)
                target_chunks = w_chunked_flat[idx_tensor]
                dq_chunks, _ = get_quantized_tensor_sim(target_chunks, fmt)
                w_dequant_flat[idx_tensor] = dq_chunks
            if len(current_formats) < total_chunks:
                w_dequant_flat[len(current_formats):] = w_chunked_flat[len(current_formats):]
            w_dequant_chunked = w_dequant_flat.view(batch_size, num_chunks, chunk_size_)
            flat = w_dequant_chunked.view(batch_size, -1)
            if pad_len > 0:
                flat = flat[:, :-pad_len]
            return flat.view(original_shape)
        elif best_type:
            w_dequant, _ = get_quantized_tensor_sim(w, best_type, chunk_size=args.weight_chunk_size)
            return w_dequant
        return w  # no quantization

    # --- Collect MHA parent names so we can reconstruct in_proj_weight ---
    # Virtual names are '<mha_name>.q_proj', '.k_proj', '.v_proj'
    mha_parents = {}  # mha_name -> {'q': ..., 'k': ..., 'v': ...}
    for lname in layer_results_map:
        for suffix in ('.q_proj', '.k_proj', '.v_proj'):
            if lname.endswith(suffix):
                parent = lname[: -len(suffix)]
                mha_parents.setdefault(parent, {})[suffix[1:]] = lname  # e.g. 'q_proj' -> full name

    # Quantize MHA in_proj_weight by reassembling q/k/v
    handled_names = set()
    for mha_name, proj_map in mha_parents.items():
        q_name = proj_map.get('q_proj')
        k_name = proj_map.get('k_proj')
        v_name = proj_map.get('v_proj')
        in_proj_key = f"{mha_name}.in_proj_weight"
        if in_proj_key not in state_dict:
            continue  # already decomposed model; skip
        w_in_proj = state_dict[in_proj_key]
        embed_dim = w_in_proj.shape[0] // 3
        slices = {'q_proj': (0, embed_dim), 'k_proj': (embed_dim, 2*embed_dim), 'v_proj': (2*embed_dim, 3*embed_dim)}
        new_in_proj = w_in_proj.clone()
        for proj_key, (start, end) in slices.items():
            vname = proj_map.get(proj_key)
            if vname and vname in layer_results_map:
                record = layer_results_map[vname]
                best_type, chunk_formats = _best_format(record)
                w_slice = w_in_proj[start:end].contiguous()
                new_in_proj[start:end] = _quantize_weight(w_slice, best_type, chunk_formats)
                quant_map[vname] = chunk_formats if chunk_formats else best_type
        state_dict[in_proj_key] = new_in_proj
        handled_names.update(proj_map.values())

    # --- Handle all other layers (including MHA out_proj if present as a real module) ---
    for name, module in model.named_modules():
        if name in layer_results_map and name not in handled_names:
            record = layer_results_map[name]
            best_type, chunk_formats = _best_format(record)
            weight_key = f"{name}.weight"
            if weight_key in state_dict:
                w = state_dict[weight_key]
                state_dict[weight_key] = _quantize_weight(w, best_type, chunk_formats)
                quant_map[name] = chunk_formats if chunk_formats else best_type

    return state_dict, quant_map

def process_single_model(args, device, metrics, base_root):
    # Initialize Database
    db = RunDatabase()
    
    # Valid model path: base_root / model_name
    model_dir = os.path.join(base_root, args.model_name)

    
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

        # --- WEIGHT QUANTIZATION MODE ---

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
                    'input_quantization': False, 
                    # 'input_quantization_type': args.input_method if args.input_method else (fmt if args.target == 'inputs' else 'fp32') 
                },
                'quantization': {
                    'format': fmt,
                    'weight_mode': 'chunk',
                    'weight_chunk_size': args.weight_chunk_size
                },
                'evaluation': eval_config,
                'dataset': dataset_base,
                'output_name': f"baseline_{fmt}"
            }
            

                
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
        
        # Calculate Total elements for normalization
        total_elements = sum(r.get('numel', 0) for r in layer_results)
        
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
                 # Accumulate correctly for global normalization
                 # L1 is already a sum, MSE is a mean and needs to be scaled by numel
                 val_raw = (val if val == val else 0.0)
                 val_to_add = val_raw * record.get('numel', 1) if m == 'mse' else val_raw
                 
                 out_name = f"baseline_{q_type}"
                 total_errors_by_config[m][out_name] = total_errors_by_config[m].get(out_name, 0.0) + val_to_add

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
                

                layer_config_m[name] = {'format': best_type}
                
                # Cross-Metric Error Calculation for Optimized Layer
                # We know the best format for metric 'm' is best_type.
                # We accumulate the error of this format for ALL metrics.
                out_name_layer = f"optimized_layer_{m}"
                for m_other in metrics:
                    err_other = record['metrics'][m_other].get(best_type, float('inf'))
                    if np.isfinite(err_other):
                        val_to_add = err_other * record.get('numel', 1) if m_other == 'mse' else err_other
                        total_errors_by_config[m_other][out_name_layer] = total_errors_by_config[m_other].get(out_name_layer, 0.0) + val_to_add

            # Per Chunk Config
            if args.per_chunk_format and args.weight_chunk_size:
                if m in record.get('chunk_winners', {}):
                    c_winners = record['chunk_winners'][m]
                    fmt_dist = {f: c_winners.count(f) for f in set(c_winners)}

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
                         
                         # We want to see how the configuration optimized for 'm' performs on ALL metrics
                         # So for each m_other, we look up the error of 'm' winners on 'm_other'
                         for m_other in metrics:
                              if m in record['chunk_cross_errors'] and m_other in record['chunk_cross_errors'][m]:
                                   err_chunk = record['chunk_cross_errors'][m][m_other]
                                   
                                   # Determine scale factor for TARGET metric (m_other)
                                   # The error stored in chunk_cross_errors is raw sum/mean from chunks
                                   # We need to normalize it to match Layer metrics for m_other
                                   sf_other = 1.0
                                   if m_other == 'mse':
                                        sf_other = args.weight_chunk_size / record['numel']
                                   elif m_other == 'cosine':
                                        sf_other = 1.0 / record.get('num_chunks', 1)
                                   
                                   if np.isfinite(err_chunk):
                                        # Accumulate into m_other's entry for this config
                                        # Scale by numel if MSE to convert layer-mean back to total sum-squared
                                        val_to_add = (err_chunk * sf_other * record.get('numel', 1)) if m_other == 'mse' else (err_chunk * sf_other)
                                        total_errors_by_config[m_other][out_name_chunk] = total_errors_by_config[m_other].get(out_name_chunk, 0.0) + val_to_add

                # For the CURRENT metric 'm', we also use the value from chunk_cross_errors[m][m]
                # This ensures consistent scaling.
                self_chunk_err = 0.0
                if 'chunk_cross_errors' in record and m in record['chunk_cross_errors'] and m in record['chunk_cross_errors'][m]:
                     # Re-calculate scale factor for 'm'
                     sf_self = 1.0
                     if m == 'mse':
                          sf_self = args.weight_chunk_size / record['numel']
                     elif m == 'cosine':
                          sf_self = 1.0 / record.get('num_chunks', 1)
                     self_chunk_err = record['chunk_cross_errors'][m][m] * sf_self
                elif np.isfinite(best_error):
                     self_chunk_err = best_error # Fallback

                # Update the main entry for this config on this metric
                # overwrite previous value to avoid double counting if the loop above already set it (it does)
                # actually, the loop above sets total_errors_by_config[m][optimized_chunk_m] when m_other == m
                # so we don't need to do anything here if the loop covers it.
                # BUT, let's be safe and ensure the self-metric is correct.
                # The loop sets it. So we are good.

            
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
            if args.per_chunk_format and chunk_format_map:
                plot_chunk_format_distribution(chunk_format_map, qt_options, metric_dir, m)
            
        # Layer-wise Error Comparison
        plot_layer_error_comparison(layer_results, qt_options, metric_dir, m)
        
        # --- Generate Configs ---
        
        # Common Config Parts
        base_adapter_config = {
            'type': 'generic', 
            'input_quantization': False,
            'quantized_ops': [],
            'quantize_first_layer': False,
        }
        

        
        dataset_cfg = dataset_base if args.run_eval else {
             'name': args.dataset_name, 'path': args.dataset_path, 
             'batch_size': args.batch_size, 'num_workers': args.num_workers
        }

        # 1. OPTIMIZED LAYERS CONFIG (Generate ONLY if NOT skipped)
        if not args.skip_layer_wise:
            # Create Quantized Weights
            q_state_dict, q_map = create_quantized_state_dict(model, layer_results_map, args, m, use_chunking=False)
            q_weights_path = os.path.join(metric_dir, "quantized_weights_layer.pt")
            torch.save(q_state_dict, q_weights_path)
            print(f"Saved quantized weights to {q_weights_path}")
            
            # Save Map
            q_map_path = os.path.join(metric_dir, "quantization_map_layer.json")
            with open(q_map_path, 'w') as f:
                json.dump(q_map, f, indent=4)
            print(f"Saved quantization map to {q_map_path}")

            # Use layer_config_m (best format per layer)
            layer_opt_config = {
                'model': {'name': args.model_name, 'weights': os.path.abspath(q_weights_path)},
                'adapter': base_adapter_config.copy(),
                'quantization': {
                    'format': 'fp8_e4m3', # Default dummy
                    'layers': {} # Empty layers config
                },
                'evaluation': eval_config,
                'dataset': dataset_cfg
            }
            


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
            # Create Quantized Weights (Chunked)
            q_state_dict_chunk, q_map_chunk = create_quantized_state_dict(model, layer_results_map, args, m, use_chunking=True)
            q_weights_path_chunk = os.path.join(metric_dir, "quantized_weights_chunk.pt")
            torch.save(q_state_dict_chunk, q_weights_path_chunk)
            print(f"Saved chunk-quantized weights to {q_weights_path_chunk}")
            
            # Save Map
            q_map_chunk_path = os.path.join(metric_dir, "quantization_map_chunk.json")
            with open(q_map_chunk_path, 'w') as f:
                json.dump(q_map_chunk, f, indent=4)
            print(f"Saved chunk quantization map to {q_map_chunk_path}")

            chunk_opt_config = {
                'model': {'name': args.model_name, 'weights': os.path.abspath(q_weights_path_chunk)},
                'adapter': base_adapter_config.copy(),
                'quantization': {
                    'format': 'fp8_e4m3',
                    'layers': {}
                },
                'evaluation': eval_config,
                'dataset': dataset_cfg
            }
            

            chunk_cfg_path = os.path.join(metric_dir, "optimized_chunk_config.yaml")
            with open(chunk_cfg_path, 'w') as f:
                yaml.dump(chunk_opt_config, f, default_flow_style=False)
            print(f"Generated Chunk-Opt config for {m} at {chunk_cfg_path}")
            
            if args.run_eval:
                run_config = chunk_opt_config.copy()
                run_config['output_name'] = f"optimized_chunk_{m}"
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
        
        # --- Log Results to Database ---
        # 1. Identify Reference (ref_fp32)
        ref_acc1, ref_acc5, ref_certainty = 0.0, 0.0, 0.0
        for res in results:
            if res.get('output_name') == 'ref_fp32':
                ref_acc1 = res.get('acc1', 0.0)
                ref_acc5 = res.get('acc5', 0.0)
                ref_certainty = res.get('certainty', 0.0)
                break
        
        # 2. Log all results except ref_fp32
        for res in results:
            out_name = res.get('output_name', '')
            if out_name == 'ref_fp32':
                continue
                
            # For weight optimization, activation is usually fp32
            # Weight DT is based on out_name (baseline_xxx or optimized_xxx)
            weight_dt = out_name.replace('baseline_', '').replace('optimized_', 'opt_')
            
            # Fetch MSE/L1 if available and normalize
            total_elements = sum(r.get('numel', 0) for r in layer_results)
            
            mse = total_errors_by_config.get('mse', {}).get(out_name)
            if mse is not None and total_elements > 0:
                 mse /= total_elements

            l1 = total_errors_by_config.get('l1', {}).get(out_name)
            if l1 is not None and total_elements > 0:
                 l1 /= total_elements
            
            # Certainty (softmax based)
            certainty = res.get('certainty', 0.0)
            acc1 = res.get('acc1', 0.0)

            db.log_run(
                model_name=res.get('model_name', args.model_name),
                weight_dt=weight_dt,
                activation_dt="fp32", # Weight experiment assumes fp32 activations
                acc1=acc1,
                acc5=res.get('acc5', 0.0),
                ref_acc1=ref_acc1,
                ref_acc5=ref_acc5,
                ref_certainty=ref_certainty,
                experiment_type="weight_quant_baseline" if out_name.startswith('baseline_') else "weight_quant_optimized",
                status=res.get('status', 'SUCCESS'),
                mse=mse,
                l1=l1,
                certainty=certainty
            )
        
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

def _analyze_weight_tensor(w, layer_name, args, metrics_to_calc, qt_options, layer_results_map):
    """
    Analyze a single weight tensor (w) against all qt_options and populate layer_results_map.
    layer_name is the key used in layer_results_map (and should match the name used during
    deployment, e.g. 'encoder.layers.0.self_attn.q_proj').
    """
    if layer_name not in layer_results_map:
        layer_results_map[layer_name] = {
            'layer': layer_name, 'shape': str(tuple(w.shape)),
            'max_val': 0.0, 'metrics': {},
            'chunk_wins': {}, 'chunk_winners': {}
        }
    record = layer_results_map[layer_name]

    for m in metrics_to_calc:
        if m not in record['metrics']:
            record['metrics'][m] = {}
            record['chunk_wins'][m] = {}
            record['chunk_winners'][m] = []

    record['numel'] = w.numel()
    if args.weight_chunk_size:
        record['num_chunks'] = (w.numel() + args.weight_chunk_size - 1) // args.weight_chunk_size

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
                    if m == 'l1':    chunk_errs = diff.abs().sum(dim=-1).view(-1)
                    elif m == 'mse': chunk_errs = diff.pow(2).mean(dim=-1).view(-1)
                    elif m == 'sqnr':
                        sig   = w_chunked.pow(2).sum(dim=-1).view(-1)
                        noise = diff.pow(2).sum(dim=-1).view(-1).clamp(min=1e-12)
                        chunk_errs = -10 * torch.log10(sig / noise)
                    elif m == 'cosine':
                        sim = torch.nn.functional.cosine_similarity(w_chunked, w_deq_chunked, dim=-1)
                        chunk_errs = (1.0 - sim).view(-1)
                    metric_chunk_errors[m][q_type] = chunk_errs.cpu().numpy()
        except:
            for m in metrics_to_calc:
                record['metrics'][m][q_type] = float('inf')

    # Process Chunk Wins
    if args.weight_chunk_size:
        for m in metrics_to_calc:
            valid_fmts = [qt for qt in qt_options if qt in metric_chunk_errors[m]]
            if not valid_fmts:
                continue
            valid_fmts_sorted = sorted(valid_fmts, key=get_format_bits)
            err_matrix = np.stack([metric_chunk_errors[m][qt] for qt in valid_fmts_sorted])
            best_indices = np.argmin(err_matrix, axis=0)  # [NumChunks]

            winners_fmts = [valid_fmts_sorted[i] for i in best_indices]
            record['chunk_winners'][m] = winners_fmts
            wins = {'total': len(best_indices)}
            for idx, qt in enumerate(valid_fmts_sorted):
                wins[qt] = int(np.sum(best_indices == idx))
            record['chunk_wins'][m] = wins

        # Chunk Cross-Errors
        if 'chunk_winners' in record:
            record['chunk_cross_errors'] = {}
            for src_m in metrics_to_calc:
                if src_m not in record['chunk_winners']:
                    continue
                winners = record['chunk_winners'][src_m]
                record['chunk_cross_errors'][src_m] = {}
                for tgt_m in metrics_to_calc:
                    total_err = 0.0
                    possible = True
                    for i, fmt in enumerate(winners):
                        if fmt in metric_chunk_errors[tgt_m]:
                            total_err += float(metric_chunk_errors[tgt_m][fmt][i])
                        else:
                            total_err = float('inf')
                            possible = False
                            break
                    record['chunk_cross_errors'][src_m][tgt_m] = total_err if possible else float('inf')

    record['max_val'] = max_val_global


def run_weight_quantization_analysis(args, model, metrics_to_calc, qt_options, layer_results_map, supported_ops):
    """
    Iterate over all supported layers and analyse weight tensors against every format in qt_options.

    Fix #5 — nn.MultiheadAttention handling:
    Native MHA has no .weight attribute; instead it packs Q/K/V projections into
    `in_proj_weight` ([3*embed, embed]) and stores the output projection as
    `out_proj.weight`.  The GenericAdapter decomposes it into four separate linear
    layers named <mha_name>.q_proj / .k_proj / .v_proj / .out_proj so the deployment
    model quantizes those four sub-layers independently.
    We replicate the same decomposition here so the analysis layer names match
    deployment layer names, and NO attention weights are silently skipped.
    """
    import torch.nn as nn

    for name, module in tqdm(model.named_modules(), desc="Analyzing Layers (Weights)"):

        # --- Special case: nn.MultiheadAttention ---
        # MHA has no .weight; it uses in_proj_weight (packed Q+K+V) + out_proj.weight.
        # The adapter decomposes these into q_proj / k_proj / v_proj / out_proj linear
        # sub-layers, so we must use the same sub-layer names here to stay aligned.
        if isinstance(module, nn.MultiheadAttention):
            # in_proj_weight: [3*embed_dim, embed_dim] — split into [embed_dim, embed_dim] each
            if module.in_proj_weight is not None:
                embed_dim = module.embed_dim
                q_w, k_w, v_w = module.in_proj_weight.data.chunk(3, dim=0)
                for proj_name, proj_w in [('q_proj', q_w), ('k_proj', k_w), ('v_proj', v_w)]:
                    virtual_name = f"{name}.{proj_name}" if name else proj_name
                    _analyze_weight_tensor(
                        proj_w, virtual_name, args, metrics_to_calc, qt_options, layer_results_map
                    )
            # out_proj is a real nn.Linear sub-module — it will be caught by the loop below
            # unless it is the MHA itself that contains it.  We handle it explicitly to
            # guarantee the name matches '<mha>.out_proj':
            if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                out_name = f"{name}.out_proj" if name else "out_proj"
                _analyze_weight_tensor(
                    module.out_proj.weight.data, out_name, args, metrics_to_calc, qt_options, layer_results_map
                )
            continue  # skip the generic .weight path for MHA

        # --- Generic path: any other supported layer with a .weight ---
        if not isinstance(module, supported_ops):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        _analyze_weight_tensor(
            module.weight.data, name, args, metrics_to_calc, qt_options, layer_results_map
        )


class OnlineParams:
    def __init__(self, metrics, formats, chunk_size=None):
        self.metrics = metrics
        self.formats = formats
        self.chunk_size = chunk_size
        self.layer_stats = {} # name -> {'count': N, 'metrics': {m: {fmt: cum_err}}, 'chunk_errors': {m: tensor}}


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
        base_root = os.path.join(os.path.dirname(__file__), "results")

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
