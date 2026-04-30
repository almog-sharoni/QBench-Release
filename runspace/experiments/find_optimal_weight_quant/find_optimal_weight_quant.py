
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
import gc
import subprocess
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.ops.quant_base import quantize_tensor
# from runspace.src.quantization.quantizer import quantize_fp_generic_i32 as quantize_i32
from runspace.src.quantization.constants import get_quantization_bias
# Late import
from runspace.core.runner import Runner
from runspace.core.report_aggregator import ReportAggregator
from runspace.src.registry.op_registry import OpRegistry
from runspace.experiments.utils.common import (
    build_fp32_runtime_config,
    build_prequantized_weight_runtime_config,
    build_runtime_weight_quant_config,
    build_weight_map_json as _build_weight_map_json,
    layer_types_from_model as _layer_types_from_model,
)

baseline_formats = [
    'fp32',
    'fp8_e1m6','fp8_e2m5','fp8_e3m4','fp8_e4m3','fp8_e5m2','fp8_e6m1','fp8_e7m0',
    'fp7_e1m5','fp7_e2m4','fp7_e3m3','fp7_e4m2','fp7_e5m1','fp7_e6m0',
    'fp6_e1m4','fp6_e2m3','fp6_e3m2','fp6_e4m1','fp6_e5m0',
    'fp5_e1m3','fp5_e2m2','fp5_e3m1','fp5_e4m0',
    'fp4_e1m2','fp4_e2m1','fp4_e3m0',
    'fp3_e1m1','fp3_e2m0',
    'fp2_e1m0'
]


def get_args():
    parser = argparse.ArgumentParser(description="Find optimal layer-wise quantization")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--include_fp32", action="store_true", help="Include FP32 in search")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: runspace/experiments/optimal_layer_quant)")
    parser.add_argument("--models_file", type=str, default=None, help="Path to a YAML file containing a list of models to run")
    
    # Validation / Metric Args
    parser.add_argument("--metrics", type=str, default="l1,mse", help="Comma-separated metrics: l1, mse, sqnr, cosine OR 'all'")
    
    # Experiment Target
    parser.add_argument("--target", type=str, default="weights", choices=['weights'], help="Target to optimize: 'weights'")
    
    # Evaluation Args
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation on FP32, FP8, and Optimized models")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--weight_chunk_size", type=int, default=128, help="Weight/Input chunk size (blocks). If set, enables chunked quantization.")
    parser.add_argument("--per_chunk_format", action="store_true", help="Enable per-chunk format selection (each chunk gets its own optimal format)")
    parser.add_argument("--plot_layers", action="store_true", help="Generate error bar plots for every single layer (warning: slow)")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit number of batches to process (default: -1 for all)")
    parser.add_argument("--baseline_formats", type=str, default=','.join(baseline_formats), help="Comma-separated list of formats to run as baselines (full eval)")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baseline evaluations (run ref + optimized only)")
    parser.add_argument("--skip_layer_wise", action="store_true", help="Skip the layer-wise optimization experiment (only run chunk/baselines)")
    parser.add_argument("--force_recalc", action="store_true", help="Force recalculation of layer errors even if results exist")
    parser.add_argument("--force_rerun", action="store_true", help="Re-run all evaluations even if already in DB")
    parser.add_argument(
        "--verify_saved_weights",
        action="store_true",
        help=(
            "After writing quantized weight files, validate each mapped weight tensor "
            "against its expected quantization format (including per-chunk maps)."
        ),
    )
    parser.add_argument(
        "--verify_atol",
        type=float,
        default=1e-7,
        help="Absolute tolerance used by --verify_saved_weights checks.",
    )

    return parser.parse_args()


def _load_existing_runs(db):
    """Fetch all runs from DB once and return a DataFrame (empty if none)."""
    runs = db.get_runs()
    return runs if not runs.empty else None


def _weight_quant_run_exists(existing_runs, model_name, experiment_type, weight_dt):
    """Return True if a successful weight-quant run already exists in the pre-fetched DataFrame."""
    if existing_runs is None:
        return False
    return not existing_runs[
        (existing_runs['model_name']      == model_name) &
        (existing_runs['experiment_type'] == experiment_type) &
        (existing_runs['weight_dt']       == weight_dt) &
        (existing_runs['activation_dt']   == 'fp32') &
        (existing_runs['status']          == 'SUCCESS')
    ].empty


def _get_ref_from_db(existing_runs, model_name):
    """Return (acc1, acc5, certainty) for the fp32 reference if it's in DB, else None."""
    if existing_runs is None:
        return None
    mask = (
        (existing_runs['model_name']      == model_name) &
        (existing_runs['experiment_type'] == 'fp32_ref') &
        (existing_runs['status']          == 'SUCCESS')
    )
    rows = existing_runs[mask]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return float(r.get('acc1', 0.0) or 0.0), float(r.get('acc5', 0.0) or 0.0), float(r.get('certainty', 0.0) or 0.0)


def _log_weight_quant_run(
    runner,
    args,
    model_name,
    model_weights,
    experiment_type,
    weight_dt,
    acc1,
    acc5,
    status,
    ref_acc1,
    ref_acc5,
    ref_certainty,
    certainty=0.0,
    mse=None,
    l1=None,
    quant_map_json=None,
    config_json=None,
    weight_source='prequantized_state_dict',
):
    if weight_source == 'fp32':
        cfg = build_fp32_runtime_config(args)
    elif weight_source == 'runtime_quantized':
        cfg = build_runtime_weight_quant_config(args, weight_dt, args.weight_chunk_size)
    else:
        cfg = build_prequantized_weight_runtime_config(args)
    if model_weights:
        cfg.setdefault('model', {})
        cfg['model']['weights'] = model_weights
    cfg.setdefault('quantization', {})
    cfg['quantization']['weight_source'] = weight_source
    cfg['experiment'] = {
        'name': 'find_optimal_weight_quant',
        'type': experiment_type,
        'weight_dt': weight_dt,
        'activation_dt': 'fp32',
        'ref_acc1': ref_acc1,
        'ref_acc5': ref_acc5,
        'ref_certainty': ref_certainty,
        'metrics': {
            'mse': mse,
            'l1': l1,
            'certainty': certainty,
        },
        'quant_map_json': quant_map_json,
        'config_json': config_json,
    }
    result = {
        'model_name': model_name,
        'status': status,
        'acc1': acc1,
        'acc5': acc5,
        'certainty': certainty if certainty is not None else 0.0,
    }
    runner.log_experiment_result(cfg, result)


def get_quantized_tensor_sim(tensor, q_type, chunk_size=None, chunk_formats=None, mode=None):
    """
    Returns the dequantized tensor (simulated quantization).
    Returns (tensor_dequant, max_val).
    """
    if chunk_size is not None:
        return quantize_tensor(tensor, q_type=q_type, mode='chunk', chunk_size=chunk_size,
                               chunk_formats=chunk_formats, rounding='nearest', validate=False)

    if mode == 'tensor':
        return quantize_tensor(tensor, q_type=q_type, mode='tensor', rounding='nearest', validate=False)

    # Default: per output-channel (dim 0) via quantize_tensor channel mode.
    # quantize_tensor(channel) preserves dim=1, so transpose [O, ...] -> [..., O]
    # to make channel axis align with output channels.
    out_channels = tensor.shape[0]
    flat = tensor.view(out_channels, -1).transpose(0, 1).contiguous()  # [K, O]
    deq_t, _ = quantize_tensor(flat, q_type=q_type, mode='channel', rounding='nearest', validate=False)
    dequant = deq_t.transpose(0, 1).contiguous().view_as(tensor)
    max_val_per_channel = tensor.view(out_channels, -1).abs().amax(dim=1, keepdim=True).clamp(min=1e-9)
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
                # IMPORTANT: keep chunk semantics (one quantized chunk per row).
                dq_chunks, _ = get_quantized_tensor_sim(
                    target_chunks,
                    fmt,
                    chunk_size=chunk_size_,
                )
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


def _resolve_weight_tensor_for_map_entry(state_dict, layer_name):
    """
    Resolve a quant-map layer key to the corresponding weight tensor in state_dict.
    Supports both regular modules ("<name>.weight") and MHA virtual entries:
    "<mha>.q_proj", "<mha>.k_proj", "<mha>.v_proj".
    """
    if layer_name.endswith('.q_proj') or layer_name.endswith('.k_proj') or layer_name.endswith('.v_proj'):
        parent = layer_name.rsplit('.', 1)[0]
        in_proj_key = f"{parent}.in_proj_weight"
        if in_proj_key not in state_dict:
            return None
        w_in = state_dict[in_proj_key]
        embed_dim = w_in.shape[0] // 3
        if layer_name.endswith('.q_proj'):
            return w_in[0:embed_dim].contiguous()
        if layer_name.endswith('.k_proj'):
            return w_in[embed_dim:2 * embed_dim].contiguous()
        return w_in[2 * embed_dim:3 * embed_dim].contiguous()

    weight_key = f"{layer_name}.weight"
    if weight_key not in state_dict:
        return None
    return state_dict[weight_key].contiguous()


def _apply_chunk_format_map(tensor, chunk_formats, chunk_size):
    """
    Re-apply the chunk-wise format map on a tensor and return the reconstructed
    dequantized tensor (same logic used when creating chunk-quantized weights).
    """
    w_chunked, original_shape, pad_len = get_chunked_tensor(tensor, chunk_size=chunk_size)
    batch_size, num_chunks, chunk_size_ = w_chunked.shape
    w_chunked_flat = w_chunked.reshape(-1, chunk_size_)
    total_chunks = w_chunked_flat.shape[0]
    w_dequant_flat = torch.zeros_like(w_chunked_flat)

    current_formats = chunk_formats[:total_chunks]
    fmt_to_indices = {}
    for idx, fmt in enumerate(current_formats):
        fmt_to_indices.setdefault(fmt, []).append(idx)

    for fmt, indices in fmt_to_indices.items():
        if not indices:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=w_chunked_flat.device)
        target_chunks = w_chunked_flat[idx_tensor]
        # Match creation path: quantize each row as one logical chunk.
        dq_chunks, _ = get_quantized_tensor_sim(
            target_chunks,
            fmt,
            chunk_size=chunk_size_,
        )
        w_dequant_flat[idx_tensor] = dq_chunks

    if len(current_formats) < total_chunks:
        w_dequant_flat[len(current_formats):] = w_chunked_flat[len(current_formats):]

    w_dequant_chunked = w_dequant_flat.view(batch_size, num_chunks, chunk_size_)
    flat = w_dequant_chunked.view(batch_size, -1)
    if pad_len > 0:
        flat = flat[:, :-pad_len]
    return flat.view(original_shape)


def verify_quantized_weights_file(weights_path, quant_map, args, atol=1e-7):
    """
    Validate that each weight tensor in `weights_path` is already quantized
    according to `quant_map`.
    """
    print(f"Verifying quantized weights file: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')

    total = 0
    passed = 0
    failed = []

    for layer_name, fmt_spec in quant_map.items():
        total += 1
        tensor = _resolve_weight_tensor_for_map_entry(state_dict, layer_name)
        if tensor is None:
            failed.append((layer_name, "missing_weight", float('inf')))
            continue

        tensor = tensor.detach().clone().contiguous()
        if not torch.isfinite(tensor).all():
            failed.append((layer_name, "non_finite_weights", float('inf')))
            continue
        try:
            if isinstance(fmt_spec, list):
                expected = _apply_chunk_format_map(
                    tensor=tensor,
                    chunk_formats=fmt_spec,
                    chunk_size=args.weight_chunk_size,
                )
            else:
                expected, _ = get_quantized_tensor_sim(
                    tensor,
                    str(fmt_spec),
                    chunk_size=args.weight_chunk_size
                )
        except Exception as e:
            failed.append((layer_name, f"quantize_error:{e}", float('inf')))
            continue
        if not torch.isfinite(expected).all():
            failed.append((layer_name, "non_finite_expected", float('inf')))
            continue

        if tensor.numel() == 0:
            max_abs_err = 0.0
        else:
            max_abs_err = (tensor - expected).abs().max().item()

        if max_abs_err <= atol:
            passed += 1
        else:
            failed.append((layer_name, fmt_spec, max_abs_err))

    if failed:
        print(f"[VERIFY] FAILED {len(failed)}/{total} layers in {weights_path}")
        for layer_name, fmt, err in failed[:20]:
            print(f"  - {layer_name}: fmt={fmt}, max_abs_err={err:.3e}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more failures")
        raise RuntimeError(
            f"Quantized weight verification failed for {len(failed)}/{total} layers "
            f"(atol={atol})."
        )

    print(f"[VERIFY] PASSED {passed}/{total} layers (atol={atol})")


def summarize_state_dict_delta(reference_state_dict, candidate_state_dict, eps=0.0):
    """
    Summarize how much `candidate_state_dict` differs from `reference_state_dict`.
    Returns a compact dict for logging/debugging.
    """
    tensors_compared = 0
    tensors_changed = 0
    elems_compared = 0
    elems_changed = 0
    max_abs_diff = 0.0
    l1_sum = 0.0

    for key, cand in candidate_state_dict.items():
        ref = reference_state_dict.get(key)
        if ref is None or not torch.is_tensor(cand) or not torch.is_tensor(ref):
            continue
        if ref.shape != cand.shape:
            continue

        tensors_compared += 1
        diff = (cand - ref).abs()
        if diff.numel() == 0:
            continue
        elems_compared += diff.numel()
        l1 = diff.sum().item()
        l1_sum += l1
        cur_max = diff.max().item()
        if cur_max > max_abs_diff:
            max_abs_diff = cur_max

        changed_mask = diff > eps
        changed_count = int(changed_mask.sum().item())
        elems_changed += changed_count
        if changed_count > 0:
            tensors_changed += 1

    mean_abs_diff = (l1_sum / elems_compared) if elems_compared > 0 else 0.0
    pct_elems_changed = (100.0 * elems_changed / elems_compared) if elems_compared > 0 else 0.0
    return {
        'tensors_compared': tensors_compared,
        'tensors_changed': tensors_changed,
        'elems_compared': elems_compared,
        'elems_changed': elems_changed,
        'pct_elems_changed': pct_elems_changed,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
    }

def process_single_model(args, device, metrics, base_root):
    runner = Runner(device)
    db = runner._get_db()
    
    # Valid model path: base_root / model_name
    model_dir = os.path.join(base_root, args.model_name)

    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Root output directory: {model_dir}")
    print(f"Target: {args.target}")
    
    # Load Model
    print(f"Loading model {args.model_name}...")
    config = build_fp32_runtime_config(args)
    
    try:
        analysis_dir = os.path.join(model_dir, "analysis_fp32")
        model, adapter, _ = runner.prepare_model_with_materialized_weights(
            config=config,
            output_dir=analysis_dir
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    layer_types = _layer_types_from_model(model)

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
        need_chunk_details = bool(args.per_chunk_format and args.weight_chunk_size)
        if need_chunk_details and os.path.exists(csv_path) and not args.force_recalc:
            print(
                f"Metric {m}: per-chunk format enabled; cached CSV lacks full chunk winner lists. "
                "Recomputing this metric."
            )
            metrics_to_calc.append(m)
            continue
        
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

    # Fetch DB state once — used for all skip checks below
    existing_runs = None if args.force_rerun else _load_existing_runs(db)

    # Always add Ref and Baselines if we are evaling (and not cached)
    if args.run_eval:
        # Ref — skip if already in DB (cached_ref_acc carries the stored values)
        cached_ref = None if args.force_rerun else _get_ref_from_db(existing_runs, args.model_name)
        if cached_ref is not None:
            print(f"[DB] Skipping ref_fp32 — already in DB (acc1={cached_ref[0]:.2f}%)")
        else:
            ref_config = build_fp32_runtime_config(args)
            ref_config['evaluation'] = eval_config
            ref_config['output_name'] = "ref_fp32"
            configs_to_run.append(ref_config)

        if args.skip_baselines:
            print("[Eval] --skip_baselines set: skipping baseline_* configs.")
        else:
            # Parse baseline formats and strip whitespace
            requested_baselines = [fmt.strip() for fmt in args.baseline_formats.split(',') if fmt.strip()]

            for fmt in requested_baselines:
                if _weight_quant_run_exists(existing_runs, args.model_name, 'weight_quant_baseline', fmt):
                    print(f"[DB] Skipping baseline_{fmt} — already in DB")
                    continue

                b_cfg = build_runtime_weight_quant_config(args, fmt, args.weight_chunk_size)
                b_cfg['evaluation'] = eval_config
                b_cfg['output_name'] = f"baseline_{fmt}"
                configs_to_run.append(b_cfg)

    # Map run output_name -> enriched quant map JSON (for DB logging / dashboard)
    weight_quant_map_json_by_output = {}

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
        dataset_cfg = dataset_base if args.run_eval else {
             'name': args.dataset_name, 'path': args.dataset_path, 
             'batch_size': args.batch_size, 'num_workers': args.num_workers
        }

        # 1. OPTIMIZED LAYERS CONFIG (Generate ONLY if NOT skipped)
        if not args.skip_layer_wise:
            # Create Quantized Weights
            q_state_dict, q_map = create_quantized_state_dict(model, layer_results_map, args, m, use_chunking=False)
            layer_delta = summarize_state_dict_delta(model.state_dict(), q_state_dict)
            print(
                "[Layer-Opt Delta] "
                f"changed_tensors={layer_delta['tensors_changed']}/{layer_delta['tensors_compared']}, "
                f"changed_elems={layer_delta['pct_elems_changed']:.2f}%, "
                f"max_abs_diff={layer_delta['max_abs_diff']:.3e}, "
                f"mean_abs_diff={layer_delta['mean_abs_diff']:.3e}"
            )
            q_weights_path = os.path.join(metric_dir, "quantized_weights_layer.pt")
            torch.save(q_state_dict, q_weights_path)
            print(f"Saved quantized weights to {q_weights_path}")
            if args.verify_saved_weights:
                verify_quantized_weights_file(
                    weights_path=q_weights_path,
                    quant_map=q_map,
                    args=args,
                    atol=args.verify_atol,
                )
            
            # Save Map
            q_map_path = os.path.join(metric_dir, "quantization_map_layer.json")
            with open(q_map_path, 'w') as f:
                json.dump(q_map, f, indent=4)
            print(f"Saved quantization map to {q_map_path}")

            # Use layer_config_m (best format per layer)
            layer_opt_config = build_prequantized_weight_runtime_config(
                args,
                weights=os.path.abspath(q_weights_path),
            )
            layer_opt_config['quantization'].update({
                'format': 'fp8_e4m3', # Default dummy
                'layers': {}, # Empty layers config
            })
            layer_opt_config['evaluation'] = eval_config
            layer_opt_config['dataset'] = dataset_cfg
            


            layer_cfg_path = os.path.join(metric_dir, "optimized_layer_config.yaml")
            with open(layer_cfg_path, 'w') as f:
                yaml.dump(layer_opt_config, f, default_flow_style=False)
            print(f"Generated Layer-Opt config for {m} at {layer_cfg_path}")
            
            if args.run_eval:
                run_config = layer_opt_config.copy()
                out_name = f"optimized_layer_{m}"
                run_config['output_name'] = out_name
                configs_to_run.append(run_config)
                weight_quant_map_json_by_output[out_name] = _build_weight_map_json(
                    q_map,
                    layer_types=layer_types,
                )
        
        
        # 2. OPTIMIZED CHUNK CONFIG (If Chunking Enabled)
        if args.weight_chunk_size and args.per_chunk_format and layer_config_per_chunk:
            # Create Quantized Weights (Chunked)
            q_state_dict_chunk, q_map_chunk = create_quantized_state_dict(model, layer_results_map, args, m, use_chunking=True)
            chunk_delta = summarize_state_dict_delta(model.state_dict(), q_state_dict_chunk)
            print(
                "[Chunk-Opt Delta] "
                f"changed_tensors={chunk_delta['tensors_changed']}/{chunk_delta['tensors_compared']}, "
                f"changed_elems={chunk_delta['pct_elems_changed']:.2f}%, "
                f"max_abs_diff={chunk_delta['max_abs_diff']:.3e}, "
                f"mean_abs_diff={chunk_delta['mean_abs_diff']:.3e}"
            )
            q_weights_path_chunk = os.path.join(metric_dir, "quantized_weights_chunk.pt")
            torch.save(q_state_dict_chunk, q_weights_path_chunk)
            print(f"Saved chunk-quantized weights to {q_weights_path_chunk}")
            if args.verify_saved_weights:
                verify_quantized_weights_file(
                    weights_path=q_weights_path_chunk,
                    quant_map=q_map_chunk,
                    args=args,
                    atol=args.verify_atol,
                )
            
            # Save Map
            q_map_chunk_path = os.path.join(metric_dir, "quantization_map_chunk.json")
            with open(q_map_chunk_path, 'w') as f:
                json.dump(q_map_chunk, f, indent=4)
            print(f"Saved chunk quantization map to {q_map_chunk_path}")

            chunk_opt_config = build_prequantized_weight_runtime_config(
                args,
                weights=os.path.abspath(q_weights_path_chunk),
            )
            chunk_opt_config['quantization'].update({
                'format': 'fp8_e4m3',
                'layers': {},
            })
            chunk_opt_config['evaluation'] = eval_config
            chunk_opt_config['dataset'] = dataset_cfg
            

            chunk_cfg_path = os.path.join(metric_dir, "optimized_chunk_config.yaml")
            with open(chunk_cfg_path, 'w') as f:
                yaml.dump(chunk_opt_config, f, default_flow_style=False)
            print(f"Generated Chunk-Opt config for {m} at {chunk_cfg_path}")
            
            if args.run_eval:
                run_config = chunk_opt_config.copy()
                out_name = f"optimized_chunk_{m}"
                run_config['output_name'] = out_name
                configs_to_run.append(run_config)
                weight_quant_map_json_by_output[out_name] = _build_weight_map_json(
                    q_map_chunk,
                    layer_types=layer_types,
                )

            
    # Evaluation Logic (same as before)
    # Evaluation Logic (same as before)
    if args.run_eval and configs_to_run:
        # ... (Execution same as original) ...
        # (For brevity, relying on original execution block or we can copy it if we replaced it)
        # I removed the full execution block in replace, so I must re-include it.
        # ...
        print("\n--- Starting Evaluation Batch ---")
        print("Using Sequential Execution (one config at a time)...")
        
        # Dedupe logic (baselines already filtered by DB at build time above)
        final_configs = []
        sigs = set()
        for c in configs_to_run:
            s = json.dumps(c, sort_keys=True)
            if s not in sigs:
                sigs.add(s)
                final_configs.append(c)

        # Secondary DB skip for optimized configs (added later in the metrics loop)
        if not args.force_rerun:
            filtered_configs = []
            for c in final_configs:
                out_name = c.get('output_name', '')
                if out_name == 'ref_fp32' or out_name.startswith('baseline_'):
                    filtered_configs.append(c)  # baselines already pre-filtered
                    continue
                weight_dt = out_name.replace('optimized_', 'opt_')
                if _weight_quant_run_exists(existing_runs, args.model_name, 'weight_quant_optimized', weight_dt):
                    print(f"[DB] Skipping {out_name} — already in DB")
                else:
                    filtered_configs.append(c)
            final_configs = filtered_configs

        results = []
        # Keep reference metrics available while logging each run immediately.
        if cached_ref is not None:
            ref_acc1, ref_acc5, ref_certainty = cached_ref
        else:
            ref_acc1, ref_acc5, ref_certainty = 0.0, 0.0, 0.0

        total_elements = sum(r.get('numel', 0) for r in layer_results)

        def _log_result_immediately(res):
            nonlocal ref_acc1, ref_acc5, ref_certainty
            out_name = res.get('output_name', '')

            if out_name == 'ref_fp32':
                if float(res.get('acc1', 0.0) or 0.0) > 0.0:
                    ref_acc1 = float(res.get('acc1', 0.0) or 0.0)
                    ref_acc5 = float(res.get('acc5', 0.0) or 0.0)
                    ref_certainty = float(res.get('certainty', 0.0) or 0.0)
                ref_status = res.get('status', 'SUCCESS')
                if ref_status in ('SUCCESS', 'NO_QUANT') and ref_acc1 > 0.0:
                    ref_status = 'SUCCESS'
                _log_weight_quant_run(
                    runner=runner,
                    args=args,
                    model_name=res.get('model_name', args.model_name),
                    model_weights=res.get('materialized_weight_path'),
                    experiment_type='fp32_ref',
                    weight_dt='fp32',
                    acc1=float(res.get('acc1', 0.0) or 0.0),
                    acc5=float(res.get('acc5', 0.0) or 0.0),
                    status=ref_status,
                    ref_acc1=ref_acc1,
                    ref_acc5=ref_acc5,
                    ref_certainty=ref_certainty,
                    certainty=float(res.get('certainty', 0.0) or 0.0),
                    quant_map_json=None,
                    weight_source='fp32',
                    config_json={
                        'model': {'name': args.model_name, 'weights': args.weights},
                        'dataset': {
                            'name': args.dataset_name,
                            'path': args.dataset_path,
                            'batch_size': args.batch_size,
                            'num_workers': args.num_workers,
                        },
                        'quantization': {'weight_source': 'fp32'},
                        'evaluation': {'output_name': 'ref_fp32'},
                    },
                )
                return

            # For weight optimization, activation is usually fp32.
            weight_dt = out_name.replace('baseline_', '').replace('optimized_', 'opt_')

            mse = total_errors_by_config.get('mse', {}).get(out_name)
            if mse is not None and total_elements > 0:
                mse /= total_elements

            l1 = total_errors_by_config.get('l1', {}).get(out_name)
            if l1 is not None and total_elements > 0:
                l1 /= total_elements

            _log_weight_quant_run(
                runner=runner,
                args=args,
                model_name=res.get('model_name', args.model_name),
                model_weights=res.get('materialized_weight_path'),
                experiment_type=(
                    "weight_quant_baseline"
                    if out_name.startswith('baseline_')
                    else "weight_quant_optimized"
                ),
                weight_dt=weight_dt,
                acc1=float(res.get('acc1', 0.0) or 0.0),
                acc5=float(res.get('acc5', 0.0) or 0.0),
                status=res.get('status', 'SUCCESS'),
                ref_acc1=ref_acc1,
                ref_acc5=ref_acc5,
                ref_certainty=ref_certainty,
                mse=mse,
                l1=l1,
                certainty=float(res.get('certainty', 0.0) or 0.0),
                quant_map_json=weight_quant_map_json_by_output.get(out_name),
                weight_source=(
                    'runtime_quantized'
                    if out_name.startswith('baseline_')
                    else 'prequantized_state_dict'
                ),
                config_json={
                    'model': {'name': args.model_name, 'weights': args.weights},
                    'dataset': {
                        'name': args.dataset_name,
                        'path': args.dataset_path,
                        'batch_size': args.batch_size,
                        'num_workers': args.num_workers,
                    },
                    'quantization': {
                        'weight_format': weight_dt,
                        'weight_mode': 'chunk',
                        'weight_chunk_size': args.weight_chunk_size,
                        'weight_source': (
                            'runtime_quantized'
                            if out_name.startswith('baseline_')
                            else 'prequantized_state_dict'
                        ),
                    },
                    'evaluation': {'output_name': out_name},
                },
            )

        total_eval_runs = len(final_configs)
        for idx, run_cfg in enumerate(final_configs, start=1):
            out_name = run_cfg.get('output_name', f'run_{idx}')
            print(f"[Eval] Running {idx}/{total_eval_runs}: {out_name}")
            res = runner.run_single(run_cfg, output_root=model_dir)
            results.append(res)
            try:
                _log_result_immediately(res)
            except Exception as log_err:
                print(f"[DB] Failed to log {out_name}: {log_err}")

        failed = [r for r in results if r.get('status') not in ('SUCCESS', 'NO_QUANT')]
        if failed:
            print(f"\n[Evaluation] {len(failed)} run(s) failed:")
            for r in failed:
                print(
                    f"  - {r.get('output_name', '<unknown>')}: "
                    f"status={r.get('status')} error={r.get('exec_error')}"
                )
        
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

    if args.models_file:
        print(f"Loading models from config: {args.models_file}")
        with open(args.models_file, 'r') as f:
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

        # Run each model in an isolated subprocess to avoid cross-model native state
        # leakage (CUDA/UCX/DataLoader workers) in long multi-model sessions.
        def _base_cli_without_model_overrides():
            raw = sys.argv[1:]
            filtered = []
            i = 0
            while i < len(raw):
                tok = raw[i]
                if tok == '--models_file':
                    i += 2
                    continue
                if tok.startswith('--models_file='):
                    i += 1
                    continue
                if tok == '--model_name':
                    i += 2
                    continue
                if tok.startswith('--model_name='):
                    i += 1
                    continue
                if tok == '--weights':
                    i += 2
                    continue
                if tok.startswith('--weights='):
                    i += 1
                    continue
                filtered.append(tok)
                i += 1
            return filtered

        base_cli = _base_cli_without_model_overrides()
        
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
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    '--model_name', str(name),
                    '--weights', str(weights),
                ] + base_cli
                print(f"[Isolated Run] {' '.join(cmd)}")
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(f"[Isolated Run] Model {name} failed with exit code {proc.returncode}")
            except Exception as e:
                print(f"Error processing model {name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Hard boundary between models to avoid cross-model residual state.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                
    else:
        # Single model mode
        process_single_model(args, device, metrics, base_root)

if __name__ == "__main__":
    main()
