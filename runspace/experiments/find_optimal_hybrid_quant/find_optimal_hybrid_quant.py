
import os
import sys
import gc
import json
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.src.quantization.quantizer import quantize
from runspace.src.ops.quant_base import quantize_tensor, calculate_scale
from runspace.src.database.handler import RunDatabase
from runspace.src.eval.metrics import compute_certainty

# Import weight-quant helpers
from runspace.experiments.find_optimal_weight_quant.find_optimal_weight_quant import (
    get_quantized_tensor_sim,
    calculate_error,
    create_quantized_state_dict,
    run_weight_quantization_analysis,
    baseline_formats as weight_baseline_formats,
)

# Import input-quant helpers
from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (
    DynamicInputQuantizer,
    candidate_formats as input_candidate_formats,
    baseline_formats as input_baseline_formats,
)

from runspace.experiments.utils.plotting import plot_accuracy_comparison


class UniformInputQuantizer:
    """
    Applies a fixed quantization format to every supported layer's input.
    Drop-in replacement for DynamicInputQuantizer when activation_dt is a
    plain format string (e.g. 'fp8_e1m6').
    """
    def __init__(self, model, fmt, chunk_size=128):
        self.model = model
        self.fmt = fmt
        self.chunk_size = chunk_size
        self.hooks = []
        self._layer_names = []
        self.supported_ops = tuple(OpRegistry.get_supported_ops().values())
        self.stats = {'sum_l1_err': 0.0, 'sum_mse_err': 0.0,
                      'sum_l1_norm': 0.0, 'sum_l2_norm': 0.0}

    def _quantize_activation(self, x):
        """Quantize activation tensor using per-chunk scale, same as DynamicInputQuantizer."""
        original_shape = x.shape
        flat = x.reshape(-1)
        pad_len = 0
        if flat.numel() % self.chunk_size != 0:
            pad_len = self.chunk_size - (flat.numel() % self.chunk_size)
            flat = torch.nn.functional.pad(flat, (0, pad_len))
        chunks = flat.view(-1, self.chunk_size)
        scale = calculate_scale(chunks.abs().amax(dim=1, keepdim=True).clamp(min=1e-5), self.fmt)
        q_chunks = quantize(chunks / scale, q_type=self.fmt, validate=False) * scale
        flat_q = q_chunks.reshape(-1)
        if pad_len > 0:
            flat_q = flat_q[:-pad_len]
        return flat_q.reshape(original_shape)

    def _make_hook(self, name):
        def hook(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return inputs
            x = inputs[0]
            x_q = self._quantize_activation(x)
            with torch.no_grad():
                diff = x - x_q
                self.stats['sum_l1_err']  += diff.abs().sum().item()
                self.stats['sum_mse_err'] += diff.pow(2).sum().item()
                self.stats['sum_l1_norm'] += x.abs().sum().item()
                self.stats['sum_l2_norm'] += x.pow(2).sum().item()
            return (x_q,) + inputs[1:]
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, self.supported_ops) or isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_pre_hook(self._make_hook(name)))
                self._layer_names.append(name)

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_final_stats(self):
        norm_l1  = self.stats['sum_l1_err']  / self.stats['sum_l1_norm']  if self.stats['sum_l1_norm']  > 0 else 0.0
        norm_mse = self.stats['sum_mse_err'] / self.stats['sum_l2_norm'] if self.stats['sum_l2_norm'] > 0 else 0.0
        return {'norm_l1': norm_l1, 'norm_mse': norm_mse,
                'total_l1': self.stats['sum_l1_err'], 'total_mse': self.stats['sum_mse_err']}

    @property
    def layer_stats(self):
        return {
            name: {'format_counts': {self.fmt: 1}, 'total_chunks': 1}
            for name in self._layer_names
        }


def get_args():
    parser = argparse.ArgumentParser(
        description="Hybrid experiment: optimal weight quant + dynamic input quant"
    )
    # Model
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--models_file", type=str, default=None,
                        help="YAML file with list of models to run on multiple models")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit_batches", type=int, default=-1,
                        help="Limit batches (-1 = all)")

    # Weight quantization options
    parser.add_argument("--weight_metrics", type=str, default="l1,mse",
                        help="Comma-separated metrics for weight format selection")
    parser.add_argument("--weight_chunk_size", type=int, default=128,
                        help="Chunk size for weight quantization blocks")
    parser.add_argument("--per_chunk_format", action="store_true",
                        help="Enable per-chunk format for weights")
    parser.add_argument("--force_recalc", action="store_true",
                        help="Force recalculation of weight errors even if cached")

    # Input quantization options
    parser.add_argument("--input_metrics", type=str, default="mse,l1",
                        help="Comma-separated metrics for dynamic input format selection")
    parser.add_argument("--input_chunk_size", type=int, default=128,
                        help="Chunk size for dynamic input quantization")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))

    # Experiment control
    parser.add_argument("--skip_weight_analysis", action="store_true",
                        help="Skip weight analysis and load cached quantized weights")
    parser.add_argument("--run_weight_only_baseline", action="store_true",
                        help="Also run the quantized-weights + fp32-input baseline")
    parser.add_argument("--run_input_only_baseline", action="store_true",
                        help="Also run the fp32-weights + dynamic-input baseline")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Re-run and re-log all experiments even if they already exist in the DB")
    parser.add_argument("--run_all_combos", action="store_true",
                        help="Run the full weight_metric × input_metric cross-product. "
                             "Default (without this flag) is to run only the best combo from DB.")
    # FP32 reference is always resolved automatically (from DB or fresh run).

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Weight Analysis Helpers
# ---------------------------------------------------------------------------

def _make_weight_args(args):
    """Build a namespace compatible with weight-quant helpers."""
    import types
    wa = types.SimpleNamespace()
    wa.model_name = args.model_name
    wa.weights = args.weights
    wa.weight_chunk_size = args.weight_chunk_size
    wa.per_chunk_format = args.per_chunk_format
    wa.dataset_name = args.dataset_name
    wa.dataset_path = args.dataset_path
    wa.batch_size = args.batch_size
    wa.num_workers = args.num_workers
    wa.limit_batches = args.limit_batches
    wa.force_recalc = args.force_recalc
    wa.plot_layers = False
    wa.skip_layer_wise = False
    wa.run_eval = False
    wa.include_fp32 = False
    wa.baseline_formats = ','.join(weight_baseline_formats)
    wa.metrics = ','.join(
        [m.strip() for m in args.weight_metrics.split(',')]
    )
    return wa


def run_weight_phase(args, device, model_dir):
    """
    Phase 1: analyse weights, build per-layer optimal quantized state dict.

    Returns:
        quant_weights_paths : dict  metric -> path to saved .pt file
        quant_maps          : dict  metric -> {layer_name: format_str}
        layer_results_map   : dict  layer_name -> analysis record
    """
    weight_metrics = [m.strip() for m in args.weight_metrics.split(',')]
    wa = _make_weight_args(args)

    # Load model once for analysis
    config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }
    adapter = create_adapter(config)
    model = adapter.model.to(device)
    model.eval()

    qt_options = weight_baseline_formats.copy()
    supported_ops = tuple(OpRegistry.get_supported_ops().keys())
    layer_results_map = {}

    if not args.skip_weight_analysis:
        print("\n[Phase 1] Analysing weight tensors ...")
        run_weight_quantization_analysis(
            wa, model, weight_metrics, qt_options, layer_results_map, supported_ops
        )
    else:
        print("\n[Phase 1] --skip_weight_analysis set; loading cached quant maps ...")
        for m in weight_metrics:
            import csv
            csv_path = os.path.join(model_dir, m, "layer_errors.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lname = row['layer']
                        if lname not in layer_results_map:
                            layer_results_map[lname] = {
                                'layer': lname, 'shape': row.get('shape', ''),
                                'max_val': float(row.get('max_val', 0)),
                                'metrics': {}, 'chunk_wins': {}, 'chunk_winners': {},
                            }
                        errs = {}
                        for k, v in row.items():
                            if k.endswith('_error'):
                                fmt = k[:-6]
                                try:
                                    errs[fmt] = float(v) if v else float('inf')
                                except ValueError:
                                    errs[fmt] = float('inf')
                        layer_results_map[lname]['metrics'][m] = errs
            else:
                print(f"  WARNING: no cached CSV at {csv_path} for metric {m}")

    # Build quantized weight files per weight metric
    quant_weights_paths = {}
    quant_maps = {}

    for m in weight_metrics:
        metric_dir = os.path.join(model_dir, f"weights_{m}")
        os.makedirs(metric_dir, exist_ok=True)

        q_state_dict, q_map = create_quantized_state_dict(
            model, layer_results_map, wa, m, use_chunking=False
        )
        q_path = os.path.join(metric_dir, "quantized_weights.pt")
        torch.save(q_state_dict, q_path)
        quant_weights_paths[m] = q_path
        quant_maps[m] = q_map

        # Persist quant map
        with open(os.path.join(metric_dir, "quantization_map.json"), 'w') as f:
            json.dump(q_map, f, indent=4)

        print(f"[Phase 1] Saved quantized weights ({m}) → {q_path}")

    # Capture layer types before model is deleted
    layer_types = {
        name: type(module).__name__
        for name, module in model.named_modules()
        if name  # skip the root module (empty name)
    }

    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()

    return quant_weights_paths, quant_maps, layer_results_map, layer_types


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _build_loader(args, device):
    config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }
    runner = Runner(device)
    loader = runner.setup_data_loader(config)
    return loader


def _run_inference(model, adapter, loader, device, args,
                   input_quantizer=None, desc=""):
    """
    Run one inference pass.  Returns (acc1, acc5, certainty, input_stats).
    input_stats is from input_quantizer.get_final_stats() if provided, else None.
    """
    correct_top1 = correct_top5 = total = 0
    total_certainty = 0.0
    batch_count = 0
    limit = args.limit_batches if args.limit_batches > 0 else float('inf')

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            if batch_count >= limit:
                break
            images, labels = adapter.prepare_batch(batch)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            total_certainty += compute_certainty(outputs) * images.size(0)

            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            correct_top1 += correct[:1].reshape(-1).float().sum().item()
            correct_top5 += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)
            batch_count += 1

    acc1 = 100.0 * correct_top1 / total if total > 0 else 0.0
    acc5 = 100.0 * correct_top5 / total if total > 0 else 0.0
    certainty = total_certainty / total if total > 0 else 0.0
    input_stats = input_quantizer.get_final_stats() if input_quantizer else None

    return acc1, acc5, certainty, input_stats


def _load_quantized_model(args, device, quant_weights_path):
    """Load model and replace state dict with quantized weights."""
    config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }
    adapter = create_adapter(config)
    model = adapter.model

    # Load quantized weights
    q_state_dict = torch.load(quant_weights_path, map_location=device)
    model.load_state_dict(q_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, adapter


def _load_fp32_model(args, device):
    """Load model with original fp32 weights."""
    config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }
    adapter = create_adapter(config)
    model = adapter.model.to(device)
    model.eval()
    return model, adapter


# ---------------------------------------------------------------------------
# Summarise quant map for DB field
# ---------------------------------------------------------------------------

def _layer_types_from_model(model):
    """Return {layer_name: class_name} for every named sub-module."""
    return {
        name: type(module).__name__
        for name, module in model.named_modules()
        if name
    }


def _build_weight_map_json(quant_map, layer_types):
    """
    Build enriched weight quant map JSON: {layer: {format, type}}.
    Falls back to "unknown" type for layers not in layer_types.
    """
    enriched = {}
    for layer, fmt in quant_map.items():
        enriched[layer] = {
            "format": fmt,
            "type": layer_types.get(layer, "unknown"),
        }
    return json.dumps(enriched)


def _build_input_map_json(layer_stats, model):
    """
    Build enriched input quant map JSON:
      {layer: {format, type, format_counts, total_chunks}}.
    Dominant format = the format chosen for the most chunks in that layer.
    """
    layer_types = _layer_types_from_model(model)
    result = {}
    for layer_name, stats in layer_stats.items():
        counts = stats.get('format_counts', {})
        if counts:
            total = sum(counts.values())
            result[layer_name] = {
                "format": max(counts, key=counts.get),
                "type": layer_types.get(layer_name, "unknown"),
                "format_counts": counts,
                "total_chunks": total,
            }
    return json.dumps(result)


def _summarise_quant_map(quant_map, w_metric=None):
    """
    Return a compact string that describes the weight format distribution,
    e.g. "opt_wl1[fp4_e1m2×12,fp8_e1m6×4]"

    Including w_metric in the prefix ensures that runs optimised with
    different metrics always produce distinct weight_dt strings, which
    prevents the dashboard's deduplication logic from silently collapsing
    them into a single row.
    """
    prefix = f"opt_w{w_metric}" if w_metric else "opt_layer"
    if not quant_map:
        return f"{prefix}_unknown"
    counts = {}
    for v in quant_map.values():
        key = str(v) if not isinstance(v, list) else "per_chunk"
        counts[key] = counts.get(key, 0) + 1
    # Sort by frequency desc
    parts = [f"{fmt}×{cnt}" for fmt, cnt in
             sorted(counts.items(), key=lambda x: -x[1])]
    return prefix + "[" + ",".join(parts[:5]) + "]"   # cap at 5 for readability


# ---------------------------------------------------------------------------
# Best-combo helpers
# ---------------------------------------------------------------------------

def _best_weight_from_db(db, model_name):
    """
    Find the weight_dt with highest acc1 among non-fp32 weight / fp32 input runs.
    Returns (weight_dt, acc1) or (None, None).
    """
    runs = db.get_runs()
    if runs.empty:
        return None, None
    candidates = runs[
        (runs['model_name']    == model_name) &
        (runs['weight_dt']     != 'fp32') &
        (runs['activation_dt'] == 'fp32') &
        (runs['status']        == 'SUCCESS') &
        (runs['acc1'].notna())
    ]
    if candidates.empty:
        return None, None
    best = candidates.loc[candidates['acc1'].idxmax()]
    return best['weight_dt'], float(best['acc1'])


def _best_input_from_db(db, model_name):
    """
    Find the activation_dt with highest acc1 among fp32 weight / non-fp32 input runs.
    Returns (activation_dt, acc1) or (None, None).
    """
    runs = db.get_runs()
    if runs.empty:
        return None, None
    candidates = runs[
        (runs['model_name']    == model_name) &
        (runs['weight_dt']     == 'fp32') &
        (runs['activation_dt'] != 'fp32') &
        (runs['status']        == 'SUCCESS') &
        (runs['acc1'].notna())
    ]
    if candidates.empty:
        return None, None
    best = candidates.loc[candidates['acc1'].idxmax()]
    return best['activation_dt'], float(best['acc1'])


def _parse_weight_metric(weight_dt):
    """'opt_wl1[...]'  →  'l1'.  Returns None if not parseable."""
    import re
    m = re.match(r'opt_w(\w+)\[', weight_dt or '')
    return m.group(1) if m else None


def _is_uniform_format(weight_dt):
    """Return True if weight_dt is a plain format string (e.g. 'fp8_e1m6'), not an opt_w* summary."""
    if not weight_dt:
        return False
    return not weight_dt.startswith('opt_w') and '[' not in weight_dt


def _build_uniform_quant_state_dict(model, fmt, chunk_size):
    """
    Build a quantized state dict where every Conv2d/Linear layer weight is
    uniformly quantized to `fmt`.  Returns (state_dict, quant_map).
    """
    supported = (torch.nn.Conv2d, torch.nn.Linear)
    state_dict = model.state_dict()
    quant_map = {}
    for name, module in model.named_modules():
        if not isinstance(module, supported):
            continue
        weight_key = f"{name}.weight"
        if weight_key not in state_dict:
            continue
        w = state_dict[weight_key]
        w_dequant, _ = get_quantized_tensor_sim(w, fmt, chunk_size=chunk_size)
        state_dict[weight_key] = w_dequant
        quant_map[name] = fmt
    return state_dict, quant_map


def _parse_input_metric(activation_dt):
    """'dyn_input_mse'  →  'mse'.  Returns None if not parseable."""
    import re
    m = re.match(r'dyn_input_(\w+)', activation_dt or '')
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# FP32 reference: fetch from DB or run inference
# ---------------------------------------------------------------------------

def _get_or_run_fp32_ref(args, device, db, model_name):
    """
    Return (ref_acc1, ref_acc5, ref_certainty).

    Strategy:
      1. Look in the DB for any successful fp32/fp32 run for this model.
      2. If found, reuse those numbers (no inference needed).
      3. If not found, run a full fp32 inference pass, log it to the DB,
         and return the results.
    """
    # --- 1. Check DB first ---
    all_runs = db.get_runs()
    if not all_runs.empty:
        fp32_rows = all_runs[
            (all_runs['model_name'] == model_name) &
            (all_runs['weight_dt'] == 'fp32') &
            (all_runs['activation_dt'] == 'fp32') &
            (all_runs['status'] == 'SUCCESS') &
            (all_runs['acc1'].notna())
        ]
        if not fp32_rows.empty:
            row = fp32_rows.iloc[0]  # newest first (get_runs ORDER BY id DESC)
            ref_acc1      = float(row['acc1'])
            ref_acc5      = float(row['acc5'])
            ref_certainty = float(row['certainty']) if row['certainty'] is not None else 0.0
            print(
                f"[FP32 ref] Found in DB (id={row['id']}): "
                f"Top1={ref_acc1:.2f}%, Top5={ref_acc5:.2f}%, "
                f"Certainty={ref_certainty:.4f}"
            )
            return ref_acc1, ref_acc5, ref_certainty

    # --- 2. Not in DB — run inference ---
    print("[FP32 ref] Not found in DB. Running fp32 inference ...")
    model, adapter = _load_fp32_model(args, device)
    loader = _build_loader(args, device)
    acc1, acc5, certainty, _ = _run_inference(
        model, adapter, loader, device, args, desc="FP32 ref"
    )
    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[FP32 ref] Top1={acc1:.2f}%, Top5={acc5:.2f}%, Certainty={certainty:.4f}")
    db.log_run(
        model_name=model_name,
        weight_dt="fp32",
        activation_dt="fp32",
        acc1=acc1, acc5=acc5,
        ref_acc1=acc1, ref_acc5=acc5, ref_certainty=certainty,
        experiment_type="fp32_ref",
        status="SUCCESS",
        certainty=certainty,
    )
    return acc1, acc5, certainty


# ---------------------------------------------------------------------------
# DB existence check
# ---------------------------------------------------------------------------

def _run_exists(db, model_name, experiment_type, weight_dt, activation_dt):
    """Return True if a successful run already exists in the DB for this combo."""
    runs = db.get_runs()
    if runs.empty:
        return False
    match = runs[
        (runs['model_name']      == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['weight_dt']       == weight_dt) &
        (runs['activation_dt']   == activation_dt) &
        (runs['status']          == 'SUCCESS')
    ]
    return not match.empty


# ---------------------------------------------------------------------------
# Main processing for a single model
# ---------------------------------------------------------------------------

def process_single_model(args, device):
    model_name = args.model_name
    weight_metrics = [m.strip() for m in args.weight_metrics.split(',')]
    input_metrics  = [m.strip() for m in args.input_metrics.split(',')]

    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    db = RunDatabase()

    print(f"\n{'='*60}")
    print(f" HYBRID EXPERIMENT: {model_name}")
    print(f"  Weight metrics : {weight_metrics}")
    print(f"  Input  metrics : {input_metrics}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Phase 1 – Weight Analysis → produce quantized weights files
    # -----------------------------------------------------------------------
    quant_weights_paths, quant_maps, layer_results_map, layer_types = run_weight_phase(
        args, device, model_dir
    )

    # -----------------------------------------------------------------------
    # FP32 reference — always resolved (from DB or fresh run)
    # -----------------------------------------------------------------------
    ref_acc1, ref_acc5, ref_certainty = _get_or_run_fp32_ref(
        args, device, db, model_name
    )

    # -----------------------------------------------------------------------
    # (Optional) Weight-only baselines  (quantized weights + fp32 inputs)
    # -----------------------------------------------------------------------
    if args.run_weight_only_baseline:
        for w_metric in weight_metrics:
            weight_dt_str_check = _summarise_quant_map(quant_maps[w_metric], w_metric=w_metric)
            if not args.force_rerun and _run_exists(
                db, model_name, f"hybrid_weight_only_{w_metric}", weight_dt_str_check, "fp32"
            ):
                print(f"\n[Baseline] Skipping weight-only ({w_metric}) — already in DB.")
                continue
            print(f"\n[Baseline] Quantized weights ({w_metric}) + FP32 inputs ...")
            q_path = quant_weights_paths[w_metric]
            model, adapter = _load_quantized_model(args, device, q_path)
            loader = _build_loader(args, device)
            acc1, acc5, certainty, _ = _run_inference(
                model, adapter, loader, device, args,
                desc=f"W-only({w_metric})"
            )
            weight_dt_str = _summarise_quant_map(quant_maps[w_metric], w_metric=w_metric)
            print(f"  W-only ({w_metric}): Top1={acc1:.2f}%, Top5={acc5:.2f}%")
            db.log_run(
                model_name=model_name,
                weight_dt=weight_dt_str,
                activation_dt="fp32",
                acc1=acc1, acc5=acc5,
                ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty,
                experiment_type=f"hybrid_weight_only_{w_metric}",
                status="SUCCESS",
                certainty=certainty,
                quant_map_json=_build_weight_map_json(quant_maps[w_metric], layer_types),
            )
            del model, adapter
            gc.collect(); torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # (Optional) Input-only baselines  (fp32 weights + dynamic input quant)
    # -----------------------------------------------------------------------
    if args.run_input_only_baseline:
        for i_metric in input_metrics:
            if not args.force_rerun and _run_exists(
                db, model_name, f"hybrid_input_only_{i_metric}", "fp32", f"dyn_input_{i_metric}"
            ):
                print(f"\n[Baseline] Skipping input-only ({i_metric}) — already in DB.")
                continue
            print(f"\n[Baseline] FP32 weights + dynamic input quant ({i_metric}) ...")
            model, adapter = _load_fp32_model(args, device)
            loader = _build_loader(args, device)

            quantizer_handler = DynamicInputQuantizer(
                model, metric=i_metric, chunk_size=args.input_chunk_size
            )
            quantizer_handler.register_hooks()

            try:
                acc1, acc5, certainty, input_stats = _run_inference(
                    model, adapter, loader, device, args,
                    input_quantizer=quantizer_handler,
                    desc=f"I-only({i_metric})"
                )
            finally:
                quantizer_handler.cleanup()

            print(f"  I-only ({i_metric}): Top1={acc1:.2f}%, Top5={acc5:.2f}%")
            db.log_run(
                model_name=model_name,
                weight_dt="fp32",
                activation_dt=f"dyn_input_{i_metric}",
                acc1=acc1, acc5=acc5,
                ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty,
                experiment_type=f"hybrid_input_only_{i_metric}",
                status="SUCCESS",
                mse=input_stats['norm_mse'] if input_stats else None,
                l1=input_stats['norm_l1'] if input_stats else None,
                certainty=certainty,
                input_map_json=_build_input_map_json(quantizer_handler.layer_stats, model),
            )
            del model, adapter
            gc.collect(); torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 2 – All-combos grid (opt-in via --run_all_combos)
    # -----------------------------------------------------------------------
    if args.run_all_combos:
        print("\n[Phase 2] All-combos grid (quantized weights × dynamic input quant) ...")

        for w_metric in weight_metrics:
            q_path = quant_weights_paths[w_metric]
            weight_dt_str = _summarise_quant_map(quant_maps[w_metric], w_metric=w_metric)
            print(f"\n  Weight metric: {w_metric}  ({weight_dt_str})")

            for i_metric in input_metrics:
                run_label = f"hybrid_w{w_metric}_i{i_metric}"
                out_dir = os.path.join(model_dir, run_label)
                os.makedirs(out_dir, exist_ok=True)

                if not args.force_rerun and _run_exists(
                    db, model_name, "hybrid_quant", weight_dt_str, f"dyn_input_{i_metric}"
                ):
                    print(f"  Skipping {run_label} — already in DB.")
                    continue

                print(f"  Input  metric: {i_metric}  → {run_label}")

                model, adapter = _load_quantized_model(args, device, q_path)
                loader = _build_loader(args, device)

                quantizer_handler = DynamicInputQuantizer(
                    model, metric=i_metric, chunk_size=args.input_chunk_size
                )
                quantizer_handler.register_hooks()

                try:
                    acc1, acc5, certainty, input_stats = _run_inference(
                        model, adapter, loader, device, args,
                        input_quantizer=quantizer_handler,
                        desc=f"Hybrid w={w_metric} i={i_metric}"
                    )
                except Exception as e:
                    print(f"  ERROR during {run_label}: {e}")
                    import traceback; traceback.print_exc()
                    quantizer_handler.cleanup()
                    del model, adapter
                    gc.collect(); torch.cuda.empty_cache()
                    db.log_run(
                        model_name=model_name,
                        weight_dt=weight_dt_str,
                        activation_dt=f"dyn_input_{i_metric}",
                        acc1=0.0, acc5=0.0,
                        ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty,
                        experiment_type="hybrid_quant",
                        status="ERROR",
                    )
                    continue
                finally:
                    quantizer_handler.cleanup()

                norm_l1  = input_stats['norm_l1']  if input_stats else None
                norm_mse = input_stats['norm_mse'] if input_stats else None
                print(
                    f"    Top1={acc1:.2f}%, Top5={acc5:.2f}%, Certainty={certainty:.4f}"
                    + (f", NormL1={norm_l1:.4e}, NormMSE={norm_mse:.4e}"
                        if norm_l1 is not None else "")
                )
                db.log_run(
                    model_name=model_name,
                    weight_dt=weight_dt_str,
                    activation_dt=f"dyn_input_{i_metric}",
                    acc1=acc1, acc5=acc5,
                    ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty,
                    experiment_type="hybrid_quant",
                    status="SUCCESS",
                    mse=norm_mse,
                    quant_map_json=_build_weight_map_json(quant_maps[w_metric], layer_types),
                    input_map_json=_build_input_map_json(quantizer_handler.layer_stats, model),
                    l1=norm_l1,
                    certainty=certainty,
                )
                stats_path = os.path.join(out_dir, "layer_stats.json")
                with open(stats_path, 'w') as f:
                    save_data = dict(quantizer_handler.layer_stats)
                    save_data['accuracy'] = {
                        'top1': acc1, 'top5': acc5, 'certainty': certainty,
                        'norm_l1': norm_l1, 'norm_mse': norm_mse,
                        'weight_metric': w_metric, 'weight_dt': weight_dt_str,
                        'input_metric': i_metric,
                    }
                    json.dump(save_data, f, indent=4)

                del model, adapter
                gc.collect()
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Phase 2 – Best-combo run (default): query DB → best weight + best input
    # -----------------------------------------------------------------------
    if not args.run_all_combos:
        print("\n[Best Combo] Querying DB for best weight and input formats ...")

        best_weight_dt, best_w_acc1 = _best_weight_from_db(db, model_name)
        best_input_dt,  best_i_acc1 = _best_input_from_db(db, model_name)

        if best_weight_dt is None:
            print("[Best Combo] No weight-only runs found in DB. "
                  "Run with --run_weight_only_baseline first.")
        elif best_input_dt is None:
            print("[Best Combo] No input-only runs found in DB. "
                  "Run with --run_input_only_baseline first.")
        else:
            print(f"[Best Combo]  Best weight: {best_weight_dt}  (acc1={best_w_acc1:.2f}%)")
            print(f"[Best Combo]  Best input : {best_input_dt}  (acc1={best_i_acc1:.2f}%)")

            # Skip if already done
            if not args.force_rerun and _run_exists(
                db, model_name, "hybrid_best_combo", best_weight_dt, best_input_dt
            ):
                print("[Best Combo] Already in DB — skipping.")
            else:
                # --- Resolve input quantization ---
                i_metric = _parse_input_metric(best_input_dt)
                is_uniform_input = _is_uniform_format(best_input_dt)

                if i_metric is None and not is_uniform_input:
                    print(f"[Best Combo] Cannot interpret input_dt '{best_input_dt}'. Skipping.")
                else:
                    # --- Resolve weight: opt_w*[...] (layer-wise) or plain uniform format ---
                    w_metric = _parse_weight_metric(best_weight_dt)
                    is_uniform = _is_uniform_format(best_weight_dt)
                    model = None
                    adapter = None
                    resolved_quant_map_json = None

                    if w_metric is None and not is_uniform:
                        print(f"[Best Combo] Cannot interpret weight_dt '{best_weight_dt}'. Skipping.")
                    elif is_uniform:
                        # Plain baseline format — build uniform quant model in-memory
                        print(f"[Best Combo] Running: uniform_weight={best_weight_dt}, input={best_input_dt}")
                        config = {
                            'model': {'name': args.model_name, 'weights': args.weights},
                            'adapter': {'type': 'generic', 'quantized_ops': []},
                            'dataset': {
                                'name': args.dataset_name, 'path': args.dataset_path,
                                'batch_size': args.batch_size, 'num_workers': args.num_workers,
                            },
                        }
                        adapter = create_adapter(config)
                        model = adapter.model
                        q_state_dict, uniform_quant_map = _build_uniform_quant_state_dict(
                            model, best_weight_dt, chunk_size=args.weight_chunk_size
                        )
                        model.load_state_dict(q_state_dict, strict=False)
                        model.to(device)
                        model.eval()
                        resolved_layer_types = _layer_types_from_model(model)
                        resolved_quant_map_json = _build_weight_map_json(uniform_quant_map, resolved_layer_types)
                    else:
                        # Layer-wise optimized — load from saved .pt file
                        q_path = os.path.join(model_dir, f"weights_{w_metric}", "quantized_weights.pt")
                        if not os.path.exists(q_path):
                            print(f"[Best Combo] Quantized weights file not found: {q_path}")
                            print("[Best Combo]  Re-running weight phase to regenerate it ...")
                            quant_weights_paths, quant_maps, layer_results_map, layer_types = run_weight_phase(
                                args, device, model_dir
                            )
                            q_path = quant_weights_paths.get(w_metric)
                        if q_path is None or not os.path.exists(str(q_path)):
                            print("[Best Combo] Quantized weights still unavailable. Skipping.")
                            model = None
                        else:
                            model, adapter = _load_quantized_model(args, device, q_path)
                            resolved_quant_map_json = _build_weight_map_json(
                                quant_maps.get(w_metric, {}), layer_types
                            )

                    # --- Shared inference for both weight paths ---
                    if model is not None:
                        loader = _build_loader(args, device)

                        if is_uniform_input:
                            quantizer_handler = UniformInputQuantizer(
                                model, fmt=best_input_dt, chunk_size=args.input_chunk_size
                            )
                        else:
                            quantizer_handler = DynamicInputQuantizer(
                                model, metric=i_metric, chunk_size=args.input_chunk_size
                            )
                        quantizer_handler.register_hooks()

                        try:
                            acc1, acc5, certainty, input_stats = _run_inference(
                                model, adapter, loader, device, args,
                                input_quantizer=quantizer_handler,
                                desc="Best Combo"
                            )
                        except Exception as e:
                            print(f"[Best Combo] ERROR: {e}")
                            import traceback; traceback.print_exc()
                            quantizer_handler.cleanup()
                            del model, adapter
                            gc.collect(); torch.cuda.empty_cache()
                            db.log_run(
                                model_name=model_name,
                                weight_dt=best_weight_dt,
                                activation_dt=best_input_dt,
                                acc1=0.0, acc5=0.0,
                                ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty,
                                experiment_type="hybrid_best_combo",
                                status="ERROR",
                            )
                        else:
                            quantizer_handler.cleanup()
                            norm_l1  = input_stats['norm_l1']  if input_stats else None
                            norm_mse = input_stats['norm_mse'] if input_stats else None
                            print(
                                f"[Best Combo] Top1={acc1:.2f}%, Top5={acc5:.2f}%, "
                                f"Certainty={certainty:.4f}"
                                + (f", NormL1={norm_l1:.4e}, NormMSE={norm_mse:.4e}"
                                   if norm_l1 is not None else "")
                            )
                            db.log_run(
                                model_name=model_name,
                                weight_dt=best_weight_dt,
                                activation_dt=best_input_dt,
                                acc1=acc1, acc5=acc5,
                                ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty,
                                experiment_type="hybrid_best_combo",
                                status="SUCCESS",
                                mse=norm_mse,
                                l1=norm_l1,
                                certainty=certainty,
                                quant_map_json=resolved_quant_map_json,
                                input_map_json=_build_input_map_json(
                                    quantizer_handler.layer_stats, model
                                ),
                            )
                            del model, adapter
                            gc.collect(); torch.cuda.empty_cache()

    print(f"\n[Done] Hybrid experiment finished for {model_name}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine models to run
    if args.models_file:
        with open(args.models_file) as f:
            yaml_models = yaml.safe_load(f)
        if not isinstance(yaml_models, list):
            print("Error: models_file must be a YAML list.")
            sys.exit(1)
        models_to_run = yaml_models
    else:
        models_to_run = [{'name': args.model_name, 'weights': args.weights}]

    print(f"Processing {len(models_to_run)} model(s).")

    for model_cfg in models_to_run:
        if isinstance(model_cfg, str):
            args.model_name = model_cfg
            args.weights = 'DEFAULT'
        else:
            args.model_name = model_cfg.get('name', args.model_name)
            args.weights = model_cfg.get('weights', 'DEFAULT')

        process_single_model(args, device)


if __name__ == "__main__":
    main()
