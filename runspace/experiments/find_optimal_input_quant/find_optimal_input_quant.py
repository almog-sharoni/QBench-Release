
import os
import sys
import torch
import argparse
import numpy as np
import yaml
import gc
import copy
import matplotlib.pyplot as plt
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.experiments.utils.common import (
    build_uniform_input_quant_cfg as _build_uniform_input_quant_cfg,
)
# from runspace.src.quantization.constants import get_quantization_bias

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# baseline_formats = ['fp32', 'fp4_e3m0','fp4_e2m1','fp4_e1m2', 'fp3_e1m1', 'fp3_e2m0','fp8_e7m0']
# baseline_formats = [ 'fp32', 'fp2_e1m0', 'fp3_e1m1', 'fp4_e1m2', 'fp5_e1m3', 'fp6_e1m4', 'fp7_e1m5', 'fp8_e1m6',
#     'fp3_e2m0', 'fp4_e3m0', 'fp5_e4m0', 'fp6_e5m0', 'fp7_e6m0', 'fp8_e7m0'
# ]
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

candidate_formats = [
    'fp32',
    'fp8_e1m6','fp8_e2m5','fp8_e3m4','fp8_e4m3','fp8_e5m2','fp8_e6m1','fp8_e7m0',
    'fp7_e1m5','fp7_e2m4','fp7_e3m3','fp7_e4m2','fp7_e5m1','fp7_e6m0',
    'fp6_e1m4','fp6_e2m3','fp6_e3m2','fp6_e4m1','fp6_e5m0',
    'fp5_e1m3','fp5_e2m2','fp5_e3m1','fp5_e4m0',
    'fp4_e1m2','fp4_e2m1','fp4_e3m0',
    'fp3_e1m1','fp3_e2m0',
    'fp2_e1m0'
]

DEFAULT_BASELINE_EXPERIMENT_TYPE = "input_quant_baseline"
DEFAULT_DYNAMIC_EXPERIMENT_TYPE = "input_quant_dynamic"


def _parse_csv_arg(value, fallback):
    if value is None:
        return list(fallback)
    parsed = [item.strip() for item in str(value).split(',') if item.strip()]
    return parsed if parsed else list(fallback)

# Keep experiments on the library replacement path (no manual tensor injection).
# `weight_quantization` will be disabled in config for input-only studies.
INPUT_ONLY_QUANTIZED_OPS = ["all"]


def _iter_quantized_modules(model):
    supported_quant_ops = tuple(OpRegistry.get_supported_ops().values())
    for module in model.modules():
        if isinstance(module, supported_quant_ops):
            yield module


def _build_input_quant_config(args, model_name, weights, default_format, quantize_first_layer=False, input_quantization=True):
    """Build the actual runtime config used by this experiment."""
    unsigned_input_sources = getattr(args, 'unsigned_input_sources', [])
    return {
        'model': {'name': model_name, 'weights': weights},
        'adapter': {
            'type': 'generic',
            'quantized_ops': INPUT_ONLY_QUANTIZED_OPS,
            'excluded_ops': args.excluded_ops,
            'quantize_first_layer': quantize_first_layer,
            'input_quantization': input_quantization,
            'weight_quantization': False,
            'input_size': getattr(args, 'input_size', 224),
        },
        'dataset': {
            'name': args.dataset_name,
            'path': args.dataset_path,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        },
        'quantization': {
            'format': default_format,
            'input_format': default_format,
            'mode': 'chunk',
            'chunk_size': args.chunk_size,
            'weight_mode': 'tensor',
            'weight_chunk_size': args.chunk_size,
            'rounding': 'nearest',
            'calib_method': 'max',
            'unsigned_input_sources': unsigned_input_sources,
            'weight_source': 'fp32',
        },
        'experiment': {
            'materialize_weights': {
                'force_rebuild': bool(getattr(args, 'force_rebuild_weights', False) or getattr(args, 'force_rerun', False)),
            },
        },
    }


def _serialize_runtime_config(config, model=None, *, experiment_type=None, activation_dt=None, metric=None, limit_batches=None):
    """Serialize the real runtime config + lightweight runtime metadata."""
    cfg = copy.deepcopy(config)
    cfg.setdefault('dataset', {})
    if limit_batches is not None:
        cfg['dataset']['limit_batches'] = limit_batches

    if model is not None:
        first_quant = next(_iter_quantized_modules(model), None)
        if first_quant is not None:
            cfg['runtime'] = {
                'sample_quant_module': first_quant.__class__.__name__,
                'q_type': str(getattr(first_quant, 'q_type', None)),
                'input_q_type': str(getattr(first_quant, 'input_q_type', None)),
                'input_mode': str(getattr(first_quant, 'input_mode', None)),
                'input_chunk_size': int(getattr(first_quant, 'input_chunk_size', 0) or 0),
                'rounding': str(getattr(first_quant, 'rounding', None)),
                'input_quantization': bool(getattr(first_quant, 'input_quantization', False)),
                'weight_quantization': bool(getattr(first_quant, 'weight_quantization', False)),
                'unsigned_input_sources': ["relu", "softmax", "quantrelu", "quantsoftmax"]
            }

    cfg['experiment'] = {
        'type': experiment_type,
        'activation_dt': activation_dt,
        'metric': metric,
    }
    return json.dumps(cfg)

def get_args():
    parser = argparse.ArgumentParser(description="Find optimal input quantization (Dynamic)")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--models_file", type=str, default=None, help="Path to models.yaml file to run on multiple models")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit number of batches to process (default: -1 for all)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"), help="Output directory")
    parser.add_argument("--metric", type=str, default="mse,l1", help="Comma-separated error metrics for dynamic selection (e.g. 'mse,l1')")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for input quantization (blocks)")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size (resolution)")
    parser.add_argument(
        "--baseline_formats",
        type=str,
        default=None,
        help="Comma-separated baseline formats to evaluate. Defaults to the script's baseline_formats list.",
    )
    parser.add_argument(
        "--candidate_formats",
        type=str,
        default=None,
        help="Comma-separated dynamic candidate formats. Defaults to the script's candidate_formats list.",
    )
    parser.add_argument(
        "--post_relu_ufp_only",
        action="store_true",
        help=(
            "Restrict dynamic candidate selection so post-ReLU layers use UFP "
            "candidates and other layers use non-UFP candidates. By default, "
            "every layer can choose from every --candidate_formats entry."
        ),
    )
    parser.add_argument(
        "--excluded_ops",
        type=str,
        default="LayerNorm",
        help="Comma-separated op names to exclude from quantization (default: LayerNorm)"
    )
    parser.add_argument("--only_dynamic", action="store_true", help="Skip baseline runs and only run dynamic optimization")
    parser.add_argument("--only_baselines", action="store_true", help="Skip dynamic runs and only run baseline runs")
    parser.add_argument("--force_rerun", action="store_true", help="Re-run all experiments even if already in DB")
    parser.add_argument(
        "--force_rebuild_weights",
        action="store_true",
        help="Force rebuilding cached materialized weights/checkpoints used by the experiment",
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default=DEFAULT_BASELINE_EXPERIMENT_TYPE,
        help="Experiment type name for baseline runs (default: input_quant_baseline)"
    )
    parser.add_argument(
        "--dynamic_experiment_type",
        type=str,
        default=None,
        help=(
            "Experiment type name for dynamic runs. Defaults to input_quant_dynamic, "
            "or to --experiment_type when --only_dynamic is set and --experiment_type "
            "was explicitly changed."
        ),
    )
    parser.add_argument(
        "--unsigned_input_sources",
        type=str,
        default="relu,softmax,quantrelu,quantsoftmax",
        help=(
            "Comma-separated activation sources that should use unsigned input/output "
            "formats. If omitted, experiment types containing 'ufp' enable "
            "relu,softmax,quantrelu,quantsoftmax."
        ),
    )
    parser.set_defaults(dynamic_unsigned_input_candidates=True)
    parser.add_argument(
        "--dynamic_unsigned_input_candidates",
        dest="dynamic_unsigned_input_candidates",
        action="store_true",
        help=(
            "Enable UFP-converted dynamic candidate formats on layers after "
            "--unsigned_input_sources sources."
        ),
    )
    parser.add_argument(
        "--no_dynamic_unsigned_input_candidates",
        dest="dynamic_unsigned_input_candidates",
        action="store_false",
        help=(
            "Disable UFP-converted dynamic candidate formats after "
            "--unsigned_input_sources sources."
        ),
    )
    # Add other args as needed
    args = parser.parse_args()
    args.excluded_ops = [op.strip() for op in args.excluded_ops.split(',') if op.strip()]
    args.baseline_formats = _parse_csv_arg(args.baseline_formats, baseline_formats)
    args.candidate_formats = _parse_csv_arg(args.candidate_formats, candidate_formats)
    if args.dynamic_experiment_type is None:
        if args.only_dynamic and args.experiment_type != DEFAULT_BASELINE_EXPERIMENT_TYPE:
            args.dynamic_experiment_type = args.experiment_type
        else:
            args.dynamic_experiment_type = DEFAULT_DYNAMIC_EXPERIMENT_TYPE
    
    args.unsigned_input_sources = [
        item.strip().lower()
        for item in args.unsigned_input_sources.split(',')
        if item.strip()
    ]
    return args


def _input_quant_run_exists(db, model_name, experiment_type, activation_dt):
    """Return True if a successful run exists in DB for this model/experiment/activation combo."""
    runs = db.get_runs()
    if runs.empty:
        return False
    return not runs[
        (runs['model_name']      == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['weight_dt']       == 'fp32') &
        (runs['activation_dt']   == activation_dt) &
        (runs['status']          == 'SUCCESS')
    ].empty


def run_baselines(args, device, formats, on_result=None):
    """
    Run baseline evaluations with strict per-format isolation.
    Each format gets a fresh adapter/model so results are independent of
    evaluation order and match single-format runs.
    """
    print(f"\n--- Running Baselines (Optimized: {formats}) ---")
    final_stats = {}
    config_json_by_fmt = {}
    runner = Runner(device)

    # Build one shared loader for all baseline formats to avoid worker respawn cost.
    loader_cfg = _build_input_quant_config(
        args,
        args.model_name,
        args.weights,
        'fp32',
        quantize_first_layer=False
    )
    loader = runner.setup_data_loader(loader_cfg)
    if loader is None:
        raise RuntimeError("Failed to build data loader for baseline runs.")

    try:
        for fmt in formats:
            input_quantization=True
            if fmt == "fp32":
                input_quantization=False
            config = _build_input_quant_config(
                args,
                args.model_name,
                args.weights,
                fmt,
                quantize_first_layer=False,
                input_quantization=input_quantization
            )
            baseline_run_dir = os.path.join(args.output_dir, args.model_name, f"baseline_{fmt}")
            model, adapter, _ = runner.prepare_model_with_materialized_weights(
                config=config,
                output_dir=baseline_run_dir
            )

            eval_results = runner.evaluate_model(
                model=model,
                data_loader=loader,
                adapter=adapter,
                max_batches=args.limit_batches,
                desc=f"Baseline ({fmt})",
                input_quant_cfg=_build_uniform_input_quant_cfg(fmt, args.chunk_size),
            )
            acc1 = eval_results.get('acc1', 0.0)
            acc5 = eval_results.get('acc5', 0.0)
            certainty = eval_results.get('certainty', 0.0)
            input_stats = eval_results.get('input_quant', {}) if fmt != 'fp32' else {}
            norm_l1 = float(input_stats.get('norm_l1', 0.0) or 0.0)
            norm_mse = float(input_stats.get('norm_mse', 0.0) or 0.0)

            final_stats[fmt] = {
                'acc1': acc1,
                'acc5': acc5,
                'certainty': certainty,
                'norm_l1': norm_l1,
                'norm_mse': norm_mse,
                'total_l1': float(input_stats.get('total_l1', 0.0) or 0.0),
                'total_mse': float(input_stats.get('total_mse', 0.0) or 0.0),
                'layer_stats': input_stats.get('layer_stats', {}) if isinstance(input_stats, dict) else {},
            }
            config_json_by_fmt[fmt] = _serialize_runtime_config(
                config,
                model=model,
                experiment_type=args.experiment_type,
                activation_dt=fmt,
                metric=None,
                limit_batches=args.limit_batches,
            )
            if on_result is not None:
                try:
                    on_result(fmt, final_stats[fmt], config_json_by_fmt[fmt])
                except Exception as e:
                    print(f"[DB] Failed to log baseline {fmt} immediately: {e}")

            print(
                f"Baseline {fmt}: Top1={acc1:.2f}%, Top5={acc5:.2f}%, "
                f"Certainty={certainty:.4f}, NormL1={norm_l1:.4e}, NormMSE={norm_mse:.4e}"
            )

            del model
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        runner._shutdown_dataloader_workers(loader)
        del loader
        gc.collect()
        torch.cuda.empty_cache()

    return final_stats, config_json_by_fmt

def plot_format_histogram(layer_stats, output_dir):
    """Generate histogram of selected formats."""
    import matplotlib.pyplot as plt
    
    print("Generating format distribution histogram...")
    
    # Aggregated counts across all layers
    total_counts = {}
    
    for layer, stats in layer_stats.items():
        if 'format_counts' in stats:
            for fmt, count in stats['format_counts'].items():
                total_counts[fmt] = total_counts.get(fmt, 0) + count
                
    if not total_counts:
        print("No format statistics found to plot.")
        return

    formats = list(total_counts.keys())
    counts = list(total_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(formats, counts, color='skyblue')
    plt.xlabel('Format')
    plt.ylabel('Total Selections (Chunks)')
    plt.title('Dynamic Input Format Selection Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "format_distribution.png"))
    plt.close()
    print(f"Saved histogram to {os.path.join(output_dir, 'format_distribution.png')}")


def plot_layer_format_distribution(layer_stats, output_dir, metric):
    """Generate a stacked bar chart of format distribution per layer."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"Generating layer-wise format distribution for {metric}...")
    
    layers = list(layer_stats.keys())
    if not layers:
        print("No layer stats found.")
        return

    # Collect all unique formats
    all_formats = set()
    for stats in layer_stats.values():
        if 'format_counts' in stats:
            all_formats.update(stats['format_counts'].keys())
    
    sorted_formats = sorted(list(all_formats))
    if not sorted_formats:
        return

    # Prepare data: [n_layers, n_formats]
    data = np.zeros((len(layers), len(sorted_formats)))
    
    for i, layer in enumerate(layers):
        counts = layer_stats[layer].get('format_counts', {})
        for j, fmt in enumerate(sorted_formats):
            data[i, j] = counts.get(fmt, 0)
            
    # Stacked bars
    # Adjust figure size based on number of layers
    plt.figure(figsize=(max(12, len(layers)*0.3), 8))
    
    bottom = np.zeros(len(layers))
    # Colormap
    cmap = plt.get_cmap('tab20', len(sorted_formats))
    
    for j, fmt in enumerate(sorted_formats):
        plt.bar(layers, data[:, j], bottom=bottom, label=fmt, color=cmap(j))
        bottom += data[:, j]
        
    plt.xlabel('Layer')
    plt.ylabel('Count (Chunks)')
    plt.title(f'Layer-wise Format Distribution ({metric.upper()})')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(output_dir, f"layer_format_distribution_{metric}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved layer distribution to {save_path}")


from runspace.experiments.utils.plotting import plot_accuracy_comparison


def process_single_model(args, model_config, device, metrics):
    """Process a single model: Baselines -> Dynamic Metrics -> Plots."""
    
    model_name = model_config['name']
    weights = model_config.get('weights', 'DEFAULT')
    
    runner = Runner(device)
    db = runner._get_db()
    
    print(f"\n###########################################################")
    print(f" PROCESSING MODEL: {model_name} (Weights: {weights})")
    print(f"###########################################################")
    
    # Model-specific output dir
    model_out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)
    
    all_results = []
    
    # --- 1. Run Baselines (Global) ---
    if not args.only_dynamic:
        args.model_name = model_name
        args.weights = weights

        # Check which baseline formats are already in DB
        cached_baseline_stats = {}
        formats_to_run = []
        if not args.force_rerun:
            for fmt in args.baseline_formats:
                if _input_quant_run_exists(db, model_name, args.experiment_type, fmt) or fmt == 'fp32' and _input_quant_run_exists(db, model_name, 'fp32_ref', 'fp32'):
                    all_runs = db.get_runs()
                    row = all_runs[
                        (all_runs['model_name'] == model_name) &
                        (all_runs['weight_dt']  == 'fp32') &
                        (all_runs['activation_dt'] == fmt) &
                        (all_runs['status'] == 'SUCCESS')
                    ]
                    if not row.empty:
                        r = row.iloc[0]
                        cached_baseline_stats[fmt] = {
                            'acc1': float(r['acc1']), 'acc5': float(r['acc5']),
                            'norm_l1': float(r['l1'] or 0), 'norm_mse': float(r['mse'] or 0),
                            'certainty': float(r['certainty'] or 0),
                        }
                        print(f"[Baseline] Skipping {fmt} — already in DB (acc1={r['acc1']:.2f}%)")
                        continue
                formats_to_run.append(fmt)
        else:
            formats_to_run = list(args.baseline_formats)

        ref_acc1_live = cached_baseline_stats.get('fp32', {}).get('acc1', 0.0)
        ref_acc5_live = cached_baseline_stats.get('fp32', {}).get('acc5', 0.0)
        ref_certainty_live = cached_baseline_stats.get('fp32', {}).get('certainty', 0.0)

        def _log_baseline_immediately(fmt, stats, cfg_json):
            nonlocal ref_acc1_live, ref_acc5_live, ref_certainty_live
            if fmt == 'fp32':
                ref_acc1_live = float(stats.get('acc1', 0.0) or 0.0)
                ref_acc5_live = float(stats.get('acc5', 0.0) or 0.0)
                ref_certainty_live = float(stats.get('certainty', 0.0) or 0.0)
                return

            log_cfg = _build_input_quant_config(
                args, model_name, weights, fmt, quantize_first_layer=False
            )
            log_cfg['experiment'] = {
                'name': 'find_optimal_input_quant',
                'type': args.experiment_type,
                'weight_dt': 'fp32',
                'activation_dt': fmt,
                'ref_acc1': ref_acc1_live,
                'ref_acc5': ref_acc5_live,
                'ref_certainty': ref_certainty_live,
                'metrics': {
                    'mse': stats.get('norm_mse', 0.0),
                    'l1': stats.get('norm_l1', 0.0),
                    'certainty': stats.get('certainty', 0.0),
                },
                'config_json': cfg_json,
            }
            runner.log_experiment_result(
                config=log_cfg,
                result={
                    'model_name': model_name,
                    'status': 'SUCCESS',
                    'acc1': stats.get('acc1', 0.0),
                    'acc5': stats.get('acc5', 0.0),
                    'certainty': stats.get('certainty', 0.0),
                    'input_quant': (
                        {
                            'mode': 'uniform',
                            'format': fmt,
                            'chunk_size': args.chunk_size,
                            'norm_l1': stats.get('norm_l1', 0.0),
                            'norm_mse': stats.get('norm_mse', 0.0),
                            'total_l1': stats.get('total_l1', 0.0),
                            'total_mse': stats.get('total_mse', 0.0),
                            'layer_stats': stats.get('layer_stats', {}),
                        }
                        if fmt != 'fp32' else {}
                    ),
                },
            )

        new_baseline_stats, _ = (
            run_baselines(
                args,
                device,
                formats_to_run,
                on_result=_log_baseline_immediately,
            )
            if formats_to_run else ({}, {})
        )
        baseline_stats = {**cached_baseline_stats, **new_baseline_stats}
        
        # Identify Reference (fp32)
        ref_acc1 = baseline_stats.get('fp32', {}).get('acc1', 0.0)
        ref_acc5 = baseline_stats.get('fp32', {}).get('acc5', 0.0)
        ref_certainty = baseline_stats.get('fp32', {}).get('certainty', 0.0)
        
        for fmt, stats in baseline_stats.items():
            # Skip logging fp32 as a visible row in the DB, but keep it in all_results for plotting
            if fmt == 'fp32':
                all_results.append({
                    'output_name': f"Base_{fmt}", 
                    'acc1': stats['acc1'],
                    'acc5': stats['acc5'],
                    'errors': {'norm_l1': stats['norm_l1'], 'norm_mse': stats['norm_mse']}
                })
                continue
                
            all_results.append({
                'output_name': f"Base_{fmt}", 
                'acc1': stats['acc1'],
                'acc5': stats['acc5'],
                'errors': {
                    'norm_l1': stats['norm_l1'],
                    'norm_mse': stats['norm_mse']
                }
            })
    else:
        print("\nSkipping Baselines (--only_dynamic set)")
        ref_acc1, ref_acc5, ref_certainty = None, None, None
        
    
    # --- 2. Run Dynamic Optimization Loop ---
    
    # Pre-load Model and Data ONCE
    print(f"\n[Optimization] Loading model and dataset once for all metrics...")
    
    config = _build_input_quant_config(
        args,
        model_name,
        weights,
        args.baseline_formats[0] if args.baseline_formats else 'fp8_e4m3',
        quantize_first_layer=False
    )
    
    # Explicitly clean up before loading
    gc.collect()
    torch.cuda.empty_cache()
    if not args.only_baselines:
        try:
            # Build loader before CUDA model initialization to keep worker start fast/stable.
            loader = runner.setup_data_loader(config)
            model, adapter, _ = runner.prepare_model_with_materialized_weights(
                config=config,
                output_dir=model_out_dir
            )
            
            for metric in metrics:
                activation_dt = f"dyn_input_{metric}"
                if not args.force_rerun and _input_quant_run_exists(db, model_name, args.dynamic_experiment_type, activation_dt):
                    print(
                        f"[Dynamic] Skipping {metric} — already in DB for {model_name} "
                        f"(experiment_type={args.dynamic_experiment_type})"
                    )
                    continue

                print(f"\n===========================================")
                print(f"Processing Metric: {metric.upper()} for {model_name}")
                print(f"=============================================")
                
                metric_out_dir = os.path.join(model_out_dir, metric)
                os.makedirs(metric_out_dir, exist_ok=True)

                try:
                    config.setdefault('evaluation', {})
                    config['evaluation']['dynamic_input_quant'] = {
                        'enabled': True,
                        'metric': metric,
                        'chunk_size': args.chunk_size,
                        'candidate_formats': args.candidate_formats,
                        'restrict_post_relu_ufp': args.post_relu_ufp_only,
                        'unsigned_input_sources': args.unsigned_input_sources,
                        'dynamic_unsigned_input_candidates': args.dynamic_unsigned_input_candidates,
                    }
                    eval_results = runner.evaluate_model(
                        model=model,
                        data_loader=loader,
                        adapter=adapter,
                        max_batches=args.limit_batches,
                        desc=f"Dynamic ({model_name}/{metric})",
                        dynamic_input_quant_cfg=config['evaluation']['dynamic_input_quant']
                    )
                    acc1 = eval_results.get('acc1', 0.0)
                    acc5 = eval_results.get('acc5', 0.0)
                    certainty = eval_results.get('certainty', 0.0)

                    dyn_stats = eval_results.get('dynamic_input_quant', {})
                    layer_stats = dyn_stats.get('layer_stats', {})
                    final_stats = {
                        'norm_l1': dyn_stats.get('norm_l1', 0.0),
                        'norm_mse': dyn_stats.get('norm_mse', 0.0),
                    }
                    
                    print(f"\nDynamic Run ({metric.upper()}): Top1={acc1:.2f}%, Top5={acc5:.2f}%, Certainty={certainty:.4f}")
                    print(f"Norm L1: {final_stats['norm_l1']:.4e}, Norm MSE: {final_stats['norm_mse']:.4e}")
                    
                    # Log Dynamic Result to Database using the actual runtime config.
                    _cfg_dyn = _serialize_runtime_config(
                        config,
                        model=model,
                        experiment_type=args.dynamic_experiment_type,
                        activation_dt=f"dyn_input_{metric}",
                        metric=metric,
                        limit_batches=args.limit_batches,
                    )
                    log_cfg = copy.deepcopy(config)
                    log_cfg['experiment'] = {
                        'name': 'find_optimal_input_quant',
                        'type': args.dynamic_experiment_type,
                        'weight_dt': 'fp32',
                        'activation_dt': f"dyn_input_{metric}",
                        'ref_acc1': ref_acc1,
                        'ref_acc5': ref_acc5,
                        'ref_certainty': ref_certainty,
                        'metrics': {
                            'mse': final_stats['norm_mse'],
                            'l1': final_stats['norm_l1'],
                            'certainty': certainty,
                        },
                        'config_json': _cfg_dyn,
                    }
                    runner.log_experiment_result(
                        config=log_cfg,
                        result={
                            'model_name': model_name,
                            'status': 'SUCCESS',
                            'acc1': acc1,
                            'acc5': acc5,
                            'certainty': certainty,
                            'input_quant': dyn_stats,
                        },
                    )
                    
                    plot_format_histogram(layer_stats, metric_out_dir)
                    plot_layer_format_distribution(layer_stats, metric_out_dir, metric)
                    
                    stats_path = os.path.join(metric_out_dir, "layer_stats.json")
                    with open(stats_path, 'w') as f:
                        # Save stats + accuracy
                        save_data = copy.deepcopy(layer_stats)
                        save_data['accuracy'] = {
                            'top1': acc1, 'top5': acc5,
                            'norm_l1': final_stats['norm_l1'],
                            'norm_mse': final_stats['norm_mse']
                        }
                        json.dump(save_data, f, indent=4)
                
                except KeyboardInterrupt:
                    print("\nInterrupted.")
                    return 
                except Exception as e:
                    print(f"Error processing {model_name} / {metric}: {e}")
                    import traceback
                    traceback.print_exc()
                    
        finally:
            # Clean up memory after ALL metrics are done for this model
            if 'model' in locals(): del model
            if 'adapter' in locals(): del adapter
            if 'loader' in locals():
                runner._shutdown_dataloader_workers(loader)
                del loader
            if 'runner' in locals(): del runner
            
            gc.collect()
            torch.cuda.empty_cache()

    # --- 3. Plot Final Comparison ---
    if all_results:
        plot_accuracy_comparison(all_results, model_out_dir)
        
        # Also save raw results to CSV/JSON for easy review
        with open(os.path.join(model_out_dir, "comparison_results.json"), 'w') as f:
             json.dump(all_results, f, indent=4)


def main():
    # Force generic cleanup at startup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse metrics
    metrics = [m.strip().lower() for m in args.metric.split(',')]
    print(f"Metrics to process: {metrics}")
    print(f"Baseline formats: {args.baseline_formats}")
    print(f"Dynamic candidate formats: {args.candidate_formats}")
    
    # Determine models to run
    models_to_run = []
    
    if args.models_file:
        print(f"Loading models from: {args.models_file}")
        with open(args.models_file, 'r') as f:
            yaml_models = yaml.safe_load(f)
            # Ensure it's a list
            if isinstance(yaml_models, list):
                models_to_run = yaml_models
            else:
                print("Error: models.yaml must contain a list of models.")
                sys.exit(1)
    else:
        # Single model from args
        models_to_run = [{'name': args.model_name, 'weights': args.weights}]
        
    print(f"Found {len(models_to_run)} models to process.")
    
    # Initialize Output Dir
    os.makedirs(args.output_dir, exist_ok=True)

    for model_config in models_to_run:
        process_single_model(args, model_config, device, metrics)

if __name__ == "__main__":
    main()
