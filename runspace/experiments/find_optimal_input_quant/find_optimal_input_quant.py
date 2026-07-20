
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
import re

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.experiments.utils.common import (
    build_uniform_input_quant_cfg as _build_uniform_input_quant_cfg,
    row_uses_encoded_activation_transport,
)
from runspace.src.quantization.dynamic_input_metrics import (
    assert_dynamic_input_metric_implemented,
    normalize_dynamic_input_metric,
    normalize_pseudo_mse3_fixed_rounding,
    normalize_pseudo_mse3_tie_break,
    validate_pseudo_mse_candidate_pairs,
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


_CANDIDATE_FORMAT_BIT_WIDTH_RE = re.compile(
    r"^(?:fp|ufp|efp|uefp)(?P<bit_width>\d+)(?:_|$)",
    re.IGNORECASE,
)
_CANDIDATE_FORMAT_EXPONENT_RE = re.compile(
    r"_e(?P<exponent>\d+)m\d+$",
    re.IGNORECASE,
)
_SUPPORTED_DYNAMIC_METRICS = {
    'l2': 'mse',
    'l1': 'l1',
    'pseudo_mse3': 'pseudo_mse3',
}
_ACTIVATION_EXPONENT_POLICIES = ('all', 'e1e2')


def _candidate_formats_by_bit_width(formats):
    """Group dynamic candidates by the total bit width in their format name."""
    grouped = {}
    for candidate in formats:
        candidate = str(candidate).strip()
        match = _CANDIDATE_FORMAT_BIT_WIDTH_RE.match(candidate)
        if match is None:
            raise ValueError(
                "Dynamic candidate format must include a bit width in its name "
                f"(for example, fp8_e4m3); got {candidate!r}."
            )
        bit_width = int(match.group('bit_width'))
        grouped.setdefault(bit_width, []).append(candidate)
    return grouped


def _candidate_format_exponent(candidate):
    match = _CANDIDATE_FORMAT_EXPONENT_RE.search(str(candidate).strip())
    if match is None:
        raise ValueError(
            "Dynamic candidate format must include exponent and mantissa widths "
            f"(for example, fp8_e2m5); got {candidate!r}."
        )
    return int(match.group('exponent'))


def _filter_candidate_formats_by_activation_exponents(formats, policy='all'):
    """Apply the activation exponent policy before splitting candidates by width."""
    policy = str(policy or 'all').strip().lower()
    if policy not in _ACTIVATION_EXPONENT_POLICIES:
        raise ValueError(
            f"Unsupported activation exponent policy: {policy!r}. "
            f"Expected one of {_ACTIVATION_EXPONENT_POLICIES}."
        )
    formats = list(formats)
    if policy == 'all':
        return formats
    return [
        candidate
        for candidate in formats
        if _candidate_format_exponent(candidate) in (1, 2)
    ]


def _normalize_requested_metrics(value):
    """Normalize the metrics supported by this accuracy experiment."""
    requested = [item.strip() for item in str(value or 'mse').split(',') if item.strip()]
    normalized_metrics = []
    for metric in requested:
        canonical = normalize_dynamic_input_metric(metric)
        normalized = _SUPPORTED_DYNAMIC_METRICS.get(canonical)
        if normalized is None:
            raise ValueError(
                "find_optimal_input_quant supports only mse, l1, and pseudo_mse3; "
                f"got {metric!r}."
            )
        assert_dynamic_input_metric_implemented(canonical)
        if normalized not in normalized_metrics:
            normalized_metrics.append(normalized)
    return normalized_metrics


def _prepare_dynamic_candidate_groups(formats, activation_exponents, metrics):
    """Filter candidates and validate pseudo-MSE3's one-pair-per-width contract."""
    filtered = _filter_candidate_formats_by_activation_exponents(
        formats,
        activation_exponents,
    )
    grouped = _candidate_formats_by_bit_width(filtered)
    skipped_widths = []
    if 'pseudo_mse3' not in metrics:
        return grouped, skipped_widths

    compatible = {}
    for bit_width, candidates in grouped.items():
        exponents = {_candidate_format_exponent(candidate) for candidate in candidates}
        if exponents != {1, 2}:
            skipped_widths.append(bit_width)
            continue
        validate_pseudo_mse_candidate_pairs(candidates)
        compatible[bit_width] = candidates

    if not compatible:
        raise ValueError(
            "pseudo_mse3 requires at least one same-width signed e1/e2 candidate pair."
        )
    return compatible, skipped_widths


def _dynamic_experiment_type_for_bit_width(base_experiment_type, bit_width):
    return f"{base_experiment_type}_{int(bit_width)}"


def _dynamic_activation_dt(metric, args):
    """Return a collision-safe database label for one selector configuration."""
    metric = str(metric).strip().lower()
    exponent_policy = str(getattr(args, 'activation_exponents', 'all') or 'all').lower()
    if metric == 'mse':
        if exponent_policy == 'all':
            return 'dyn_input_mse'
        return f"dyn_input_mse_{exponent_policy}"
    if metric == 'l1':
        if exponent_policy == 'all':
            return 'dyn_input_l1'
        return f"dyn_input_l1_{exponent_policy}"
    if metric == 'pseudo_mse3':
        bits_to_take = int(getattr(args, 'bits_to_take', 0) or 0)
        fixed_rounding = normalize_pseudo_mse3_fixed_rounding(
            getattr(args, 'pseudo_mse3_fixed_rounding', 'floor')
        )
        tie_break = normalize_pseudo_mse3_tie_break(
            getattr(args, 'pseudo_mse3_tie_break', 'exp1')
        )
        chunk_size = int(getattr(args, 'chunk_size', 128) or 128)
        return (
            f"dyn_input_pseudo_mse3_{exponent_policy}_btt{bits_to_take}_"
            f"{fixed_rounding}_{tie_break}_c{chunk_size}"
        )
    raise ValueError(f"Unsupported dynamic metric label: {metric!r}")


def _build_dynamic_input_quant_cfg(args, metric, candidates, model_name):
    """Build the runtime selector config, including pseudo-MSE3 hardware controls."""
    cfg = {
        'enabled': True,
        'mode': 'dynamic',
        'transport': 'encoded',
        'metric': metric,
        'chunk_size': args.chunk_size,
        'candidate_formats': list(candidates),
        'restrict_post_relu_ufp': args.post_relu_ufp_only,
        'unsigned_input_sources': args.unsigned_input_sources,
        'dynamic_unsigned_input_candidates': args.dynamic_unsigned_input_candidates,
        'use_cache_sim_db': args.use_cache_sim_db,
        'model_name': model_name,
        'activation_exponents': args.activation_exponents,
    }
    if metric == 'pseudo_mse3':
        cfg.update({
            'metric_param': int(args.bits_to_take),
            'pseudo_mse3_fixed_rounding': args.pseudo_mse3_fixed_rounding,
            'pseudo_mse3_tie_break': args.pseudo_mse3_tie_break,
        })
    return cfg


def _validate_cache_sim_candidate_groups(candidate_groups, use_cache_sim_db):
    incompatible_widths = [
        bit_width for bit_width in candidate_groups if int(bit_width) != 8
    ]
    if use_cache_sim_db and incompatible_widths:
        raise ValueError(
            "--use_cache_sim_db assigns FP8 candidates to on-chip activations and "
            "cannot produce isolated non-8-bit dynamic runs. Remove that option or "
            "provide only 8-bit --candidate_formats."
        )

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
            'quantize_first_layer': args.fold_input_norm if hasattr(args, 'fold_input_norm') else quantize_first_layer,
            'fold_input_norm': args.fold_input_norm if hasattr(args, 'fold_input_norm') else True,
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
    unsigned_input_sources = cfg.get('quantization', {}).get('unsigned_input_sources', [])
    if isinstance(unsigned_input_sources, str):
        unsigned_input_sources = [s.strip() for s in unsigned_input_sources.split(',') if s.strip()]

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
                'unsigned_input_sources': list(unsigned_input_sources or [])
            }

    cfg['experiment'] = {
        'type': experiment_type,
        'activation_dt': activation_dt,
        'metric': metric,
    }
    return json.dumps(cfg)

def get_args(argv=None):
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
    parser.add_argument(
        "--metric",
        type=str,
        default="mse",
        help="Comma-separated dynamic selection metrics: mse, l1, or pseudo_mse3.",
    )
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for input quantization (blocks)")
    parser.add_argument(
        "--activation-exponents",
        "--activation_exponents",
        choices=_ACTIVATION_EXPONENT_POLICIES,
        default="all",
        help=(
            "Activation candidate exponent policy. pseudo_mse3 automatically uses "
            "e1e2; all preserves the complete candidate set for MSE and L1."
        ),
    )
    parser.add_argument(
        "--bits-to-take",
        "--bits_to_take",
        dest="bits_to_take",
        type=int,
        default=0,
        help=(
            "pseudo_mse3 fixed-point difference bits. 0 keeps the exact "
            "floating-point squared-error difference."
        ),
    )
    parser.add_argument(
        "--pseudo-mse3-fixed-rounding",
        "--pseudo_mse3_fixed_rounding",
        "--fixed-rounding",
        dest="pseudo_mse3_fixed_rounding",
        type=normalize_pseudo_mse3_fixed_rounding,
        choices=("floor", "nearest"),
        default="floor",
        help="pseudo_mse3 fixed-point conversion policy.",
    )
    parser.add_argument(
        "--pseudo-mse3-tie-break",
        "--pseudo_mse3_tie_break",
        "--tie-break",
        dest="pseudo_mse3_tie_break",
        type=normalize_pseudo_mse3_tie_break,
        choices=("exp1", "exp2"),
        default="exp1",
        help="pseudo_mse3 exact chunk-sum tie policy.",
    )
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
        default="",
        help="Comma-separated op names to exclude from quantization (default: none)"
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
        "--skip_input_error_stats",
        action="store_true",
        help=(
            "Skip expensive per-layer input quantization error reductions for "
            "uniform baseline runs. Accuracy and layer format counts are still logged."
        ),
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
            "Base experiment type name for dynamic runs; each bit width is stored as "
            "<name>_<bits>. Defaults to input_quant_dynamic, "
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
    parser.add_argument(
        "--use_cache_sim_db",
        action="store_true",
        help=(
            "Fetch cache simulation results from the database instead of a file. "
            "This is compatible only with an 8-bit dynamic candidate pool."
        ),
    )
    parser.add_argument("--fold_input_norm", action="store_true", default=True,
                        help="Fold input normalization into first layer weights and quantize first layer")
    parser.add_argument("--no_fold_input_norm", action="store_false", dest="fold_input_norm",
                        help="Disable input normalization folding and first layer quantization")
    # Add other args as needed
    args = parser.parse_args(argv)
    args.excluded_ops = [op.strip() for op in args.excluded_ops.split(',') if op.strip()]
    args.baseline_formats = _parse_csv_arg(args.baseline_formats, baseline_formats)
    args.candidate_formats = _parse_csv_arg(args.candidate_formats, candidate_formats)
    try:
        args.metrics = _normalize_requested_metrics(args.metric)
    except (ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
    if args.bits_to_take < 0:
        parser.error("--bits-to-take must be non-negative")
    if 'pseudo_mse3' in args.metrics and args.activation_exponents != 'e1e2':
        print("[pseudo_mse3] forcing --activation-exponents e1e2")
        args.activation_exponents = 'e1e2'
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
    matches = runs[
        (runs['model_name']      == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['weight_dt']       == 'fp32') &
        (runs['activation_dt']   == activation_dt) &
        (runs['status']          == 'SUCCESS')
    ]
    if str(activation_dt).strip().lower() != 'fp32':
        matches = matches[
            matches.apply(row_uses_encoded_activation_transport, axis=1)
        ]
    return not matches.empty


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
            baseline_input_quant_cfg = _build_uniform_input_quant_cfg(
                fmt,
                args.chunk_size,
                unsigned_input_sources=args.unsigned_input_sources,
                use_unsigned_input_candidates=args.dynamic_unsigned_input_candidates,
                collect_error_stats=not args.skip_input_error_stats,
            )
            if baseline_input_quant_cfg is not None:
                config.setdefault('evaluation', {})['input_quant'] = copy.deepcopy(
                    baseline_input_quant_cfg
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
                input_quant_cfg=baseline_input_quant_cfg,
            )
            acc1 = eval_results.get('acc1', 0.0)
            acc5 = eval_results.get('acc5', 0.0)
            certainty = eval_results.get('certainty', 0.0)
            input_stats = eval_results.get('input_quant', {}) if fmt != 'fp32' else {}
            norm_mse = float(input_stats.get('norm_mse', 0.0) or 0.0)

            final_stats[fmt] = {
                'acc1': acc1,
                'acc5': acc5,
                'certainty': certainty,
                'norm_mse': norm_mse,
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
                f"Certainty={certainty:.4f}, NormMSE={norm_mse:.4e}"
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
                            'norm_mse': float(r['mse'] or 0),
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
                            'norm_mse': stats.get('norm_mse', 0.0),
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
                    'errors': {'norm_mse': stats['norm_mse']}
                })
                continue
                
            all_results.append({
                'output_name': f"Base_{fmt}", 
                'acc1': stats['acc1'],
                'acc5': stats['acc5'],
                'errors': {
                    'norm_mse': stats['norm_mse']
                }
            })
    else:
        print("\nSkipping Baselines (--only_dynamic set)")
        ref_acc1, ref_acc5, ref_certainty = None, None, None
        
    
    # --- 2. Run Dynamic Optimization Loop ---
    
    candidate_groups = {}
    if not args.only_baselines:
        candidate_groups, skipped_widths = _prepare_dynamic_candidate_groups(
            args.candidate_formats,
            args.activation_exponents,
            metrics,
        )
        if skipped_widths:
            print(
                "[pseudo_mse3] Skipping bit width(s) without a complete signed "
                f"e1/e2 pair: {skipped_widths}"
            )
        _validate_cache_sim_candidate_groups(
            candidate_groups,
            args.use_cache_sim_db,
        )

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
            
            for bit_width, bit_width_candidates in candidate_groups.items():
                dynamic_experiment_type = _dynamic_experiment_type_for_bit_width(
                    args.dynamic_experiment_type,
                    bit_width,
                )
                for metric in metrics:
                    activation_dt = _dynamic_activation_dt(metric, args)
                    if not args.force_rerun and _input_quant_run_exists(
                        db,
                        model_name,
                        dynamic_experiment_type,
                        activation_dt,
                    ):
                        print(
                            f"[Dynamic {bit_width}-bit] Skipping {metric} — already in DB "
                            f"for {model_name} (experiment_type={dynamic_experiment_type})"
                        )
                        continue

                    print(f"\n===========================================")
                    print(
                        f"Processing {bit_width}-bit Metric: {metric.upper()} "
                        f"for {model_name}"
                    )
                    print(f"Candidates: {bit_width_candidates}")
                    print(f"=============================================")

                    metric_out_dir = os.path.join(
                        model_out_dir,
                        dynamic_experiment_type,
                        activation_dt.removeprefix('dyn_input_'),
                    )
                    os.makedirs(metric_out_dir, exist_ok=True)

                    try:
                        config.setdefault('evaluation', {})
                        config['evaluation']['dynamic_input_quant'] = (
                            _build_dynamic_input_quant_cfg(
                                args,
                                metric,
                                bit_width_candidates,
                                model_name,
                            )
                        )
                        eval_results = runner.evaluate_model(
                            model=model,
                            data_loader=loader,
                            adapter=adapter,
                            max_batches=args.limit_batches,
                            desc=f"Dynamic ({model_name}/{bit_width}-bit/{metric})",
                            dynamic_input_quant_cfg=config['evaluation']['dynamic_input_quant']
                        )
                        acc1 = eval_results.get('acc1', 0.0)
                        acc5 = eval_results.get('acc5', 0.0)
                        certainty = eval_results.get('certainty', 0.0)

                        dyn_stats = eval_results.get('dynamic_input_quant', {})
                        layer_stats = dyn_stats.get('layer_stats', {})
                        final_stats = {
                            'norm_mse': dyn_stats.get('norm_mse', 0.0),
                        }

                        print(
                            f"\nDynamic {bit_width}-bit Run ({metric.upper()}): "
                            f"Top1={acc1:.2f}%, Top5={acc5:.2f}%, "
                            f"Certainty={certainty:.4f}"
                        )
                        print(f"Norm MSE: {final_stats['norm_mse']:.4e}")

                        # Log Dynamic Result to Database using the actual runtime config.
                        _cfg_dyn = _serialize_runtime_config(
                            config,
                            model=model,
                            experiment_type=dynamic_experiment_type,
                            activation_dt=activation_dt,
                            metric=metric,
                            limit_batches=args.limit_batches,
                        )
                        log_cfg = copy.deepcopy(config)
                        log_cfg['experiment'] = {
                            'name': 'find_optimal_input_quant',
                            'type': dynamic_experiment_type,
                            'weight_dt': 'fp32',
                            'activation_dt': activation_dt,
                            'ref_acc1': ref_acc1,
                            'ref_acc5': ref_acc5,
                            'ref_certainty': ref_certainty,
                            'metrics': {
                                'mse': final_stats['norm_mse'],
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
                                'norm_mse': final_stats['norm_mse']
                            }
                            json.dump(save_data, f, indent=4)

                    except KeyboardInterrupt:
                        print("\nInterrupted.")
                        return
                    except Exception as e:
                        print(
                            f"Error processing {model_name} / {bit_width}-bit / {metric}: {e}"
                        )
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
    
    metrics = list(args.metrics)
    print(f"Metrics to process: {metrics}")
    print(f"Activation exponent policy: {args.activation_exponents}")
    if 'pseudo_mse3' in metrics:
        print(
            "pseudo_mse3 controls: "
            f"bits_to_take={args.bits_to_take}, "
            f"fixed_rounding={args.pseudo_mse3_fixed_rounding}, "
            f"tie_break={args.pseudo_mse3_tie_break}"
        )
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
