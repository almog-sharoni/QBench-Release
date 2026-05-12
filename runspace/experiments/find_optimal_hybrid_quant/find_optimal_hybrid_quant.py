
import os
import sys
import gc
import json
import yaml
import argparse
import torch
import copy
import types
import csv
from tqdm import tqdm

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.experiments.utils.common import (
    build_dynamic_input_quant_cfg as _build_dynamic_input_quant_cfg,
    build_loader as _build_loader,
    build_runtime_config as _base_runtime_config,
    build_uniform_input_quant_cfg as _build_uniform_input_quant_cfg,
    build_weight_map_json as _build_weight_map_json,
    get_or_run_fp32_ref as _get_or_run_fp32_ref_common,
    layer_types_from_model as _layer_types_from_model,
    load_fp32_model as _load_fp32_model,
    run_inference as _run_inference,
)

from runspace.experiments.find_optimal_weight_quant.find_optimal_weight_quant import (
    get_quantized_tensor_sim,
    create_quantized_state_dict,
    run_weight_quantization_analysis,
    baseline_formats as weight_baseline_formats,
)

from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (
    candidate_formats as input_candidate_formats,
)

HYBRID_WEIGHT_FORMATS = list(weight_baseline_formats)
HYBRID_INPUT_CANDIDATE_FORMATS = list(input_candidate_formats)
DEFAULT_HYBRID_EXPERIMENT_TYPE = "hybrid_quant_optimal"


def _parse_csv_arg(value, fallback):
    if value is None:
        return list(fallback)
    parsed = [item.strip() for item in str(value).split(',') if item.strip()]
    return parsed if parsed else list(fallback)


def get_args():
    parser = argparse.ArgumentParser(
        description="Hybrid experiment: configure specific weight and input quantizations directly."
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

    # Weight quantization
    parser.add_argument("--weight_mode", type=str, choices=["fixed", "optimized"], required=True,
                        help="Mode for weight quantization.")
    parser.add_argument("--weight_format", type=str, default="fp8_e4m3",
                        help="Fixed weight format (e.g. 'fp8_e4m3') if weight_mode is 'fixed'.")
    parser.add_argument("--weight_metric", type=str, default="mse",
                        help="Metric (e.g. 'mse', 'l1') if weight_mode is 'optimized'.")
    parser.add_argument("--weight_candidate_formats", type=str, default=None,
                        help="Comma-separated candidates for optimized weights.")
    parser.add_argument("--weight_chunk_size", type=int, default=128,
                        help="Chunk size for weight quantization blocks")
    parser.add_argument("--weight_act_batches", type=int, default=10,
                        help="Calibration batches for activation-aware weight metrics such as act_mse")
    parser.add_argument("--per_chunk_format", action="store_true",
                        help="Enable per-chunk format for weights in optimized mode")
    parser.add_argument("--force_recalc", action="store_true",
                        help="Force recalculation of weight errors even if cached (for optimized mode)")
    parser.add_argument("--skip_weight_analysis", action="store_true",
                        help="Skip weight analysis and load cached quantized weights (for optimized mode)")

    # Input quantization
    parser.add_argument("--input_mode", type=str, choices=["fixed", "dynamic"], required=True,
                        help="Mode for input quantization.")
    parser.add_argument("--input_format", type=str, default="fp8_e4m3",
                        help="Fixed input format (e.g. 'fp8_e4m3') if input_mode is 'fixed'.")
    parser.add_argument("--input_metric", type=str, default="mse",
                        help="Metric if input_mode is 'dynamic'. Only 'mse' is supported.")
    parser.add_argument("--input_candidate_formats", type=str, default=None,
                        help="Comma-separated input candidate formats for dynamic input selection.")
    parser.add_argument("--input_chunk_size", type=int, default=128,
                        help="Chunk size for dynamic input quantization")
    parser.add_argument("--use_cache_sim_db", action="store_true",
                        help="Use cache simulation results from DB for residency-aware quantization")
    parser.add_argument("--unsigned_input_sources", type=str, default=None,
                        help="Comma-separated list of ops whose output is always unsigned (e.g. 'relu,softmax')")
    parser.add_argument("--dynamic_unsigned_input_candidates", action="store_true", default=True,
                        help="Allow using unsigned formats (UFP) for layers with unsigned inputs")
    parser.add_argument("--no_dynamic_unsigned_input_candidates", action="store_false", dest="dynamic_unsigned_input_candidates",
                        help="Disable using unsigned formats (UFP) for layers with unsigned inputs")
    parser.add_argument("--skip_depthwise_input_quant", action="store_true",
                        help="Ablation: leave depthwise Conv2d inputs in FP32 while keeping other dynamic input quantization enabled")
    parser.add_argument("--fold_input_norm", action="store_true", default=True,
                        help="Fold input normalization into first layer weights and quantize first layer")
    parser.add_argument("--no_fold_input_norm", action="store_false", dest="fold_input_norm",
                        help="Disable input normalization folding and first layer quantization")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--experiment_type", type=str, default=DEFAULT_HYBRID_EXPERIMENT_TYPE,
                        help="Experiment type label for database logging")

    args = parser.parse_args()
    args.weight_candidate_formats = _parse_csv_arg(args.weight_candidate_formats, HYBRID_WEIGHT_FORMATS)
    args.input_candidate_formats = _parse_csv_arg(args.input_candidate_formats, HYBRID_INPUT_CANDIDATE_FORMATS)
    # Unsigned sources
    args.unsigned_input_sources = _parse_csv_arg(args.unsigned_input_sources, [])
    
    return args


def _make_weight_args(args):
    """Build a namespace compatible with weight-quant helpers."""
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
    wa.baseline_formats = ','.join(args.weight_candidate_formats)
    wa.metrics = args.weight_metric
    return wa


def run_weight_phase(runner, args, device, model_dir, base_config):
    """
    Analyse weights and build per-layer/chunk optimal quantized state dict.
    Returns: (quantized_weights_path, quant_map, layer_types)
    """
    m = args.weight_metric
    wa = _make_weight_args(args)

    analysis_dir = os.path.join(model_dir, "weight_phase_fp32")
    model, adapter, _ = runner.prepare_model_with_materialized_weights(
        config=base_config,
        output_dir=analysis_dir,
    )

    qt_options = list(args.weight_candidate_formats)
    supported_ops = tuple(OpRegistry.get_supported_ops().keys())
    layer_results_map = {}

    if not args.skip_weight_analysis:
        print(f"\n[Weight Phase] Analysing weight tensors for metric: {m} ...")
        run_weight_quantization_analysis(
            wa, model, [m], qt_options, layer_results_map, supported_ops
        )
    else:
        print("\n[Weight Phase] --skip_weight_analysis set; loading cached quant maps ...")
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
                            if fmt not in qt_options:
                                continue
                            try:
                                errs[fmt] = float(v) if v else float('inf')
                            except ValueError:
                                errs[fmt] = float('inf')
                    layer_results_map[lname]['metrics'][m] = errs
        else:
            print(f"  WARNING: no cached CSV at {csv_path} for metric {m}")

    metric_dir = os.path.join(model_dir, f"weights_{m}")
    os.makedirs(metric_dir, exist_ok=True)

    q_state_dict, q_map = create_quantized_state_dict(
        model, layer_results_map, wa, m, use_chunking=args.per_chunk_format
    )
    q_state_dict = _materialize_weight_buffers_from_map(model, q_state_dict, q_map, args)
    q_path = os.path.join(metric_dir, "quantized_weights.pt")
    torch.save(q_state_dict, q_path)

    with open(os.path.join(metric_dir, "quantization_map.json"), 'w') as f:
        json.dump(q_map, f, indent=4)

    print(f"[Weight Phase] Saved optimized quantized weights ({m}) → {q_path}")

    layer_types = {
        name: type(module).__name__
        for name, module in model.named_modules()
        if name
    }

    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()

    return q_path, q_map, layer_types


def _iter_weight_modules(model):
    supported = (torch.nn.Conv2d, torch.nn.Linear)
    for name, module in model.named_modules():
        if name and isinstance(module, supported) and getattr(module, 'weight', None) is not None:
            yield name, module


def _disable_runtime_io_quant(model):
    for module in model.modules():
        if hasattr(module, 'input_quantization'):
            module.input_quantization = False
        if hasattr(module, 'output_quantization'):
            module.output_quantization = False
        if hasattr(module, 'weight_fp8'):
            module.weight_fp8 = None
        if hasattr(module, 'weight_scale'):
            module.weight_scale = None
        if hasattr(module, 'weight_scale_packed'):
            module.weight_scale_packed = None


def _activation_weight_error_for_module(module, inputs, ref_output, quantized_weights):
    import torch.nn.functional as F

    x = inputs[0] if isinstance(inputs, tuple) else inputs
    if not isinstance(x, torch.Tensor) or not isinstance(ref_output, torch.Tensor):
        return {}

    errors = {}
    with torch.no_grad():
        for fmt, q_weight in quantized_weights.items():
            try:
                if isinstance(module, torch.nn.Conv2d):
                    q_out = F.conv2d(
                        x,
                        q_weight,
                        module.bias,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                    )
                elif isinstance(module, torch.nn.Linear):
                    q_out = F.linear(x, q_weight, module.bias)
                else:
                    continue
                diff = ref_output - q_out
                errors[fmt] = (diff.pow(2).sum().detach(), diff.numel())
            except Exception:
                errors[fmt] = (torch.tensor(float('inf'), device=x.device), 1)
    return errors


def _materialize_weight_buffers_from_map(model, q_state_dict, q_map, args):
    """
    Keep saved `weight`, `weight_fp8`, and `weight_scale` mutually consistent.

    The evaluation path prefers weight_fp8/weight_scale when present. A state dict
    with quantized `.weight` tensors but stale FP8 buffers can therefore evaluate a
    different model than the one described by the quantization map.
    """
    modules = dict(model.named_modules())
    model.load_state_dict(q_state_dict, strict=False)
    for layer_name, selected_format in q_map.items():
        module = modules.get(layer_name)
        if module is None or not hasattr(module, 'calibrate_weights'):
            continue
        module.weight_quantization = True
        module.weight_chunk_size = args.weight_chunk_size
        if isinstance(selected_format, list):
            module.chunk_formats = selected_format
            module.weight_mode = 'chunk'
            module.q_type = selected_format[0] if selected_format else getattr(module, 'q_type', 'fp8_e1m6')
        else:
            module.chunk_formats = None
            module.weight_mode = 'channel'
            module.q_type = selected_format
        module.calibrate_weights()
    return model.state_dict()


def run_activation_weight_phase(runner, args, device, model_dir, base_config):
    """
    Activation-aware layer-wise weight selection.

    For each Conv2d/Linear-like quantized module, run a small calibration window
    through the FP32-weight model, and choose the weight format that minimizes the
    module's local output MSE for the observed inputs. This intentionally ignores
    per-chunk format selection; it answers whether layer sensitivity, not raw
    weight reconstruction, is driving the MobileNetV3 drop.
    """
    metric = args.weight_metric
    analysis_dir = os.path.join(model_dir, f"weight_phase_{metric}")
    model, adapter, _ = runner.prepare_model_with_materialized_weights(
        config=base_config,
        output_dir=analysis_dir,
    )
    model.eval()
    _disable_runtime_io_quant(model)

    loader = _build_loader(args, device, runner, config_builder=lambda _: base_config)
    calib_batches = int(args.weight_act_batches if args.weight_act_batches > 0 else 10)
    if args.limit_batches and args.limit_batches > 0:
        calib_batches = min(calib_batches, int(args.limit_batches))

    qt_options = [fmt for fmt in args.weight_candidate_formats if str(fmt).lower() != 'fp32']
    layer_results_map = {}

    print(
        f"\n[Weight Phase] Activation-aware weight search ({metric}) "
        f"over {calib_batches} calibration batches ..."
    )

    for layer_name, module in tqdm(list(_iter_weight_modules(model)), desc="Analyzing Layers (ActMSE)"):
        q_weights = {}
        for fmt in qt_options:
            try:
                q_w, _ = get_quantized_tensor_sim(
                    module.weight.detach(),
                    fmt,
                    chunk_size=args.weight_chunk_size,
                )
                q_weights[fmt] = q_w.detach()
            except Exception:
                pass
        if not q_weights:
            continue

        sum_err = {fmt: 0.0 for fmt in q_weights}
        sum_numel = {fmt: 0 for fmt in q_weights}

        def hook_fn(mod, inputs, output):
            batch_errors = _activation_weight_error_for_module(mod, inputs, output, q_weights)
            for fmt, (err_sum, numel) in batch_errors.items():
                sum_err[fmt] += float(err_sum.item())
                sum_numel[fmt] += int(numel)

        handle = module.register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= calib_batches:
                        break
                    inputs, targets = adapter.prepare_batch(batch)
                    inputs = runner._to_device(inputs)
                    targets = runner._to_device(targets)
                    adapter.forward(model, (inputs, targets))
        finally:
            handle.remove()

        metrics = {
            fmt: (sum_err[fmt] / sum_numel[fmt] if sum_numel[fmt] > 0 else float('inf'))
            for fmt in q_weights
        }
        best_fmt = min(metrics, key=metrics.get)
        layer_results_map[layer_name] = {
            'layer': layer_name,
            'shape': tuple(module.weight.shape),
            'max_val': float(module.weight.detach().abs().max().item()),
            'numel': int(module.weight.numel()),
            'metrics': {metric: metrics},
            'best_error': metrics[best_fmt],
        }

    metric_dir = os.path.join(model_dir, f"weights_{metric}")
    os.makedirs(metric_dir, exist_ok=True)
    q_state_dict, q_map = create_quantized_state_dict(
        model, layer_results_map, args, metric, use_chunking=False
    )
    q_state_dict = _materialize_weight_buffers_from_map(model, q_state_dict, q_map, args)
    q_path = os.path.join(metric_dir, "quantized_weights.pt")
    torch.save(q_state_dict, q_path)
    with open(os.path.join(metric_dir, "quantization_map.json"), 'w') as f:
        json.dump(q_map, f, indent=4)

    print(f"[Weight Phase] Saved activation-aware quantized weights ({metric}) → {q_path}")

    layer_types = _layer_types_from_model(model)
    del model, adapter, loader
    gc.collect()
    torch.cuda.empty_cache()
    return q_path, q_map, layer_types


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


def _summarise_quant_map(quant_map, prefix="opt"):
    if not quant_map:
        return f"{prefix}_unknown"
    counts = {}
    for v in quant_map.values():
        key = str(v) if not isinstance(v, list) else "per_chunk"
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{fmt}x{cnt}" for fmt, cnt in sorted(counts.items(), key=lambda x: -x[1])]
    return prefix + "[" + ",".join(parts[:5]) + "]"


def _log_hybrid_run(
    runner,
    base_config,
    model_name,
    weight_dt,
    activation_dt,
    acc1,
    acc5,
    status,
    experiment_name='hybrid_quant',
    ref_acc1=None,
    ref_acc5=None,
    ref_certainty=None,
    certainty=None,
    mse=None,
    quant_map_json=None,
    input_map_json=None,
    input_quant_stats=None,
):
    cfg = copy.deepcopy(base_config)
    cfg['experiment'] = {
        'name': experiment_name,
        'type': runner.args.experiment_type if hasattr(runner, 'args') and hasattr(runner.args, 'experiment_type') else 'hybrid_quant',
        'weight_dt': weight_dt,
        'activation_dt': activation_dt,
        'ref_acc1': ref_acc1,
        'ref_acc5': ref_acc5,
        'ref_certainty': ref_certainty,
        'metrics': {
            'mse': mse,
            'certainty': certainty,
        },
        'quant_map_json': quant_map_json,
        'input_map_json': input_map_json,
    }
    result = {
        'model_name': model_name,
        'status': status,
        'acc1': acc1,
        'acc5': acc5,
        'certainty': certainty if certainty is not None else 0.0,
    }
    if input_quant_stats is not None:
        result['input_quant'] = input_quant_stats
    runner.log_experiment_result(cfg, result)


def process_single_model(args, device):
    args.input_metric = "mse"
    model_name = args.model_name
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    base_config = _base_runtime_config(args, model_name=model_name, weights=args.weights)

    # We must explicitly turn ON quantized_ops and input_quantization in the base config
    # so the adapter actually builds Quant layers (including functional replacements like QuantAdd)
    # AND decomposes complex layers like MHA into q_proj/k_proj/v_proj so the weight shapes match.
    base_config.setdefault('adapter', {})
    base_config['adapter']['quantized_ops'] = ['all']
    base_config['adapter']['input_quantization'] = True
    base_config['adapter']['weight_quantization'] = True
    base_config['adapter']['fold_input_norm'] = args.fold_input_norm
    base_config['adapter']['quantize_first_layer'] = args.fold_input_norm

    runner = Runner(device)
    runner.args = args # Store args in runner for logging access
    db = runner._get_db()

    print(f"\n{'='*60}")
    print(f" HYBRID EXPERIMENT: {model_name}")
    print(f"  Weight Mode  : {args.weight_mode}")
    if args.weight_mode == "fixed":
        print(f"  Weight Format: {args.weight_format}")
    else:
        print(f"  Weight Metric: {args.weight_metric}")
        print(f"  Weight Cands : {args.weight_candidate_formats}")

    print(f"  Input Mode   : {args.input_mode}")
    if args.input_mode == "fixed":
        print(f"  Input Format : {args.input_format}")
    else:
        print(f"  Input Metric : {args.input_metric}")
        print(f"  Input Cands  : {args.input_candidate_formats}")
    print(f"{'='*60}")

    # Determine experiment name: hybrid_quant_<w_bits>/<i_bits>[_w_cache_sim]
    # Based on the first (highest) candidate bit-widths.
    w_primary = args.weight_format if args.weight_mode == "fixed" else (args.weight_candidate_formats[0] if args.weight_candidate_formats else "unknown")
    i_primary = args.input_format if args.input_mode == "fixed" else (args.input_candidate_formats[0] if args.input_candidate_formats else "unknown")
    
    def _get_bits(fmt):
        s = str(fmt)
        return s.split('_')[0] if '_' in s else s

    experiment_name = f"hybrid_quant_{_get_bits(w_primary)}/{_get_bits(i_primary)}"
    if args.use_cache_sim_db:
        experiment_name += "_w_cache_sim"
        
        # Check if model exists in cache simulation DB
        sim = db.get_latest_cache_simulation(model_name)
        if sim is None:
            print(f"\n[CacheSim] Model '{model_name}' not found in cache simulation database.")
            print(f"[CacheSim] Triggering automatic cache simulation...")
            try:
                from runspace.experiments.asic_cache_simulation.simulate_cache import run_simulation as run_cache_sim
                # Create dummy args for simulation
                sim_args = types.SimpleNamespace()
                sim_args.model_name = model_name
                sim_args.cache_size = 2.0  # Default from simulate_cache.py
                sim_args.num_banks = 16    # Default
                sim_args.metadata_bits = 0 # Default
                sim_args.batch_size = 1    # Default for residency analysis
                sim_args.device = str(device)
                
                # Execute simulation (this will also upload to DB)
                run_cache_sim(sim_args)
                print(f"[CacheSim] Simulation completed and uploaded to DB.\n")
            except Exception as e:
                print(f"[CacheSim] Error during automatic simulation: {e}")
                import traceback
                traceback.print_exc()

    print(f"  Experiment Name: {experiment_name}")

    ref_acc1, ref_acc5, ref_certainty = _get_or_run_fp32_ref_common(
        runner, args, device, db, model_name, experiment_name=experiment_name
    )

    # Prepare Weights
    if args.weight_mode == "fixed":
        weight_dt_str = args.weight_format
        print(f"\n[Weights] Building uniformly quantized weights ({weight_dt_str}) ...")
        model_fp32, adapter_fp32 = _load_fp32_model(runner, args, device, config_builder=lambda _: base_config)
        q_state_dict, quant_map = _build_uniform_quant_state_dict(
            model_fp32, args.weight_format, chunk_size=args.weight_chunk_size
        )
        layer_types = _layer_types_from_model(model_fp32)
        q_path = os.path.join(model_dir, "fixed_weights.pt")
        torch.save(q_state_dict, q_path)
        del model_fp32, adapter_fp32
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[Weights] Running optimized weight quantization (metric={args.weight_metric}) ...")
        if args.weight_metric == "act_mse":
            q_path, quant_map, layer_types = run_activation_weight_phase(
                runner, args, device, model_dir, base_config
            )
        else:
            q_path, quant_map, layer_types = run_weight_phase(runner, args, device, model_dir, base_config)
        weight_dt_str = _summarise_quant_map(quant_map, prefix=f"opt_w{args.weight_metric}")

    quant_map_json = _build_weight_map_json(quant_map, layer_types)

    # Load Model with Quantized Weights
    model, adapter = runner.load_model_from_weight_file(
        config=base_config,
        weight_file_path=q_path,
        skip_calibration=True
    )

    # Prepare Inputs
    if args.input_mode == "fixed":
        activation_dt_str = args.input_format
        input_quant_cfg = _build_uniform_input_quant_cfg(
            args.input_format,
            args.input_chunk_size,
        )
    else:
        activation_dt_str = f"dyn_input_{args.input_metric}"
        input_quant_cfg = _build_dynamic_input_quant_cfg(
            metric=args.input_metric,
            chunk_size=args.input_chunk_size,
            candidate_formats=args.input_candidate_formats,
            use_cache_sim_db=args.use_cache_sim_db,
            model_name=args.model_name,
            unsigned_input_sources=args.unsigned_input_sources,
            dynamic_unsigned_input_candidates=args.dynamic_unsigned_input_candidates,
            skip_depthwise_input_quant=args.skip_depthwise_input_quant,
        )

    loader = _build_loader(args, device, runner)

    print(f"\n[Inference] Running Hybrid Setup: Weight={weight_dt_str}, Input={activation_dt_str} ...")
    try:
        acc1, acc5, certainty, input_stats = _run_inference(
            runner, model, adapter, loader, args,
            input_quant_cfg=input_quant_cfg,
            desc=f"Hybrid W={args.weight_mode} I={args.input_mode}"
        )
    except Exception as e:
        print(f"  ERROR during inference: {e}")
        import traceback; traceback.print_exc()
        del model, adapter
        gc.collect(); torch.cuda.empty_cache()
        _log_hybrid_run(
            runner=runner, base_config=base_config, model_name=model_name,
            weight_dt=weight_dt_str, activation_dt=activation_dt_str,
            acc1=0.0, acc5=0.0, status="ERROR",
            experiment_name=experiment_name,
            ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty
        )
        return

    norm_mse = input_stats['norm_mse'] if input_stats else None

    print(
        f"  Result: Top1={acc1:.2f}%, Top5={acc5:.2f}%, Certainty={certainty:.4f}"
        + (f", NormMSE={norm_mse:.4e}" if norm_mse is not None else "")
    )

    _log_hybrid_run(
        runner=runner,
        base_config=base_config,
        model_name=model_name,
        weight_dt=weight_dt_str,
        activation_dt=activation_dt_str,
        acc1=acc1,
        acc5=acc5,
        status="SUCCESS",
        experiment_name=experiment_name,
        ref_acc1=ref_acc1,
        ref_acc5=ref_acc5,
        ref_certainty=ref_certainty,
        certainty=certainty,
        mse=norm_mse,
        quant_map_json=quant_map_json,
        input_quant_stats=input_stats,
    )

    # Log layer stats (contains chunk win rates for dynamic inputs)
    run_label = f"hybrid_{args.weight_mode}_{args.input_mode}"
    out_dir = os.path.join(model_dir, run_label)
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "layer_stats.json")
    with open(stats_path, 'w') as f:
        save_data = dict(input_stats.get('layer_stats', {}) if input_stats else {})
        
        # Merge weight format counts into the layer stats for visibility
        for layer_name, spec in quant_map.items():
            if layer_name not in save_data:
                save_data[layer_name] = {}
            if isinstance(spec, list):
                counts = {}
                for fmt in spec:
                    counts[str(fmt)] = counts.get(str(fmt), 0) + 1
                save_data[layer_name]['weight_format_counts'] = counts
                save_data[layer_name]['weight_total_chunks'] = len(spec)
                if counts:
                    save_data[layer_name]['weight_format'] = sorted(counts.items(), key=lambda x: -x[1])[0][0]
            else:
                save_data[layer_name]['weight_format'] = str(spec)
                save_data[layer_name]['weight_format_counts'] = {str(spec): 1}
                save_data[layer_name]['weight_total_chunks'] = 1
                
        save_data['accuracy'] = {
            'top1': acc1, 'top5': acc5, 'certainty': certainty,
            'norm_mse': norm_mse,
            'weight_mode': args.weight_mode, 'weight_dt': weight_dt_str,
            'input_mode': args.input_mode, 'input_dt': activation_dt_str,
        }
        json.dump(save_data, f, indent=4)

    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[Done] Hybrid experiment finished for {model_name}.")


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
