import os
import sys

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import math
import argparse
import json
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.registry.op_registry import OpRegistry
from runspace.core.runner import Runner
from runspace.experiments.find_optimal_weight_quant.find_optimal_weight_quant import get_quantized_tensor_sim
from runspace.experiments.utils.common import build_fp32_runtime_config
from runspace.experiments.utils.plotting import get_format_bits
from runspace.src.database.handler import RunDatabase

from runspace.experiments.asic_cache_simulation.simulate_cache import (
    analyze_model,
    evaluate_stay,
    get_footprint_elements,
    round_to_banks,
    fmt_elems,
    optimize_layer_bits
)

try:
    from runspace.experiments.asic_cache_simulation.rules import RULES
except ImportError:
    RULES = {}

# ============================================================
# DynamicInputQuantizer Monkey-Patch Setup
# ============================================================
# Import from both paths to ensure we monkey-patch both copies of the class
# due to python namespace duplication (runspace.src vs src).
try:
    from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer as DIQ1
    from runspace.src.quantization.dynamic_input_quantizer import DEFAULT_DYNAMIC_INPUT_CANDIDATES
except ImportError:
    DIQ1 = None
    DEFAULT_DYNAMIC_INPUT_CANDIDATES = []

try:
    from src.quantization.dynamic_input_quantizer import DynamicInputQuantizer as DIQ2
    if not DEFAULT_DYNAMIC_INPUT_CANDIDATES:
        from src.quantization.dynamic_input_quantizer import DEFAULT_DYNAMIC_INPUT_CANDIDATES
except ImportError:
    DIQ2 = None

def patched_candidates_for_layer(self, layer_name, module=None, input_index=0):
    """Silent custom candidates picker that uses per-layer input bit-width for off-chip layers."""
    is_unsigned = (
        self.use_unsigned_input_candidates
        and (
            layer_name in self.post_unsigned_layers
            or layer_name in self.unsigned_passthrough_layers
        )
    ) or (
        module is not None
        and self._module_uses_unsigned_input(module)
    )

    if input_index == 1:
        residual_bits = self.layer_residual_input_bits_map.get(layer_name)
        if residual_bits is not None:
            candidates = get_input_formats_for_bits(residual_bits)
            return self._make_unsigned_candidates(candidates) if is_unsigned else candidates

    stays_on_chip = self.cache_sim_map.get(layer_name, True)

    if stays_on_chip:
        return self.unsigned_all_fp8_formats if is_unsigned else self.all_fp8_formats

    input_bits = self.layer_input_bits_map.get(layer_name, 8)
    candidates = get_input_formats_for_bits(input_bits)

    if is_unsigned:
        return self._make_unsigned_candidates(candidates)

    if self.restrict_post_relu_ufp:
        ufp_for_bits = [f for f in candidates if f.startswith('ufp')]
        non_ufp_for_bits = [f for f in candidates if not f.startswith('ufp')]
        if layer_name in self.post_relu_layers:
            return ufp_for_bits or self._make_unsigned_candidates(non_ufp_for_bits)
        return non_ufp_for_bits or ufp_for_bits

    return candidates

for DIQ in (DIQ1, DIQ2):
    if DIQ is not None:
        # Override candidates method
        DIQ._candidates_for_layer = patched_candidates_for_layer
        
        # Override init to load global maps
        _orig_init = DIQ.__init__
        def make_new_init(orig_init):
            def new_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                self.cache_sim_map = getattr(self.__class__, '_global_cache_sim_map', self.cache_sim_map)
                self.layer_input_bits_map = getattr(self.__class__, '_global_layer_input_bits_map', {})
                self.layer_residual_input_bits_map = getattr(
                    self.__class__,
                    '_global_layer_residual_input_bits_map',
                    getattr(self, 'layer_residual_input_bits_map', {}),
                )
            return new_init
        DIQ.__init__ = make_new_init(_orig_init)


# ============================================================
# Signed Formats By Bit Width
# ============================================================
SIGNED_FORMATS_BY_BITS = {
    8: ['fp8_e4m3', 'fp8_e5m2', 'fp8_e3m4', 'fp8_e2m5', 'fp8_e1m6', 'fp8_e6m1', 'fp8_e7m0'],
    7: ['fp7_e1m5', 'fp7_e2m4', 'fp7_e3m3', 'fp7_e4m2', 'fp7_e5m1', 'fp7_e6m0'],
    6: ['fp6_e1m4', 'fp6_e2m3', 'fp6_e3m2', 'fp6_e4m1', 'fp6_e5m0'],
    5: ['fp5_e1m3', 'fp5_e2m2', 'fp5_e3m1', 'fp5_e4m0'],
    4: ['fp4_e1m2', 'fp4_e2m1', 'fp4_e3m0'],
    3: ['fp3_e1m1', 'fp3_e2m0'],
    2: ['fp2_e1m0']
}

# ============================================================
# Helpers
# ============================================================

def get_input_formats_for_bits(b):
    """Filter input formats to retrieve candidates with exactly b bits."""
    candidates = []
    for fmt in DEFAULT_DYNAMIC_INPUT_CANDIDATES:
        if get_format_bits(fmt) == b:
            candidates.append(fmt)
    # Ensure signed standard formats of b bits are also included
    if b in SIGNED_FORMATS_BY_BITS:
        for fmt in SIGNED_FORMATS_BY_BITS[b]:
            if fmt not in candidates:
                candidates.append(fmt)
    return candidates


def get_best_weight_format(weight_tensor, formats, chunk_size=128):
    """Find the best quantization format for a weight tensor by minimizing MSE error."""
    if not formats:
        return None
    best_fmt = None
    best_err = float('inf')
    for fmt in formats:
        try:
            w_deq, _ = get_quantized_tensor_sim(weight_tensor, fmt, chunk_size=chunk_size)
            err = (weight_tensor - w_deq).pow(2).mean().item()
            if err < best_err:
                best_err = err
                best_fmt = fmt
        except Exception:
            pass
    return best_fmt if best_fmt else formats[0]


def build_bandwidth_fp32_config(args, model_name, weights):
    """Build the model-load config used to materialize the FP32 reference.

    The adapter is configured to mirror the standard hybrid_quant pipeline so the
    module graph matches the eval pipeline exactly: Quant wrappers everywhere,
    decomposed MHA into q_proj/k_proj/v_proj, folded input normalization, first
    layer wrapped. Quantization is left OFF at the gate level
    (input_quantization=False, weight_quantization=False) so weights stay FP32 —
    they are later calibrated per-layer in `create_bandwidth_aware_state_dict`.
    """
    config = build_fp32_runtime_config(args, model_name=model_name, weights=weights)
    adapter_cfg = config.setdefault('adapter', {})
    adapter_cfg.update({
        'type': 'generic',
        'quantized_ops': ['all'],
        'build_quantized': True,
        'input_quantization': False,
        'weight_quantization': False,
        'fold_input_norm': True,
        'quantize_first_layer': True,
    })
    config.setdefault('evaluation', {})
    config['evaluation'].update({
        'mode': 'evaluate',
        'max_batches': args.limit_batches,
    })
    return config


def get_cached_fp32_acc1(model_name):
    """Return cached FP32 top-1 accuracy from the run DB, or None if unavailable."""
    try:
        ref_metrics = RunDatabase().get_reference_metrics(model_name)
    except Exception as exc:
        print(f"[FP32 ref] Could not read cached reference from DB: {exc}")
        return None

    if not ref_metrics:
        return None

    ref_acc1 = float(ref_metrics[0] or 0.0)
    if ref_acc1 <= 0.0:
        return None

    print(f"[FP32 ref] Using cached DB reference: Top-1 Acc = {ref_acc1:.3f}%")
    return ref_acc1


def run_cache_simulation(model_name, cache_size_M, batch_size=1, num_banks=16, metadata_bits=0, device="cuda"):
    """Run cache simulation to classify layers as on-chip or off-chip."""
    cache_elements = int(cache_size_M * 1_000_000)
    bank_size = cache_elements // num_banks if num_banks > 0 else 0
    
    # 1. Analyze model
    layers = analyze_model(model_name, batch_size=batch_size, device=device,
                           cache_elements=cache_elements, bank_size=bank_size, metadata_bits=metadata_bits)
    
    # 2. Evaluate stay status
    results = []
    cache_sim_map = {}
    for i, layer in enumerate(layers):
        next_layer = layers[i + 1] if i + 1 < len(layers) else None
        
        weight_elems = get_footprint_elements(layer['weight_elems'], metadata_bits)
        input_elems  = get_footprint_elements(layer['input_elems'],  metadata_bits)
        output_elems = get_footprint_elements(layer['output_elems'], metadata_bits)
        
        next_xin_elems = (
            get_footprint_elements(next_layer['input_elems'], metadata_bits)
            if next_layer else 0
        )
        
        output_banked   = round_to_banks(output_elems,   bank_size)
        input_banked    = round_to_banks(input_elems,    bank_size)
        weight_banked   = round_to_banks(weight_elems,   bank_size)
        next_xin_banked = round_to_banks(next_xin_elems, bank_size)
        
        ctx = {
            'input_banked':    input_banked,
            'output_banked':   output_banked,
            'weight_banked':   weight_banked,
            'next_xin_banked': next_xin_banked,
            'cache_elements':  cache_elements,
            'bank_size':       bank_size,
            'filter_height':   layer.get('filter_height', 0),
            'filter_width':    layer.get('filter_width', 0),
            'in_channels':     layer.get('in_channels', 0),
            'out_channels':    layer.get('out_channels', 0),
            'input_channel_height':  layer.get('input_channel_height', 0),
            'input_channel_width':   layer.get('input_channel_width', 0),
            'output_channel_height': layer.get('output_channel_height', 0),
            'output_channel_width':  layer.get('output_channel_width', 0),
            'jump_back_size_in_banks': layer.get('jump_back_size_in_banks', 0),
        }
        
        stay_on_chip, perm_elems, possible, rule_name = evaluate_stay(
            layer, ctx, next_layer, metadata_bits, bank_size, cache_elements
        )
        xin_from_cache = RULES.get(rule_name, {}).get('xin_from_cache', True)
        
        layer_copy = dict(layer)
        layer_copy['stay_on_chip'] = stay_on_chip
        layer_copy['xin_from_cache'] = xin_from_cache
        results.append(layer_copy)
        cache_sim_map[layer['name']] = stay_on_chip
        
    return results, cache_sim_map


def compute_model_runtime(layers_with_stay_status, b_bits, bandwidth=1.0):
    """
    Compute model total runtime (execution cycles) under bandwidth-constrained
    conditions using layer-wide transfer bit-width optimization.

    For each layer, total transfer cycles are checked against compute cycles.
    If the layer is BW-limited, every transferred component for that layer is
    reduced by 1 bit together (floor = b_bits) until the layer is compute-
    limited or all transferred components hit the floor. On-chip layers still
    optimize weights because weights are always streamed in.

    Returns
    -------
    total_runtime : float
    layer_input_bits : dict  layer_name -> input bit-width
    layer_weight_bits : dict layer_name -> weight bit-width
    layer_output_bits : dict layer_name -> output bit-width
    layer_residual_input_bits : dict layer_name -> residual operand bit-width
    """
    total_runtime = 0.0
    prev_stay_on_chip = False
    layer_input_bits = {}
    layer_weight_bits = {}
    layer_output_bits = {}
    layer_residual_input_bits = {}
    
    for idx, layer in enumerate(layers_with_stay_status):
        stay_on_chip = layer.get('stay_on_chip', False)
        lname = layer['name']
        xin_from_cache = layer.get('xin_from_cache', True)
        
        need_input_transfer = (idx == 0 or not prev_stay_on_chip or not xin_from_cache)
        need_weight_transfer = True
        need_output_transfer = not stay_on_chip
        residual_bits = 3
        residual_input_stream_elems = layer.get('residual_input_stream_elems', 0)
        residual_output_elems = layer.get('residual_output_elems', 0)
        fixed_transfers = []
        forced_bits = {}
        residual_output_uses_main_stream = (
            residual_output_elems > 0
            and need_output_transfer
            and residual_output_elems == layer.get('output_elems', 0)
        )

        if residual_output_elems > 0:
            if residual_output_uses_main_stream:
                forced_bits['output'] = residual_bits
            else:
                fixed_transfers.append({
                    'name': 'residual_output',
                    'elems': residual_output_elems,
                    'bits': residual_bits,
                })
        if residual_input_stream_elems > 0:
            fixed_transfers.append({
                'name': 'residual_input',
                'elems': residual_input_stream_elems,
                'bits': residual_bits,
            })
            layer_residual_input_bits[lname] = residual_bits
        
        in_b, w_b, out_b, cycles = optimize_layer_bits(
            layer, bandwidth,
            need_input_transfer, need_weight_transfer, need_output_transfer,
            min_bits=b_bits,
            fixed_transfers=fixed_transfers,
            forced_bits=forced_bits,
        )
        
        layer_input_bits[lname] = in_b
        layer_weight_bits[lname] = w_b
        layer_output_bits[lname] = out_b
        total_runtime += cycles
        
        prev_stay_on_chip = stay_on_chip
        
    return total_runtime, layer_input_bits, layer_weight_bits, layer_output_bits, layer_residual_input_bits


def create_bandwidth_aware_state_dict(model, cache_sim_map, per_layer_weight_bits, chunk_size=128):
    """Calibrate per-layer weights via the Quant layer's own machinery.

    The model is built with `quantized_ops:['all']`, so each compute layer is a
    Quant{Conv2d,Linear} wrapper with `weight_fp8`/`weight_scale` buffers, and
    MHA is already decomposed into separate q_proj/k_proj/v_proj Linear
    sub-modules whose names match the cache_sim_map keys directly.

    For each layer in `cache_sim_map`:
      - pick the best signed format among `SIGNED_FORMATS_BY_BITS[bits]`
      - set the module's q_type and call `calibrate_weights()` to populate
        `weight_fp8` and `weight_scale` consistently — matching the
        find_optimal_hybrid_quant pipeline's `_materialize_weight_buffers_from_map`.
    """
    modules = dict(model.named_modules())
    quant_map = {}

    for layer_name in cache_sim_map:
        module = modules.get(layer_name)
        if module is None or not hasattr(module, 'calibrate_weights'):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        bits = per_layer_weight_bits.get(layer_name, 8)
        formats = SIGNED_FORMATS_BY_BITS.get(bits, [])
        best_fmt = get_best_weight_format(module.weight.detach(), formats, chunk_size=chunk_size)
        if not best_fmt:
            continue

        module.weight_quantization = True
        module.weight_chunk_size = chunk_size
        module.weight_mode = 'channel'
        module.chunk_formats = None
        module.q_type = best_fmt
        module.calibrate_weights()
        quant_map[layer_name] = best_fmt

    return model.state_dict(), quant_map


# ============================================================
# Main Experiment Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Recreation of bandwidth-aware quantization sweeps.")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name or path to a YAML file with a list of models")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for accuracy evaluation")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataset loader")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit number of evaluation batches (-1 = all)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--bandwidth", type=float, default=1.0, help="Memory bandwidth in bytes/cycle")
    args = parser.parse_args()

    # Verify CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is requested but not available. Falling back to cpu.")
        args.device = "cpu"

    # Resolve models list
    models_to_run = []
    if args.model_name.endswith('.yaml') or args.model_name.endswith('.yml'):
        with open(args.model_name, 'r') as f:
            yaml_content = yaml.safe_load(f)
            if isinstance(yaml_content, list):
                for item in yaml_content:
                    if isinstance(item, dict):
                        models_to_run.append((item.get('name'), item.get('weights', 'DEFAULT')))
                    elif isinstance(item, str):
                        models_to_run.append((item, 'DEFAULT'))
            elif isinstance(yaml_content, dict):
                models_to_run.append((yaml_content.get('name'), yaml_content.get('weights', 'DEFAULT')))
    else:
        models_to_run.append((args.model_name, args.weights))

    runner = Runner(device=args.device)

    for model_name, model_weights in models_to_run:
        print(f"\n============================================================")
        print(f"RUNNING BANDWIDTH-AWARE EXPERIMENT FOR MODEL: {model_name}")
        print(f"============================================================")

        # Resolve output directory for this model
        if args.output_dir:
            if len(models_to_run) > 1:
                model_output_dir = os.path.join(args.output_dir, model_name)
            else:
                model_output_dir = args.output_dir
        else:
            model_output_dir = os.path.join(PROJECT_ROOT, "runspace/experiments/bandwidth_aware_quant/results", model_name)

        os.makedirs(model_output_dir, exist_ok=True)

        print(f"Results will be stored in: {model_output_dir}")

        # 1. Load the reference FP32 model and obtain weight state dict
        ref_config = build_bandwidth_fp32_config(args, model_name, model_weights)
        model_order_dir = os.path.join(model_output_dir, "ref_fp32_loader")
        os.makedirs(model_order_dir, exist_ok=True)
        model, _, _ = runner.prepare_model_with_materialized_weights(config=ref_config, output_dir=model_order_dir)
        model.to(args.device)

        # 2. Setup results tracking structures
        cache_sizes = [0.0, 2.0, 4.0]  # Cache sizes in Millions of elements
        min_bits_list = [3]  # We sweep starting min_bits = 3 only
        results_data = {min_bits: {cs: [] for cs in cache_sizes} for min_bits in min_bits_list}

        # Cache simulations for all cache sizes (run only once per cache size!)
        cache_sims = {}
        for cs in cache_sizes:
            print(f"\n--- Running cache simulation for Cache Size: {cs}MB ---")
            sim_layers, cache_sim_map = run_cache_simulation(model_name, cs, batch_size=1, device=args.device)
            cache_sims[cs] = (sim_layers, cache_sim_map)
            
            # Log summary of stay decisions
            on_chip_cnt = sum(1 for stay in cache_sim_map.values() if stay)
            off_chip_cnt = len(cache_sim_map) - on_chip_cnt
            print(f"Cache {cs}MB: {on_chip_cnt} layers stay on-chip, {off_chip_cnt} layers stream off-chip.")

        # 3. Evaluate reference FP32 model (no quantization on weights or inputs)
        #    Compute cycles with 32-bit element transfers (no reduction — min=max=32).
        print(f"\n--- Evaluating reference FP32 model ---")
        ref_cycles_per_cs = {}
        for cs in cache_sizes:
            sim_layers, _ = cache_sims[cs]
            total_cyc = 0.0
            prev_stay = False
            for idx, layer in enumerate(sim_layers):
                stay = layer.get('stay_on_chip', False)
                xin_cache = layer.get('xin_from_cache', True)
                need_in = (idx == 0 or not prev_stay or not xin_cache)
                need_out = not stay
                _, _, _, cyc = optimize_layer_bits(
                    layer, args.bandwidth,
                    need_in, True, need_out,
                    min_bits=32, max_bits=32
                )
                total_cyc += cyc
                prev_stay = stay
            ref_cycles_per_cs[cs] = total_cyc
            print(f"  Cache {cs}MB FP32 cycles (32b): {total_cyc:,}")

        ref_baseline_cycles = ref_cycles_per_cs.get(0.0)
        if ref_baseline_cycles is None:
            ref_baseline_cycles = next(iter(ref_cycles_per_cs.values()), 1.0)
        print(f"  FP32 0MB reference baseline cycles: {ref_baseline_cycles:,}")

        ref_acc1 = get_cached_fp32_acc1(model_name)
        if ref_acc1 is None:
            try:
                print("Running reference FP32 evaluation...")
                ref_eval_results = runner.run_single(ref_config, output_root=model_output_dir)
                ref_acc1 = ref_eval_results.get('acc1', 0.0)
                print(f"Reference FP32: Top-1 Acc = {ref_acc1:.3f}%")
            except Exception as e:
                print(f"Error during reference evaluation: {e}")
                ref_acc1 = 0.0
                import traceback
                traceback.print_exc()

        # 4. Sweep loops
        unique_runs = []
        for cs in cache_sizes:
            for b in range(2, 9):
                unique_runs.append((cs, b))

        # Sort runs to keep cache size changes minimal (saves reloading models frequently)
        unique_runs = sorted(unique_runs, key=lambda x: (x[0], x[1]))

        evaluated_points = {}  # (cache_size, b_bits) -> (accuracy, cycles)

        temp_weights_dir = os.path.join(model_output_dir, "temp_weights")
        os.makedirs(temp_weights_dir, exist_ok=True)

        for cs, b in unique_runs:
            print(f"\n============================================================")
            print(f"Evaluating Cache: {cs}MB | Bit-Width of Off-Chip Layers: {b}-bits")
            print(f"============================================================")
            
            sim_layers, cache_sim_map = cache_sims[cs]
            
            # 3.1. Compute execution runtime (cycles) and obtain layer-specific transfer bits
            (
                cycles,
                layer_input_bits,
                layer_weight_bits,
                layer_output_bits,
                layer_residual_input_bits,
            ) = compute_model_runtime(sim_layers, b, bandwidth=args.bandwidth)
            print(f"Calculated compute time: {cycles:,} cycles")

            # 3.2. Quantize model weights based on per-layer weight bit-width
            print("Quantizing model weights...")
            q_state_dict, _ = create_bandwidth_aware_state_dict(model, cache_sim_map, per_layer_weight_bits=layer_weight_bits, chunk_size=128)
            
            # Save quantized weights to temporary file
            temp_weights_path = os.path.join(temp_weights_dir, f"weights_cs_{cs}_b_{b}.pt")
            torch.save(q_state_dict, temp_weights_path)
            
            # 3.3. Inject cache sim map and per-layer input bits to global attributes of both classes
            if DIQ1 is not None:
                DIQ1._global_cache_sim_map = cache_sim_map
                DIQ1._global_layer_input_bits_map = layer_input_bits
                DIQ1._global_layer_residual_input_bits_map = layer_residual_input_bits
            if DIQ2 is not None:
                DIQ2._global_cache_sim_map = cache_sim_map
                DIQ2._global_layer_input_bits_map = layer_input_bits
                DIQ2._global_layer_residual_input_bits_map = layer_residual_input_bits

            # 3.4. Build evaluation config — mirror the standard hybrid_quant
            # pipeline (Quant wrappers + decomposed MHA + folded input norm) so the
            # loaded state_dict (with weight_fp8/weight_scale populated by
            # create_bandwidth_aware_state_dict) is consumed by the same module
            # graph. weight_quantization=True lets calibrate_weights run during
            # build, but the materialized state_dict's weight_fp8/weight_scale
            # buffers take precedence over re-calibration on load.
            eval_config = {
                'model': {'name': model_name, 'weights': os.path.abspath(temp_weights_path)},
                'adapter': {
                    'type': 'generic',
                    'quantized_ops': ['all'],
                    'build_quantized': True,
                    'input_quantization': True,
                    'weight_quantization': True,
                    'fold_input_norm': True,
                    'quantize_first_layer': True,
                },
                'evaluation': {
                    'mode': 'evaluate',
                    'max_batches': args.limit_batches,
                    'dynamic_input_quant': {
                        'enabled': True,
                        'chunk_size': 128,
                        'candidate_formats': get_input_formats_for_bits(b),
                        'use_cache_sim_db': False,
                        'unsigned_input_sources': ['relu', 'relu6', 'softmax', 'quantrelu', 'quantsoftmax','quantrelu6'],
                    }
                },
                'dataset': {
                    'name': args.dataset_name,
                    'path': args.dataset_path,
                    'batch_size': args.batch_size,
                    'num_workers': args.num_workers,
                }
            }

            # 3.5. Run accuracy evaluation
            try:
                print("Running evaluation forward pass...")
                eval_results = runner.run_single(eval_config, output_root=model_output_dir)
                acc1 = eval_results.get('acc1', 0.0)
                acc5 = eval_results.get('acc5', 0.0)
                print(f"Accuracy results: Top-1 Acc = {acc1:.3f}%, Top-5 Acc = {acc5:.3f}%")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                acc1 = 0.0
                acc5 = 0.0
                import traceback
                traceback.print_exc()

            # Save evaluated point
            evaluated_points[(cs, b)] = (acc1, cycles)

            # Cleanup temporary weight file
            if os.path.exists(temp_weights_path):
                try:
                    os.remove(temp_weights_path)
                except OSError:
                    pass

        # 4. Map evaluated points to the results_data sweeps structure
        for min_bits in min_bits_list:
            for cs in cache_sizes:
                for b in range(min_bits, 9):
                    acc1, cycles = evaluated_points[(cs, b)]
                    results_data[min_bits][cs].append((b, acc1, cycles))

        # 5. Save results to JSON file
        results_path = os.path.join(model_output_dir, "bandwidth_aware_quant_results.json")
        serializable_results = {
            'model_name': model_name,
            'ref_fp32': {
                'accuracy': ref_acc1,
                'baseline_cycles': ref_baseline_cycles,
                'cycles_per_cache_size': ref_cycles_per_cs,
            },
            'min_bits_sweeps': {
                str(mb): {
                    str(cs): [{'b': p[0], 'accuracy': p[1], 'cycles': p[2]} for p in points]
                    for cs, points in cache_data.items()
                }
                for mb, cache_data in results_data.items()
            }
        }
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nAll results saved to {results_path}")

        # 6. Generate plots for each starting min_bits threshold
        print("\n--- Generating plots for each starting min_bits threshold ---")
        colors = {0.0: 'red', 2.0: 'blue', 4.0: 'green'}

        # Normalization factor: FP32 0MB cache is the 1.0x speedup baseline.
        norm = ref_cycles_per_cs.get(0.0, 1.0)
        if norm <= 0:
            norm = 1.0

        def normalized_speedup(cycles):
            if cycles <= 0:
                return 0.0
            return norm / cycles

        def find_axis_break(x_values, x_min, x_max):
            unique_x = sorted(set(round(x, 6) for x in x_values if math.isfinite(x)))
            if len(unique_x) < 3:
                return None

            gaps = [(right - left, left, right) for left, right in zip(unique_x, unique_x[1:])]
            largest_gap, gap_left, gap_right = max(gaps, key=lambda item: item[0])
            total_span = max(x_max - x_min, 1e-9)
            if largest_gap < max(0.20, total_span * 0.35):
                return None

            gap_pad = min(max(largest_gap * 0.08, 0.01), 0.04)
            left_xlim = (x_min, gap_left + gap_pad)
            right_xlim = (gap_right - gap_pad, x_max)
            if left_xlim[1] >= right_xlim[0]:
                return None
            return left_xlim, right_xlim

        def draw_axis_break_marks(left_ax, right_ax):
            d = 0.012
            kwargs = dict(color='black', clip_on=False, linewidth=1.0)
            left_ax.plot((1 - d, 1 + d), (-d, +d), transform=left_ax.transAxes, **kwargs)
            left_ax.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=left_ax.transAxes, **kwargs)
            right_ax.plot((-d, +d), (-d, +d), transform=right_ax.transAxes, **kwargs)
            right_ax.plot((-d, +d), (1 - d, 1 + d), transform=right_ax.transAxes, **kwargs)

        def dedup_legend(axes):
            handles = []
            labels = []
            seen = set()
            for plot_ax in axes:
                ax_handles, ax_labels = plot_ax.get_legend_handles_labels()
                for handle, label in zip(ax_handles, ax_labels):
                    if not label or label.startswith('_') or label in seen:
                        continue
                    handles.append(handle)
                    labels.append(label)
                    seen.add(label)
            return handles, labels

        global_min_acc = 100.0
        global_max_acc = 0.0
        global_min_norm = float('inf')
        global_max_norm = float('-inf')
        for min_bits, cache_data in results_data.items():
            for cs, points in cache_data.items():
                for b, acc, cyc in points:
                    nx = normalized_speedup(cyc)
                    global_min_norm = min(global_min_norm, nx)
                    global_max_norm = max(global_max_norm, nx)
                    global_min_acc = min(global_min_acc, acc)
                    global_max_acc = max(global_max_acc, acc)
        for cs in cache_sizes:
            rc = ref_cycles_per_cs.get(cs)
            if rc and rc > 0:
                nx = normalized_speedup(rc)
                global_min_norm = min(global_min_norm, nx)
                global_max_norm = max(global_max_norm, nx)
        if ref_acc1 > 0:
            global_min_acc = min(global_min_acc, ref_acc1)
            global_max_acc = max(global_max_acc, ref_acc1)
        if global_min_acc == 100.0:
            global_min_acc, global_max_acc = 0.0, 100.0
        if global_min_norm == float('inf'):
            global_min_norm, global_max_norm = 0, 1
        ylim_min = max(0.0, global_min_acc - 5.0)
        ylim_max = min(100.0, global_max_acc + 5.0)
        xpad = (global_max_norm - global_min_norm) * 0.05
        xlim_min = max(0.0, global_min_norm - xpad)
        xlim_max = global_max_norm + xpad

        linestyles = {0.0: '-', 2.0: '-', 4.0: '--'}
        markers = {0.0: 'o', 2.0: 's', 4.0: '^'}
        jitter = {0.0: 1.0, 2.0: 0.98, 4.0: 1.02}

        for min_bits, cache_data in results_data.items():
            x_values = []
            for cs in cache_sizes:
                ref_cyc = ref_cycles_per_cs.get(cs)
                if ref_cyc is not None and ref_cyc > 0 and ref_acc1 > 0:
                    x_values.append(normalized_speedup(ref_cyc))
            for cs, points in cache_data.items():
                j_factor = jitter.get(cs, 1.0)
                x_values.extend(normalized_speedup(cyc) * j_factor for _, _, cyc in points)

            axis_break = find_axis_break(x_values, xlim_min, xlim_max)
            if axis_break:
                fig, (ax_left, ax_right) = plt.subplots(
                    1, 2, figsize=(12, 7), sharey=True,
                    gridspec_kw={'width_ratios': [0.55, 3.5], 'wspace': 0.05}
                )
                axes = [ax_left, ax_right]
                ax_left.set_xlim(*axis_break[0])
                ax_right.set_xlim(*axis_break[1])
                ax_left.set_xticks([1.0])
                ax_left.set_xticklabels(["1.0"])
                ax_left.spines['right'].set_visible(False)
                ax_right.spines['left'].set_visible(False)
                ax_right.tick_params(axis='y', left=False, labelleft=False)
                draw_axis_break_marks(ax_left, ax_right)
                print(
                    f"  Using x-axis break from {axis_break[0][1]:.3f} "
                    f"to {axis_break[1][0]:.3f}"
                )
            else:
                fig, ax_left = plt.subplots(figsize=(12, 7))
                axes = [ax_left]
                ax_left.set_xlim(xlim_min, xlim_max)

            def axis_for_x(x):
                if len(axes) == 1:
                    return axes[0]
                return axes[0] if x <= axis_break[0][1] else axes[1]

            ref_label_added = False

            # FP32 reference diamonds
            for cs in cache_sizes:
                ref_cyc = ref_cycles_per_cs.get(cs)
                if ref_cyc is not None and ref_cyc > 0 and ref_acc1 > 0:
                    nx = normalized_speedup(ref_cyc)
                    plot_ax = axis_for_x(nx)
                    color = colors.get(cs, 'black')
                    lbl = "Ref FP32" if not ref_label_added else None
                    ref_label_added = True
                    plot_ax.scatter([nx], [ref_acc1], marker='D', color=color,
                                    s=120, zorder=10, edgecolors='black', linewidths=1.5,
                                    label=lbl)
                    plot_ax.annotate(
                        fmt_elems(int(ref_cyc)),
                        (nx, ref_acc1),
                        textcoords="offset points",
                        xytext=(10, -12),
                        ha='left', va='top',
                        fontsize=8, color='#495057'
                    )

            # Sweep curves
            for cs, points in cache_data.items():
                if not points:
                    continue
                points = sorted(points, key=lambda x: x[0])
                cyc_range = f"{fmt_elems(int(min(p[2] for p in points)))}–{fmt_elems(int(max(p[2] for p in points)))}"
                acc_range = f"{min(p[1] for p in points):.2f}–{max(p[1] for p in points):.2f}%"
                print(f"  Cache {cs}MB: {len(points)} pts, acc {acc_range}, cyc {cyc_range}")
                bits = [p[0] for p in points]
                accs = [p[1] for p in points]
                cycles = [p[2] for p in points]

                j_factor = jitter.get(cs, 1.0)
                plot_x = [normalized_speedup(cyc) * j_factor for cyc in cycles]

                label = f"Cache {cs}MB ({cyc_range} cyc)"
                color = colors.get(cs, 'black')
                linestyle = linestyles.get(cs, '-')
                marker = markers.get(cs, 'o')

                for plot_ax in axes:
                    plot_ax.plot(plot_x, accs, marker=marker, label=label, color=color,
                                 linestyle=linestyle, linewidth=2.5, markersize=18,
                                 markeredgewidth=1.2, markeredgecolor='black')

                unique_coords = []
                for idx, (b, acc, cyc) in enumerate(points):
                    px = plot_x[idx]
                    found = False
                    for group in unique_coords:
                        if abs(group['cyc'] - cyc) < 1e-2 and abs(group['acc'] - acc) < 1e-3:
                            group['bits'].append(b)
                            group['indices'].append(idx)
                            found = True
                            break
                    if not found:
                        unique_coords.append({
                            'px': px, 'cyc': cyc, 'acc': acc,
                            'bits': [b], 'indices': [idx]
                        })

                for group in unique_coords:
                    group_bits = sorted(group['bits'])
                    acc = group['acc']
                    px = group['px']
                    cyc = group['cyc']

                    if len(group_bits) > 1:
                        bw_text = f"{group_bits[0]}-{group_bits[-1]}"
                        b_val = group_bits[0]
                    else:
                        bw_text = f"{group_bits[0]}"
                        b_val = group_bits[0]

                    plot_ax = axis_for_x(px)
                    plot_ax.text(
                        px, acc, bw_text,
                        ha='center', va='center',
                        fontsize=7.5 if len(bw_text) > 1 else 8.5,
                        color='white', weight='bold',
                        zorder=12
                    )

            axes[-1].text(
                0.03, 0.05,
                f"x-axis: time_ref / time  (1.0x = FP32 0MB baseline, higher = faster)\n"
                f"Memory Bandwidth = {args.bandwidth} bytes/cycle.\n"
                f"Numbers: bit-width inside markers; sweep cycle ranges in legend.",
                transform=axes[-1].transAxes, ha='left', va='bottom',
                fontsize=9.5, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                          edgecolor='#ced4da', alpha=0.9)
            )

            fig.suptitle(
                f"Accuracy vs. Normalized Speedup vs FP32 0MB (Starting Min Bits = {min_bits})\n"
                f"Memory Bandwidth = {args.bandwidth} bytes/cycle"
            )
            xlabel = f"Normalized Speedup vs FP32 0MB  (1.0x = {fmt_elems(int(norm))} cyc baseline, higher = faster)"
            if len(axes) == 1:
                axes[0].set_xlabel(xlabel)
            else:
                fig.text(0.5, 0.035, xlabel, ha='center', va='center')
            axes[0].set_ylabel("Top-1 Accuracy (%)")
            for plot_ax in axes:
                plot_ax.set_ylim(ylim_min, ylim_max)
                plot_ax.grid(True, linestyle='--', alpha=0.6)
            handles, labels = dedup_legend(axes)
            axes[-1].legend(handles, labels)
            fig.tight_layout(rect=[0, 0.06 if len(axes) > 1 else 0, 1, 0.93])
            plot_path = os.path.join(model_output_dir, f"accuracy_vs_speedup_min_bits_{min_bits}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved plot: {plot_path}")

        # Cleanup temp directory
        try:
            os.rmdir(temp_weights_dir)
        except OSError:
            pass

    print(f"\nDone! Recreated bandwidth_aware_quant experiment complete.")


if __name__ == "__main__":
    main()
