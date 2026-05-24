import os
import sys

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import math
import argparse
import json
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
from runspace.experiments.utils.plotting import get_format_bits

from runspace.experiments.asic_cache_simulation.simulate_cache import (
    analyze_model,
    evaluate_stay,
    get_footprint_elements,
    round_to_banks
)

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

def patched_candidates_for_layer(self, layer_name, module=None):
    """Silent custom candidates picker that defaults missing layers to stay_on_chip=True (FP8)."""
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

    # Silence warnings: default missing layers (e.g. activations, pooling, residuals) to stays_on_chip=True (FP8)
    stays_on_chip = self.cache_sim_map.get(layer_name, True)

    if stays_on_chip:
        return self.unsigned_all_fp8_formats if is_unsigned else self.all_fp8_formats

    if is_unsigned:
        return self.unsigned_candidate_formats

    if self.restrict_post_relu_ufp:
        if layer_name in self.post_relu_layers:
            return self.ufp_candidates or self._make_unsigned_candidates(self.non_ufp_candidates)
        return self.non_ufp_candidates or self.ufp_candidates

    return self.candidate_formats

for DIQ in (DIQ1, DIQ2):
    if DIQ is not None:
        # Override candidates method
        DIQ._candidates_for_layer = patched_candidates_for_layer
        
        # Override init to load global map
        _orig_init = DIQ.__init__
        def make_new_init(orig_init):
            def new_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                self.cache_sim_map = getattr(self.__class__, '_global_cache_sim_map', {})
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
        
        layer_copy = dict(layer)
        layer_copy['stay_on_chip'] = stay_on_chip
        results.append(layer_copy)
        cache_sim_map[layer['name']] = stay_on_chip
        
    return results, cache_sim_map


def compute_model_runtime(layers_with_stay_status, b_bits, bandwidth=1.0):
    """
    Compute model total runtime (execution cycles) under bandwidth-constrained conditions.
    Accumulates cycles across parent operations and all of their collapsed operations.
    """
    total_runtime = 0.0
    prev_stay_on_chip = False  # Track if the previous layer's output stayed on-chip
    
    for idx, layer in enumerate(layers_with_stay_status):
        stay_on_chip = layer.get('stay_on_chip', False)
        
        # 1. Compute cycles
        l_type = layer['type']
        compute_cycles = 0
        
        if l_type == 'Conv2d':
            out_c = layer.get('out_channels', 0)
            in_c = layer.get('in_channels', 0)
            groups = layer.get('groups', 1)
            fh = layer.get('filter_height', 0)
            fw = layer.get('filter_width', 0)
            oh = layer.get('output_channel_height', 0)
            ow = layer.get('output_channel_width', 0)
            macs = out_c * (in_c / groups) * fh * fw * oh * ow
            compute_cycles = math.ceil(macs / 128)
        elif l_type == 'Linear':
            weight_elems = layer.get('weight_elems', 0)
            macs = weight_elems * 1  # batch_size is 1
            compute_cycles = math.ceil(macs / 128)
        else:
            out_elems = layer.get('output_elems', 0)
            compute_cycles = math.ceil(out_elems / 128)
            
        # Add collapsed child compute cycles
        for collapsed in layer.get('collapsed_layers', []):
            collapsed_out_elems = collapsed.get('output_elems', 0)
            compute_cycles += math.ceil(collapsed_out_elems / 128)
            
        # 2. Memory transfer cycles (assuming 1.0 elements/cycle bandwidth by default)
        if stay_on_chip:
            transfer_cycles = 0
        else:
            # Off-chip layer transfers weights, inputs (if needed), and outputs
            # Weights are quantized to b_bits
            weight_transfer = layer.get('weight_elems', 0) * (b_bits / 8.0)
            
            # Input transfer: if first layer or previous layer was off-chip
            if idx == 0 or not prev_stay_on_chip:
                input_transfer = layer.get('input_elems', 0)
            else:
                input_transfer = 0
                
            # Output transfer
            output_transfer = layer.get('output_elems', 0)
            
            transfer_cycles = (weight_transfer + input_transfer + output_transfer) / bandwidth
            
        # Runtime is max(compute, transfer)
        layer_runtime = max(compute_cycles, transfer_cycles)
        total_runtime += layer_runtime
        
        # Save stay status for the next layer's input check
        prev_stay_on_chip = stay_on_chip
        
    return total_runtime


def create_bandwidth_aware_state_dict(model, cache_sim_map, b_bits, chunk_size=128):
    """Reassemble state dict with weights quantized based on stay_on_chip status and target bit-width."""
    state_dict = model.state_dict()
    quant_map = {}
    
    # 1. First, identify if there are MHA virtual layers
    # MHA virtual names are like '<mha_name>.q_proj', '<mha_name>.k_proj', '<mha_name>.v_proj'
    # in the cache sim map.
    mha_parents = {}
    for lname in cache_sim_map:
        for suffix in ('.q_proj', '.k_proj', '.v_proj'):
            if lname.endswith(suffix):
                parent = lname[:-len(suffix)]
                mha_parents.setdefault(parent, {})[suffix[1:]] = lname
                
    handled_names = set()
    
    # Slices for MHA in_proj_weight
    for mha_name, proj_map in mha_parents.items():
        in_proj_key = f"{mha_name}.in_proj_weight"
        if in_proj_key not in state_dict:
            continue
        w_in_proj = state_dict[in_proj_key]
        embed_dim = w_in_proj.shape[0] // 3
        slices = {
            'q_proj': (0, embed_dim),
            'k_proj': (embed_dim, 2 * embed_dim),
            'v_proj': (2 * embed_dim, 3 * embed_dim),
        }
        new_in_proj = w_in_proj.clone()
        for proj_key, (start, end) in slices.items():
            vname = proj_map.get(proj_key)
            if vname and vname in cache_sim_map:
                stay_on_chip = cache_sim_map[vname]
                w_slice = w_in_proj[start:end].contiguous()
                
                # Determine bits and candidates
                bits = 8 if stay_on_chip else b_bits
                formats = SIGNED_FORMATS_BY_BITS.get(bits, [])
                best_fmt = get_best_weight_format(w_slice, formats, chunk_size=chunk_size)
                
                if best_fmt:
                    w_deq, _ = get_quantized_tensor_sim(w_slice, best_fmt, chunk_size=chunk_size)
                    new_in_proj[start:end] = w_deq
                    quant_map[vname] = best_fmt
                    
        state_dict[in_proj_key] = new_in_proj
        handled_names.update(proj_map.values())
        
    # 2. For all other layers in cache_sim_map
    for name, module in model.named_modules():
        if name in cache_sim_map and name not in handled_names:
            stay_on_chip = cache_sim_map[name]
            weight_key = f"{name}.weight"
            if weight_key in state_dict:
                w = state_dict[weight_key]
                
                # Determine bits and candidates
                bits = 8 if stay_on_chip else b_bits
                formats = SIGNED_FORMATS_BY_BITS.get(bits, [])
                best_fmt = get_best_weight_format(w, formats, chunk_size=chunk_size)
                
                if best_fmt:
                    w_deq, _ = get_quantized_tensor_sim(w, best_fmt, chunk_size=chunk_size)
                    state_dict[weight_key] = w_deq
                    quant_map[name] = best_fmt
                    
    return state_dict, quant_map


# ============================================================
# Main Experiment Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Recreation of bandwidth-aware quantization sweeps.")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name")
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for accuracy evaluation")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for dataset loader")
    parser.add_argument("--limit_batches", type=int, default=-1, help="Limit number of evaluation batches (-1 = all)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--bandwidth", type=float, default=1.0, help="Memory bandwidth in elements/cycle")
    args = parser.parse_args()

    # Verify CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is requested but not available. Falling back to cpu.")
        args.device = "cpu"

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "runspace/experiments/bandwidth_aware_quant/results", args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Bandwidth-Aware mixed-precision quantization experiment for {args.model_name}")
    print(f"Results will be stored in: {output_dir}")

    runner = Runner(device=args.device)

    # 1. Load the reference FP32 model and obtain weight state dict
    ref_config = {
        'model': {'name': args.model_name, 'weights': args.weights},
        'adapter': {'type': 'generic', 'quantized_ops': []},
        'dataset': {
            'name': args.dataset_name, 'path': args.dataset_path,
            'batch_size': args.batch_size, 'num_workers': args.num_workers,
        },
    }
    model_order_dir = os.path.join(output_dir, "ref_fp32_loader")
    os.makedirs(model_order_dir, exist_ok=True)
    model, _, _ = runner.prepare_model_with_materialized_weights(config=ref_config, output_dir=model_order_dir)
    model.to(args.device)

    # 2. Setup results tracking structures
    cache_sizes = [0.0, 2.0, 4.0]  # Cache sizes in Millions of elements
    min_bits_list = [2, 3, 4, 5, 6, 7]  # We sweep starting min_bits from 2 to 7
    results_data = {min_bits: {cs: [] for cs in cache_sizes} for min_bits in min_bits_list}

    # Cache simulations for all cache sizes (run only once per cache size!)
    cache_sims = {}
    for cs in cache_sizes:
        print(f"\n--- Running cache simulation for Cache Size: {cs}MB ---")
        sim_layers, cache_sim_map = run_cache_simulation(args.model_name, cs, batch_size=1, device=args.device)
        cache_sims[cs] = (sim_layers, cache_sim_map)
        
        # Log summary of stay decisions
        on_chip_cnt = sum(1 for stay in cache_sim_map.values() if stay)
        off_chip_cnt = len(cache_sim_map) - on_chip_cnt
        print(f"Cache {cs}MB: {on_chip_cnt} layers stay on-chip, {off_chip_cnt} layers stream off-chip.")

    # 3. Sweep loops
    # To optimize execution, we can group runs by (cache_size, b_bits).
    # Since accuracy of a run is fully determined by (cache_size, b_bits), we can evaluate
    # each unique pair of (cache_size, b_bits) once and store it.
    unique_runs = []
    for cs in cache_sizes:
        for b in range(2, 9):
            unique_runs.append((cs, b))

    # Sort runs to keep cache size changes minimal (saves reloading models frequently)
    unique_runs = sorted(unique_runs, key=lambda x: (x[0], x[1]))

    evaluated_points = {}  # (cache_size, b_bits) -> (accuracy, cycles)

    temp_weights_dir = os.path.join(output_dir, "temp_weights")
    os.makedirs(temp_weights_dir, exist_ok=True)

    for cs, b in unique_runs:
        print(f"\n============================================================")
        print(f"Evaluating Cache: {cs}MB | Bit-Width of Off-Chip Layers: {b}-bits")
        print(f"============================================================")
        
        sim_layers, cache_sim_map = cache_sims[cs]
        
        # 3.1. Compute execution runtime (cycles)
        cycles = compute_model_runtime(sim_layers, b, bandwidth=args.bandwidth)
        print(f"Calculated compute time: {cycles:,} cycles")

        # 3.2. Quantize model weights based on stay status and bit width
        print("Quantizing model weights...")
        q_state_dict, _ = create_bandwidth_aware_state_dict(model, cache_sim_map, b_bits=b, chunk_size=128)
        
        # Save quantized weights to temporary file
        temp_weights_path = os.path.join(temp_weights_dir, f"weights_cs_{cs}_b_{b}.pt")
        torch.save(q_state_dict, temp_weights_path)
        
        # 3.3. Inject cache sim map to global attributes of both classes
        if DIQ1 is not None:
            DIQ1._global_cache_sim_map = cache_sim_map
        if DIQ2 is not None:
            DIQ2._global_cache_sim_map = cache_sim_map

        # 3.4. Build evaluation config
        eval_config = {
            'model': {'name': args.model_name, 'weights': os.path.abspath(temp_weights_path)},
            'adapter': {
                'type': 'generic',
                'quantized_ops': ['-1'],
                'input_quantization': False,
            },
            'evaluation': {
                'mode': 'evaluate',
                'max_batches': args.limit_batches,
                'dynamic_input_quant': {
                    'enabled': True,
                    'chunk_size': 128,
                    'candidate_formats': get_input_formats_for_bits(b),
                    'use_cache_sim_db': False,
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
            eval_results = runner.run_single(eval_config, output_root=output_dir)
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
    # For a given starting min_bits threshold:
    # We sweep b from min_bits to 8.
    for min_bits in min_bits_list:
        for cs in cache_sizes:
            for b in range(min_bits, 9):
                acc1, cycles = evaluated_points[(cs, b)]
                results_data[min_bits][cs].append((b, acc1, cycles))

    # 5. Save results to JSON file
    results_path = os.path.join(output_dir, "bandwidth_aware_quant_results.json")
    serializable_results = {
        'model_name': args.model_name,
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
    
    # Enforce same scale: find global min/max across all min_bits and cache sizes
    global_min_cycles = float('inf')
    global_max_cycles = float('-inf')
    global_min_acc = 100.0
    global_max_acc = 0.0

    for min_bits, cache_data in results_data.items():
        for cs, points in cache_data.items():
            for b, acc, cyc in points:
                global_min_cycles = min(global_min_cycles, cyc)
                global_max_cycles = max(global_max_cycles, cyc)
                global_min_acc = min(global_min_acc, acc)
                global_max_acc = max(global_max_acc, acc)

    if global_min_cycles == float('inf'):
        global_min_cycles, global_max_cycles = 0, 1000000
    if global_min_acc == 100.0:
        global_min_acc, global_max_acc = 0.0, 100.0

    # Margins for same scaling
    xlim_min = global_min_cycles * 0.9
    xlim_max = global_max_cycles * 1.1
    ylim_min = max(0.0, global_min_acc - 5.0)
    ylim_max = min(100.0, global_max_acc + 5.0)

    for min_bits, cache_data in results_data.items():
        plt.figure(figsize=(10, 6))
        
        for cs, points in cache_data.items():
            if not points:
                continue
            # Sort points by bit width b
            points = sorted(points, key=lambda x: x[0])
            bits = [p[0] for p in points]
            accs = [p[1] for p in points]
            cycles = [p[2] for p in points]
            
            label = f"Cache {cs}MB"
            color = colors.get(cs, 'black')
            
            plt.plot(cycles, accs, marker='o', label=label, color=color, linewidth=2)
            
            # Annotate points with bit-width b
            for b, acc, cyc in points:
                plt.annotate(f"{b}b", (cyc, acc), textcoords="offset points", 
                             xytext=(0, 10), ha='center', fontsize=8, color=color, weight='bold')
                
        plt.title(f"Accuracy vs. Compute Time (Starting Min Bits = {min_bits})\nBandwidth = {args.bandwidth} elements/cycle")
        plt.xlabel("Compute Time (Cycles)")
        plt.ylabel("Top-1 Accuracy (%)")
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"accuracy_vs_compute_time_min_bits_{min_bits}.png")
        plt.savefig(plot_path, dpi=150)
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
