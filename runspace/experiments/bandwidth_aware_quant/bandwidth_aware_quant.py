import os
import sys

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import math
import argparse
import json
import sqlite3
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
from runspace.src.quantization.chunking import chunk_weight_by_context
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

CANONICAL_INPUT_FORMATS_BY_BITS = {
    2: 'fp2_e1m0',
    3: 'fp3_e1m1',
    4: 'fp4_e2m1',
    5: 'fp5_e2m2',
    6: 'fp6_e3m2',
    7: 'fp7_e3m3',
    8: 'fp8_e4m3',
}

def patched_candidates_for_layer(self, layer_name, module=None, input_index=0):
    """Silent custom candidates picker that uses per-layer input bit-width for off-chip layers."""
    is_unsigned = self._layer_uses_unsigned_input(layer_name, input_index=input_index)
    input_format_policy = getattr(self, 'layer_input_format_policy', 'all')

    if input_index == 1:
        residual_bits = self.layer_residual_input_bits_map.get(layer_name)
        if residual_bits is not None:
            candidates = get_input_formats_for_bits(
                residual_bits,
                policy=input_format_policy,
                unsigned=is_unsigned,
            )
            return (
                self._make_unsigned_candidates(candidates)
                if is_unsigned and input_format_policy != 'typed'
                else candidates
            )

    stays_on_chip = self.cache_sim_map.get(layer_name, True)
    input_bits = self.layer_input_bits_map.get(layer_name, 8)

    if input_bits == 8:
        candidates = get_input_formats_for_bits(
            8,
            policy=input_format_policy,
            unsigned=is_unsigned,
        )
    elif stays_on_chip:
        candidates = get_input_formats_for_bits(
            8,
            policy=input_format_policy,
            unsigned=is_unsigned,
        )
    else:
        candidates = get_input_formats_for_bits(
            input_bits,
            policy=input_format_policy,
            unsigned=is_unsigned,
        )

    if is_unsigned and input_format_policy != 'typed':
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
                self.layer_need_input_transfer_map = getattr(
                    self.__class__,
                    '_global_layer_need_input_transfer_map',
                    getattr(self, 'layer_need_input_transfer_map', {}),
                )
                self.layer_input_format_policy = getattr(
                    self.__class__,
                    '_global_layer_input_format_policy',
                    getattr(self, 'layer_input_format_policy', 'all'),
                )
            return new_init
        DIQ.__init__ = make_new_init(_orig_init)


# ============================================================
# Signed Formats By Bit Width
# ============================================================
SIGNED_FORMATS_BY_BITS = {
    8: ['fp8_e4m3', 'fp8_e5m2', 'fp8_e3m4', 'fp8_e2m5', 'fp8_e1m6'],# 'fp8_e6m1', 'fp8_e7m0'],
    7: ['fp7_e1m5', 'fp7_e2m4', 'fp7_e3m3', 'fp7_e4m2'], #'fp7_e5m1', 'fp7_e6m0'],
    6: ['fp6_e1m4', 'fp6_e2m3', 'fp6_e3m2'], #'fp6_e4m1', 'fp6_e5m0'],
    5: ['fp5_e1m3', 'fp5_e2m2', 'fp5_e3m1'], #'fp5_e4m0'],
    4: ['fp4_e1m2', 'fp4_e2m1', 'fp4_e3m0'],
    3: ['fp3_e1m1', 'fp3_e2m0'],
    2: ['fp2_e1m0']
}

# ============================================================
# Helpers
# ============================================================

def get_input_formats_for_bits(b, policy='all', unsigned=False):
    """Filter input formats to retrieve candidates with exactly b bits."""
    if policy == 'canonical':
        fmt = CANONICAL_INPUT_FORMATS_BY_BITS.get(b)
        return [fmt] if fmt else []

    candidates = []
    for fmt in DEFAULT_DYNAMIC_INPUT_CANDIDATES:
        if get_format_bits(fmt) == b:
            candidates.append(fmt)
    # Ensure signed standard formats of b bits are also included
    if b in SIGNED_FORMATS_BY_BITS:
        for fmt in SIGNED_FORMATS_BY_BITS[b]:
            if fmt not in candidates:
                candidates.append(fmt)
    if policy == 'typed':
        if unsigned:
            typed_candidates = [fmt for fmt in candidates if fmt.startswith('ufp')]
        else:
            typed_candidates = [fmt for fmt in candidates if not fmt.startswith('ufp')]
        return typed_candidates or candidates
    return candidates


def get_best_weight_format(weight_tensor, formats, chunk_size=128, layer_name=None):
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
        except Exception as exc:
            lname = layer_name or "<unknown>"
            raise RuntimeError(
                f"Failed to score weight format {fmt} for layer {lname} "
                f"with shape {tuple(weight_tensor.shape)} and chunk_size={chunk_size}"
            ) from exc
    return best_fmt if best_fmt else formats[0]


def get_best_chunk_formats(weight_tensor, formats, chunk_size=128, layer_name=None):
    """Per-chunk MSE format selection (one MSE-best format per context chunk).

    Mirrors the proven per-chunk winner computation in find_optimal_weight_quant
    (run_weight_quantization_analysis): chunk the weight by context, quantize it
    with each candidate format, then pick, for every chunk independently, the
    format with the lowest per-chunk MSE. Returns a flat list of length
    num_contexts * num_chunks (matching the order calibrate_weights consumes via
    chunk_weight_by_context(...).reshape(-1, chunk_size)).
    """
    if not formats:
        return None

    w_chunked, _, _ = chunk_weight_by_context(weight_tensor, chunk_size)
    num_chunks_total = w_chunked.shape[0] * w_chunked.shape[1]

    per_format_errs = []
    valid_formats = []
    for fmt in formats:
        try:
            w_deq, _ = get_quantized_tensor_sim(weight_tensor, fmt, chunk_size=chunk_size)
        except Exception as exc:
            lname = layer_name or "<unknown>"
            raise RuntimeError(
                f"Failed to score per-chunk weight format {fmt} for layer {lname} "
                f"with shape {tuple(weight_tensor.shape)} and chunk_size={chunk_size}"
            ) from exc
        w_deq_chunked, _, _ = chunk_weight_by_context(w_deq, chunk_size)
        chunk_errs = (w_chunked - w_deq_chunked).pow(2).mean(dim=-1).view(-1)
        per_format_errs.append(chunk_errs)
        valid_formats.append(fmt)

    if not valid_formats:
        return None

    err_matrix = torch.stack(per_format_errs)  # [num_formats, num_chunks_total]
    best_indices = err_matrix.argmin(dim=0).tolist()
    return [valid_formats[i] for i in best_indices][:num_chunks_total]


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
        need_input_transfer = (i == 0 or not results[-1].get('stay_on_chip', False) or not xin_from_cache)
        
        layer_copy = dict(layer)
        layer_copy['stay_on_chip'] = stay_on_chip
        layer_copy['xin_from_cache'] = xin_from_cache
        layer_copy['need_input_transfer'] = need_input_transfer
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
    layer_need_input_transfer : dict layer_name -> whether layer input is externally transferred
    """
    total_runtime = 0.0
    prev_stay_on_chip = False
    layer_input_bits = {}
    layer_weight_bits = {}
    layer_output_bits = {}
    layer_residual_input_bits = {}
    layer_need_input_transfer = {}
    
    for idx, layer in enumerate(layers_with_stay_status):
        stay_on_chip = layer.get('stay_on_chip', False)
        lname = layer['name']
        xin_from_cache = layer.get('xin_from_cache', True)
        
        need_input_transfer = (idx == 0 or not prev_stay_on_chip or not xin_from_cache)
        layer_need_input_transfer[lname] = need_input_transfer
        need_weight_transfer = True
        need_output_transfer = not stay_on_chip
        residual_bits = b_bits
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
        
    return (
        total_runtime,
        layer_input_bits,
        layer_weight_bits,
        layer_output_bits,
        layer_residual_input_bits,
        layer_need_input_transfer,
    )


def create_bandwidth_aware_state_dict(model, cache_sim_map, per_layer_weight_bits, chunk_size=128, best_weight_map_by_bits=None):
    """Calibrate per-layer weights via the Quant layer's own machinery.

    The model is built with `quantized_ops:['all']`, so each compute layer is a
    Quant{Conv2d,Linear} wrapper with `weight_fp8`/`weight_scale` buffers, and
    MHA is already decomposed into separate q_proj/k_proj/v_proj Linear
    sub-modules whose names match the cache_sim_map keys directly.

    For each layer in `cache_sim_map`:
      - If `best_weight_map_by_bits` is provided, look up the best format for the
        *allocated* bit-width (from `per_layer_weight_bits`).
      - Otherwise, pick the best signed format among `SIGNED_FORMATS_BY_BITS[bits]`
      - set the module's q_type and call `calibrate_weights()` to populate
        `weight_fp8` and `weight_scale` consistently — matching the
        find_optimal_hybrid_quant pipeline's `_materialize_weight_buffers_from_map`.
    """
    modules = dict(model.named_modules())
    quant_map = {}
    used_best = 0
    used_fallback = 0

    for layer_name in cache_sim_map:
        module = modules.get(layer_name)
        if module is None or not hasattr(module, 'calibrate_weights'):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        bits = per_layer_weight_bits.get(layer_name, 8)
        best_fmt = None

        if best_weight_map_by_bits is not None:
            layer_map = best_weight_map_by_bits.get(layer_name)
            if layer_map:
                best_fmt = layer_map.get(bits)
                if best_fmt:
                    print(f"[best_weights] {layer_name}: using best {bits}-bit format {best_fmt}")
                    used_best += 1
                else:
                    print(f"[best_weights] {layer_name}: no best {bits}-bit format in map, falling back to SIGNED_FORMATS_BY_BITS")
            else:
                print(f"[best_weights] {layer_name}: not in best-weight map, falling back to SIGNED_FORMATS_BY_BITS")

        if best_fmt is None:
            formats = SIGNED_FORMATS_BY_BITS.get(bits, [])
            best_fmt = get_best_weight_format(
                module.weight.detach(),
                formats,
                chunk_size=chunk_size,
                layer_name=layer_name,
            )
            if not best_fmt:
                continue
            if best_weight_map_by_bits is not None:
                used_fallback += 1

        module.weight_quantization = True
        module.weight_chunk_size = chunk_size
        module.weight_mode = 'chunk'
        module.chunk_formats = None
        module.q_type = best_fmt
        module.calibrate_weights()
        quant_map[layer_name] = best_fmt

    if best_weight_map_by_bits is not None:
        total_layers = used_best + used_fallback
        print(f"[best_weights] Summary: {used_best}/{total_layers} layers used best weights, {used_fallback}/{total_layers} fell back to SIGNED_FORMATS_BY_BITS")
    
    return model.state_dict(), quant_map


MSE_POLICY = 'mse'  # sentinel: per-chunk MSE selection among SIGNED_FORMATS_BY_BITS[bits]


def create_descent_state_dict(model, cache_sim_map, per_layer_weight_bits,
                              policy_by_bits, chunk_size=128):
    """Materialize weights for the greedy-descent mode.

    Each layer is assigned a weight bit-width (`per_layer_weight_bits`) and uses
    the policy decided for that bit-width (`policy_by_bits[bits]`). A policy is
    either a fixed format string (applied to the whole layer) or the sentinel
    `MSE_POLICY` (per-chunk MSE among SIGNED_FORMATS_BY_BITS[bits]).

    Returns (state_dict, quant_map) where quant_map maps layer -> the fixed
    format string, or `"mse:<dominant_fmt>"` for per-chunk MSE layers.
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
        policy = policy_by_bits.get(bits)
        if policy is None:
            # No decided policy for this width yet — fall back to per-layer MSE-best.
            policy = get_best_weight_format(
                module.weight.detach(), SIGNED_FORMATS_BY_BITS.get(bits, []),
                chunk_size=chunk_size, layer_name=layer_name,
            )
            if not policy:
                continue

        module.weight_quantization = True
        module.weight_chunk_size = chunk_size
        module.weight_mode = 'chunk'

        if policy == MSE_POLICY:
            chunk_formats = get_best_chunk_formats(
                module.weight.detach(), SIGNED_FORMATS_BY_BITS.get(bits, []),
                chunk_size=chunk_size, layer_name=layer_name,
            )
            if not chunk_formats:
                continue
            module.chunk_formats = chunk_formats
            module.q_type = chunk_formats[0]
            module.calibrate_weights()
            dominant = max(set(chunk_formats), key=chunk_formats.count)
            quant_map[layer_name] = f"mse:{dominant}"
        else:
            module.chunk_formats = None
            module.q_type = policy
            module.calibrate_weights()
            quant_map[layer_name] = policy

    return model.state_dict(), quant_map


def get_best_weight_map_from_db(model_name, db_path=None):
    """Fetch the best per-layer weight format map from the database.

    Looks for the latest `weight_quant_optimized` run for the model and
    returns the `quant_map_json` as a dict mapping layer_name -> format.
    Returns None if no optimized run is found.
    """
    try:
        db = RunDatabase(db_path=db_path) if db_path else RunDatabase()
        run = db.get_latest_run(
            model_name=model_name,
            experiment_type='weight_quant_optimized',
            status='SUCCESS',
        )
    except Exception as exc:
        print(f"[best_weights] Could not query DB for best weight map: {exc}")
        return None

    if not run:
        print(f"[best_weights] No weight_quant_optimized run found for {model_name}")
        return None

    raw_map = run.get('quant_map_json')
    if not raw_map:
        print(f"[best_weights] weight_quant_optimized run for {model_name} has no quant_map_json")
        return None

    try:
        best_map = json.loads(raw_map)
        if not isinstance(best_map, dict):
            print(f"[best_weights] quant_map_json is not a dict for {model_name}")
            return None
    except Exception as exc:
        print(f"[best_weights] Failed to parse quant_map_json for {model_name}: {exc}")
        return None

    # The map may contain nested dicts like {"format": "fp4_e2m1", ...}.
    # Flatten to layer_name -> format string.
    flat_map = {}
    for layer, value in best_map.items():
        if isinstance(value, dict):
            fmt = value.get("format")
            if fmt:
                flat_map[layer] = fmt
        elif isinstance(value, str):
            flat_map[layer] = value

    print(f"[best_weights] Loaded best weight map for {model_name} from run id {run.get('id')}: {len(flat_map)} layers")
    return flat_map


def _extract_optimized_layer_formats(q_map):
    """Flatten an optimized run's quant_map_json into (layer_formats, bits).

    Optimized (`weight_quant_optimized_*`) runs store each layer's `format` as a
    per-chunk **list** plus a `dominant_format` string; older/simple runs may
    store a plain format string. We collapse each layer to one representative
    format (its dominant per-chunk format) and infer the run's bit-width as the
    most common format bit-width across layers.

    Returns
    -------
    layer_formats : dict  layer_name -> format string
    bits : int | None     the run's bit-width (None if undeterminable)
    """
    layer_formats = {}
    bit_votes = {}
    for layer, value in q_map.items():
        fmt = None
        if isinstance(value, dict):
            fmt = value.get("dominant_format")
            if not fmt:
                raw = value.get("format")
                if isinstance(raw, list) and raw:
                    fmt = max(set(raw), key=raw.count)  # most common chunk format
                elif isinstance(raw, str):
                    fmt = raw
        elif isinstance(value, str):
            fmt = value
        if not fmt or not isinstance(fmt, str):
            continue
        bits = get_format_bits(fmt)
        if bits is None or bits <= 0:
            continue
        layer_formats[layer] = fmt
        bit_votes[bits] = bit_votes.get(bits, 0) + 1

    run_bits = max(bit_votes, key=bit_votes.get) if bit_votes else None
    return layer_formats, run_bits


def load_best_weight_map_by_bits(model_name, db_path=None):
    """Load per-layer best weight format for each bit-width from DB runs.

    For every bit-width the two candidate sources are compared by *actual top-1
    accuracy* and the winner is chosen:
      - `weight_quant_baseline`        → single best fixed format at that width
      - `weight_quant_optimized_<N>bit` → per-chunk run (one best run per width)

    Decision per bit-width:
      - baseline wins  → every layer uses that one fixed format
      - optimized wins → every layer uses its own per-layer (dominant) format,
                         falling back to the fixed format for any layer the
                         optimized run did not cover.

    Returns a dict: layer -> {bits -> best_fmt}, or None if no usable data.
    """
    try:
        db = RunDatabase(db_path=db_path) if db_path else RunDatabase()
    except Exception as exc:
        print(f"[best_weights_by_bits] Could not open DB: {exc}")
        return None

    # ------------------------------------------------------------------
    # 1. Fetch baseline + optimized runs in one connection.
    # ------------------------------------------------------------------
    try:
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT weight_dt, acc1
                FROM runs
                WHERE model_name = ?
                  AND experiment_type LIKE 'weight_quant_baseline%'
                  AND status = 'SUCCESS'
                  AND acc1 > 0
                ORDER BY id DESC
                """,
                (model_name,),
            )
            baseline_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT acc1, quant_map_json
                FROM runs
                WHERE model_name = ?
                  AND experiment_type LIKE 'weight_quant_optimized%'
                  AND status = 'SUCCESS'
                  AND acc1 > 0
                ORDER BY id DESC
                """,
                (model_name,),
            )
            optimized_rows = cursor.fetchall()
    except Exception as exc:
        print(f"[best_weights_by_bits] Error reading runs for {model_name}: {exc}")
        return None

    # ------------------------------------------------------------------
    # 2. Baseline: best fixed format per bit-width by acc1.
    # ------------------------------------------------------------------
    baseline_best_by_bits = {}  # bits -> (acc1, fmt)
    for row in baseline_rows:
        fmt = row["weight_dt"]
        if not fmt or fmt.lower() == "fp32":
            continue
        bits = get_format_bits(fmt)
        if bits is None or bits <= 0:
            continue
        acc1 = float(row["acc1"] or 0.0)
        if bits not in baseline_best_by_bits or acc1 > baseline_best_by_bits[bits][0]:
            baseline_best_by_bits[bits] = (acc1, fmt)

    # ------------------------------------------------------------------
    # 3. Optimized: best per-chunk run per bit-width by acc1.
    # ------------------------------------------------------------------
    optimized_best_by_bits = {}  # bits -> (acc1, {layer: fmt})
    for row in optimized_rows:
        raw_map = row["quant_map_json"]
        if not raw_map:
            continue
        try:
            q_map = json.loads(raw_map)
        except Exception:
            continue
        if not isinstance(q_map, dict):
            continue
        layer_formats, run_bits = _extract_optimized_layer_formats(q_map)
        if run_bits is None or not layer_formats:
            continue
        acc1 = float(row["acc1"] or 0.0)
        if run_bits not in optimized_best_by_bits or acc1 > optimized_best_by_bits[run_bits][0]:
            optimized_best_by_bits[run_bits] = (acc1, layer_formats)

    if not baseline_best_by_bits and not optimized_best_by_bits:
        print(f"[best_weights_by_bits] No baseline/optimized data for {model_name} in DB")
        return None

    # ------------------------------------------------------------------
    # 4. Per bit-width, pick each layer's format from the right source.
    #
    # The per-layer choice MUST come from the optimized run's per-layer
    # `dominant_format`, NOT from the baseline fixed format ranked by
    # whole-model accuracy. Baseline accuracy is measured with EVERY layer
    # forced to one format, so it rewards range over precision for the single
    # worst layer — the opposite of what an isolated low-bit layer (embedded in
    # an otherwise-high-precision model) needs. The optimized run already picks
    # the per-layer MSE-best format, which is what we want here. The fixed
    # baseline format is only used to fill layers/widths the optimized run did
    # not cover.
    # ------------------------------------------------------------------
    all_layers = set()
    for _, layer_formats in optimized_best_by_bits.values():
        all_layers.update(layer_formats.keys())

    all_bits = sorted(set(baseline_best_by_bits) | set(optimized_best_by_bits))
    best_map_by_bits = {layer: {} for layer in all_layers}

    print(f"[best_weights_by_bits] Source decision per bit-width for {model_name}:")
    for bits in all_bits:
        base = baseline_best_by_bits.get(bits)   # (acc1, fmt)
        opt = optimized_best_by_bits.get(bits)   # (acc1, {layer: fmt})
        base_fmt = base[1] if base else None

        if opt is not None:
            opt_layer_formats = opt[1]
            covered = sum(1 for layer in all_layers if layer in opt_layer_formats)
            for layer in all_layers:
                best_map_by_bits[layer][bits] = opt_layer_formats.get(layer) or base_fmt
            print(
                f"[best_weights_by_bits]   {bits}-bit: OPTIMIZED per-layer "
                f"({covered}/{len(all_layers)} layers; "
                f"fixed fallback={base_fmt or 'none'})"
            )
        elif base_fmt is not None:
            for layer in all_layers:
                best_map_by_bits[layer][bits] = base_fmt
            print(
                f"[best_weights_by_bits]   {bits}-bit: BASELINE fixed {base_fmt} "
                f"(no optimized run at this width)"
            )
        # else: neither source available — caller falls back to SIGNED_FORMATS_BY_BITS

    # Drop layers that received no bit-width assignments at all.
    best_map_by_bits = {layer: m for layer, m in best_map_by_bits.items() if m}
    if not best_map_by_bits:
        print(f"[best_weights_by_bits] No usable best-weight map for {model_name}")
        return None

    print(
        f"[best_weights_by_bits] Built best map for {model_name}: "
        f"{len(best_map_by_bits)} layers, bits={all_bits}, "
        f"baseline bits={sorted(baseline_best_by_bits)}, "
        f"optimized bits={sorted(optimized_best_by_bits)}"
    )
    return best_map_by_bits


def set_diq_globals(cache_sim_map, layer_input_bits, layer_residual_input_bits,
                    layer_need_input_transfer, input_format_policy):
    """Inject the per-(cache_size, b) maps the patched DynamicInputQuantizer reads."""
    for DIQ in (DIQ1, DIQ2):
        if DIQ is None:
            continue
        DIQ._global_cache_sim_map = cache_sim_map
        DIQ._global_layer_input_bits_map = layer_input_bits
        DIQ._global_layer_residual_input_bits_map = layer_residual_input_bits
        DIQ._global_layer_need_input_transfer_map = layer_need_input_transfer
        DIQ._global_layer_input_format_policy = input_format_policy


def build_eval_config(model_name, args, weights_path, b, layer_need_input_transfer):
    """Build the evaluation config for a materialized weights file.

    Mirrors the standard hybrid_quant pipeline (Quant wrappers + decomposed MHA +
    folded input norm) so the loaded state_dict's weight_fp8/weight_scale buffers
    are consumed by the same module graph. Inputs always use dynamic input quant.
    """
    return {
        'model': {'name': model_name, 'weights': os.path.abspath(weights_path)},
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
            'mode': args.mode,
            'max_batches': args.limit_batches,
            'compare_batches': args.compare_batches,
            'compare_mode': args.compare_mode,
            'dynamic_input_quant': {
                'enabled': True,
                'chunk_size': 128,
                'candidate_formats': get_input_formats_for_bits(b, policy=args.input_format_policy),
                'input_transfer_map': layer_need_input_transfer,
                'use_cache_sim_db': False,
                'collect_error_stats': False,
                'collect_format_stats': False,
                'unsigned_input_sources': ['relu', 'relu6', 'softmax', 'quantrelu', 'quantsoftmax', 'quantrelu6'],
            }
        },
        'dataset': {
            'name': args.dataset_name,
            'path': args.dataset_path,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        }
    }


def run_descent_for_cache(model, model_name, args, runner, sim_layers, cache_sim_map,
                          model_output_dir, temp_weights_dir):
    """Greedy weight-format descent for one cache size.

    Descends b = 8 -> 3. At each level, layers above b keep their already-decided
    format (frozen in `policy_by_bits`); layers newly at exactly b bits get their
    policy chosen by sweeping all fixed b-bit formats plus per-chunk MSE, keeping
    whichever maximizes full-model acc1.

    Returns
    -------
    points : dict   b -> (acc1, cycles)
    policy_by_bits : dict  bits -> winning policy (fixed fmt string or MSE_POLICY)
    per_level : dict  b -> {'layer_weight_bits': {...}, 'layer_formats': {...}}
    """
    points = {}
    policy_by_bits = {}
    per_level = {}

    for b in range(8, 2, -1):  # 8, 7, 6, 5, 4, 3
        print(f"\n------------------------------------------------------------")
        print(f"[descent] Cache stage | Bit-width level b={b}")
        print(f"------------------------------------------------------------")

        (
            cycles,
            layer_input_bits,
            layer_weight_bits,
            layer_output_bits,
            layer_residual_input_bits,
            layer_need_input_transfer,
        ) = compute_model_runtime(sim_layers, b, bandwidth=args.bandwidth)
        print(f"[descent]   compute time: {cycles:,} cycles")

        # DIQ input maps depend only on (cache, b); set once for all candidates.
        set_diq_globals(cache_sim_map, layer_input_bits, layer_residual_input_bits,
                        layer_need_input_transfer, args.input_format_policy)

        candidates = list(SIGNED_FORMATS_BY_BITS.get(b, [])) + [MSE_POLICY]
        best_policy = None
        best_acc1 = -1.0
        best_quant_map = None

        for cand in candidates:
            trial_policy = dict(policy_by_bits)
            trial_policy[b] = cand
            q_state_dict, quant_map = create_descent_state_dict(
                model, cache_sim_map, layer_weight_bits, trial_policy, chunk_size=128,
            )
            temp_weights_path = os.path.join(temp_weights_dir, f"weights_b_{b}_cand_{cand.replace(':', '_')}.pt")
            torch.save(q_state_dict, temp_weights_path)

            eval_config = build_eval_config(model_name, args, temp_weights_path, b, layer_need_input_transfer)
            try:
                eval_results = runner.run_single(eval_config, output_root=model_output_dir)
                acc1 = eval_results.get('acc1', 0.0)
            except Exception as e:
                print(f"[descent]   candidate {cand}: eval error: {e}")
                acc1 = 0.0
                import traceback
                traceback.print_exc()
            print(f"[descent]   b={b} candidate {cand!r}: Top-1 = {acc1:.3f}%")

            if os.path.exists(temp_weights_path):
                try:
                    os.remove(temp_weights_path)
                except OSError:
                    pass

            if acc1 > best_acc1:
                best_acc1 = acc1
                best_policy = cand
                best_quant_map = quant_map

        policy_by_bits[b] = best_policy
        points[b] = (best_acc1, cycles)
        per_level[b] = {
            'layer_weight_bits': dict(layer_weight_bits),
            'layer_formats': best_quant_map or {},
        }
        print(f"[descent]   => b={b} WINNER: {best_policy!r} (Top-1 = {best_acc1:.3f}%)")

    return points, policy_by_bits, per_level


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
    parser.add_argument("--mode", type=str, default="evaluate", choices=["evaluate", "compare"],
                        help="'evaluate' = accuracy only; 'compare' = layer-by-layer comparison vs FP32 reference")
    parser.add_argument("--compare_batches", type=int, default=50,
                        help="Number of batches to use in compare mode (-1 = full dataset)")
    parser.add_argument("--compare_mode", type=str, default="propagated", choices=["propagated", "isolated"],
                        help="In compare mode: 'propagated' = cumulative end-to-end divergence per layer; "
                             "'isolated' = teacher-forced per-layer output error (no inherited drift)")
    parser.add_argument("--cache_size", type=float, default=None,
                        help="Pin a single cache size (e.g. 4.0) instead of sweeping [0, 2, 4]")
    parser.add_argument("--cache_sizes", type=float, nargs="+", default=None,
                        help="Explicit list of cache sizes to run (e.g. --cache_sizes 0 2). "
                             "Overrides --cache_size and the default [0, 2, 4] sweep.")
    parser.add_argument("--b_bits", type=int, default=None,
                        help="Pin a single min bit-width (e.g. 8) instead of sweeping 2-8")
    parser.add_argument("--input_format_policy", choices=["all", "typed", "canonical"], default="all",
                        help="'all' searches all formats for each bit-width; 'typed' searches signed/unsigned formats separately; 'canonical' uses one balanced format per bit-width.")
    parser.add_argument("--use_best_weights", action="store_true",
                        help="Use the per-layer best weight formats from the latest weight_quant_optimized DB run instead of SIGNED_FORMATS_BY_BITS.")
    parser.add_argument("--descent", action="store_true",
                        help="Greedy weight-format descent: from 8 down to 3 bits, pick (by full-model acc1) the best weight policy "
                             "for layers newly limited to each width — among all fixed formats of that width plus a per-chunk-MSE option — "
                             "freezing the formats already decided for higher-bit layers.")
    args = parser.parse_args()

    if args.descent and args.use_best_weights:
        parser.error("--descent and --use_best_weights are mutually exclusive.")

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
            if args.descent:
                default_results_dir = "runspace/experiments/bandwidth_aware_quant/results_descent"
            elif args.use_best_weights:
                default_results_dir = "runspace/experiments/bandwidth_aware_quant/results_best_weights"
            else:
                default_results_dir = "runspace/experiments/bandwidth_aware_quant/results"
            model_output_dir = os.path.join(PROJECT_ROOT, default_results_dir, model_name)

        os.makedirs(model_output_dir, exist_ok=True)

        print(f"Results will be stored in: {model_output_dir}")

        # 1. Load the reference FP32 model and obtain weight state dict
        ref_config = build_bandwidth_fp32_config(args, model_name, model_weights)
        model_order_dir = os.path.join(model_output_dir, "ref_fp32_loader")
        os.makedirs(model_order_dir, exist_ok=True)
        model, _, _ = runner.prepare_model_with_materialized_weights(config=ref_config, output_dir=model_order_dir)
        model.to(args.device)

        # 2. Setup results tracking structures
        # Treat 0 as "not set" for b_bits so the dashboard default (0) triggers a sweep.
        if args.cache_sizes is not None:
            cache_sizes = list(args.cache_sizes)
        elif args.cache_size is not None:
            cache_sizes = [args.cache_size]
        else:
            cache_sizes = [0.0, 2.0, 4.0]
        # Descent mode always sweeps the full 8->3 range; --b_bits is ignored.
        effective_b_bits = None if args.descent else (args.b_bits if args.b_bits not in (None, 0) else None)
        min_bits_list = [effective_b_bits] if effective_b_bits is not None else [3]
        results_data = {min_bits: {cs: [] for cs in cache_sizes} for min_bits in min_bits_list}

        # Load best weight map from DB if requested
        best_weight_map_by_bits = None
        if args.use_best_weights:
            best_weight_map_by_bits = load_best_weight_map_by_bits(model_name)
            if best_weight_map_by_bits is None:
                print(f"[WARNING] --use_best_weights requested but no per-bit-width best weight map found for {model_name}. "
                      "Falling back to SIGNED_FORMATS_BY_BITS.")

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
        evaluated_points = {}  # (cache_size, b_bits) -> (accuracy, cycles)
        descent_data = {}      # cache_size -> {'policy_by_bits':..., 'per_level':...}

        temp_weights_dir = os.path.join(model_output_dir, "temp_weights")
        os.makedirs(temp_weights_dir, exist_ok=True)

        if args.descent:
            # Greedy descent owns its own per-candidate eval loop; the legacy
            # per-(cs,b) loop below is skipped (unique_runs empty).
            for cs in cache_sizes:
                print(f"\n============================================================")
                print(f"[descent] Cache: {cs}MB")
                print(f"============================================================")
                sim_layers, cache_sim_map = cache_sims[cs]
                points, policy_by_bits, per_level = run_descent_for_cache(
                    model, model_name, args, runner, sim_layers, cache_sim_map,
                    model_output_dir, temp_weights_dir,
                )
                for b, (acc1, cyc) in points.items():
                    evaluated_points[(cs, b)] = (acc1, cyc)
                descent_data[cs] = {'policy_by_bits': policy_by_bits, 'per_level': per_level}
            unique_runs = []
        else:
            unique_runs = []
            for cs in cache_sizes:
                b_range = [min_bits_list[0]] if effective_b_bits is not None else range(2, 9)
                for b in b_range:
                    unique_runs.append((cs, b))

            # Sort runs to keep cache size changes minimal (saves reloading models frequently)
            unique_runs = sorted(unique_runs, key=lambda x: (x[0], x[1]))

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
                layer_need_input_transfer,
            ) = compute_model_runtime(sim_layers, b, bandwidth=args.bandwidth)
            print(f"Calculated compute time: {cycles:,} cycles")

            # 3.2. Quantize model weights based on per-layer weight bit-width
            print("Quantizing model weights...")
            q_state_dict, quant_map = create_bandwidth_aware_state_dict(
                model, cache_sim_map, per_layer_weight_bits=layer_weight_bits,
                chunk_size=128, best_weight_map_by_bits=best_weight_map_by_bits,
            )
            
            # Save quantized weights to temporary file
            temp_weights_path = os.path.join(temp_weights_dir, f"weights_cs_{cs}_b_{b}.pt")
            torch.save(q_state_dict, temp_weights_path)
            
            # 3.3. Inject cache sim map and per-layer input bits to global attributes of both classes
            if DIQ1 is not None:
                DIQ1._global_cache_sim_map = cache_sim_map
                DIQ1._global_layer_input_bits_map = layer_input_bits
                DIQ1._global_layer_residual_input_bits_map = layer_residual_input_bits
                DIQ1._global_layer_need_input_transfer_map = layer_need_input_transfer
                DIQ1._global_layer_input_format_policy = args.input_format_policy
            if DIQ2 is not None:
                DIQ2._global_cache_sim_map = cache_sim_map
                DIQ2._global_layer_input_bits_map = layer_input_bits
                DIQ2._global_layer_residual_input_bits_map = layer_residual_input_bits
                DIQ2._global_layer_need_input_transfer_map = layer_need_input_transfer
                DIQ2._global_layer_input_format_policy = args.input_format_policy

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
                    'mode': args.mode,
                    'max_batches': args.limit_batches,
                    'compare_batches': args.compare_batches,
                    'compare_mode': args.compare_mode,
                    'dynamic_input_quant': {
                        'enabled': True,
                        'chunk_size': 128,
                        'candidate_formats': get_input_formats_for_bits(b, policy=args.input_format_policy),
                        'input_transfer_map': layer_need_input_transfer,
                        'use_cache_sim_db': False,
                        'collect_error_stats': False,
                        'collect_format_stats': False,
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

            # Save weight-choice verification for this (cache_size, b) run
            verification_path = os.path.join(model_output_dir, f"weight_choice_verification_cs{cs}_b{b}.json")
            with open(verification_path, 'w') as f:
                json.dump(quant_map, f, indent=4, sort_keys=True)
            # Only echo the per-layer choices when running with best weights —
            # otherwise these [best_weights] lines are misleading for the default
            # MSE-optimized path. The JSON is always saved for inspection.
            if args.use_best_weights:
                print(f"[best_weights] Saved weight choice verification to {verification_path}")
                for layer, fmt in sorted(quant_map.items()):
                    print(f"[best_weights]   final choice: {layer} -> {fmt}")

            # Cleanup temporary weight file (skip in compare mode — runner may need it)
            if args.mode != 'compare' and os.path.exists(temp_weights_path):
                try:
                    os.remove(temp_weights_path)
                except OSError:
                    pass

        # 4. Map evaluated points to the results_data sweeps structure
        for min_bits in min_bits_list:
            for cs in cache_sizes:
                b_range = [effective_b_bits] if effective_b_bits is not None else range(min_bits, 9)
                for b in b_range:
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
        if best_weight_map_by_bits is not None:
            serializable_results['best_weight_map_by_bits'] = best_weight_map_by_bits
            serializable_results['used_best_weights'] = True
        if args.descent:
            serializable_results['used_descent'] = True
            serializable_results['descent'] = {
                str(cs): {
                    'policy_by_bits': {str(bits): pol for bits, pol in data['policy_by_bits'].items()},
                    'per_level': {
                        str(b): {
                            'layer_weight_bits': lvl['layer_weight_bits'],
                            'layer_formats': lvl['layer_formats'],
                        }
                        for b, lvl in data['per_level'].items()
                    },
                }
                for cs, data in descent_data.items()
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
