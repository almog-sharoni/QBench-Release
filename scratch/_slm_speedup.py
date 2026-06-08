"""
Bandwidth/cache speedup analysis for the SLM (opt-125m), reusing the
asic_cache_simulation + bandwidth_aware_quant cycle model. No inference is run —
only a one-shot shape trace + the analytical cycle/transfer model.

speedup(cs) = ref_cycles(FP32 @ 0MB cache) / cycles(cs, b_bits)   [exactly the
experiment's normalization: FP32 0MB = 1.0x].
"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from runspace.experiments.asic_cache_simulation.simulate_cache import (
    analyze_model, get_footprint_elements, round_to_banks, evaluate_stay,
    optimize_layer_bits,
)
from runspace.experiments.asic_cache_simulation.rules import RULES
from runspace.experiments.bandwidth_aware_quant.bandwidth_aware_quant import compute_model_runtime

B_BITS    = int(os.environ.get("B_BITS", "8"))     # off-chip transfer bit-width
BATCH     = int(os.environ.get("BATCH", "1"))
SEQ       = int(os.environ.get("SEQ", "512"))
BANDWIDTH = float(os.environ.get("BANDWIDTH", "1.0"))
NUM_BANKS = 16
METADATA_BITS = 0
CACHE_SIZES_M = [0.0, 2.0, 4.0]   # millions of elements (~MB at 1 byte/elem)
VOCAB = 50000
DEVICE = "cuda"

model_cfg = {"name": "facebook/opt-125m", "source": "huggingface"}
adapter_cfg = {"type": "slm", "build_quantized": True,
               "quantized_ops": ["all"], "input_quantization": True}

dummy_ids = torch.randint(0, VOCAB, (BATCH, SEQ), dtype=torch.long, device=DEVICE)


def stay_eval_layers(layers, cache_elements, bank_size):
    """Replicates run_cache_simulation's per-layer stay evaluation."""
    out = []
    for i, layer in enumerate(layers):
        next_layer = layers[i + 1] if i + 1 < len(layers) else None
        we = get_footprint_elements(layer['weight_elems'], METADATA_BITS)
        ie = get_footprint_elements(layer['input_elems'],  METADATA_BITS)
        oe = get_footprint_elements(layer['output_elems'], METADATA_BITS)
        nxe = get_footprint_elements(next_layer['input_elems'], METADATA_BITS) if next_layer else 0
        ctx = {
            'input_banked':  round_to_banks(ie, bank_size),
            'output_banked': round_to_banks(oe, bank_size),
            'weight_banked': round_to_banks(we, bank_size),
            'next_xin_banked': round_to_banks(nxe, bank_size),
            'cache_elements': cache_elements, 'bank_size': bank_size,
            'filter_height': layer.get('filter_height', 0),
            'filter_width': layer.get('filter_width', 0),
            'in_channels': layer.get('in_channels', 0),
            'out_channels': layer.get('out_channels', 0),
            'input_channel_height': layer.get('input_channel_height', 0),
            'input_channel_width': layer.get('input_channel_width', 0),
            'output_channel_height': layer.get('output_channel_height', 0),
            'output_channel_width': layer.get('output_channel_width', 0),
            'jump_back_size_in_banks': layer.get('jump_back_size_in_banks', 0),
        }
        stay, _perm, _ok, rule = evaluate_stay(layer, ctx, next_layer, METADATA_BITS,
                                               bank_size, cache_elements)
        lc = dict(layer)
        lc['stay_on_chip'] = stay
        lc['xin_from_cache'] = RULES.get(rule, {}).get('xin_from_cache', True)
        lc['rule'] = rule
        out.append(lc)
    return out


def fp32_ref_cycles(sim_layers):
    """FP32 reference: 32-bit transfers everywhere (min=max=32)."""
    total, prev_stay = 0.0, False
    for idx, layer in enumerate(sim_layers):
        stay = layer.get('stay_on_chip', False)
        need_in = (idx == 0 or not prev_stay or not layer.get('xin_from_cache', True))
        need_out = not stay
        _, _, _, cyc = optimize_layer_bits(layer, BANDWIDTH, need_in, True, need_out,
                                           min_bits=32, max_bits=32)
        total += cyc
        prev_stay = stay
    return total


print(f"SLM bandwidth/cache speedup — opt-125m | batch={BATCH} seq={SEQ} | "
      f"b={B_BITS}-bit off-chip | bandwidth={BANDWIDTH} B/cyc | banks={NUM_BANKS}")

rows = []
ref_cycles_per_cs, quant_cycles_per_cs, onchip_counts = {}, {}, {}
for cs in CACHE_SIZES_M:
    cache_elements = int(cs * 1_000_000)
    bank_size = cache_elements // NUM_BANKS if NUM_BANKS > 0 else 0
    layers = analyze_model(model_cfg, batch_size=BATCH, device=DEVICE,
                           adapter_cfg=adapter_cfg, cache_elements=cache_elements,
                           bank_size=bank_size, metadata_bits=METADATA_BITS,
                           dummy_input=dummy_ids)
    sim_layers = stay_eval_layers(layers, cache_elements, bank_size)
    on_chip = sum(1 for l in sim_layers if l['stay_on_chip'])
    onchip_counts[cs] = (on_chip, len(sim_layers))
    ref_cycles_per_cs[cs] = fp32_ref_cycles(sim_layers)
    quant_cycles_per_cs[cs] = compute_model_runtime(sim_layers, B_BITS, BANDWIDTH)[0]

baseline = ref_cycles_per_cs[0.0]   # FP32 @ 0MB == 1.0x

print(f"\nFP32 0MB baseline cycles: {baseline:,.0f}\n")
hdr = f"{'Cache':>7} | {'on-chip':>10} | {'FP32 cyc':>14} | {'b8 cyc':>14} | {'FP32 cache-only':>16} | {'b8 speedup':>11}"
print(hdr); print("-" * len(hdr))
for cs in CACHE_SIZES_M:
    on, tot = onchip_counts[cs]
    fp32_x = baseline / ref_cycles_per_cs[cs]
    b8_x = baseline / quant_cycles_per_cs[cs]
    print(f"{cs:>5}MB | {on:>3}/{tot:<6} | {ref_cycles_per_cs[cs]:>14,.0f} | "
          f"{quant_cycles_per_cs[cs]:>14,.0f} | {fp32_x:>15.2f}x | {b8_x:>10.2f}x")
