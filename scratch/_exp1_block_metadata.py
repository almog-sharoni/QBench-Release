"""
Exp 1 — Cache/bandwidth-aware microscaling: does the latency-optimal block size
depend on cache residency / regime?

Microscaling shares 1 scale per block. Finer block -> more scales -> more
metadata bytes that must be STREAMED with the tensor (free if the tensor is
cache-resident). Latency per layer = max(compute, (data+metadata)/BW), so the
metadata penalty only bites where a tensor is BOTH streamed AND memory-bound.

We sweep block size for three regimes and report latency normalized to block=128
(the MX-ish default) + the metadata share of streamed bytes. No inference, no
training — pure cost model on the existing simulator.
"""
import os, sys, math
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from runspace.experiments.asic_cache_simulation.simulate_cache import (
    analyze_model, _compute_layer_cycles, get_footprint_elements,
    round_to_banks, evaluate_stay,
)
from runspace.experiments.asic_cache_simulation.rules import RULES

B          = int(os.environ.get("B_BITS", "4"))     # data bit-width (MXFP4 regime)
SCALE_BITS = int(os.environ.get("SCALE_BITS", "8")) # E8M0 scale = 8 bits/block
BW         = 1.0
BLOCKS     = [256, 128, 64, 32, 16, 8]
CACHE_MB   = float(os.environ.get("CACHE_MB", "4.0"))
CTX        = int(os.environ.get("CTX", "2048"))
DEVICE     = "cuda"


def stream_meta_cycles(elems, block):
    """metadata transfer cycles for a streamed tensor (1 scale / block)."""
    if elems <= 0:
        return 0.0
    return math.ceil(elems / block) * (SCALE_BITS / 8) / BW


def stay_eval(layers, cache_elements, bank_size):
    out, prev = [], False
    for i, layer in enumerate(layers):
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        ie = get_footprint_elements(layer['input_elems'], 0)
        oe = get_footprint_elements(layer['output_elems'], 0)
        we = get_footprint_elements(layer['weight_elems'], 0)
        nxe = get_footprint_elements(nxt['input_elems'], 0) if nxt else 0
        ctx = {'input_banked': round_to_banks(ie, bank_size),
               'output_banked': round_to_banks(oe, bank_size),
               'weight_banked': round_to_banks(we, bank_size),
               'next_xin_banked': round_to_banks(nxe, bank_size),
               'cache_elements': cache_elements, 'bank_size': bank_size,
               'jump_back_size_in_banks': layer.get('jump_back_size_in_banks', 0)}
        for k in ('filter_height','filter_width','in_channels','out_channels',
                  'input_channel_height','input_channel_width',
                  'output_channel_height','output_channel_width'):
            ctx[k] = layer.get(k, 0)
        stay, _p, _ok, rule = evaluate_stay(layer, ctx, nxt, 0, bank_size, cache_elements)
        lc = dict(layer); lc['stay_on_chip'] = stay
        lc['xin_from_cache'] = RULES.get(rule, {}).get('xin_from_cache', True)
        out.append(lc)
    return out


def prefill_latency(model_cfg, adapter_cfg, dummy, block):
    """Sum of per-layer max(compute, data+metadata transfer) over a prefill pass."""
    cache_elements = int(CACHE_MB * 1e6)
    bank = cache_elements // 16
    layers = analyze_model(model_cfg, batch_size=1, device=DEVICE, adapter_cfg=adapter_cfg,
                           cache_elements=cache_elements, bank_size=bank,
                           metadata_bits=0, dummy_input=dummy)
    sim = stay_eval(layers, cache_elements, bank)
    total = data_b = meta_b = 0.0
    on_chip = 0
    prev = False
    for i, l in enumerate(sim):
        compute = _compute_layer_cycles(l)
        stay = l['stay_on_chip']; on_chip += int(stay)
        need_in = (i == 0 or not prev or not l['xin_from_cache'])
        streamed = [(l.get('weight_elems', 0), True),
                    (l.get('input_elems', 0),  need_in),
                    (l.get('output_elems', 0), not stay)]
        data = meta = 0.0
        for e, s in streamed:
            if s and e > 0:
                data += e * B / 8 / BW
                meta += stream_meta_cycles(e, block)
        total += max(compute, data + meta)
        data_b += data; meta_b += meta
        prev = stay
    return total, (meta_b / (data_b + meta_b) if data_b + meta_b else 0.0), on_chip, len(sim)


def decode_latency(W, KV, block):
    """Decode is memory-bound: latency = (weights+KV) data + metadata, all streamed."""
    data = (W + KV) * B / 8 / BW
    meta = (math.ceil(W / block) + math.ceil(KV / block)) * (SCALE_BITS / 8) / BW
    return data + meta, meta / (data + meta)


def edge_roofline(mid):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(mid)
    h, L, heads = c.hidden_size, c.num_hidden_layers, c.num_attention_heads
    kv = getattr(c, "num_key_value_heads", heads)
    hd = getattr(c, "head_dim", h // heads); ffn = c.intermediate_size; vocab = c.vocab_size
    W = L * (h*(heads*hd) + (heads*hd)*h + 2*h*(kv*hd) + 3*h*ffn) + vocab*h
    KV = L * 2 * CTX * (kv * hd)
    return W, KV


def report(title, fn):
    print(f"\n=== {title} ===  (data b={B}-bit, scale={SCALE_BITS}-bit/block)")
    base = None
    rows = []
    for blk in BLOCKS:
        lat, metafrac = fn(blk)
        rows.append((blk, lat, metafrac))
        if blk == 128:
            base = lat
    print(f"{'block':>6} | {'norm latency (vs 128)':>21} | {'metadata % of streamed':>22}")
    print("-" * 56)
    for blk, lat, mf in rows:
        print(f"{blk:>6} | {lat/base:>20.3f}x | {mf*100:>21.1f}%")


# --- PREFILL regimes ---
report("ViT-B/16 prefill (cache 4MB)",
       lambda blk: prefill_latency({"name": "vit_b_16", "weights": None},
                                   {"type": "generic", "build_quantized": True}, None, blk)[:2])

opt_ids = torch.randint(0, 50000, (1, 512), dtype=torch.long, device=DEVICE)
report("OPT-125M prefill seq=512 (cache 4MB)",
       lambda blk: prefill_latency({"name": "facebook/opt-125m", "source": "huggingface"},
                                   {"type": "slm", "build_quantized": True,
                                    "quantized_ops": ["all"], "input_quantization": True},
                                   opt_ids, blk)[:2])

# --- DECODE regime ---
for mid in ["Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3.5-mini-instruct"]:
    W, KV = edge_roofline(mid)
    report(f"DECODE {mid.split('/')[-1]} ctx={CTX}", lambda blk, W=W, KV=KV: decode_latency(W, KV, blk))
