"""
Edge-LLM DECODE roofline (analytical — no weights, no inference).

Decode (autoregressive, batch=1, 1 token/step) is memory-bound: per token the
accelerator streams every weight once and re-reads the whole KV cache. Compute
is negligible (1 token). So time/token ≈ DRAM_bytes / bandwidth.

per-token DRAM(b, C, Lc) = weights@b + KV@b(context Lc), minus whatever fits in
the on-chip cache C (weights cached first — reused every token). Speedup is vs
the FP16, no-cache baseline. MB = 1e6 bytes (matches the repo's element≈byte
convention).
"""
import os
from transformers import AutoConfig

CTX        = int(os.environ.get("CTX", "2048"))     # context length for KV
B_BITS     = int(os.environ.get("B_BITS", "8"))     # quantized decode bit-width
BASE_BITS  = 16                                      # FP16 baseline (edge native)
CACHES_MB  = [0.0, 2.0, 4.0]
BW_GBs     = float(os.environ.get("BW_GBS", "100")) # edge LPDDR5x ~100 GB/s

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
# Gated (config not fetchable without auth) — public architecture specs hardcoded.
HARDCODED = {
    "meta-llama/Llama-3.2-1B (specs)": dict(h=2048, L=16, heads=32, kv=8, hd=64,
                                            ffn=8192, vocab=128256, tie=True),
}


def specs_from_config(c):
    h = c.hidden_size
    heads = c.num_attention_heads
    return dict(
        h=h, L=c.num_hidden_layers, heads=heads,
        kv=getattr(c, "num_key_value_heads", heads),
        hd=getattr(c, "head_dim", h // heads),
        ffn=c.intermediate_size, vocab=c.vocab_size,
        tie=getattr(c, "tie_word_embeddings", False),
    )


def roofline(s):
    h, L, heads, kv, hd, ffn, vocab = (s[k] for k in ("h", "L", "heads", "kv", "hd", "ffn", "vocab"))
    qo = h * (heads * hd) + (heads * hd) * h
    kvp = 2 * h * (kv * hd)
    mlp = 3 * h * ffn                      # SwiGLU: gate + up + down
    per_layer = qo + kvp + mlp
    W = L * per_layer + vocab * h          # + lm_head projection (streamed per token)
    KV = L * 2 * CTX * (kv * hd)           # K and V over the whole context
    return W, KV


def per_token_dram(W, KV, bits, cache_bytes):
    w_b = W * bits / 8
    kv_b = KV * bits / 8
    w_dram = max(0.0, w_b - cache_bytes)            # weights cached first (reused every token)
    rem = max(0.0, cache_bytes - w_b)
    kv_dram = max(0.0, kv_b - rem)
    return w_dram + kv_dram


def load_all():
    out = []
    for mid in MODELS:
        try:
            out.append((mid.split("/")[-1], specs_from_config(AutoConfig.from_pretrained(mid))))
        except Exception as e:
            print(f"skip {mid}: {type(e).__name__}")
    for name, s in HARDCODED.items():
        out.append((name, s))
    return out


B_LIST = [3, 4, 5, 6, 7, 8]
models = load_all()
print(f"\nEdge-LLM DECODE roofline | context={CTX} | b-sweep {B_LIST} vs FP16 | "
      f"C=0 (2-4MB cache is negligible here) | BW={BW_GBs} GB/s")

# 1) tokens/sec matrix (model x bit-width), plus FP16
print("\n--- tokens/sec (BW={:.0f} GB/s) ---".format(BW_GBs))
hdr = f"{'model':<26} | {'params':>6} | {'FP16':>6} | " + " | ".join(f"{'b'+str(b):>5}" for b in B_LIST)
print(hdr); print("-" * len(hdr))
for name, s in models:
    W, KV = roofline(s)
    fp16_toks = BW_GBs * 1e9 / per_token_dram(W, KV, BASE_BITS, 0.0)
    cells = []
    for b in B_LIST:
        toks = BW_GBs * 1e9 / per_token_dram(W, KV, b, 0.0)
        cells.append(f"{toks:>5.0f}")
    print(f"{name:<26} | {W/1e9:>5.2f}B | {fp16_toks:>6.0f} | " + " | ".join(cells))

# 2) speedup vs FP16 (uniform across models when cache is negligible -> 16/b)
print("\n--- speedup vs FP16 (memory-bound -> 16/b, ~identical across models) ---")
print("  " + " | ".join(f"b{b}: {16.0/b:.2f}x" for b in B_LIST))

# 3) cache sweep 0..20 MB (step 2): speedup vs FP16 as cache grows
CACHE_SWEEP = [c * 2.0 for c in range(0, 11)]   # 0,2,...,20 MB
for b in (8, 4):
    print(f"\n--- speedup vs FP16 at b={b}, cache 0..20 MB ---")
    hdr = f"{'model':<24} | " + " | ".join(f"{int(c):>4}" for c in CACHE_SWEEP) + "   | gain@20MB"
    print(hdr); print("-" * len(hdr))
    for name, s in models:
        W, KV = roofline(s)
        base = per_token_dram(W, KV, BASE_BITS, 0.0)
        vals = [base / per_token_dram(W, KV, b, c * 1e6) for c in CACHE_SWEEP]
        gain = (vals[-1] / vals[0] - 1.0) * 100.0
        print(f"{name[:24]:<24} | " + " | ".join(f"{v:>4.2f}" for v in vals) + f"   | +{gain:.1f}%")

# 4) what cache size is needed to actually hold weights / KV (the real thresholds)
print("\n--- thresholds @b=8, ctx={} : 20MB as a fraction of decode traffic ---".format(CTX))
for name, s in models:
    W, KV = roofline(s)
    total_b8 = (W + KV) * 8 / 8          # bytes
    frac = 20e6 / total_b8 * 100
    print(f"  {name[:24]:<24}: KV={KV/1e6:6.1f} MB | weights={W/1e6:6.0f} MB "
          f"| 20MB = {frac:4.1f}% of total decode bytes")
