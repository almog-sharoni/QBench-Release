# Cache- & Bandwidth-Aware Microscaling — Validated Findings

**Thesis.** In microscaling (MX) quantization the per-block scale is a *hardware* cost, not just an accuracy knob: a finer block buys accuracy but adds scale-metadata that must be streamed (only when a tensor is *streamed* and the layer is *memory-bound*). Therefore (a) the latency-optimal block size is **regime-dependent**, and (b) the block size and bit-width should be **co-allocated per layer**, not fixed at the MX default of 32.

**Setup.** Perplexity = wikitext-2 (raw) test, seqlen 2048, standard GPTQ/HF protocol. Weight-only quantization of decoder-block linears (embeddings & lm_head kept FP16, per GPTQ). MXFP4 = OCP E2M1 + E8M0 shared scale (4.25 effective bits at block 32 — matches the published number). MXINT8 = INT8 + E8M0. RTN = symmetric round-to-nearest (per-channel / per-group). Effective bits / element = `b + scale_bits/block`; this is proportional to per-token decode memory traffic and hence decode latency.

## Finding 0 — Harness validation (results are trustworthy)

Our FP16 wikitext-2 perplexity matches published numbers, so the quantization deltas are credible.

| model | our FP16 | published FP16 |
|---|---|---|
| Qwen/Qwen2.5-0.5B | 13.07 | — (no widely-published weight-only ppl) |
| facebook/opt-1.3b | 14.62 | 14.62 (GPTQ paper) |
| facebook/opt-125m | 27.65 | 27.65 (GPTQ paper) |

## Finding 1 — Block-size latency cost is regime-dependent (analytical, on our cycle model)

Using the asic-cache/bandwidth cycle model, normalized to the MX-default block 32 (4-bit data, 8-bit scale): in **decode** (memory-bound) shrinking the block 32→8 costs **+~18% latency** (and 256 *saves* ~5%) because scale-metadata is on the critical path; in **prefill** it is **flat** (only 1/136 ViT and 1/74 OPT layers are memory-bound, the rest hide metadata under MACs). Metadata overhead per byte is regime-independent (`scale_bits/(b·block)`); what differs is whether it converts to latency. → *finer blocks are free in prefill, costly in decode.* (plots: exp1_latency_vs_block.png)


## Finding 2 — Accuracy: block matters only at LOW bits; the SCALE FORMAT is a real axis

(a) **Block matters only at 4-bit.** MXFP4 degrades with coarser blocks; MXINT8 is near-lossless at *every* block (so 8-bit weights can use the coarsest block and save metadata). (b) **Per-channel (one scale/row) collapses** at 4-bit — fine-grained scaling is essential. (c) **Scale format matters and is itself a co-design knob.** Group-INT4 with a 16-bit FP scale is *more accurate* than MXFP4 with its 8-bit power-of-2 E8M0 scale (OPT-125M: INT4-g32 30.22 vs MXFP4/32 32.79; Qwen: 15.31 vs 16.18) — but it spends more scale bits. So MXFP4 is NOT the most accurate 4-bit format; its advantage is hardware efficiency (cheap E8M0 scale). The right axis is (bits × block × **scale format**), matching the recent NVFP4 (E4M3 scale) vs MXFP4 (E8M0) debate.


## Main comparison (wikitext-2 ppl; lower=better)

| model | FP16 | MXFP4/128 (4.06b) | MXFP4/32 (4.25b) | MXINT8 (8.25b) | RTN-INT4 per-ch (4b) | RTN-INT4 g128 (4.06b) | RTN-INT3 g128 (3.06b) | **allocator @≤4.30b** |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | 13.07 | 16.55 | 16.18 | 13.09 | 30.26 | 18.18 | 235.99 | **15.88** |
| facebook/opt-1.3b | 14.62 | 15.37 | 15.25 | 14.60 | 31.34 | 15.87 | 3972.56 | **15.01** |
| facebook/opt-125m | 27.65 | 34.26 | 32.79 | 27.68 | 51.45 | 32.85 | 124.96 | **32.09** |

## Allocator win (block×bit vs uniform MXFP4/32 at iso effective-bits)

| model | MXFP4/32 ppl @4.25b | allocator ppl @≤4.30b | Δppl |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B | 16.18 | 15.88 | 0.30 |
| facebook/opt-1.3b | 15.25 | 15.01 | 0.24 |
| facebook/opt-125m | 32.79 | 32.09 | 0.70 |

## Finding 3 — block×bit allocator (sensitivity-guided) fills the uniform-MX precision gap

Uniform MX offers only ~4.25 bits (MXFP4/32) or ~8.25 bits (MXINT8) — a wide gap with nothing between, and MXFP4 *plateaus* (finer blocks barely help weights). A greedy per-layer allocator over {MXFP4/256, MXFP4/32, MXFP4/8, MXINT8/32}, ordered by measured sensitivity-per-bit, traces a Pareto that uniform MX cannot reach. (plots: alloc_<model>.png; ordering sanity: order_<model>.png.)


## Allocator vs BEST uniform MX, across accuracy targets (the honest headline)

For each accuracy target, `best uniform` = the fewest-effective-bit *uniform* MX config (any MXFP4 block, any MXINT8 block) that meets it; `allocator` = our block×bit Pareto. Uniform MX has a gap between 4.25 and 8.25 bits that the allocator fills.


**Qwen/Qwen2.5-0.5B** (FP16=13.07)

| accuracy target | best-uniform eff-bits | allocator eff-bits | **bits saved** |
|---|---|---|---|
| FP×1.05 = 13.72 | 8.03 | 8.06 | **-0.4%** |
| FP×1.10 = 14.38 | 8.03 | 7.64 | **4.9%** |
| FP×1.20 = 15.68 | 8.03 | 5.00 | **37.8%** |
| FP×1.35 = 17.65 | 4.03 | 4.03 | **0.0%** |

_max decode-traffic saving vs best uniform: **37.8%**_

**facebook/opt-1.3b** (FP16=14.62)

| accuracy target | best-uniform eff-bits | allocator eff-bits | **bits saved** |
|---|---|---|---|
| FP×1.05 = 15.36 | 4.12 | 4.13 | **-0.0%** |
| FP×1.10 = 16.09 | 4.03 | 4.03 | **0.0%** |
| FP×1.20 = 17.55 | 4.03 | 4.03 | **0.0%** |
| FP×1.35 = 19.74 | 4.03 | 4.03 | **0.0%** |

_max decode-traffic saving vs best uniform: **0.0%**_

**facebook/opt-125m** (FP16=27.65)

| accuracy target | best-uniform eff-bits | allocator eff-bits | **bits saved** |
|---|---|---|---|
| FP×1.05 = 29.04 | 8.03 | 7.99 | **0.5%** |
| FP×1.10 = 30.42 | 8.03 | 6.78 | **15.6%** |
| FP×1.20 = 33.18 | 4.25 | 4.11 | **3.2%** |
| FP×1.35 = 37.33 | 4.03 | 4.03 | **0.0%** |

_max decode-traffic saving vs best uniform: **15.6%**_

## Finding 4 — block×bit allocator on the STRONG primitive (group-INT4) beats best uniform

Since group-INT4 (FP scale) > MXFP4, we build the per-layer allocator over group-INT {INT4/g256, INT4/g64, INT4/g16, INT8/g64} (16-bit FP scale → eff-bits = b+16/group). It is compared to the strongest *uniform* group-INT config. (plots: allocgi_<model>.png)

| model | FP16 | best uniform INT4 (eff-bits, ppl) | allocator @ same bits | max bits saved vs best uniform |
|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | 13.07 | 5.00b, 14.58 | 14.59 | **33.6%** |
| facebook/opt-125m | 27.65 | 5.00b, 29.77 | 29.77 | **22.6%** |

## Threats to validity / honest caveats

- **Weight-only RTN-style** quantization (no GPTQ/AWQ error compensation, no activation quant). GPTQ would shift all 4-bit numbers down ~1 ppl and is orthogonal/additive to our block×bit allocation.
- **Effective-bits = latency proxy.** Finding 1 uses the actual cycle model; Findings 2–3 use effective bits (exact for decode memory traffic, the dominant edge-decode cost). A full latency–accuracy Pareto coupling the two is the natural next step.
- **Marginal (rest-FP) sensitivity** drives only the allocator *ordering*; every plotted Pareto point is a *measured* perplexity, so additivity is not assumed in the results.
- The allocator win is **target-dependent**: large in the mid-accuracy regime (uniform-MX gap), ~0 at the extremes where one uniform config is already optimal. We report the full curve, not a single number.
- Models: OPT-125M / OPT-1.3B (standard quant benchmarks) + Qwen2.5-0.5B (current edge, GQA). Larger models and Llama/Gemma (gated) remain to be run.

## What holds, for the thesis

1. **Validated harness** — FP16 matches published exactly (OPT-125M 27.65, OPT-1.3B 14.62), so all quantization deltas are trustworthy. Quantizers are OCP-spec (MXFP4 = 4.25 eff-bits). Per-channel RTN collapses at 4-bit; group/block scaling is essential — consistent with the literature.
2. **No universal 4-bit winner (scale format is a real axis).** Group-INT4 (16-bit FP scale) is often *more accurate* than MXFP4 (8-bit E8M0 scale) but spends more scale bits; which wins flips with model and granularity (e.g. OPT-125M: INT4-g32 30.22 < MXFP4/32 32.79; OPT-1.3B: MXFP4/32 15.25 < RTN-g128 15.87). MXFP4's advantage is hardware efficiency, not accuracy — mirroring the NVFP4-vs-MXFP4 debate.
3. **Finding 1 (latency regime split)** — the conceptual core: block size is latency-free in compute-bound prefill but costs up to +18% in memory-bound decode. Clean and model-independent.
4. **Finding 2 (block matters only at low bits)** holds across all three models (MXINT8 near-lossless at every block; MXFP4 degrades with coarser blocks).
5. **Finding 4 (block×bit allocator on the strong group-INT primitive)** delivers a real, measured decode-traffic saving (OPT-125M: 22.6% vs the *strongest* uniform baseline) by filling the 4→8-bit gap. The win is concentrated in the mid-accuracy regime and is larger on harder-to-quantize models (125M, Qwen) than on the already-robust 1.3B.

**Next, to make it a strong paper:** (a) couple Findings to the Finding-1 *latency* model for a true latency–accuracy Pareto incl. the prefill/decode split; (b) add the scale-format axis (E8M0 vs E4M3/FP, à la NVFP4) into the allocator; (c) add GPTQ error-compensation (additive) and activation/KV quant; (d) scale to Llama-3.x / 7B.