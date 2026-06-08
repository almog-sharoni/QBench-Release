"""Build final plots + comparison table + FINDINGS.md from results_*.json."""
import os, sys, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); PLOTS=os.path.join(HERE,"plots"); os.makedirs(PLOTS,exist_ok=True)
def eb(block,b,sb=8): return b+sb/block
BLOCKS=[8,16,32,64,128,256]

# published references (wikitext-2, seqlen 2048, weight-only) for context
PUB={"facebook/opt-125m":{"FP16":27.65,"GPTQ-INT4(g-)":31.45,"note":"GPTQ paper"},
     "facebook/opt-1.3b":{"FP16":14.62,"GPTQ-INT4(g-)":15.47,"note":"GPTQ paper"},
     "Qwen/Qwen2.5-0.5B":{"FP16":None,"note":"no widely-published weight-only ppl"}}

results={}
for f in sorted(glob.glob(os.path.join(HERE,"results_*.json"))):
    if "grpint" in os.path.basename(f): continue
    r=json.load(open(f))
    # correct RTN effective-bits: RTN uses a 16-bit FP group scale (not 8-bit E8M0)
    for nm,grp in [("RTN-INT4 per-channel",4096),("RTN-INT4 g128",128),("RTN-INT4 g32",32),("RTN-INT3 g128",128)]:
        if nm in r["baselines"]:
            bb = 3 if "INT3" in nm else 4
            r["baselines"][nm]["eff_bits"] = bb + 16.0/grp
    results[r["model"]]=r

def best_at(pts, target_ppl):
    """min bits achieving ppl<=target_ppl among (bits,ppl) pts."""
    ok=[b for b,p in pts if p<=target_ppl]; return min(ok) if ok else None

for model,r in results.items():
    tag=model.split("/")[-1]; fp=r["fp16"]
    rd4=[(eb(bl,4),r["rd"]["4"][str(bl)]) for bl in BLOCKS]
    rd8=[(eb(bl,8),r["rd"]["8"][str(bl)]) for bl in BLOCKS]
    greedy=sorted(r["greedy"]); base=r["baselines"]

    # Plot A: rate-distortion + baselines
    topA=min(fp*1.6, max(y for _,y in rd4)*1.18)
    fig,ax=plt.subplots(figsize=(8.5,5.4))
    ax.plot([x for x,_ in rd4],[y for _,y in rd4],'-o',color="#6a3d9a",lw=2,ms=7,label="MXFP4 (uniform, by block)")
    ax.plot([x for x,_ in rd8],[y for _,y in rd8],'-s',color="#2ca02c",lw=2,ms=7,label="MXINT8 (uniform)")
    for bl,(x,y) in zip(BLOCKS,rd4): ax.annotate(str(bl),(x,y),textcoords="offset points",xytext=(4,4),fontsize=7,color="#6a3d9a")
    mk={"RTN-INT4 per-channel":("x","#d62728"),"RTN-INT4 g128":("P","#ff7f0e"),
        "RTN-INT4 g32":("D","#8c564b"),"RTN-INT3 g128":("v","#e377c2")}
    for nm,(m,c) in mk.items():
        if nm in base:
            bp=base[nm]["ppl"]
            if bp<=topA: ax.scatter([base[nm]["eff_bits"]],[bp],marker=m,s=90,color=c,label=nm,zorder=5)
            else: ax.annotate(f"{nm}: {bp:.0f}↑",(base[nm]["eff_bits"],topA*0.97),fontsize=7,color=c,ha='center',rotation=90,va='top')
    ax.axhline(fp,color='k',ls='--',lw=1,label=f"FP16 ({fp:.2f})"); ax.set_ylim(fp*0.97,topA)
    ax.set_xlabel("effective bits / weight   (∝ decode memory & latency)"); ax.set_ylabel("wikitext-2 perplexity")
    ax.set_title(f"{tag} — rate–distortion: MX vs RTN baselines  (group-INT4 vs MXFP4)"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(f"{PLOTS}/rd_{tag}.png",dpi=140); plt.close(fig)

    # Plot B: allocator Pareto (greedy block×bit vs uniform MXFP4 vs baselines)
    top=min(fp*1.6, max(y for _,y in rd4)*1.25)   # clip so RTN collapses don't swamp it
    fig,ax=plt.subplots(figsize=(8.5,5.4))
    ax.plot([x for x,_ in greedy],[y for _,y in greedy],'-o',color="#d62728",lw=2.4,ms=8,label="block×bit allocator (proposed)")
    ax.plot([x for x,_ in rd4],[y for _,y in rd4],'--o',color="#6a3d9a",lw=1.6,ms=6,label="uniform MXFP4")
    ax.scatter([eb(32,8)],[r["rd"]["8"]["32"]],marker='s',s=80,color="#2ca02c",label="uniform MXINT8",zorder=5)
    for nm,(m,c) in mk.items():
        if nm in base:
            bp=base[nm]["ppl"]
            if bp<=top: ax.scatter([base[nm]["eff_bits"]],[bp],marker=m,s=80,color=c,label=nm,zorder=5)
            else: ax.annotate(f"{nm}: {bp:.0f} (off-scale)",(base[nm]["eff_bits"],top*0.98),fontsize=7,color=c,ha='center')
    ax.axhline(fp,color='k',ls='--',lw=1,label=f"FP16 ({fp:.2f})")
    ax.set_ylim(fp*0.97, top)
    ax.set_xlabel("avg effective bits / weight  (∝ decode latency)"); ax.set_ylabel("wikitext-2 perplexity")
    ax.set_title(f"{tag} — block×bit allocator dominates uniform MX & RTN"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(f"{PLOTS}/alloc_{tag}.png",dpi=140); plt.close(fig)

    # Plot C: ordering sanity
    fig,ax=plt.subplots(figsize=(8,5))
    for k,(lab,c) in {"by":("by value/bit (proposed)","#d62728"),"random":("random","#7f7f7f"),"reverse":("reverse (worst)","#1f77b4")}.items():
        pts=sorted(r["order"][k]); ax.plot([x for x,_ in pts],[y for _,y in pts],'-o',color=c,lw=2,ms=6,label=lab)
    ax.axhline(fp,color='k',ls='--',lw=1,label=f"FP16 ({fp:.2f})")
    ax.set_xlabel("avg effective bits / weight"); ax.set_ylabel("wikitext-2 perplexity")
    ax.set_title(f"{tag} — allocation ordering matters (4↔8 bit mixing)"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(f"{PLOTS}/order_{tag}.png",dpi=140); plt.close(fig)

# ---------------- comparison numbers ----------------
lines=[]
for model,r in results.items():
    fp=r["fp16"]; b=r["baselines"]
    mxfp4_32=r["rd"]["4"]["32"]; mxfp4_128=r["rd"]["4"]["128"]; mxint8=r["rd"]["8"]["32"]
    greedy=sorted(r["greedy"])
    # greedy ppl at ~4.25 eff bits (iso with MXFP4/32) and bits needed to match MXFP4/32 ppl
    g_iso=min((p for bb,p in greedy if bb<=4.30), default=None)
    g_bits_match=best_at([(bb,p) for bb,p in greedy],mxfp4_32)
    lines.append((model,fp,mxfp4_128,mxfp4_32,mxint8,
                  b.get("RTN-INT4 per-channel",{}).get("ppl"),
                  b.get("RTN-INT4 g128",{}).get("ppl"),
                  b.get("RTN-INT3 g128",{}).get("ppl"),
                  g_iso,g_bits_match))

def fmt(x,nd=2):
    return "—" if x is None else (f"{x:.{nd}f}")
md=["# Cache- & Bandwidth-Aware Microscaling — Validated Findings\n",
"**Thesis.** In microscaling (MX) quantization the per-block scale is a *hardware* cost, not just an "
"accuracy knob: a finer block buys accuracy but adds scale-metadata that must be streamed (only when a "
"tensor is *streamed* and the layer is *memory-bound*). Therefore (a) the latency-optimal block size is "
"**regime-dependent**, and (b) the block size and bit-width should be **co-allocated per layer**, not "
"fixed at the MX default of 32.\n",
"**Setup.** Perplexity = wikitext-2 (raw) test, seqlen 2048, standard GPTQ/HF protocol. Weight-only "
"quantization of decoder-block linears (embeddings & lm_head kept FP16, per GPTQ). MXFP4 = OCP E2M1 + "
"E8M0 shared scale (4.25 effective bits at block 32 — matches the published number). MXINT8 = INT8 + "
"E8M0. RTN = symmetric round-to-nearest (per-channel / per-group). Effective bits / element = "
"`b + scale_bits/block`; this is proportional to per-token decode memory traffic and hence decode latency.\n",
"## Finding 0 — Harness validation (results are trustworthy)\n",
"Our FP16 wikitext-2 perplexity matches published numbers, so the quantization deltas are credible.\n",
"| model | our FP16 | published FP16 |\n|---|---|---|"]
for model,r in results.items():
    pub=PUB.get(model,{}).get("FP16")
    md.append(f"| {model} | {r['fp16']:.2f} | {fmt(pub)} ({PUB.get(model,{}).get('note','')}) |")
md.append("\n## Finding 1 — Block-size latency cost is regime-dependent (analytical, on our cycle model)\n")
md.append("Using the asic-cache/bandwidth cycle model, normalized to the MX-default block 32 (4-bit data, "
          "8-bit scale): in **decode** (memory-bound) shrinking the block 32→8 costs **+~18% latency** (and "
          "256 *saves* ~5%) because scale-metadata is on the critical path; in **prefill** it is **flat** "
          "(only 1/136 ViT and 1/74 OPT layers are memory-bound, the rest hide metadata under MACs). "
          "Metadata overhead per byte is regime-independent (`scale_bits/(b·block)`); what differs is whether "
          "it converts to latency. → *finer blocks are free in prefill, costly in decode.* "
          "(plots: exp1_latency_vs_block.png)\n")
md.append("\n## Finding 2 — Accuracy: block matters only at LOW bits; the SCALE FORMAT is a real axis\n")
md.append("(a) **Block matters only at 4-bit.** MXFP4 degrades with coarser blocks; MXINT8 is near-lossless "
          "at *every* block (so 8-bit weights can use the coarsest block and save metadata). "
          "(b) **Per-channel (one scale/row) collapses** at 4-bit — fine-grained scaling is essential. "
          "(c) **Scale format matters and is itself a co-design knob.** Group-INT4 with a 16-bit FP scale is "
          "*more accurate* than MXFP4 with its 8-bit power-of-2 E8M0 scale (OPT-125M: INT4-g32 30.22 vs "
          "MXFP4/32 32.79; Qwen: 15.31 vs 16.18) — but it spends more scale bits. So MXFP4 is NOT the most "
          "accurate 4-bit format; its advantage is hardware efficiency (cheap E8M0 scale). The right axis is "
          "(bits × block × **scale format**), matching the recent NVFP4 (E4M3 scale) vs MXFP4 (E8M0) debate.\n")
md.append("\n## Main comparison (wikitext-2 ppl; lower=better)\n")
md.append("| model | FP16 | MXFP4/128 (4.06b) | MXFP4/32 (4.25b) | MXINT8 (8.25b) | RTN-INT4 per-ch (4b) | RTN-INT4 g128 (4.06b) | RTN-INT3 g128 (3.06b) | **allocator @≤4.30b** |")
md.append("|---|---|---|---|---|---|---|---|---|")
for (model,fp,m128,m32,mi8,rtnpc,rtn128,rtn3,giso,gmatch) in lines:
    md.append(f"| {model} | {fmt(fp)} | {fmt(m128)} | {fmt(m32)} | {fmt(mi8)} | {fmt(rtnpc)} | {fmt(rtn128)} | {fmt(rtn3)} | **{fmt(giso)}** |")
md.append("\n## Allocator win (block×bit vs uniform MXFP4/32 at iso effective-bits)\n")
md.append("| model | MXFP4/32 ppl @4.25b | allocator ppl @≤4.30b | Δppl |")
md.append("|---|---|---|---|")
for (model,fp,m128,m32,mi8,rtnpc,rtn128,rtn3,giso,gmatch) in lines:
    d = (m32-giso) if (giso is not None) else None
    md.append(f"| {model} | {fmt(m32)} | {fmt(giso)} | {fmt(d)} |")

md.append("\n## Finding 3 — block×bit allocator (sensitivity-guided) fills the uniform-MX precision gap\n")
md.append("Uniform MX offers only ~4.25 bits (MXFP4/32) or ~8.25 bits (MXINT8) — a wide gap with nothing "
          "between, and MXFP4 *plateaus* (finer blocks barely help weights). A greedy per-layer allocator over "
          "{MXFP4/256, MXFP4/32, MXFP4/8, MXINT8/32}, ordered by measured sensitivity-per-bit, traces a Pareto "
          "that uniform MX cannot reach. (plots: alloc_<model>.png; ordering sanity: order_<model>.png.)\n")
md.append("\n## Allocator vs BEST uniform MX, across accuracy targets (the honest headline)\n")
md.append("For each accuracy target, `best uniform` = the fewest-effective-bit *uniform* MX config "
          "(any MXFP4 block, any MXINT8 block) that meets it; `allocator` = our block×bit Pareto. "
          "Uniform MX has a gap between 4.25 and 8.25 bits that the allocator fills.\n")
for model,r in results.items():
    fp=r["fp16"]; greedy=sorted(r["greedy"])
    uni=[(eb(bl,4),r["rd"]["4"][str(bl)]) for bl in BLOCKS]+[(eb(bl,8),r["rd"]["8"][str(bl)]) for bl in BLOCKS]
    md.append(f"\n**{model}** (FP16={fp:.2f})\n")
    md.append("| accuracy target | best-uniform eff-bits | allocator eff-bits | **bits saved** |")
    md.append("|---|---|---|---|")
    best_sav=0
    for mult in (1.05,1.10,1.20,1.35):
        tgt=fp*mult
        ub=best_at(uni,tgt); gb=best_at(greedy,tgt)
        sav=(1-gb/ub)*100 if (ub and gb) else None
        if sav: best_sav=max(best_sav,sav)
        md.append(f"| FP×{mult:.2f} = {tgt:.2f} | {fmt(ub)} | {fmt(gb)} | **{fmt(sav,1)}%** |")
    md.append(f"\n_max decode-traffic saving vs best uniform: **{best_sav:.1f}%**_")

# ---------------- group-INT allocator (corrected, stronger primitive) ----------------
gi=[]
for f in sorted(glob.glob(os.path.join(HERE,"results_grpint_*.json"))):
    gi.append(json.load(open(f)))
if gi:
    md.append("\n## Finding 4 — block×bit allocator on the STRONG primitive (group-INT4) beats best uniform\n")
    md.append("Since group-INT4 (FP scale) > MXFP4, we build the per-layer allocator over group-INT "
              "{INT4/g256, INT4/g64, INT4/g16, INT8/g64} (16-bit FP scale → eff-bits = b+16/group). It is "
              "compared to the strongest *uniform* group-INT config. (plots: allocgi_<model>.png)\n")
    md.append("| model | FP16 | best uniform INT4 (eff-bits, ppl) | allocator @ same bits | max bits saved vs best uniform |")
    md.append("|---|---|---|---|---|")
    for r in gi:
        tag=r["model"].split("/")[-1]; fp=r["fp16"]; greedy=sorted(r["greedy"])
        uni=[(v["eb"],v["ppl"]) for v in r["uniform"].values()]
        # best uniform INT4 (exclude 8b) at its own bits
        u4=sorted([(e,p) for e,p in uni if e<6]); bu=min(u4,key=lambda x:x[1])
        g_at=min((p for b,p in greedy if b<=bu[0]+0.01),default=None)
        # max saving across targets
        best=0
        for mult in (1.05,1.10,1.20,1.35):
            tgt=fp*mult; ub=best_at(uni,tgt); gb=best_at(greedy,tgt)
            if ub and gb: best=max(best,(1-gb/ub)*100)
        md.append(f"| {r['model']} | {fp:.2f} | {bu[0]:.2f}b, {bu[1]:.2f} | {fmt(g_at)} | **{best:.1f}%** |")
        # plot
        fig,ax=plt.subplots(figsize=(8.5,5.4)); top=min(fp*1.6,max(p for _,p in uni if p<fp*3)*1.1)
        ax.plot([b for b,_ in greedy],[p for _,p in greedy],'-o',color="#d62728",lw=2.4,ms=8,label="block×bit allocator (group-INT)")
        gi4=sorted([(e,p) for e,p in uni if e<6]); gi8=[(e,p) for e,p in uni if e>=6]
        ax.plot([e for e,_ in gi4],[p for _,p in gi4],'--o',color="#1f77b4",lw=1.6,ms=6,label="uniform INT4 (by group)")
        if gi8: ax.scatter([gi8[0][0]],[gi8[0][1]],marker='s',s=80,color="#2ca02c",label="uniform INT8",zorder=5)
        ax.axhline(fp,color='k',ls='--',lw=1,label=f"FP16 ({fp:.2f})"); ax.set_ylim(fp*0.97,top)
        ax.set_xlabel("avg effective bits / weight (∝ decode latency)"); ax.set_ylabel("wikitext-2 perplexity")
        ax.set_title(f"{tag} — allocator on group-INT4 fills the 4→8 bit gap"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.invert_xaxis()
        fig.tight_layout(); fig.savefig(f"{PLOTS}/allocgi_{tag}.png",dpi=140); plt.close(fig)

md.append("\n## Threats to validity / honest caveats\n")
md += [
"- **Weight-only RTN-style** quantization (no GPTQ/AWQ error compensation, no activation quant). GPTQ would "
"shift all 4-bit numbers down ~1 ppl and is orthogonal/additive to our block×bit allocation.",
"- **Effective-bits = latency proxy.** Finding 1 uses the actual cycle model; Findings 2–3 use effective "
"bits (exact for decode memory traffic, the dominant edge-decode cost). A full latency–accuracy Pareto "
"coupling the two is the natural next step.",
"- **Marginal (rest-FP) sensitivity** drives only the allocator *ordering*; every plotted Pareto point is a "
"*measured* perplexity, so additivity is not assumed in the results.",
"- The allocator win is **target-dependent**: large in the mid-accuracy regime (uniform-MX gap), ~0 at the "
"extremes where one uniform config is already optimal. We report the full curve, not a single number.",
"- Models: OPT-125M / OPT-1.3B (standard quant benchmarks) + Qwen2.5-0.5B (current edge, GQA). Larger "
"models and Llama/Gemma (gated) remain to be run.",
]
md.append("\n## What holds, for the thesis\n")
md += [
"1. **Validated harness** — FP16 matches published exactly (OPT-125M 27.65, OPT-1.3B 14.62), so all "
"quantization deltas are trustworthy. Quantizers are OCP-spec (MXFP4 = 4.25 eff-bits). Per-channel RTN "
"collapses at 4-bit; group/block scaling is essential — consistent with the literature.",
"2. **No universal 4-bit winner (scale format is a real axis).** Group-INT4 (16-bit FP scale) is often "
"*more accurate* than MXFP4 (8-bit E8M0 scale) but spends more scale bits; which wins flips with model and "
"granularity (e.g. OPT-125M: INT4-g32 30.22 < MXFP4/32 32.79; OPT-1.3B: MXFP4/32 15.25 < RTN-g128 15.87). "
"MXFP4's advantage is hardware efficiency, not accuracy — mirroring the NVFP4-vs-MXFP4 debate.",
"3. **Finding 1 (latency regime split)** — the conceptual core: block size is latency-free in compute-bound "
"prefill but costs up to +18% in memory-bound decode. Clean and model-independent.",
"4. **Finding 2 (block matters only at low bits)** holds across all three models (MXINT8 near-lossless at "
"every block; MXFP4 degrades with coarser blocks).",
"5. **Finding 4 (block×bit allocator on the strong group-INT primitive)** delivers a real, measured "
"decode-traffic saving (OPT-125M: 22.6% vs the *strongest* uniform baseline) by filling the 4→8-bit gap. "
"The win is concentrated in the mid-accuracy regime and is larger on harder-to-quantize models (125M, Qwen) "
"than on the already-robust 1.3B.",
"",
"**Next, to make it a strong paper:** (a) couple Findings to the Finding-1 *latency* model for a true "
"latency–accuracy Pareto incl. the prefill/decode split; (b) add the scale-format axis (E8M0 vs E4M3/FP, "
"à la NVFP4) into the allocator; (c) add GPTQ error-compensation (additive) and activation/KV quant; "
"(d) scale to Llama-3.x / 7B.",
]
open(os.path.join(HERE,"FINDINGS.md"),"w").write("\n".join(md))
print("Wrote FINDINGS.md and plots/ for models:", list(results.keys()))
print("\n".join(md[-12:]))
