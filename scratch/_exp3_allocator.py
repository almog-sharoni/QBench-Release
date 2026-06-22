"""
B: rate-distortion with two bit-width families (MXFP4 + MXINT8) vs block.
C: Exp 3 — per-tensor (bit x block) allocator guided by per-layer sensitivity,
   validated against reverse-ranked and random allocation on the
   accuracy vs effective-bits (decode-latency) Pareto.
"""
import os, sys, math, random
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch, numpy as np, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE="cuda"; SCALE_BITS=8; SEQ=512; NB=100
OUT=os.path.join(REPO,"scratch","plots"); os.makedirs(OUT,exist_ok=True)
random.seed(0)
def eff_bits(block,b): return b + SCALE_BITS/block

# ---- MX quantizers ----
E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],device=DEVICE)
def mx_fp4(w,block):
    shp=w.shape; flat=w.reshape(-1); pad=(-flat.numel())%block
    if pad: flat=torch.cat([flat,flat.new_zeros(pad)])
    blk=flat.reshape(-1,block); amax=blk.abs().amax(1,keepdim=True).clamp(min=1e-12)
    scale=2.0**torch.ceil(torch.log2(amax/6.0)); x=blk/scale
    idx=(x.abs().unsqueeze(-1)-E2M1).abs().argmin(-1)
    return ((x.sign()*E2M1[idx])*scale).reshape(-1)[:shp.numel()].reshape(shp)
def mx_int8(w,block,nbits=8):
    qmax=2**(nbits-1)-1; shp=w.shape; flat=w.reshape(-1); pad=(-flat.numel())%block
    if pad: flat=torch.cat([flat,flat.new_zeros(pad)])
    blk=flat.reshape(-1,block); amax=blk.abs().amax(1,keepdim=True).clamp(min=1e-12)
    scale=2.0**torch.ceil(torch.log2(amax/qmax))
    q=torch.clamp(torch.round(blk/scale),-qmax,qmax)*scale
    return q.reshape(-1)[:shp.numel()].reshape(shp)
def quant(w,b,block):  return mx_fp4(w,block) if b==4 else mx_int8(w,block)

# ---- model + data ----
tok=AutoTokenizer.from_pretrained("facebook/opt-125m")
model=AutoModelForCausalLM.from_pretrained("facebook/opt-125m",torch_dtype=torch.float32,
                                           attn_implementation="eager").to(DEVICE).eval()
ref={k:v.detach().clone() for k,v in model.state_dict().items()}
linears=[(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear)]
params={n:m.weight.numel() for n,m in linears}; total_p=sum(params.values())
text="\n\n".join(load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"])
allids=tok(text,return_tensors="pt")["input_ids"].squeeze(0)
nb=min(NB,allids.numel()//SEQ); ids=allids[:nb*SEQ].view(nb,SEQ).to(DEVICE)
@torch.no_grad()
def ppl():
    loss=0.0; nt=0
    for i in range(0,nb,8):
        bt=ids[i:i+8]; lg=model(input_ids=bt).logits
        sl=lg[:,:-1,:].reshape(-1,lg.size(-1)); lb=bt[:,1:].reshape(-1)
        loss+=torch.nn.functional.cross_entropy(sl,lb,reduction='sum').item(); nt+=lb.numel()
    return math.exp(loss/nt)
@torch.no_grad()
def apply(cfg):  # cfg: name -> (b,block) or None for FP
    model.load_state_dict(ref)
    for n,m in linears:
        c=cfg.get(n)
        if c is not None: m.weight.data=quant(m.weight.data,c[0],c[1])
fp=ppl(); print(f"FP ref ppl={fp:.3f}  ({nb} blocks, {len(linears)} linears)")

# ================= B: rate-distortion, two families =================
BLOCKS=[8,16,32,64,128,256]
rd={4:{},8:{}}
for b in (4,8):
    for bl in BLOCKS:
        apply({n:(b,bl) for n,_ in linears}); rd[b][bl]=ppl()
        print(f"  uniform {b}b blk{bl:>3}  eff={eff_bits(bl,b):.3f}  ppl={rd[b][bl]:.3f}")
apply({})  # restore

# ================= C: per-layer sensitivity (knapsack: value per bit) =================
print("\nMeasuring per-layer sensitivity (quantize each layer alone -> 4b/blk32)...")
sens={}
for n,_ in linears:
    apply({n:(4,32)}); sens[n]=ppl()-fp
EXTRA_BITS = eff_bits(32,8)-eff_bits(32,4)   # cost/param to promote 4b->8b
# value density = ppl recovered per extra bit spent (knapsack ratio)
density={n: sens[n]/(params[n]*EXTRA_BITS) for n,_ in linears}
order=sorted(linears,key=lambda nm:density[nm[0]],reverse=True)  # best value-per-bit first
def short(n): return ".".join(n.split(".")[-3:])
print("  top-5 value/bit:", [(short(n), round(sens[n],2)) for n,_ in order[:5]])
print("  bot-5 value/bit:", [(short(n), round(sens[n],2)) for n,_ in order[-5:]])

def sweep(order_list):
    """Sweep k = #layers promoted to 8b/blk32 (rest 4b/blk32). Returns (avg_eff_bits, ppl)."""
    pts=[]
    ks=sorted(set(list(range(0,len(order_list)+1,4))+[len(order_list)]))
    for k in ks:
        promoted=set(n for n,_ in order_list[:k])
        cfg={n:((8,32) if n in promoted else (4,32)) for n,_ in linears}
        apply(cfg); p=ppl()
        avg=sum(params[n]*eff_bits(cfg[n][1],cfg[n][0]) for n,_ in linears)/total_p
        pts.append((avg,p))
    return pts

print("\nExp3 allocations: by value/bit / reverse / random ...")
by_sens=sweep(order)
reverse=sweep(list(reversed(order)))
rnd_order=linears[:]; random.shuffle(rnd_order); random_a=sweep(rnd_order)
apply({})

# ================= Plots =================
# Fig 1 updated: two families rate-distortion
fig,ax=plt.subplots(figsize=(8.5,5.4))
for b,col,mk in [(4,"#6a3d9a","o"),(8,"#2ca02c","s")]:
    xs=[eff_bits(bl,b) for bl in BLOCKS]; ys=[rd[b][bl] for bl in BLOCKS]
    ax.plot(xs,ys,'-'+mk,color=col,lw=2.2,ms=8,label=f"{'MXFP4' if b==4 else 'MXINT8'} weights")
    for bl,x,y in zip(BLOCKS,xs,ys): ax.annotate(f"{bl}",(x,y),textcoords="offset points",xytext=(5,5),fontsize=7,color=col)
ax.axhline(fp,color='k',ls='--',lw=1.0,label=f"FP32 ({fp:.1f})")
ax.set_xlabel("effective bits / element  (= b + 8/block)  ∝ decode memory traffic")
ax.set_ylabel("wikitext-2 perplexity")
ax.set_title("B — Rate–distortion, two MX families\n4-bit: block matters a lot   |   8-bit: near-lossless, block ~ irrelevant")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_rate_distortion.png",dpi=140); plt.close(fig)

# Exp3: allocator Pareto
fig,ax=plt.subplots(figsize=(8.5,5.4))
for pts,lab,col,st in [(by_sens,"by sensitivity (proposed)","#d62728","-o"),
                       (random_a,"random","#7f7f7f","-s"),
                       (reverse,"reverse (worst)","#1f77b4","-^")]:
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,st,color=col,lw=2.2,ms=7,label=lab)
ax.axhline(fp,color='k',ls='--',lw=1.0,label=f"FP32 ({fp:.1f})")
ax.set_xlabel("avg effective bits / weight  (param-weighted)  ∝ decode latency")
ax.set_ylabel("wikitext-2 perplexity")
ax.set_title("Exp 3 — Per-layer (bit×block) allocation\nsensitivity-guided dominates: same accuracy at fewer bits / lower decode latency")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.invert_xaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/exp3_allocator_pareto.png",dpi=140); plt.close(fig)

# quantify the win: at a fixed ppl target, bits saved by sensitivity vs random
def bits_at_ppl(pts,target):
    pts=sorted(pts,key=lambda p:p[0])
    best=None
    for avg,p in pts:
        if p<=target and (best is None or avg<best): best=avg
    return best
tgt=fp*1.10  # 10% over FP
bs=bits_at_ppl(by_sens,tgt); br=bits_at_ppl(random_a,tgt)
print(f"\nAt ppl<= {tgt:.1f}: by-sensitivity needs {bs:.2f} eff-bits, random {br:.2f}  "
      f"(saving {100*(1-bs/br):.1f}% decode traffic)" if bs and br else "\n(target not bracketed)")
print("Saved: fig1_rate_distortion.png  exp3_allocator_pareto.png")
