"""
Microscaling block-size co-design — Exp 1 (latency) + Exp 2 (accuracy) + plots.
Rebuilt around effective-bits/element = data_bits + scale_bits/block, which IS the
decode memory cost (sanity: block=32,b=4 -> 4.25 bits == published MXFP4).
"""
import os, sys, math
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runspace.experiments.asic_cache_simulation.simulate_cache import (
    analyze_model, _compute_layer_cycles, get_footprint_elements,
    round_to_banks, evaluate_stay)
from runspace.experiments.asic_cache_simulation.rules import RULES

BLOCKS=[8,16,32,64,128,256]; SCALE_BITS=8; BW=1.0; CACHE_MB=4.0; CTX=2048; DEVICE="cuda"
REF_BLK=32  # normalize to MXFP4 default
OUT=os.path.join(REPO,"scratch","plots"); os.makedirs(OUT,exist_ok=True)

def eff_bits(block,b): return b + SCALE_BITS/block

# ---- sanity check: reproduce MXFP4 = 4.25 bits ----
assert abs(eff_bits(32,4)-4.25)<1e-9, "effective-bits model wrong"
print("SANITY  effective bits/elem:  b=4 ->", {bl:round(eff_bits(bl,4),3) for bl in BLOCKS})
print("        (block=32,b=4 = 4.25 == published MXFP4)  ✓")

# ============================ Exp 1: latency ============================
def stay_eval(layers,ce,bank):
    out,prev=[],False
    for i,layer in enumerate(layers):
        nxt=layers[i+1] if i+1<len(layers) else None
        ctx={'input_banked':round_to_banks(get_footprint_elements(layer['input_elems'],0),bank),
             'output_banked':round_to_banks(get_footprint_elements(layer['output_elems'],0),bank),
             'weight_banked':round_to_banks(get_footprint_elements(layer['weight_elems'],0),bank),
             'next_xin_banked':round_to_banks(get_footprint_elements(nxt['input_elems'],0) if nxt else 0,bank),
             'cache_elements':ce,'bank_size':bank,'jump_back_size_in_banks':layer.get('jump_back_size_in_banks',0)}
        for k in ('filter_height','filter_width','in_channels','out_channels',
                  'input_channel_height','input_channel_width','output_channel_height','output_channel_width'):
            ctx[k]=layer.get(k,0)
        stay,_p,_o,rule=evaluate_stay(layer,ctx,nxt,0,bank,ce)
        lc=dict(layer); lc['stay_on_chip']=stay; lc['xin_from_cache']=RULES.get(rule,{}).get('xin_from_cache',True)
        out.append(lc)
    return out

def trace_prefill(model_cfg,adapter_cfg,dummy):
    ce=int(CACHE_MB*1e6); bank=ce//16
    layers=analyze_model(model_cfg,batch_size=1,device=DEVICE,adapter_cfg=adapter_cfg,
                         cache_elements=ce,bank_size=bank,metadata_bits=0,dummy_input=dummy)
    sim=stay_eval(layers,ce,bank); traced=[]; prev=False
    for i,l in enumerate(sim):
        compute=_compute_layer_cycles(l); stay=l['stay_on_chip']
        need_in=(i==0 or not prev or not l['xin_from_cache']); streamed=[]
        if l.get('weight_elems',0)>0: streamed.append(l['weight_elems'])
        if need_in and l.get('input_elems',0)>0: streamed.append(l['input_elems'])
        if (not stay) and l.get('output_elems',0)>0: streamed.append(l['output_elems'])
        traced.append((compute,streamed)); prev=stay
    return traced

def prefill_lat(traced,block,b):
    total=0.0; membound=0
    for compute,streamed in traced:
        data=sum(e*b/8/BW for e in streamed)
        meta=sum(math.ceil(e/block)*(SCALE_BITS/8)/BW for e in streamed)
        if data+meta>compute: membound+=1
        total+=max(compute,data+meta)
    return total,membound,len(traced)

print("\n=== Exp 1: latency ===")
tr_vit=trace_prefill({"name":"vit_b_16","weights":None},{"type":"generic","build_quantized":True},None)
ids=torch.randint(0,50000,(1,512),dtype=torch.long,device=DEVICE)
tr_opt=trace_prefill({"name":"facebook/opt-125m","source":"huggingface"},
                     {"type":"slm","build_quantized":True,"quantized_ops":["all"],"input_quantization":True},ids)
# prefill normalized latency at b=4 + memory-bound layer counts
prefill={}
for nm,tr in [("ViT-B/16 prefill",tr_vit),("OPT-125M prefill",tr_opt)]:
    base,mb,n=prefill_lat(tr,REF_BLK,4)
    vals=[prefill_lat(tr,bl,4)[0]/base for bl in BLOCKS]
    prefill[nm]=vals
    print(f"  {nm}: memory-bound layers @block8 = {prefill_lat(tr,8,4)[1]}/{n}  -> norm latency flat={max(vals)-min(vals)<0.02}")
# decode normalized latency for b in {4,8} (model-independent -> compute once)
decode={}
for b in (4,8):
    base=eff_bits(REF_BLK,b)
    decode[b]=[eff_bits(bl,b)/base for bl in BLOCKS]
print(f"  decode b4 norm: {[round(x,3) for x in decode[4]]}  (model-independent)")

# ============================ Exp 2: accuracy ============================
print("\n=== Exp 2: MXFP4 weight perplexity ===")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn
E2M1=torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.],device=DEVICE)
def mx_fp4(w,block):
    shp=w.shape; flat=w.reshape(-1); pad=(-flat.numel())%block
    if pad: flat=torch.cat([flat,flat.new_zeros(pad)])
    blk=flat.reshape(-1,block); amax=blk.abs().amax(1,keepdim=True).clamp(min=1e-12)
    scale=2.0**torch.ceil(torch.log2(amax/6.0)); x=blk/scale
    idx=(x.abs().unsqueeze(-1)-E2M1).abs().argmin(-1)
    return ((x.sign()*E2M1[idx])*scale).reshape(-1)[:shp.numel()].reshape(shp)

tok=AutoTokenizer.from_pretrained("facebook/opt-125m")
model=AutoModelForCausalLM.from_pretrained("facebook/opt-125m",torch_dtype=torch.float32,
                                           attn_implementation="eager").to(DEVICE).eval()
ref={k:v.detach().clone() for k,v in model.state_dict().items()}
text="\n\n".join(load_dataset("wikitext","wikitext-2-raw-v1",split="test")["text"])
allids=tok(text,return_tensors="pt")["input_ids"].squeeze(0); seq=512
nb=min(150,allids.numel()//seq); blk_ids=allids[:nb*seq].view(nb,seq).to(DEVICE)
@torch.no_grad()
def ppl():
    loss=0.0; nt=0
    for i in range(0,nb,8):
        bt=blk_ids[i:i+8]; lg=model(input_ids=bt).logits
        sl=lg[:,:-1,:].reshape(-1,lg.size(-1)); lb=bt[:,1:].reshape(-1)
        loss+=torch.nn.functional.cross_entropy(sl,lb,reduction='sum').item(); nt+=lb.numel()
    return math.exp(loss/nt)
fp_ppl=ppl(); print(f"  FP32 ref ppl={fp_ppl:.3f}  ({nb} blocks)")
exp2={}
for bl in BLOCKS:
    model.load_state_dict(ref)
    for m in model.modules():
        if isinstance(m,nn.Linear): m.weight.data=mx_fp4(m.weight.data,bl)
    exp2[bl]=ppl(); print(f"  block={bl:>3}  eff_bits={eff_bits(bl,4):.3f}  ppl={exp2[bl]:.3f}")
model.load_state_dict(ref)

# ============================ Plots ============================
xt=np.log2(np.array(BLOCKS))
# Plot 1: latency vs block — decode (b4,b8) rising, prefill flat
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(xt,decode[4],'-o',color="#d62728",lw=2.2,ms=8,label="decode, 4-bit data (any model)")
ax.plot(xt,decode[8],'-o',color="#ff7f0e",lw=2.2,ms=7,label="decode, 8-bit data (any model)")
ax.plot(xt,prefill["ViT-B/16 prefill"],'--s',color="#1f77b4",lw=2,ms=7,label="ViT-B/16 prefill (4-bit)")
ax.plot(xt,prefill["OPT-125M prefill"],'--^',color="#17becf",lw=2,ms=7,label="OPT-125M prefill (4-bit)")
ax.axhline(1.0,color='k',lw=0.8,ls=':'); ax.axvline(np.log2(REF_BLK),color='gray',lw=0.8,ls=':')
ax.text(np.log2(REF_BLK)+0.05,0.965,"MXFP4 default\n(block 32)",fontsize=8,color='gray')
ax.set_xticks(xt); ax.set_xticklabels(BLOCKS)
ax.set_xlabel("microscaling block size (elements / scale)")
ax.set_ylabel(f"latency  (normalized to block={REF_BLK})")
ax.set_title("Exp 1 — Block-size latency cost is regime-dependent\ndecode (memory-bound) pays; prefill (compute-bound) is free")
ax.annotate("finer → +decode latency",xy=(xt[0],decode[4][0]),xytext=(xt[1],1.13),
            color="#d62728",fontsize=9,fontweight='bold')
ax.legend(fontsize=8,loc='center right'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/exp1_latency_vs_block.png",dpi=140); plt.close(fig)

# Plot 2: ppl vs block
fig,ax=plt.subplots(figsize=(8,5)); yp=[exp2[b] for b in BLOCKS]
ax.plot(xt,yp,'-o',color="#2ca02c",lw=2.2,ms=8,label="MXFP4 weights (OPT-125M)")
ax.axhline(fp_ppl,color='k',ls='--',lw=1.2,label=f"FP32 reference ({fp_ppl:.1f})")
ax.axvline(np.log2(REF_BLK),color='gray',lw=0.8,ls=':')
ax.set_xticks(xt); ax.set_xticklabels(BLOCKS)
ax.set_xlabel("microscaling block size (elements / scale)")
ax.set_ylabel("wikitext-2 perplexity  (lower = better)")
ax.set_title(f"Exp 2 — Accuracy improves with finer blocks\n(OPT-125M, MXFP4 weights, {nb} wikitext blocks)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/exp2_accuracy_vs_block.png",dpi=140); plt.close(fig)

# Plot 3: Fig 1 — rate-distortion (the rigorous money plot)
fig,ax=plt.subplots(figsize=(8.5,5.4))
eb=[eff_bits(b,4) for b in BLOCKS]
ax.plot(eb,yp,'-o',color="#6a3d9a",lw=2.4,ms=9)
for b,x,y in zip(BLOCKS,eb,yp):
    ax.annotate(f"blk {b}",(x,y),textcoords="offset points",xytext=(6,6),fontsize=8)
ax.axhline(fp_ppl,color='k',ls='--',lw=1.0,label=f"FP32 ({fp_ppl:.1f})")
ax.axvline(4.25,color='gray',lw=0.8,ls=':'); ax.text(4.27,max(yp)*0.97,"MXFP4 std\n4.25 bits",fontsize=8,color='gray')
ax.set_xlabel("effective bits / element   =  4  +  8/block   (∝ decode memory traffic & latency)")
ax.set_ylabel("wikitext-2 perplexity")
ax.set_title("Fig 1 — Rate–distortion of block size (MXFP4 weights)\n"
             "decode: x-axis IS latency → pick the knee   |   prefill: x-axis is FREE → pick min-ppl (finest)")
ax.invert_xaxis()  # left = cheaper
ax.legend(fontsize=9,loc='upper left'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_rate_distortion.png",dpi=140); plt.close(fig)
print("\nSaved: exp1_latency_vs_block.png  exp2_accuracy_vs_block.png  fig1_rate_distortion.png")
