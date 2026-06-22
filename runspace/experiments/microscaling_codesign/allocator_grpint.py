"""
Corrected headline allocator built on the STRONGER 4-bit primitive
(group-INT4 with FP16 scale), since Finding 2 showed group-INT4 > MXFP4.
Co-allocates group-size x bit-width per layer; compares to the best uniform
(INT4-g64) and the gap to INT8-g64. Saves results_grpint_<tag>.json.
"""
import os, sys, json, time, random
os.environ.setdefault("HF_DATASETS_OFFLINE","1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from mx_lib import rtn_int, decoder_linears, load_wikitext2_enc, ppl_from_enc
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV="cuda"; random.seed(0)
MODEL=sys.argv[1] if len(sys.argv)>1 else "facebook/opt-125m"
SW=int(sys.argv[2]) if len(sys.argv)>2 else 40
tag=MODEL.split("/")[-1]
# group-INT configs with FP16 (16-bit) scale: (bits, group); eff = bits + 16/group
CFGS=[(4,256),(4,64),(4,16),(8,64)]
def eb(b,g): return b + 16.0/g
CFG_EB=[eb(b,g) for (b,g) in CFGS]

tok=AutoTokenizer.from_pretrained(MODEL, use_fast=("opt" not in MODEL.lower()))
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
ref={k:v.detach().clone() for k,v in model.state_dict().items()}
lins=decoder_linears(model); params={n:m.weight.numel() for n,m in lins}; tot=sum(params.values())
ENC=load_wikitext2_enc(tok)
def PPL(w=None): return ppl_from_enc(model,ENC,DEV,2048,w)
def apply(cfg):
    model.load_state_dict(ref)
    for n,m in lins:
        c=cfg.get(n)
        if c is not None: m.weight.data=rtn_int(m.weight.data.float(),c[0],c[1]).half()
fp=PPL(); print(f"{MODEL} FP16={fp:.3f}  lins={len(lins)}")
t0=time.time()

# uniform group-INT4 rate-distortion (best-primitive) + INT8
uni={}
for (b,g) in [(4,256),(4,128),(4,64),(4,32),(4,16),(8,64)]:
    apply({n:(b,g) for n,_ in lins}); uni[f"{b}b/g{g}"]={"ppl":PPL(),"eb":eb(b,g)}
apply({})
print("uniform:",{k:round(v['ppl'],2) for k,v in uni.items()})

# per-(layer,config) marginal sensitivity
apply({}); fpw=PPL(SW); sens={}
for n,_ in lins:
    row=[]
    for (b,g) in CFGS: apply({n:(b,g)}); row.append(PPL(SW)-fpw)
    sens[n]=row
apply({}); print(f"sensitivity done {time.time()-t0:.0f}s")

# greedy block(group)xbit allocation
def avgb(st): return sum(params[n]*CFG_EB[st[n]] for n,_ in lins)/tot
st={n:0 for n,_ in lins}; path=[(avgb(st),dict(st))]
while any(st[n]<len(CFGS)-1 for n,_ in lins):
    best=None
    for n,_ in lins:
        c=st[n]
        if c<len(CFGS)-1:
            r=(sens[n][c]-sens[n][c+1])/((CFG_EB[c+1]-CFG_EB[c])*params[n])
            if best is None or r>best[0]: best=(r,n)
    st[best[1]]+=1; path.append((avgb(st),dict(st)))
idxs=sorted(set(round(i*(len(path)-1)/9) for i in range(10)))
greedy=[]
for i in idxs:
    ab,s=path[i]; apply({n:CFGS[s[n]] for n,_ in lins}); greedy.append((ab,PPL()))
apply({})

res={"model":MODEL,"fp16":fp,"uniform":uni,"cfgs":CFGS,"cfg_eb":CFG_EB,"greedy":greedy}
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"results_grpint_{tag}.json"),"w"),indent=2)
print("greedy:",[(round(a,2),round(p,2)) for a,p in greedy])
print(f"done {time.time()-t0:.0f}s")
