"""
Full co-design run for one model: rate-distortion (B), baselines, and the
block x bit greedy allocator (C). Saves results JSON. Validated harness (mx_lib).

Usage: run_codesign.py <model_id> [sens_windows]
"""
import os, sys, json, math, time, random
os.environ.setdefault("HF_DATASETS_OFFLINE","1")   # wikitext from cache; models may still download
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from mx_lib import mxfp4, mxint8, rtn_int, eff_bits, decoder_linears, load_wikitext2_enc, ppl_from_enc
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV="cuda"; random.seed(0)
MODEL = sys.argv[1] if len(sys.argv)>1 else "facebook/opt-125m"
SENS_WIN = int(sys.argv[2]) if len(sys.argv)>2 else 40      # windows for sensitivity (speed)
BLOCKS = [8,16,32,64,128,256]
# candidate (bits, block) configs for the allocator, increasing eff-bits/accuracy
CFGS = [(4,256),(4,32),(4,8),(8,32)]
CFG_EB = [eff_bits(bl,b) for (b,bl) in CFGS]
tag = MODEL.split("/")[-1]
OUTJSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{tag}.json")

print(f"=== {MODEL} ===")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=("opt" not in MODEL.lower()))
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
ref = {k:v.detach().clone() for k,v in model.state_dict().items()}
lins = decoder_linears(model)
params = {n:m.weight.numel() for n,m in lins}; tot=sum(params.values())
print(f"decoder linears: {len(lins)}  params(quantized): {tot/1e6:.1f}M")

def quant(w,b,block): return (mxfp4(w,block) if b==4 else mxint8(w,block))
def apply(cfgmap):     # name -> (b,block) or None
    model.load_state_dict(ref)
    for n,m in lins:
        c=cfgmap.get(n)
        if c is not None: m.weight.data = quant(m.weight.data.float(), c[0], c[1]).half()
ENC = load_wikitext2_enc(tok)
def PPL(win=None): return ppl_from_enc(model, ENC, DEV, seqlen=2048, max_windows=win)

t0=time.time()
fp = PPL(); print(f"FP16 ppl = {fp:.3f}")

# ---------------- B: rate-distortion + baselines ----------------
res = {"model":MODEL, "fp16":fp, "nparams_q":tot, "nlin":len(lins),
       "rd":{"4":{}, "8":{}}, "baselines":{}, "cfgs":CFGS, "cfg_eb":CFG_EB}
for b in (4,8):
    for bl in BLOCKS:
        apply({n:(b,bl) for n,_ in lins}); res["rd"][str(b)][bl]=PPL()
    print(f"  uniform {b}b done")
for name,fn,eb in [("RTN-INT4 per-channel", lambda w: rtn_int(w,4,None), 4.0),
                   ("RTN-INT4 g128",        lambda w: rtn_int(w,4,128),  eff_bits(128,4)),
                   ("RTN-INT4 g32",         lambda w: rtn_int(w,4,32),   eff_bits(32,4)),
                   ("RTN-INT3 g128",        lambda w: rtn_int(w,3,128),  eff_bits(128,3))]:
    model.load_state_dict(ref)
    for n,m in lins: m.weight.data = fn(m.weight.data.float()).half()
    res["baselines"][name]={"ppl":PPL(),"eff_bits":eb}
apply({})
print(f"  baselines done ({time.time()-t0:.0f}s)")

# ---------------- C: per-(layer,config) marginal sensitivity ----------------
print("measuring marginal sensitivity (layer x config, rest FP)...")
apply({}); fp_win = PPL(SENS_WIN)     # FP ppl at the (smaller) sensitivity window count
sens={}  # name -> [dppl for each cfg]
for n,_ in lins:
    row=[]
    for (b,bl) in CFGS:
        apply({n:(b,bl)}); row.append(PPL(SENS_WIN)-fp_win)
    sens[n]=row
apply({})
print(f"  sensitivity done ({time.time()-t0:.0f}s)")

# ---------------- greedy block x bit allocation ----------------
def avg_bits(state): return sum(params[n]*CFG_EB[state[n]] for n,_ in lins)/tot
state={n:0 for n,_ in lins}                 # start cheapest (MXFP4/256)
greedy=[(avg_bits(state), dict(state))]
while any(state[n]<len(CFGS)-1 for n,_ in lins):
    best=None
    for n,_ in lins:
        c=state[n]
        if c<len(CFGS)-1:
            dppl=sens[n][c]-sens[n][c+1]                       # ppl reduction
            dbits=(CFG_EB[c+1]-CFG_EB[c])*params[n]
            r=dppl/dbits
            if best is None or r>best[0]: best=(r,n)
    _,bn=best; state[bn]+=1
    greedy.append((avg_bits(state), dict(state)))
# measure REAL ppl at ~10 checkpoints along the greedy path
idxs=sorted(set([round(i*(len(greedy)-1)/9) for i in range(10)]))
greedy_pts=[]
for i in idxs:
    ab,st=greedy[i]; apply({n:CFGS[st[n]] for n,_ in lins})
    greedy_pts.append((ab, PPL()))
res["greedy"]=greedy_pts
apply({})

# ---------------- ordering sanity: value vs random vs reverse (bit-mix) ----------------
EXTRA=eff_bits(32,8)-eff_bits(32,4)
dens={n: (sens[n][1]/(params[n]*EXTRA)) for n,_ in lins}   # value/bit of MXFP4/32 layer
def sweep(order):
    pts=[]
    for k in range(0,len(order)+1,max(1,len(order)//12)):
        prom=set(n for n,_ in order[:k])
        cfg={n:((8,32) if n in prom else (4,32)) for n,_ in lins}
        apply(cfg); ab=sum(params[n]*eff_bits(cfg[n][1],cfg[n][0]) for n,_ in lins)/tot
        pts.append((ab,PPL()))
    return pts
by=sorted(lins,key=lambda nm:dens[nm[0]],reverse=True)
rv=list(reversed(by)); rnd=lins[:]; random.shuffle(rnd)
res["order"]={"by":sweep(by),"reverse":sweep(rv),"random":sweep(rnd)}
apply({})

json.dump(res, open(OUTJSON,"w"), indent=2)
print(f"saved {OUTJSON}  (total {time.time()-t0:.0f}s)")
print(f"  MXFP4/32={res['rd']['4'][32]:.2f}  MXINT8/32={res['rd']['8'][32]:.2f}  "
      f"greedy@~4.5b={[p for p in greedy_pts if p[0]<4.7][-1] if any(p[0]<4.7 for p in greedy_pts) else greedy_pts[0]}")
