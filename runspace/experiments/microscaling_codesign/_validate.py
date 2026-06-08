import os, sys, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mx_lib import mxfp4, mxint8, rtn_int, eff_bits, decoder_linears, wikitext2_ppl
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV="cuda"
print("== quantizer sanity ==")
print(" eff_bits MXFP4 blk32 =", eff_bits(32,4), "(want 4.25)")
w = torch.randn(256,256, device=DEV)*0.1
q = mxfp4(w, 32)
uniq = torch.unique((q/ (2.0**torch.floor(torch.log2(q.abs().amax())))).abs())
print(" MXFP4 quantized; max|q|=%.4f  #unique vals(sample)=%d" % (q.abs().max().item(), torch.unique(q).numel()))
print(" MXINT8 err vs MXFP4 err (rand):", round((w-mxint8(w,32)).pow(2).mean().item(),6),
      round((w-mxfp4(w,32)).pow(2).mean().item(),6))

print("\n== load OPT-125M ==")
tok = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16).to(DEV).eval()
fp = wikitext2_ppl(model, tok, DEV, seqlen=2048)
print(f" OPT-125M FP16 wikitext2 ppl = {fp:.2f}   (published: 27.65)")

ref = {k:v.detach().clone() for k,v in model.state_dict().items()}
lins = decoder_linears(model); print(f" decoder linears quantized: {len(lins)}")
def apply(fn):
    model.load_state_dict(ref)
    for n,m in lins: m.weight.data = fn(m.weight.data.float()).half()
for name,fn in [("MXFP4 blk32", lambda w: mxfp4(w,32)),
                ("MXFP4 blk128", lambda w: mxfp4(w,128)),
                ("MXINT8 blk32", lambda w: mxint8(w,32)),
                ("RTN-INT4 per-channel", lambda w: rtn_int(w,4,None)),
                ("RTN-INT4 g128", lambda w: rtn_int(w,4,128))]:
    apply(fn); p = wikitext2_ppl(model, tok, DEV, seqlen=2048)
    print(f"   {name:<22} ppl = {p:.2f}")
model.load_state_dict(ref)
