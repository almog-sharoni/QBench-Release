import torch, torchvision as tv
from runspace.src.ops.quant_base import quantize_tensor
from runspace.experiments.find_optimal_weight_quant.find_optimal_weight_quant import get_quantized_tensor_sim
SIGNED={8:['fp8_e4m3','fp8_e5m2','fp8_e3m4','fp8_e2m5','fp8_e1m6','fp8_e6m1','fp8_e7m0'],
7:['fp7_e1m5','fp7_e2m4','fp7_e3m3','fp7_e4m2','fp7_e5m1','fp7_e6m0'],
6:['fp6_e1m4','fp6_e2m3','fp6_e3m2','fp6_e4m1','fp6_e5m0'],
5:['fp5_e1m3','fp5_e2m2','fp5_e3m1','fp5_e4m0'],
4:['fp4_e1m2','fp4_e2m1','fp4_e3m0'],3:['fp3_e1m1','fp3_e2m0']}
def flat_best(w,fmts):
    be,bf=1e30,None
    for f in fmts:
        dq,_=quantize_tensor(w,q_type=f,mode='chunk',chunk_size=128,validate=False)
        e=(w-dq).pow(2).mean().item()
        if e<be: be,bf=e,f
    return bf
def ctx_best(w,fmts):
    be,bf=1e30,None
    for f in fmts:
        dq,_=get_quantized_tensor_sim(w,f,chunk_size=128)
        e=(w-dq).pow(2).mean().item()
        if e<be: be,bf=e,f
    return bf
for mname,ctor in [("resnet18",tv.models.resnet18),("efficientnet_b0",tv.models.efficientnet_b0)]:
    m=ctor(weights="DEFAULT").cuda().eval()
    tot=flips=0; flip_layers=[]
    for n,mod in m.named_modules():
        if isinstance(mod,(torch.nn.Conv2d,torch.nn.Linear)) and mod.weight is not None:
            w=mod.weight.detach().float().cuda()
            if w.dim()<2: continue
            for b in [8,7,6,5,4,3]:
                ff=flat_best(w,SIGNED[b]); cf=ctx_best(w,SIGNED[b]); tot+=1
                if ff!=cf:
                    flips+=1; flip_layers.append((n,b,ff,cf))
    print(f"\n### {mname}: {flips}/{tot} (layer,bit) format picks FLIP flat->context  ({100*flips/tot:.1f}%)")
    for n,b,ff,cf in flip_layers[:12]:
        print(f"   {n:42s} {b}b: {ff} -> {cf}")
