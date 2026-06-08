"""
Microscaling co-design study — core library.

Quantizers are OCP-MX-spec faithful where applicable:
  MXFP4 : E2M1 elements (values {0,.5,1,1.5,2,3,4,6}), shared E8M0 scale
          X = 2^(floor(log2(amax)) - emax_elem), emax(E2M1)=2, saturating.
  MXINT8: INT8 elements, shared E8M0 scale.
Baselines for literature comparison:
  RTN-INT4 per-channel and per-group(128), symmetric round-to-nearest.

Perplexity uses the standard GPTQ/HF wikitext-2 protocol (seqlen 2048,
non-overlapping windows) so FP numbers are directly comparable to papers.
"""
import math, torch, torch.nn as nn

# ----------------------------- quantizers -----------------------------
_E2M1 = None
def _e2m1(device):
    global _E2M1
    if _E2M1 is None or _E2M1.device != device:
        _E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.], device=device)
    return _E2M1

def mxfp4(w, block):
    """OCP MXFP4 (E2M1 + E8M0), block along the flattened tensor."""
    vals = _e2m1(w.device); shp = w.shape; flat = w.reshape(-1)
    pad = (-flat.numel()) % block
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.reshape(-1, block)
    amax = blk.abs().amax(1, keepdim=True)
    shared_exp = torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2.0      # emax(E2M1)=2
    shared_exp = torch.where(amax > 0, shared_exp, torch.zeros_like(shared_exp))
    X = torch.exp2(shared_exp.clamp(-127, 127))
    x = blk / X
    idx = (x.abs().unsqueeze(-1) - vals).abs().argmin(-1)                   # nearest E2M1 (saturates at 6)
    q = (x.sign() * vals[idx]) * X
    return q.reshape(-1)[:shp.numel()].reshape(shp)

def mxint8(w, block, nbits=8):
    qmax = 2**(nbits-1) - 1
    shp = w.shape; flat = w.reshape(-1); pad = (-flat.numel()) % block
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    blk = flat.reshape(-1, block)
    amax = blk.abs().amax(1, keepdim=True).clamp(min=1e-30)
    shared_exp = torch.floor(torch.log2(amax)) - math.floor(math.log2(qmax))
    X = torch.exp2(shared_exp.clamp(-127, 127))
    q = torch.clamp(torch.round(blk / X), -qmax, qmax) * X
    return q.reshape(-1)[:shp.numel()].reshape(shp)

def rtn_int(w, nbits=4, group=None):
    """Symmetric round-to-nearest INT. group=None -> per output channel (row)."""
    qmax = 2**(nbits-1) - 1
    if group is None:                                   # per-channel (per row)
        amax = w.abs().amax(1, keepdim=True).clamp(min=1e-12)
        s = amax / qmax
        return torch.clamp(torch.round(w / s), -qmax, qmax) * s
    shp = w.shape; flat = w.reshape(-1); pad = (-flat.numel()) % group
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.reshape(-1, group); amax = g.abs().amax(1, keepdim=True).clamp(min=1e-12)
    s = amax / qmax
    q = torch.clamp(torch.round(g / s), -qmax, qmax) * s
    return q.reshape(-1)[:shp.numel()].reshape(shp)

def eff_bits(block, b, scale_bits=8):
    return b + scale_bits / block

# ----------------------------- model utils -----------------------------
def decoder_linears(model):
    """Linear layers inside decoder blocks (exclude lm_head / embeddings), GPTQ-style."""
    out = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and ('lm_head' not in n) and ('embed' not in n):
            out.append((n, m))
    return out

# ----------------------------- perplexity -----------------------------
def load_wikitext2_enc(tok):
    """Tokenize the wikitext-2 (raw) test set ONCE; reuse the tensor for all evals."""
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return tok("\n\n".join(test["text"]), return_tensors="pt").input_ids

@torch.no_grad()
def ppl_from_enc(model, enc, device="cuda", seqlen=2048, max_windows=None):
    """Standard GPTQ/HF wikitext-2 perplexity from a pre-tokenized id tensor."""
    nwin = enc.numel() // seqlen
    if max_windows: nwin = min(nwin, max_windows)
    nlls = []
    for i in range(nwin):
        batch = enc[:, i*seqlen:(i+1)*seqlen].to(device)
        logits = model(batch).logits
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
        lb = batch[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(sl, lb)
        nlls.append(loss.float() * (seqlen - 1))
    return math.exp(torch.stack(nlls).sum().item() / (nwin * (seqlen - 1)))

@torch.no_grad()
def wikitext2_ppl(model, tok, device="cuda", seqlen=2048, max_windows=None):
    return ppl_from_enc(model, load_wikitext2_enc(tok), device, seqlen, max_windows)
