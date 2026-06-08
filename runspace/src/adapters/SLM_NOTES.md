# SLM / LLM Quantization (SLMAdapter) — Notes & Known Concerns

Support for quantizing real decoder-only LLMs (first target: `facebook/opt-125m`)
through the existing PyTorch quant ops. Entry points:

- `slm_adapter.py` — `SLMAdapter` (adapter type `slm`)
- `../ops/quant_opt_attention.py` — `QuantOPTAttention` (quantized attention score path)
- `../datasets/wikitext.py` — `wikitext2_lm` loader (tokenized fixed-length blocks)
- `../configs/opt125m_fp8.yaml` — example config (fp8_e4m3, per-context chunking)
- `scratch/_slm_smoke.py` — FP vs FP8 perplexity harness
- `scratch/_slm_compliance.py` — coverage / FP8 compliance report

## How it works

`AutoModelForCausalLM` loads the model with `attn_implementation="eager"`. The
GenericAdapter recursive replace swaps `nn.Linear → QuantLinear`,
`nn.LayerNorm → QuantLayerNorm`, `nn.ReLU → QuantReLU`. Then `SLMAdapter._quantize_attention`
swaps each `OPTAttention → QuantOPTAttention`, routing QKᵀ / softmax / attn·V
through `QuantBMM` / `QuantSoftmax`. Perplexity comes from `MetricsEngine`'s 3-D
logits branch.

Recommended config: per-context chunking, **separate** for weights and inputs —
`mode: chunk` + `chunk_size: 128` (inputs), `weight_mode: chunk` +
`weight_chunk_size: 128` (weights). chunk_size is fixed at 128 by the CUDA codec.

Verified result (full wikitext-2 test): full fp8 W+A+attention, chunked = **+3.24
ppl** (38.5 → 41.8). Coverage 146/148 (98.6%), **supported-but-unquantized = 0**.

---

## Known concerns / limitations

1. **Attention `bmm` chunking is not contraction-aligned.** Attention operands
   have `head_dim = 64`; the per-context chunker pads them to the 128-wide codec
   block. It quantizes correctly, but for the **K** operand the chunk axis is not
   the matmul contraction axis, so that scale granularity is approximate. The
   Linear layers (≈99% of compute/params) chunk cleanly along the true
   contraction. A contraction-aligned attention chunker is a TODO.

2. **Per-tensor activation quant is effectively broken for LLMs.** Activation
   outliers blow out a single per-tensor scale (full per-tensor W+A+attention
   gives ppl in the hundreds–thousands). Per-context **chunk** mode is required
   for usable activation quantization — treat per-tensor as weights-only.

3. **Masked-softmax quantization uses a clamp hack.** HF adds `finfo.min`
   (~-3.4e38) to masked logits; a per-tensor/per-context quantized softmax would
   let that value capture the entire scale and zero out the real logits (ppl
   exploded to ~5500 before the fix). `QuantOPTAttention.softmax_mask_floor = -30`
   clamps masked logits to a representable floor. Caveat: a *valid* logit below
   -30 (rare after `* head_dim**-0.5` scaling) would also be clamped, and the
   floor value is tied to the fp8 range — revisit if using a very different
   format or much larger logits.

4. **`QuantOPTAttention.forward` is a verbatim copy of transformers 4.46.3.** It
   will silently diverge (or break) if `transformers` is upgraded — re-sync the
   forward body against the installed version. It is also **OPT-specific**; other
   architectures (Llama, GPT-NeoX, …) need their own attention subclass.

5. **GPT-2 won't quantize.** It uses HF's `Conv1D` (not `nn.Linear`), which the
   op registry doesn't match, so its projections stay FP. Use models built from
   `nn.Linear` (OPT, Pythia/GPT-NeoX, GPT-Neo).

6. **`Runner.run_single` not validated for SLM.** The verified paths are
   `Runner.evaluate_model` (eval/perplexity) and `LayerComparator` directly (the
   compliance report). The full `run_single` flow (DB logging, weight
   materialization, reference-model comparison wiring) has **not** been exercised
   end-to-end for the `slm` adapter.

7. **Coverage is measured by module iteration, not FX.** HF decoder forwards are
   not `torch.fx`-traceable, so `_verify_coverage_fx` falls back to counting leaf
   **modules**. Functional ops are invisible to it — which is exactly why the
   attention score ops had to be made real modules (`QuantBMM`/`QuantSoftmax`).
   "Supported-but-unquantized = 0" is the meaningful signal; the two embedding
   tables are (correctly) counted as unsupported and left FP.

8. **Environment is non-standard.** `transformers`/`datasets`/`tokenizers` are
   installed in `runspace/.pylibs` (the sandbox image is read-only), with
   `huggingface_hub` pinned `<1.0` to satisfy transformers 4.46. `apptainer.sh`
   was modified to add `.pylibs` to `PYTHONPATH` and set a project-local
   `HF_HOME`. Re-running `pip install --target=runspace/.pylibs` is how to add
   packages.

9. **Evaluation methodology.** `wikitext2_lm` uses non-overlapping fixed-length
   blocks (no sliding window / context carryover), so absolute perplexity is
   higher than published numbers (OPT-125M ≈ 27 with the standard stride recipe).
   FP-vs-FP8 deltas are the meaningful comparison, not the absolute ppl.

10. **Decode/KV-cache path untested.** Eval runs a single full-sequence forward;
    the `past_key_value` / incremental-generation path through `QuantOPTAttention`
    is implemented (copied) but not exercised or validated.

---

## Embedding placement on a pipelined-accelerator + RISC-V architecture

(Design discussion — not implemented. Relevant to whether/how to quantize the
embedding table; see concern about embeddings above.)

**The embedding lookup belongs on the RISC-V (orchestrating DMA), not on the PE
array.** An `nn.Embedding` is a *gather*, not a MAC: for token id `t` you fetch
row `t` of the table (768 values). There is no contraction dimension, no
accumulation — nothing for a 128-PE systolic array to do. Forcing it onto the
accelerator means a `one_hot(vocab) × table` matmul (a 50272-wide multiply per
token to read one row — absurd). It is also a **data-dependent address** (the id
comes from the input), and irregular indexed gathers break the dense, statically
scheduled dataflow the pipeline wants. That is exactly scalar-core + DMA work:

- RISC-V computes the address (`id × d_model × bytes`) and programs a DMA descriptor;
- DMA streams the row(s) from DRAM/HBM into the activation buffer;
- token + position vectors are added elementwise (RISC-V / small vector unit, or
  folded into the first layer's input load).

**How many times it happens: once per token position, once total — NOT multiplied
by the number of layers.** This is the key contrast with Linear/attention, which
run per token *per layer*.

| Phase | Embedding lookups |
|---|---|
| Prefill (prompt length S) | S token gathers + S position gathers, once |
| Decode (generate T tokens) | 1 gather per step → T total |
| Full generation | **S + T** |
| Perplexity eval (single forward, batch B × seq L) | B × L gathers, one pass |

Cost is `O(S)` row reads of `d_model` elements — negligible next to the
`O(S · d² · L)` MACs on the accelerator, and in decode it is a single
`d_model`-vector fetch per step, fully hidden under layer compute.

**Asymmetry with the output side:** `lm_head` (vocab projection, tied to the same
table in OPT) *is* a real matmul `[S×d] × [d×vocab]` and runs **on the
accelerator**, once per emitted position. So the same matrix is touched two ways —
input side = RISC-V gather, output side = PE-array matmul.

**Implication for quantization:** storing the table in fp8 mainly saves DRAM
footprint/bandwidth on that large table (~31% of OPT-125M params). The dequant is
a short scalar/vector op the RISC-V does once per token — never on the critical
MAC path — so the *cost* case is favorable precisely because it is RISC-V/DMA
work. The open question is purely *accuracy* (most sensitive tensor). Sane design:
quantize the table for storage, dequant on the RISC-V during the gather, validate
the ppl hit, and keep it FP / higher-mantissa if it does not pay off — and keep it
consistent with the tied `lm_head`.
