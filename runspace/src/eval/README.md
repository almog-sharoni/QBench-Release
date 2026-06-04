# Evaluation

This directory contains the model evaluation / comparison machinery and its
configuration files.

## `LayerComparator` (`comparator.py`)

`LayerComparator` runs an FP32 **reference** model and a **quantized** model on
the same data and produces a per-layer report (`comparison_report.txt`):
accuracy, per-layer activation/weight error (MSE + cosine similarity), FP8
compliance, dynamic-range (XMax) stats, and chunk-wise scale exponents.

### How it works

1. **Hook registration.**
   - On the **reference** model, a `forward_pre_hook` on every supported op
     records the layer's input into `ref_activations[name]`, and its weight into
     `ref_weights[name]`.
   - On the **quantized** model, leaf modules get a `forward_hook` that records
     the (quantized) input into `quant_activations[name]` and appends to an
     ordered `call_log`. Quant layers also set `capture_activations = True`, so
     each layer stashes `last_quant_input` (the input *after* quantization,
     scaled back), `last_quant_weight`, scales, and XMax buffers during forward.

2. **Comparison loop** (`compare`). For each batch:
   - the reference model runs a full forward pass (inputs un-quantized);
   - the quantized model runs a full forward pass under `quantized_dispatch`;
   - `_compute_layer_metrics()` matches layers **by module name** and accumulates
     MSE/cosine and the other stats.

3. **Report** (`_generate_report`) averages the accumulated metrics over all
   batches and writes the tables to `comparison_report.txt`.

> **Layer matching is by name.** The reference model is built with the same
> structure as the quantized model (same op replacement, BN/input-norm folding,
> MHA decomposition) so that `named_modules()` line up one-to-one.

> **Cosine similarity is magnitude-invariant.** Per-output-channel weight
> quantization preserves direction, so `Weight CosSim ≈ 1.0000` is expected and
> healthy — read the `Weight MSE` column for the actual weight error.

### Compare modes

Selected via `evaluation.compare_mode` in the run config (or the constructor's
`compare_mode=` argument). The headline two columns of the *Layer Comparison*
table change meaning with the mode; everything else is identical.

| | `propagated` (default) | `isolated` (teacher-forced) |
|---|---|---|
| **Data flow** | Both models run independently, end-to-end. Each layer sees the activation that its *own* network produced upstream. | The reference runs first; then each quantized layer is **fed the FP32 reference input** for that same layer (via an injection `forward_pre_hook`). |
| **Headline columns** | `Input MSE` / `Input CosSim` — the divergence of the activation *arriving at* each layer. | `Out MSE` / `Out CosSim` — the divergence of the activation *leaving* each layer. |
| **What it measures** | **Cumulative** divergence: a layer's number includes all error inherited from every layer before it, plus amplification. | **Isolated** per-layer error: weight quant + input quant + that layer's compute, with **no inherited drift**. |
| **Answers** | "How far apart have the two networks drifted by this point?" | "How much error does *this specific layer* introduce on a clean input?" |
| **End-to-end accuracy** | Real quantized-model accuracy. | **Not meaningful** — every layer was teacher-forced, so the final logits don't reflect the real model. Use the per-layer table; the report prints a warning. |

#### Why both exist

In `propagated` mode a clean input (`CosSim ≈ 0.999`) followed by a sharply
degraded next-layer reading does *suggest* a local amplifier, but it conflates
the layer's own quantization with its amplification of inherited drift — you
cannot attribute the error to one layer. `isolated` mode removes the inherited
component: a high `Out MSE` there means the layer genuinely struggles to quantize
its (clean) input, independent of upstream error. Use `propagated` to see where
error *accumulates* in a real run, and `isolated` to find which layers are the
intrinsic *sources*.

#### Implementation notes for `isolated` mode

- The reference hooks additionally capture each layer's **full input tuple**
  (`ref_layer_inputs`) and **output** (`ref_layer_outputs`); the quant hooks
  capture the teacher-forced **output** (`quant_layer_outputs`). All three reset
  per batch.
- Injection replaces every positional tensor arg whose shape matches the
  reference, so multi-operand ops (`QuantAdd`, `QuantMatMul`) are teacher-forced
  on all operands.
- Because each leaf layer's input is overwritten, the quant model's actual
  data-flow between layers is discarded — only the single injected layer's output
  is measured. This is intentional and is what makes the per-layer error isolated.

### Usage

```yaml
# in a run config
evaluation:
  mode: compare
  compare_mode: isolated   # or 'propagated' (default)
  compare_batches: 50
```

From the bandwidth-aware experiment:

```bash
./apptainer.sh runspace/experiments/bandwidth_aware_quant/bandwidth_aware_quant.py \
    --model_name mobilevit_s --cache_size 4.0 --b_bits 8 \
    --mode compare --compare_mode isolated --compare_batches 50
```

## `compliance_config.yaml`

This file controls which parameters `LayerComparator` checks for quantization compliance.

### Structure

*   `quantized_params`: A list of attribute names to check on each layer module.
    *   If a module has an attribute with one of these names (and it is not None), the comparator will verify that its values are valid according to the layer's quantization type (e.g., FP8 E4M3).
    *   Common attributes include:
        *   `weight_fp8`: Explicitly quantized weights.
        *   `weight`: Standard weights (checked if `weight_fp8` is missing).
        *   `last_quant_rm`: Quantized running mean (from BatchNorm).
        *   `last_quant_rv`: Quantized running variance (from BatchNorm).

### Adding New Parameters

To add a new quantized parameter to the check (e.g., if you implement a new layer that stores `my_param_quant`), simply add `my_param_quant` to the `quantized_params` list in `compliance_config.yaml`. No code changes are required in `comparator.py`.
