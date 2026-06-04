# ASIC Cache Simulation Experiment

Simulates a model layer-by-layer to determine which outputs can remain on chip, which transfers must stream through external memory, and how compute/transfer cycles behave under the hardware model.

## Methodology

### Units
All sizes are in **number of FP8 elements** (1 element = 8 bits).  
The cache is specified in **millions of elements** (`--cache_size 2.0` = 2,000,000 elements).

### Element Footprint Calculation
Each tensor's footprint is rounded to the simulator's 128-element transfer chunks, with optional metadata overhead per chunk.

**Formula:**
- `num_chunks = ceil(num_elements / 128)`
- `chunk_elems = num_chunks * 128`
- `metadata_elems = ceil(num_chunks × metadata_bits / 8)`  *(metadata bytes as element-equivalents)*
- `total_footprint = chunk_elems + metadata_elems`

### Cache Banking
The cache is divided into `num_banks` equal banks.  
Any allocation is rounded up to the nearest bank boundary.
- `bank_size = cache_elements / num_banks`  *(in elements)*
- `allocated = ceil(footprint / bank_size) × bank_size`

### Layer Capture
Layers are captured via forward hooks in execution order.  
Captured types include **Conv**, **Linear**, pooling, normalization, residual skip entries, registered quantized ops, and supported functional arithmetic ops such as `QuantMatMul`, `QuantBMM`, `QuantAdd`, and `QuantCat`.

Residual entries record the identity tensor (`xin`) that must be held in cache across the entire residual block, plus the merged output size.

Activations registered in `OpRegistry` are treated as pipeline operations for compute-cycle accounting. They still participate in tensor shape/output propagation, but they contribute `0` compute cycles. If an activation is collapsed into a previous layer, its collapsed compute contribution is also `0`.

### Compute-Cycle Model
The hardware model assumes 128 processing units (PUs) with a two-stage multiply/accumulate pipeline.

Reduction-style ops use 128-wide chunks per output:

- Conv: `output_elems * ceil((in_channels / groups * filter_height * filter_width) / 128) + 1`
- Linear: `output_elems * ceil(in_features / 128) + 1`
- MatMul/BMM: `output_elems * ceil(reduction_dim / 128) + 1`

Elementwise ops such as add/sub/mul/div and residual merges use `ceil(output_elems / 128)`.
Concatenation and registry-marked activations use `0` compute cycles because they are modeled as pipeline/data-movement operations.

Transfer cycles use the same 128-element chunk granularity:

- `transfer_cycles = ceil(elems / 128) * (16 * bits) / bandwidth`

The per-layer cycle estimate is:

- `total_cycles = max(compute_cycles, total_transfer_cycles)`

---

## Rule System

Decisions are made by a **priority-ordered rule cascade**. Each rule specifies:

| Field | Meaning |
|---|---|
| `on_chip` | `True` → output stays in cache after this layer; `False` → output streamed to external memory |
| `xin_from_cache` | `True` → rule requires xin to be fully resident in cache; `False` → xin streamed from external via a 2-bank buffer |
| `match` | predicate — when this rule can apply to a layer |
| `stay` | capacity check — can the layer execute under this rule? |
| `perm` | elements that must remain resident after execution (permanents for the next layer) |

Rules are evaluated in order. A rule is **confirmed** only if:
1. `match(layer)` is true, and
2. `stay(layer)` is true, and
3. The **next layer** has at least one compatible rule given the xin source implied by `on_chip` (1-level lookahead).

If no rule confirms → the layer is **FLAGGED** (cannot execute under any known schedule).

### Rules (in priority order)

| Rule | `on_chip` | `xin_from_cache` | Condition | Notes |
|---|---|---|---|---|
| **r1_global_fit** | ✓ | ✓ | `xin + xout + 2 banks ≤ cache` | Everything fits; both tensors stay on chip |
| **r2_residual** | ✓ | ✓ | Residual or downsample layer; `xin + 3 banks ≤ cache` | xin (skip tensor) in cache; x_residual + xout use 3 banks |
| **r2_conv_output_dominated** | ✓ | ✓ | Conv2d, `xout ≥ xin`; `xout + weights + 2 banks ≤ cache` | xout is larger — hold xout + weights; xin read from cache and freed |
| **r2_conv_input_dominated** | ✓ | ✓ | Conv2d, `xin > xout`; `xin + weights + 2 banks ≤ cache` | xin is larger — hold xin + weights across sliding-window sweep |
| **r2_stream_xin_keep_xout** | ✓ | ✗ | Conv2d; `xout + weights + 2 banks ≤ cache` | xin arrives from external (previous layer evicted); xout still kept on chip |
| **r3_weights_plus_4banks** | ✗ | ✗ | All layers; `weights + 4 banks ≤ cache` | Full streaming: xin and xout both via external; output is QUANTIZE |

> **Linear layers** intentionally skip rules 2a–2d and fall through directly to rule 3.

### xin Source Propagation

`on_chip` from the current layer determines the xin source for the **next** layer:

- Current `on_chip=True` → next layer's xin comes from cache → next layer must match a `xin_from_cache=True` rule.
- Current `on_chip=False` → next layer's xin comes from external → next layer must match a `xin_from_cache=False` rule.

This is enforced by the 1-level lookahead in `evaluate_stay`.

---

## Output

Results are saved to `simulation_results.json` with four sections:

| Section | Contents |
|---|---|
| `metadata` | Cache params, model, bandwidth, timestamp |
| `summary` | Layer counts, quantize/flagged totals |
| `layers` | Per-layer element counts, bank usage, `stay_on_chip`, `xin_from_cache`, transfer bits, BW-limited flags, compute cycles, total cycles, `rule`, `reason` |
| `off_chip_layers` | Layer names whose outputs must be quantized for external memory |

Console output columns: `Layer Name | Type | Input | Weights | Output | Banked | NextXin | OnChip | inB | wB | outB | Reason`

---

## Runner Integration

Point any run config to the simulation results. Off-chip layers inherit the run's
`format` and `mode` — no quantization format is baked into the simulation file.

```yaml
quantization:
  format: fp8_e4m3   # off-chip layers inherit this format
  mode: chunk
  cache_simulation_path: runspace/experiments/asic_cache_simulation/simulation_results.json
```

Explicit entries under `quantization.layers` always take priority over simulation overrides.

---

## Parameters

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `resnet18` | Model to analyze |
| `--cache_size` | `2.0` | Cache size in **millions of elements** |
| `--num_banks` | `16` | Number of cache banks |
| `--metadata_bits` | `0` | Extra metadata bits per 128-element chunk |
| `--batch_size` | `1` | Batch size for activation shape calculation |
| `--device` | `cuda` | Device for the dummy forward pass |
| `--bandwidth` | `1.0` | Memory bandwidth in bytes/cycle for transfer-cycle and BW-limitation analysis |

## Running

```bash
python runspace/experiments/asic_cache_simulation/simulate_cache.py \
    --model_name resnet18 --cache_size 2.0 --metadata_bits 16 --bandwidth 1.0
```
