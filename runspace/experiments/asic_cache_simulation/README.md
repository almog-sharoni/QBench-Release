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

Every tensor operand records its runtime producer, consumer input index, size,
storage identity, and whether it is an explicit model input or model state.
Residual roles are classified after the complete graph is known, so projected
skips and attention layout operations do not depend on hook execution order.

Before a cache graph is stored, fail-fast validation rejects unresolved
internal tensors, backward producer edges, incomplete `QuantAdd` operands,
duplicate residual/hold lifetimes, edge-size drift, non-contiguous resident
lifetimes, or an active cache column without a graph arrow to highlight. Only
explicit forward inputs may produce an `Input` node. Stored graph metadata
contains `graph_validation=passed` and `unresolved_internal_inputs=0`; older
graph schema versions are regenerated automatically.

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

## Producer-Consumer Lifetime Policy

Cache placement uses the same explicit runtime graph as the cache map and
dashboard architecture graph. It is not limited to the previous and next
layer.

1. Each produced activation is allocated once.
2. Its lifetime ends after its final consumer.
3. A layer input that references a resident producer reuses that allocation;
   it is not counted again as `x_in`.
4. Fan-out values such as LayerNorm→Q/K/V and long residual skips remain live
   through every consumer.
5. When the live set exceeds capacity, the simulator evicts the buffer whose
   next use is farthest in the future. Evicted values are written externally
   and reloaded if a later consumer needs them.
6. Model parameters and buffers are transfer costs, not long-lived activation
   allocations.

`producer_consumer_resident` means the output survives all required cache
decisions without an external write. `producer_consumer_spill` means it is
written externally immediately, has no future consumer, or is evicted later.
The JSON records the live producer indices, evictions, reloads, required cache,
and resident cache before and after every layer.

### Rule-aware operator workspace

Producer-consumer lifetimes are combined with the ordered hardware rules in
`rules.py`. A rule may reuse an input allocation for the output only when the
current layer is that tensor's final consumer. Convolution rules account for
the shared input/output region, a read/write pipeline-boundary bank, and the
full or scaled jumpback allocation. QuantAdd may overwrite the largest
resident final-use operand. Streamed weights/model state use two banks. The
greedy connection-placement pass evaluates every candidate against these
rule-aware totals rather than `x_in + x_out` unconditionally.

### Pointwise activation fusion

Unary, shape-preserving pointwise activations such as ReLU, GELU, SiLU,
Hardswish, Sigmoid, and Tanh are modeled as part of their direct producer's
hardware pipeline. Their standalone row/node is removed, consumers are
rewired to the producer's post-activation output, and the dashboard hover card
lists the fused activation. Fusion is applied only when no other consumer
needs the producer's raw pre-activation value. Normalization operations and
Softmax remain explicit scheduled layers.

### Greedy cache-map residency selection

The generated cache-map CSV applies a second, connection-level greedy pass to
the residual and `hold_N` lifetimes:

1. Round every activation allocation up to a whole cache bank.
2. Reserve two banks for streamed weights/model state.
3. Mark layers whose current bank requirement exceeds the configured cache as
   red.
4. Temporarily stream each remaining residual/hold candidate and count how many
   currently red layers become green.
5. Select the candidate solving the most layers. On a tie, select the smaller
   bank-rounded tensor; if the sizes also tie, select the first cache-map
   column.
6. Replace the selected tensor's resident lifetime with a two-bank buffer at
   its producer and at each required reload consumer, then repeat. For a
   residual, the immediately adjacent pipeline consumer still receives the
   producer's normal `x_out`→`x_in` handoff and is not falsely treated as a
   residual reload.
7. Stop when all layers fit or no remaining candidate solves a red layer.

The connection column contains its full bank-rounded allocation while
resident. For a selected streamed connection it contains two banks only on
stream-out/stream-in rows and zero between them. `weight_stream` exposes the
two-bank weight/model-state buffer, so `total_cache_needed_kb` reflects the
optimized bank allocation.

Dashboard red arrows combine these connection placements with the full
producer-consumer cache plan. A direct output that cannot remain resident is
marked as an `x_out` stream to the exact consumer that reloads it. A streamed
residual marks only its residual add/reload arrow; it does not color an
otherwise on-chip next-layer edge red.

---

## Output

Results are saved to `simulation_results.json` with these sections:

| Section | Contents |
|---|---|
| `metadata` | Cache params, model, bandwidth, timestamp |
| `summary` | Layer counts, quantize/flagged totals |
| `layers` | Per-layer element counts, bank usage, `stay_on_chip`, `xin_from_cache`, transfer bits, BW-limited flags, compute cycles, total cycles, `rule`, `reason` |
| `cache_map` | Numeric matrix with one row per layer and fixed `x_in`, `x_out`, and `residual_N` columns |
| `off_chip_layers` | Layer names whose outputs must be quantized for external memory |

Console output columns: `Layer Name | Type | Input | Weights | Output | Banked | Required | Resident | OnChip | inB | wB | outB | Reason`

The cache map is also printed after the layer report and saved to
`cache_map.csv` plus a model-specific `cache_map_<model>.csv`. Every numeric
data cell is in decimal KB (`1 KB = 1,000 bytes`, and one FP8 element is one
byte). In CLI-generated maps, values are whole-bank allocations and
`total_cache_needed_kb` reports the optimized resident and streaming bank
requirement. A resident residual column contains its tensor allocation from
the producer layer through the consuming add layer and contains `0` elsewhere;
a streamed residual instead contains two banks only at its transfer endpoints. Residual
adds are identified from activation-tensor provenance, so
the bypass operand may appear on either side of the add; static parameters such
as ViT positional embeddings are excluded. The cache map retains explicit
LayerNorm, activation, and softmax rows; the main simulation now uses this same
un-collapsed runtime schedule.

The tracer also creates `hold_N_<producer>_to_<consumer>` columns automatically
for non-adjacent or multi-consumer activation tensors. Detection uses runtime
producer lineage and follows tensor views, so attention dependencies such as
LayerNorm fan-out to Q/K/V, Q and K retention through scaled dot product, and V
retention through attention-weighted values require no model-specific rules.

Each physical tensor is shown in only one data column per row. It appears as
`x_out` on its producer row, then moves to its named residual/hold column. If a
later layer uses that held tensor as its first operand, `x_in` is `0` on that
row because the same allocation is already represented by the named column.
This convention makes the displayed cache-size columns additive and prevents
visual double-counting; the connection name still identifies the input source.

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
| `--cache_map_only` | disabled | Write only `cache_map_<model>.csv`; skip the cache simulation, JSON files, database upload, and full report |

## Running

```bash
python runspace/experiments/asic_cache_simulation/simulate_cache.py \
    --model_name resnet18 --cache_size 2.0 --metadata_bits 16 --bandwidth 1.0
```

To produce only the cache-map CSV:

```bash
python runspace/experiments/asic_cache_simulation/simulate_cache.py \
    --model_name resnet18 --device cpu --cache_map_only
```

This still performs the model forward trace required to discover tensor sizes
and residual connections, but it does not run the cache rules or write any
other result files.
