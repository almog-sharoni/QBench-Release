# Bandwidth-Aware Quantization Experiment

This experiment evaluates model accuracy under a bandwidth-aware mixed-precision scheme guided by cache simulation analysis. It profiles layer-wide transfer quantization (sweeping precision floors from 3 to 8 bits) and plots Top-1 Accuracy against normalized speedup vs FP32 0MB.

## Key Features

1. **ASIC Cache Simulation Integration**
   - Integrates model stay decisions (`stay_on_chip = True/False`) and `xin_from_cache` flags for cache configurations (0MB, 2MB, 4MB).
   - On-chip layers use FP8 input activation candidates; transferred layer traffic uses a layer-wide optimized bit-width.
   - Input transfer detection accounts for the rule's `xin_from_cache` flag — layers streaming input from external are always flagged for input transfer even if the previous layer's output was cached.

2. **Layer-Wide Transfer Bit-Width Optimization**
   - For each layer, total transfer cycles for required input, weight, and output traffic are compared against compute cycles.
   - Inputs are transferred when needed by execution order and cache rules; outputs are transferred only when the layer does not stay on chip.
   - **Weights are always streamed in** from external memory and are always part of the transfer check.
   - Starting from 8 bits, every transferred component in the layer is reduced by 1 bit together while total transfer remains bandwidth-limited.
   - Reduction stops when the layer becomes compute-limited or every transferred component reaches the sweep floor (`b_bits`).
   - The code still records input/weight/output bit-width maps separately because some components may not be transferred, but all transferred components for a layer step down together.

3. **Cycle-Accurate Hardware Modeling**
   - Conv, Linear, and MatMul compute cycles use 128-wide reduction chunks per output (`num_outputs * ceil(reduction_dim / 128) + 1`), matching the two-stage multiply/accumulate pipeline.
   - Elementwise ops use `ceil(output_elems / 128)`, while concatenation and registry-marked activation layers use `0` compute cycles because they are modeled as pipeline/data-movement operations.
   - Transfer cycles are computed in 128-element chunks (`ceil(elems / 128) * 16 * bits / bandwidth`), reflecting the chunk granularity of the memory subsystem.
   - Core runtime per layer is modeled as `max(compute_cycles, total_transfer_cycles)`.
   - Quantized layer types (`QuantConv2d`, `QuantLinear`, `QuantMatMul`, `QuantBMM`, arithmetic ops) are handled with the same hardware model as their native or functional counterparts.

4. **FP32 Reference Baseline**
   - A full-precision (FP32, unquantized) evaluation is run once per model.
   - Reference cycles are computed at 32 bits (4 bytes/element) with min=max=32 (no optimization).
   - Plotted as a black-edged diamond on the accuracy-vs-normalized-speedup chart for each cache size.

5. **Unsigned Input Activation Hooking**
   - Detects layers following `relu`, `relu6`, and `softmax` activations to enable Unsigned Floating Point (UFP) candidates.
   - Per-layer input candidates are computed from the layer's optimized input bit-width rather than the sweep-wide global bit-width.

6. **Advanced Plotting & Annotation**
   - X-axis uses normalized speedup vs FP32 0MB: `time_ref / time`, where FP32 with 0MB cache is the `1.0x` baseline and higher values are faster.
   - A broken x-axis is added automatically when the FP32 0MB baseline is separated from the main result cluster by a large empty range.
   - Jitters Cache 2MB (0.98x) and 4MB (1.02x) curves to distinguish overlapping points.
   - Deduplicates flat region annotations (grouped as range `3-8`) to prevent text blobs.
   - Uses large cache-specific markers with bit-width labels centered inside them; sweep cycle ranges are shown in the legend while FP32 reference cycles remain beside the reference diamonds.

## Command Line Usage

### Run on a single model:
```bash
./apptainer.sh runspace/experiments/bandwidth_aware_quant/bandwidth_aware_quant.py \
    --model_name resnet18 \
    --limit_batches 2
```

### Run on a list of models defined in a YAML file:
```bash
./apptainer.sh runspace/experiments/bandwidth_aware_quant/bandwidth_aware_quant.py \
    --model_name runspace/inputs/models.yaml \
    --limit_batches 2
```

### Plot per-layer bit allocations without inference:
```bash
./apptainer.sh runspace/experiments/bandwidth_aware_quant/plot_layer_bit_allocations.py \
    --model_name resnet18 \
    --cache_sizes 0 2 4 \
    --thresholds 2 3 4 5 6 7 8
```

This generates one `layer_weight_bits_cache_*.png` plot per cache size. The
x-axis is layer number, the y-axis is the per-layer bit-width, and each line is
a minimum bit threshold. The lowest threshold is plotted last as a solid black
line so it stays visually in front; the other thresholds are dashed.

### Options:
- `--model_name`: Model name (e.g. `resnet18`) or path to a YAML file containing a list of models.
- `--bandwidth`: Memory bandwidth in bytes/cycle (default: `1.0`).
- `--limit_batches`: Number of evaluation batches (default: `-1` for all).
- `--device`: Target execution device (`cuda` or `cpu`).
- `--batch_size`: Batch size for evaluation (default: `128`).
