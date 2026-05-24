# Bandwidth-Aware Quantization Experiment

This experiment evaluates model accuracy under a bandwidth-aware mixed-precision scheme guided by cache simulation analysis. It profiles off-chip layer quantization (sweeping precision from 2 to 8 bits) and plots Top-1 Accuracy against execution time (cycles).

## Key Features

1. **ASIC Cache Simulation Integration**
   - Integrates model stay decisions (`stay_on_chip = True/False`) for cache configurations (0MB, 2MB, 4MB).
   - On-chip layers stay in FP8 format; off-chip layers are quantized dynamically.

2. **Compute Time Cycle Modeling**
   - Sums compute cycles of parent operations and their collapsed child operations (activations, normalization, pooling, softmax) to model pipeline execution.
   - Calculates memory transfer cycles under bandwidth constraints (configurable via `--bandwidth`).
   - Weights are transferred at target bit precision, while input/output activations are transferred at 8-bit precision.
   - Core runtime is modeled as `max(compute_cycles, transfer_cycles)`.

3. **Dynamic Precision Selection (Off-Chip)**
   - If an off-chip layer's memory transfer bottleneck becomes compute-limited at a higher precision $k$ than the minimum target $b$, the script dynamically elevates the layer to use $k$ bits.
   - This maximizes accuracy without increasing execution runtime (cycles).

4. **Unsigned Input Activation Hooking**
   - Detects layers following `relu`, `relu6`, and `softmax` activations to enable Unsigned Floating Point (UFP) candidates.

5. **Advanced Plotting & Annotation**
   - Unified scales for log-scaled cycles and accuracy.
   - Jitters Cache 2MB (0.98x) and 4MB (1.02x) curves to distinguish overlapping points.
   - Deduplicates flat region annotations (grouped as range `2-8`) to prevent text blobs.
   - Colors cycle count labels in charcoal gray (`#495057`) separate from bold bit-width labels.

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

### Options:
- `--model_name`: Model name (e.g. `resnet18`) or path to a YAML file containing a list of models.
- `--bandwidth`: Elements transferred per cycle (default: `1.0`).
- `--limit_batches`: Number of evaluation batches (default: `-1` for all).
- `--device`: Target execution device (`cuda` or `cpu`).
- `--batch_size`: Batch size for evaluation (default: `128`).
