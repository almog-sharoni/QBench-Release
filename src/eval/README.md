# Evaluation Configuration

This directory contains configuration files for the evaluation and compliance checking process.

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
