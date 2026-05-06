# QBench Configuration Guide

This document explains all available configuration options for the QBench framework. Configuration files are in YAML format.

## Structure

The configuration file is divided into the following sections:
- `model`: Defines the base model to load.
- `adapter`: Configures the quantization adapter and layer replacement.
- `quantization`: Specifies global quantization parameters.
- `dataset`: Configures the dataset for evaluation.
- `evaluation`: Controls the evaluation process.

---

## 1. Model Configuration (`model`)

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `name` | string | Name of the model architecture. | `resnet18`, `vit_b_16` |
| `source` | string | Source of the model (`torchvision` or `custom`). | `torchvision` |
| `weights` | string | Pretrained weights identifier. | `IMAGENET1K_V1` |

## 2. Adapter Configuration (`adapter`)

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `type` | string | Type of adapter to use. Currently supports `generic`. | `generic` |
| `quantize_first_layer` | bool | Whether to quantize the input of the first layer. | `true` |
| `quantized_ops` | list[str] | List of operations to replace with quantized versions. | `["Conv2d", "Linear", "ReLU"]` |
| `input_quantization` | bool | Whether to quantize layer inputs. | `true` |
| `weight_quantization` | bool | Whether to quantize layer weights. | `true` |
| `output_quantization` | bool | Whether to quantize layer outputs (default `false`). | `true` |

## 3. Quantization Configuration (`quantization`)

This section controls how quantization is applied.

### Global Settings

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `format` | string | Target quantization format. Supported: `fp8_e4m3`, `fp8_e5m2`, `int8`, `fp4_e2m1`. | `fp8_e4m3` |
| `bias` | int | Optional exponent bias override. If null, uses format default. | `null` |
| `calib_method` | string | Calibration method (currently unused/placeholder). | `max` |

### Input Quantization Modes (Conv2d, Linear)

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `mode` | string | Quantization mode for inputs. Options: `tensor`, `channel`, `chunk`. | `tensor` |
| `chunk_size` | int | Chunk size if `mode` is `chunk`. | `128` |

### Weight Quantization Modes

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `weight_mode` | string | Quantization mode for weights. Options: `tensor`, `channel`, `chunk`. | `channel` |
| `weight_chunk_size` | int | Chunk size if `weight_mode` is `chunk`. | `64` |

### Activation Quantization Modes (ReLU, Softmax, etc.)

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `act_mode` | string | Quantization mode for activation layers. Options: `tensor`, `channel`, `chunk`. | `tensor` |
| `act_chunk_size` | int | Chunk size if `act_mode` is `chunk`. | `null` |

### Output Quantization Modes

Applies to every quantized op (Conv/Linear/BN/MatMul/Activations/Softmax/...) when
`adapter.output_quantization` is `true`. Off by default — existing experiments are
byte-identical until the flag is enabled.

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `output_format` | string | Output quantization format. Defaults to `format` if unset. | `fp8_e5m2` |
| `output_mode` | string | Quantization mode for outputs. Options: `tensor`, `channel`, `chunk`. | `tensor` |
| `output_chunk_size` | int | Chunk size when `output_mode` is `chunk`. | `null` |

### Per-layer output_quantization override

In a `quantization.layers.<name>` block, output quantization can be enabled per-layer
even when `adapter.output_quantization` is `false` globally. Precedence:

1. Explicit `output_quantization: true|false` per-layer wins.
2. Otherwise, presence of any `output_format` / `output_mode` / `output_chunk_size`
   in the layer block **implicitly enables** output quantization for that layer.
3. Otherwise, the layer inherits the global `adapter.output_quantization` flag.

```yaml
adapter:
  output_quantization: false   # off globally
quantization:
  format: fp8_e4m3
  layers:
    layer4.1.conv2:
      output_format: fp8_e4m3   # implicitly enables output_quantization for THIS layer only
    fc:
      output_quantization: true # explicit form; takes precedence
      output_format: fp8_e5m2
```

### Strict-mode compliance check

`quantization.strict_format_check` (bool, default `false`) controls how the comparator
report validates captured tensors against their declared formats:

- `false` — fast path, mantissa-bit count only (`_check_mantissa_precision`). Catches
  simulator bugs that produce too-precise values. Cheap.
- `true` — full per-value validation through the codec (`check_fp8_compliance`).
  Catches mantissa precision PLUS exponent range, NaN/Inf, sign violations, and
  special-value handling. Slower but thorough.

The same flag also drives the **runtime format detection** that populates the
report's `Pre-Quant`, `Input Fmt`, and `Output Fmt` columns. In strict mode the
detector tries each candidate format with full compliance; in non-strict it picks
the narrowest format whose mantissa-bit budget covers the captured tensor.

### Comparator report columns

The comparator report's per-layer table uses **runtime-captured tensors** to
determine each cell — values reflect what *actually happened* this batch, not
what was configured:

| Column | Meaning |
| :--- | :--- |
| `Weight Fmt` | Configured weight format (or `FP32` when weight quantization off / `N/A` for ops without weights). |
| `Input Q?` | `Yes` iff the layer actually populated `last_quant_input_unscaled` this batch (= the layer's forward called `quantize_input`). |
| `Pre-Quant` | Runtime-detected format of the tensor that arrived at the layer. |
| `Input Fmt` | Runtime-detected format of the tensor fed to the compute kernel (post-quantize-input if called). |
| `Output Q?` | `Yes` iff `module.output_quantization` is enabled for this layer. |
| `Output Fmt` | Runtime-detected format of the layer's natural output (captured pre-quantize-output). For value-preserving ops like MaxPool/ReLU this remains the input format even with output quantization off. |
| `Weight Check` / `Input Check` / `Output Check` | Compliance check on the captured tensor (mantissa-bit only when non-strict, full check when strict). |

#### Technical Details: How Activations are Identified

The framework distinguishes between "Layers" (weights) and "Activations" based on the class inheritance:
- **Layers**: Operations that inherit from `QuantizedLayerMixin` (e.g., `QuantConv2d`, `QuantLinear`, `QuantBatchNorm2d`). These use `mode` (for inputs) and `weight_mode` (for weights).
- **Activations**: Operations that do **not** inherit from `QuantizedLayerMixin` (e.g., `QuantReLU`, `QuantSoftmax`, `QuantSiLU`). These use `act_mode` and `act_chunk_size`.

### Layer-Specific Overrides (`layers`)

You can override any of the above settings for specific layers by their name.

```yaml
quantization:
  layers:
    layer1.0.conv1:
      mode: chunk
      chunk_size: 32
      weight_mode: tensor
    fc:
      format: int8
    layer4.1.conv2:
      output_format: fp8_e4m3
      output_mode: tensor
```

Per-layer overrides for outputs (`output_format`, `output_mode`,
`output_chunk_size`) follow the same pattern as inputs/weights and override the
global `quantization.output_*` defaults.

## 4. Dataset Configuration (`dataset`)

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `name` | string | Name of the dataset. | `imagenet` |
| `path` | string | Absolute path to the dataset root. | `/data/imagenet` |
| `batch_size` | int | Batch size for evaluation. | `64` |
| `num_workers` | int | Number of data loading workers. | `4` |

## 5. Evaluation Configuration (`evaluation`)

| Option | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `mode` | string | Evaluation mode. `evaluate` (quantized model accuracy only) or `compare` (reference vs quantized comparison). | `compare` |
| `compare_batches` | int | Number of batches to use for comparison. `-1` for all. | `1` |
| `max_samples` | int | Optional limit on number of samples to evaluate. | `1000` |
| `generate_graph_svg` | bool | Whether to generate the quantization graph SVG. Default: `true`. | `true` |

---

## Example Configuration

```yaml
model:
  name: resnet18
  source: torchvision
  weights: IMAGENET1K_V1

adapter:
  type: generic
  quantize_first_layer: true
  quantized_ops: ["Conv2d", "Linear", "BatchNorm2d", "ReLU"]

quantization:
  format: fp8_e4m3
  
  # Inputs (Conv/Linear)
  mode: chunk
  chunk_size: 128
  
  # Weights
  weight_mode: channel
  
  # Activations (ReLU/Softmax)
  act_mode: tensor

dataset:
  name: imagenet
  path: /data/imagenet/val
  batch_size: 64
  num_workers: 4

evaluation:
  mode: compare
  compare_batches: 10
```
