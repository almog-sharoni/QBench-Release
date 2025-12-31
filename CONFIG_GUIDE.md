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
```

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
