# Quantized Pooling

This document describes the quantized pooling layers available in `quant_pooling.py`. These modules are designed to wrap standard PyTorch pooling layers and integrate them into the quantized inference pipeline.

## Overview

The pooling layers are implemented as `nn.Module` wrappers that replace standard PyTorch pooling operations. They support:
- **FP8/INT8 Input Quantization**: Ensuring inputs are quantized before pooling.
- **Standard Pooling Logic**: Leveraging PyTorch's optimized pooling kernels on quantized data.
- **Activation Capturing**: Optional capturing of input/output tensors for debugging or calibration.

---

## Available Pooling Layers

### 1. QuantMaxPool2d

**Class:** `QuantMaxPool2d`  
**Original Op:** `nn.MaxPool2d`

Standard 2D Max Pooling.

#### Mathematical Formula
For a window $W$ of input values $x$:
$$
y = \max_{x \in W} (x)
$$

#### Method
- **Input Quantization**: The input tensor is quantized to the configured format (e.g., FP8) before pooling.
- **Operation**: The standard `nn.MaxPool2d` operation is applied to the quantized input.
- **Output**: Since the max of a set of quantized values is always one of those values, the output preserves the quantization grid (assuming no padding with different values affects the max).

---

### 2. QuantAdaptiveAvgPool2d

**Class:** `QuantAdaptiveAvgPool2d`  
**Original Op:** `nn.AdaptiveAvgPool2d`

Adaptive Average Pooling, where the output size is fixed, and the stride/kernel size are computed automatically.

#### Mathematical Formula
For a window $W$ of size $N$ containing input values $x_i$:
$$
y = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

#### Method
- **Input Quantization**: The input tensor is quantized to the configured format.
- **Operation**: The standard `nn.AdaptiveAvgPool2d` is applied.
- **Output**: The output is the average of quantized values.
    - *Note*: The average of FP8/INT8 values is not necessarily representable in the same quantized format (it becomes a higher-precision float).
    - Currently, the layer returns this higher-precision result. If the next layer is quantized, it will re-quantize this input.

---

### 3. QuantAvgPool2d

**Class:** `QuantAvgPool2d`  
**Original Op:** `nn.AvgPool2d`

Standard 2D Average Pooling with fixed kernel size and stride.

#### Mathematical Formula
For a window $W$ of size $N$ containing input values $x_i$:
$$
y = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

#### Method
- **Input Quantization**: The input tensor is quantized to the configured format.
- **Operation**: The standard `nn.AvgPool2d` is applied.
- **Output**: Similar to Adaptive AvgPool, the output is a floating-point average of the quantized inputs. It is not explicitly re-quantized at the output of this layer, leaving that to the subsequent layer if needed.
