# Quantized Pooling

This document describes the quantized pooling layers available in `quant_pooling.py`. These modules are designed to wrap standard PyTorch pooling layers and integrate them into the quantized inference pipeline.

## Overview

The pooling layers are implemented as `nn.Module` wrappers that replace standard PyTorch pooling operations. In hardware activation-transport mode, each pooling layer remains a named FX compute stage: its producer supplies the quantized input packet and the pooling result is encoded exactly once for downstream consumers. They support:
- **Producer-Stage Input Transport**: Inputs arrive through the configured reference or encoded hardware transport.
- **Standard Pooling Logic**: Leveraging PyTorch's optimized pooling kernels on quantized data.
- **Accurate Activation Capturing**: Natural pooling outputs are captured separately from transmitted quantized outputs.

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
- **Input Quantization**: The input tensor is decoded from the producer-stage packet before pooling.
- **Operation**: The standard `nn.MaxPool2d` operation is applied to the quantized input.
- **Output**: The pooling result is encoded once as this stage's output packet. `return_indices=True` is rejected in hardware transport mode because the packet ABI currently transports activation tensors, not activation/index tuples.

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
- **Input Quantization**: The input tensor is decoded from the producer-stage packet.
- **Operation**: The standard `nn.AdaptiveAvgPool2d` is applied.
- **Output**: The natural average is encoded once as this stage's output packet. The average need not already lie on the input format's grid.

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
- **Input Quantization**: The input tensor is decoded from the producer-stage packet.
- **Operation**: The standard `nn.AvgPool2d` is applied.
- **Output**: As with adaptive average pooling, the natural average is encoded exactly once at the producer-stage boundary.
