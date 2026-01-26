# Quantized Normalization Layers

This document describes the current implementation of quantized normalization layers in QBench.

## 1. QuantBatchNorm2d

**Class:** `QuantBatchNorm2d`  
**Source:** `src/ops/quant_bn.py`  
**Registry Name:** `QuantBatchNorm2d`  
**Original Class:** `torch.nn.BatchNorm2d`

### Description
The `QuantBatchNorm2d` layer simulates the behavior of a BatchNorm layer in a quantized environment. It is designed to maintain high precision for stability while allowing input and weight quantization.

### Quantization Scheme

*   **Input:** Quantized to the target format (e.g., FP8, INT8) using `quantize_input`.
*   **Weights (Gamma):** Quantized to the target format (e.g., FP8, INT8). They are dequantized to FP32 for the actual calculation.
*   **Bias (Beta):** **Not Quantized** (remains FP32).
*   **Running Mean:** **Not Quantized** (remains FP32).
*   **Running Variance:** **Not Quantized** (remains FP32).

### Rationale for FP32 Stats
The `running_mean` and `running_var` statistics are crucial for the stability of Batch Normalization. Quantizing `running_var` (especially with a single tensor-wide scale) can lead to underflow for channels with small variances. Since BatchNorm divides by $\sqrt{\sigma^2 + \epsilon}$, any error near zero in the variance can result in massive amplification of the output, destroying model accuracy. Therefore, these statistics are kept in FP32.

### Operation Flow
1.  **Input Quantization:** $X_{q} = Quantize(X)$
2.  **Weight Dequantization:** $\gamma_{dq} = Dequantize(\gamma_{q})$
3.  **BatchNorm Computation:**
    $$ Y = \frac{X_{q} - \mu_{running}}{\sqrt{\sigma_{running}^2 + \epsilon}} \cdot \gamma_{dq} + \beta $$
    *(Where $\mu_{running}$, $\sigma_{running}^2$, and $\beta$ are FP32)*

---

## 2. QuantLayerNorm

**Class:** `QuantLayerNorm`  
**Source:** `src/ops/quant_ln.py`  
**Registry Name:** `QuantLayerNorm`  
**Original Class:** `torch.nn.LayerNorm`  
**Status:** 🚧 UNDER CONSTRUCTION

### Description
The `QuantLayerNorm` layer implements a quantized version of Layer Normalization. It supports input and weight quantization and also quantizes the output, which is typical for Transformer blocks where LayerNorm is followed by other quantized operations.

### Quantization Scheme

*   **Input:** Quantized to the target format (e.g., FP8, INT8).
*   **Weights (Gamma):** Quantized to the target format. Dequantized to FP32 for calculation.
*   **Bias (Beta):** **Not Quantized** (remains FP32).
*   **Output:** Quantized to the target format.

### Operation Flow
1.  **Input Quantization:** $X_{q} = Quantize(X)$
2.  **Weight Dequantization:** $\gamma_{dq} = Dequantize(\gamma_{q})$
3.  **LayerNorm Computation:**
    $$ Y_{fp32} = LayerNorm(X_{q}, \gamma_{dq}, \beta) $$
4.  **Output Quantization:**
    $$ Y_{out} = Quantize(Y_{fp32}) $$

### Note on QuantLayerNorm2d
There is also a `QuantLayerNorm2d` implementation (used in ConvNeXt) which wraps `QuantLayerNorm` but handles the channel permutation `(N, C, H, W) -> (N, H, W, C)` required for LayerNorm to operate on the channel dimension correctly.
