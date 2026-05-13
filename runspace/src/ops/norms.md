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

### Description
The `QuantLayerNorm` layer implements a hardware-friendly quantized version of Layer Normalization. It follows a multi-stage pipeline designed for low-precision arithmetic, using Look-Up Tables (LUTs) for complex operations like reciprocal square root.

### Quantization Scheme

*   **Input:** Quantized to the target format (e.g., FP8).
*   **Mean:** The negative mean is calculated and quantized before subtraction.
*   **Variance (Inverse):** Calculated using a LUT-based approximation and quantized.
*   **Weights (Gamma):** Quantized to the target format.
*   **Bias (Beta):** Quantized to the target format.
*   **Output:** Quantized to the target format.

### Operation Flow
1.  **Input Quantization:** $X_{q} = Quantize(X)$
2.  **Mean Calculation:** $\mu = \text{mean}(X_{q})$, then $X_{\text{centered}} = X_{q} + Quantize(-\mu)$
3.  **Variance Approximation:**
    *   $S = \sum (X_{\text{centered}})^2$
    *   $\text{inv\_std} = \text{LUT\_Approx}(S, n) \approx \frac{1}{\sqrt{\frac{1}{n} S + \epsilon}}$
    *   $\text{inv\_std}_{q} = Quantize(\text{inv\_std})$
4.  **Normalization:** $X_{\text{norm}} = Quantize(X_{\text{centered}} \cdot \text{inv\_std}_{q})$
5.  **Affine Transform:**
    *   $\gamma_{q} = Quantize(\gamma)$, $\beta_{q} = Quantize(\beta)$
    *   $Y = X_{\text{norm}} \cdot \gamma_{q} + \beta_{q}$
6.  **Output Quantization:** $Y_{out} = Quantize(Y)$

### Hardware-Friendly Variance Approximation (LUT)
To avoid high-latency square root and division, the layer approximates $1/\sqrt{(1/n) \cdot S}$ using a 256-entry Look-Up Table:
1.  **Decomposition:** The sum of squares $S$ is decomposed into exponent $E$ and mantissa $M \in [1, 2)$ such that $S = 2^E \cdot M$.
2.  **LUT Lookup:** The mantissa $M$ is used to index a LUT that stores precomputed values of $1/\sqrt{(1/n) \cdot M}$.
3.  **Reconstruction:** The final reciprocal square root is reconstructed as $2^{-E/2} \cdot \text{LUT}[M]$, where the exponent shift is a simple bitwise operation in hardware.


### Note on QuantLayerNorm2d
There is also a `QuantLayerNorm2d` implementation (used in ConvNeXt) which wraps `QuantLayerNorm` but handles the channel permutation `(N, C, H, W) -> (N, H, W, C)` required for LayerNorm to operate on the channel dimension correctly.
