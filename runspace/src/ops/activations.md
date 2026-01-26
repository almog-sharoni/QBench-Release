# Quantized Activations

This document describes the quantized activation functions available in `quant_activations.py`. These modules are designed to wrap standard PyTorch activations and optimize them for quantized inference, often using Look-Up Tables (LUTs) or piecewise approximations.

## Overview

The activations are implemented as `nn.Module` wrappers that can replace standard PyTorch activations. They support:
- **FP8/INT8 Quantization**: Handling inputs and outputs in quantized formats.
- **LUT-based Execution**: Precomputing results for low-precision inputs (e.g., FP8) to avoid runtime floating-point math.
- **Piecewise Approximations**: Efficiently approximating complex functions like GELU.

---

## Available Activations

### 1. QuantReLU

**Class:** `QuantReLU`  
**Original Op:** `nn.ReLU`

Standard Rectified Linear Unit. In the current implementation, this module wraps `nn.functional.relu`.

#### Mathematical Formula
$$
f(x) = \max(0, x)
$$

#### Method
- The forward pass applies the standard ReLU function.
- Supports capturing activation statistics (`last_quant_input`, `last_quant_output_unscaled`) for calibration or debugging.

---

### 2. QuantReLU6

**Class:** `QuantReLU6`  
**Original Op:** `nn.ReLU6`

Rectified Linear Unit capped at 6.

#### Mathematical Formula
$$
f(x) = \min(\max(0, x), 6)
$$

#### Method
- The forward pass applies the standard ReLU6 function.
- Supports capturing activation statistics.

---

### 3. QuantGELU

**Class:** `QuantGELU`  
**Original Op:** `nn.GELU`

Gaussian Error Linear Unit. This implementation uses a **piecewise approximation** combined with a small Look-Up Table (LUT) for the non-linear region. This allows for efficient execution without expensive `erf` calculations at runtime.

#### Mathematical Formula (Approximation)

The function is approximated over three regions defined by a threshold parameter $A$ (default $A=3.0$):

$$
f(x) \approx \begin{cases} 
0 & \text{if } x \le -A \\
x & \text{if } x \ge A \\
\text{LUT}[i(x)] & \text{if } -A < x < A 
\end{cases}
$$

Where the index $i(x)$ maps the input range $[-A, A]$ to the LUT table size (256 bins):

$$
t = \frac{x + A}{2A}
$$
$$
i(x) = \lfloor \text{clamp}(t, 0, 1) \times 255 \rfloor
$$

#### Method
1.  **LUT Construction (Initialization)**:
    - A table of 256 values is precomputed.
    - Each entry corresponds to `GELU(x_i)` where $x_i$ is the midpoint of the $i$-th bin in $[-A, A]$.
    - Continuity is enforced by setting $\text{LUT}[0] = 0$ and $\text{LUT}[255] = A$.

2.  **Forward Pass**:
    - For inputs outside $[-A, A]$, the function is linear (identity or zero).
    - For inputs inside $[-A, A]$, the value is retrieved from the precomputed `piecewise_lut` using the calculated index.
    - This avoids FP32 transcendental operations during inference.

---

## LUTActivation Mixin

**Class:** `LUTActivation`

A helper mixin used to build and apply Look-Up Tables for FP8 inference.

#### Method
- **`build_lut`**: Generates a table of outputs for all possible 256 bit patterns of an FP8/INT8 type.
- **`apply_lut`**: Quantizes the input tensor to the target type (e.g., FP8), interprets the bits as indices, and retrieves the output from the table. This allows the entire activation to be executed as a memory lookup.

*Note: `QuantGELU` uses a specialized piecewise LUT approach rather than the generic `LUTActivation` full-range LUT, as it handles a wider dynamic range by clamping.*
