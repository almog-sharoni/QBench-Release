# pseudo_MSE Metric

`pseudo_MSE` is a pairwise activation-format selection metric for comparing
same-width floating-point candidates with exponent widths `1` and `2`.

For a given total bit width, the candidate pair is:

```text
exp=1: fpB_e1mM
exp=2: fpB_e2m(M-1)
```

For unsigned post-activation candidates, the same relationship applies:

```text
exp=1: ufpB_e1mM
exp=2: ufpB_e2m(M-1)
```

`M` is the mantissa width of the exp=1 format. The exp=2 format must have
exactly one fewer mantissa bit, so both candidates have the same total width.
Both candidates must use the same signedness.

## Per-Element Definition

For each scaled activation value `x`, the metric computes the exact signed
difference between the squared reconstruction errors of the two candidate
formats:

```text
q_exp1 = decode_emb(encode_emb(x, e=1, m=M,   sgn), e=1, m=M,   sgn)
q_exp2 = decode_emb(encode_emb(x, e=2, m=M-1, sgn), e=2, m=M-1, sgn)

diff = (x - q_exp2)^2 - (x - q_exp1)^2
```

The sign is meaningful:

```text
diff < 0  => exp=2 has lower squared error for this element
diff > 0  => exp=1 has lower squared error for this element
diff == 0 => tie for this element
```

This is not the old bit-level pseudo-unit approximation. The implementation
now compares the actual reconstructed values produced by `encode_emb` and
`decode_emb`.

## Mathematical Function

Let `Q_{e,m,s}(x)` be the reconstructed value after encoding and decoding `x`
with exponent width `e`, mantissa width `m`, and signedness `s`:

```math
Q_{e,m,s}(x) =
\operatorname{decode\_emb}(
    \operatorname{encode\_emb}(x, e, m, s),
    e, m, s
)
```

For an exp=1 / exp=2 same-width pair:

```math
m_2 = m_1 - 1
```

The per-element pseudo_MSE difference is:

```math
\Delta_{\mathrm{pseudo\_MSE}}(x; m_1, s)
=
\left(x - Q_{2,m_1-1,s}(x)\right)^2
-
\left(x - Q_{1,m_1,s}(x)\right)^2
```

For a chunk `C`, the chunk-level decision value is:

```math
D(C; m_1, s)
=
\sum_{x_i \in C}
\Delta_{\mathrm{pseudo\_MSE}}(x_i; m_1, s)
```

The selected format is:

```math
\operatorname{select}(C) =
\begin{cases}
\mathrm{exp}=2, & D(C; m_1, s) < 0 \\
\mathrm{exp}=1, & D(C; m_1, s) \ge 0
\end{cases}
```

## Actual Function

The CUDA metric function is:

```cpp
__device__ __forceinline__ float pseudo_mse_sqerr_diff(
    float scaled_v,
    int m1,
    int m2,
    int sgn)
{
    const uint32_t p1 = encode_emb(scaled_v, 1, m1, sgn);
    const uint32_t p2 = encode_emb(scaled_v, 2, m2, sgn);

    const float q1 = decode_emb(p1, 1, m1, sgn);
    const float q2 = decode_emb(p2, 2, m2, sgn);

    const float d1 = scaled_v - q1;
    const float d2 = scaled_v - q2;

    return d2 * d2 - d1 * d1;
}
```

The Python reference uses the same definition:

```python
def pseudo_mse_sqerr_diff_from_scaled(
    scaled_values,
    exp1_mantissa_width,
    exp2_mantissa_width,
    is_signed,
):
    m1 = int(exp1_mantissa_width)
    m2 = int(exp2_mantissa_width)
    if m2 != m1 - 1:
        raise ValueError(f"pseudo_MSE requires m2 == m1 - 1; got m1={m1}, m2={m2}")

    q1 = pseudo_mse_reconstruct_scaled_python(scaled_values, 1, m1, is_signed)
    q2 = pseudo_mse_reconstruct_scaled_python(scaled_values, 2, m2, is_signed)

    d1 = scaled_values.to(torch.float32) - q1
    d2 = scaled_values.to(torch.float32) - q2
    return d2 * d2 - d1 * d1
```

## Chunk Selection Rule

Dynamic activation quantization selects one format per chunk. For a chunk,
`pseudo_MSE` sums the signed per-element differences:

```text
chunk_diff = sum_i [(x_i - q_exp2_i)^2 - (x_i - q_exp1_i)^2]
```

The selected format is:

```text
if chunk_diff < 0:
    choose exp=2
else:
    choose exp=1
```

Ties choose exp=1.

## Constraints

`pseudo_MSE` is defined for one same-width e1/e2 pair at a time:

```text
cands_e[e1] == 1
cands_e[e2] == 2
cands_sgn[e1] == cands_sgn[e2]
cands_m[e2] == cands_m[e1] - 1
```

Examples:

```text
fp8_e1m6  vs fp8_e2m5
fp7_e1m5  vs fp7_e2m4
fp6_e1m4  vs fp6_e2m3
fp5_e1m3  vs fp5_e2m2
fp4_e1m2  vs fp4_e2m1

ufp8_e1m7 vs ufp8_e2m6
ufp7_e1m6 vs ufp7_e2m5
ufp6_e1m5 vs ufp6_e2m4
ufp5_e1m4 vs ufp5_e2m3
ufp4_e1m3 vs ufp4_e2m2
```

Unsupported cases:

```text
mixed signed and unsigned candidates
exp widths other than 1 or 2
missing e1/e2 partner
multiple bit-width pairs in one selector call
m2 != m1 - 1
```

## Implementation

The CUDA path computes `diff` per lane as a float, reduces it across the chunk
with a signed sum, then chooses exp=2 only when the reduced value is negative.

The Python reference path mirrors the same `encode_emb` and `decode_emb`
behavior with vectorized PyTorch bit operations and is tested against the CUDA
selector.
