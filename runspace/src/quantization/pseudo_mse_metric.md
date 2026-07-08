# pseudo_MSE Metric

`pseudo_MSE` is a pairwise activation-format selection metric for comparing
same-width floating-point candidates with exponent widths `1` and `2`.
It is intentionally not the same selector as MSE: each element votes for the
candidate using a bit-level pseudo `err2-err1` signal, then the exp=2 vote
count is divided by an integer divisor before the chunk decision. The default
divisor is `4`, matching the original `>> 2` behavior. A divisor of `2` is
available as an L1-equivalence verification mode.

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
Both candidates must use the same signedness.  pseudo_MSE uses mantissa
truncation for the selected quantized output in both Python and CUDA paths.

## Per-Element Definition

For each scaled activation value `x`, the metric computes a signed bit-level
pseudo difference `diff = err2 - err1`.  Let `a = abs(x)`, and define the
exponent depth `d` by:

```text
d=0 for 1.0 <= a < 2.0
d=1 for 0.5 <= a < 1.0
d=2 for 0.25 <= a < 0.5
...
```

Let `X_k` be the kth normalized mantissa bit after the hidden leading `1`, and
let `M` be the higher mantissa width from the exp=1 format.  The per-element
signal is:

```text
d == 0       => +X_M
d == 1       => 0
1 < d < M+1  => -X_(M+1-d)
d == M+1     => -1   # hidden leading 1
d > M+1      => 0
```

The sign convention is:

```text
diff < 0  => exp=2 wins this element
diff > 0  => exp=1 wins this element
diff == 0 => tie for this element, no vote
```

## Mathematical Function

For an exp=1 / exp=2 same-width pair:

```math
m_2 = m_1 - 1
```

For a chunk `C`, define:

```math
W_1(C) = |\{x_i \in C : \mathrm{diff}_i > 0\}|
```

```math
W_2(C) = |\{x_i \in C : \mathrm{diff}_i < 0\}|
```

Exact zero differences are ties and are excluded from both counts. The selected
format uses divisor `D`, where `D=4` by default and `D=2` for the L1
verification mode:

```math
\operatorname{select}(C) =
\begin{cases}
\mathrm{exp}=2, & \left\lfloor W_2(C) / D \right\rfloor \ge W_1(C) \\
\mathrm{exp}=1, & \left\lfloor W_2(C) / D \right\rfloor < W_1(C)
\end{cases}
```

## Chunk Selection Rule

Dynamic activation quantization selects one format per chunk. For a chunk,
`pseudo_MSE` counts the signed per-element winners:

```text
exp2_wins = count_i(diff_i < 0)
exp1_wins = count_i(diff_i > 0)
exp2_wins_shifted = floor(exp2_wins / e2_win_divisor)
```

The selected format is:

```text
if exp2_wins_shifted >= exp1_wins:
    choose exp=2
else:
    choose exp=1
```

The unshifted `exp2_wins` count is still useful for debug and hardware-vector
reporting, but only `exp2_wins_shifted` drives the decision. Hardware-vector
exports also keep the legacy `expected_e2_wins_shift2` divide-by-4 diagnostic.
Ties, including all-zero-vote chunks, choose exp=2.

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

The CUDA path computes the bit-level `diff` per lane from the FP32 exponent and
mantissa bits, packs exp=1 and exp=2 votes into one exact reduction, divides
the exp=2 count by the pseudo_MSE divisor, then chooses exp=2 when the shifted
count is at least the exp=1 count.  The selected value is encoded with
truncation.  The Python reference mirrors the same FP32 bit operations and
truncating encode path, and is tested against the CUDA selector for both
supported divisors.
