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
Both candidates must use the same signedness.  The pseudo_MSE family uses the
same round-to-nearest quantized output path as the generic MSE selector in both
Python and CUDA paths.

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
\mathrm{exp}=2, & W_2(C) / D \ge W_1(C) \\
\mathrm{exp}=1, & W_2(C) / D < W_1(C)
\end{cases}
```

## pseudo_MSE2 Variant

`pseudo_MSE2` is a separate pairwise metric using CUDA metric code `8` and
canonical metric name `pseudo_mse2`. It uses the same e1/e2 candidate
constraints, rounded quantized output, and tie rule as `pseudo_MSE`, but its
per-element `diff = err2 - err1` is an int32 fixed-point window vote.

The weighted per-element signal is:

```text
d == 0       => +X_M * window_int(X_M, X_(M+1), X_(M+2), ...)
d == 1       => 0
1 < d < M+1  => -X_k * (window_int(X_k, X_(k+1), X_(k+2), ...) >> 2), where k = M+1-d
d == M+1     => -(window_int(1, X_1, X_2, ...) >> 2)
d > M+1      => 0
```

The selected window is represented as the old weighted value scaled by `2^24`,
leaving two extra low bits before the exp=2 shift. For a full window this means
`X_n` and `X_(n+1)` both have weight `2^24`, `X_(n+2)` has weight `2^23`, and
so on down to weight `4`. Exp=2 shifts this value right by two before
accumulation, preserving the old `2^22`-scaled precision. The default window
size is 24, covering the full FP32 significand window.

Implementations may limit this with `mantissa_window_bits=N`.  In that mode,
the `M`/`k` cases use `X_n` through `X_(n+N-1)`, and the hidden-leading case
uses the same total window size: hidden `1` plus `N-1` explicit mantissa bits.
The default uses all remaining FP32 mantissa bits, i.e. 24 total hidden-case
terms.

```text
exp1_wins = sum_i(max(diff_i, 0))
exp2_wins = sum_i(max(-diff_i, 0))
```

The decision is:

```text
if exp2_wins > exp1_wins:
    choose exp=2
else:
    choose exp=1
```

The chunk-level `e2_win_divisor` adjustment remains part of `pseudo_MSE`, but
does not drive `pseudo_MSE2`; pseudo_MSE2 applies the exp=2 shift before
accumulating per-chunk sums.

## pseudo_MSE3

`pseudo_MSE3` is a pairwise metric using CUDA metric code `9` and canonical
metric name `pseudo_mse3`. It uses the same e1/e2 candidate constraints and
rounded quantized output, but its per-element signal is the exact squared
error difference:

```text
diff_i = err2_i^2 - err1_i^2
```

The chunk decision is:

```text
if sum_i(diff_i) < 0:
    choose exp=2
else:
    choose exp=1
```

The reference implementation asserts that `diff_i * 2^(2M)` is in the
rounded-path range `[-1/4, 3)`. The CUDA search computes the same exact
squared-error diff directly from the rounded e1/e2 reconstructions.

The `runspace/experiments/pseudo_mse3/pseudo_mse.py` experiment also accepts
`--bits-to-take N`. The default `N=0` keeps the exact floating-point
`err2^2 - err1^2` sum. Positive `N` converts each per-value signal to fixed
point before the chunk sum. `--fixed-rounding floor` preserves the original
`floor(diff_i * 2^N)` behavior. `--fixed-rounding nearest` matches activation
`encode_emb` rounding by rounding magnitude to nearest with exact half cases
away from zero. The fused CUDA selector receives `N` as `metric_param` and a
separate fixed-rounding mode code, and applies the same conversion.

The decision tie policy is independently configurable. `--tie-break exp1`
uses the legacy strict comparison `sum(diff_i) < 0`, so exact accumulator ties
select e1. `--tie-break exp2` changes only the comparison to
`sum(diff_i) <= 0`, so exact accumulator ties select e2.

## Chunk Selection Rule

Dynamic activation quantization selects one format per chunk. For a chunk,
`pseudo_MSE` counts the signed per-element winners:

```text
exp2_wins = count_i(diff_i < 0)
exp1_wins = count_i(diff_i > 0)
exp2_wins_shifted = exp2_wins / e2_win_divisor
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
mantissa bits, separately reduces the exp=1 and exp=2 vote sums, divides the
exp=2 sum by the pseudo_MSE divisor, then chooses exp=2 when the shifted sum is
at least the exp=1 sum. The selected value is encoded with round-to-nearest,
matching the generic MSE selector. The Python reference mirrors the same FP32
bit operations and rounded encode path, and is tested against the CUDA selector
for both supported divisors.
