# Known Bugs

## BUG-001: Quantizer does not clamp underflow values to zero

**File:** `src/quantization/quantizer.py` — `quantize_fp_generic`

**Symptom:** `weight_fp8` (and quantized activations) contain values smaller than the minimum representable subnormal of the target format. For example, FP8 E4M3 (bias=7) has a minimum subnormal of `2^(-9) = 0.001953125`, but `weight_fp8` may contain values like `0.00048828125 = 2^(-11)`. These values fail the FP8 compliance check in `_generate_report`.

**Root cause:** `quantize_fp_generic` works by bit-manipulating the FP32 representation to truncate mantissa bits. It does not constrain the exponent to the target format's range. When a value's magnitude falls below the minimum subnormal of the target format, the function preserves the value (with reduced mantissa precision) instead of clamping it to zero.

**Impact:** All formats, all model types. The compliance check (`check_fp8_compliance`) correctly identifies these as invalid — they are not representable in the target format.

**Fix:** In `quantize_fp_generic`, after mantissa truncation, clamp any value whose exponent falls below the minimum representable exponent of the target format to zero. Specifically, zero out values where the resulting FP8 exponent would be negative (i.e., below the subnormal threshold).

---

## BUG-002: Compliance check uses wrong lookup table for non-standard FP formats

**File:** `src/eval/comparator.py` — `_generate_report`

**Symptom:** For any quantization format other than `fp8_e4m3`, `fp8_e5m2`, `fp4_e2m1`, `fp4_e3m0`, or `int8`, the compliance check silently falls back to the `fp8_e4m3` lookup table. This causes every layer to report `❌ FAIL` even when values are correctly quantized to the actual target format.

**Root cause:** The comparator pre-generates five hard-coded tables at report time and selects among them with `if/elif` branches. Any format outside this set (e.g., `fp8_e1m6`, `fp8_e3m4`, `fp8_e0m7`) falls through to the `else` branch, which uses the `fp8_e4m3` table. The `check_fp8_compliance` function itself can generate a correct table for any format generically, but this path is never reached when the comparator supplies a pre-built (wrong) table as the `valid_table` argument.

**Impact:** All custom/non-standard FP formats. Every layer in models quantized to formats like `fp8_e1m6` will show compliance failures regardless of whether the quantization is correct.

**Fix:** Remove the pre-built table selection logic. Instead, pass `q_type` to `check_fp8_compliance` and let it generate the correct table internally (the function already supports this path when `valid_table=None`). The pre-generation optimization can be preserved by caching tables by `q_type` key rather than by format name.
