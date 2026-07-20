from dataclasses import dataclass
import re
from typing import Callable

import torch


PSEUDO_MSE_DISPLAY_NAME = "pseudo_MSE"
PSEUDO_MSE_CANONICAL_NAME = "pseudo_mse"
PSEUDO_MSE2_DISPLAY_NAME = "pseudo_MSE2"
PSEUDO_MSE2_CANONICAL_NAME = "pseudo_mse2"
PSEUDO_MSE3_DISPLAY_NAME = "pseudo_MSE3"
PSEUDO_MSE3_CANONICAL_NAME = "pseudo_mse3"
PSEUDO_MSE3_FIXED_ROUNDING_FLOOR = "floor"
PSEUDO_MSE3_FIXED_ROUNDING_NEAREST = "nearest"
PSEUDO_MSE3_FIXED_ROUNDING_CODES = {
    PSEUDO_MSE3_FIXED_ROUNDING_FLOOR: 0,
    PSEUDO_MSE3_FIXED_ROUNDING_NEAREST: 1,
}
PSEUDO_MSE3_FIXED_ROUNDING_ALIASES = {
    "floor": PSEUDO_MSE3_FIXED_ROUNDING_FLOOR,
    "nearest": PSEUDO_MSE3_FIXED_ROUNDING_NEAREST,
    "rtn": PSEUDO_MSE3_FIXED_ROUNDING_NEAREST,
    "round_to_nearest": PSEUDO_MSE3_FIXED_ROUNDING_NEAREST,
    "activation": PSEUDO_MSE3_FIXED_ROUNDING_NEAREST,
}
PSEUDO_MSE3_TIE_BREAK_EXP1 = "exp1"
PSEUDO_MSE3_TIE_BREAK_EXP2 = "exp2"
PSEUDO_MSE3_TIE_BREAK_CODES = {
    PSEUDO_MSE3_TIE_BREAK_EXP1: 0,
    PSEUDO_MSE3_TIE_BREAK_EXP2: 1,
}
PSEUDO_MSE3_TIE_BREAK_ALIASES = {
    "exp1": PSEUDO_MSE3_TIE_BREAK_EXP1,
    "lt": PSEUDO_MSE3_TIE_BREAK_EXP1,
    "strict": PSEUDO_MSE3_TIE_BREAK_EXP1,
    "exp2": PSEUDO_MSE3_TIE_BREAK_EXP2,
    "le": PSEUDO_MSE3_TIE_BREAK_EXP2,
    "less_equal": PSEUDO_MSE3_TIE_BREAK_EXP2,
}
PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR = 4
PSEUDO_MSE_SUPPORTED_E2_WIN_DIVISORS = (2, 4)


@dataclass(frozen=True)
class DynamicInputMetricSpec:
    name: str
    display_name: str
    cuda_code: int | None
    reducer: Callable[[torch.Tensor, float], torch.Tensor] | None
    pairwise: bool = False

    @property
    def implemented(self):
        return self.reducer is not None or self.pairwise

    @property
    def cuda_implemented(self):
        return self.cuda_code is not None and self.implemented


@dataclass(frozen=True)
class PseudoMseCandidatePair:
    bit_width: int
    e1_index: int
    e2_index: int
    e1_format: str
    e2_format: str
    exp1_mantissa_width: int
    is_signed: bool


def _reduce_l2(diff, metric_param):
    return diff.pow(2).sum(dim=1)


def _reduce_l1(diff, metric_param):
    return diff.abs().sum(dim=1)


def _reduce_linf(diff, metric_param):
    return diff.abs().max(dim=1).values


def _reduce_bias(diff, metric_param):
    return diff.sum(dim=1).abs()


def _reduce_l0(diff, metric_param):
    return (diff != 0).to(diff.dtype).sum(dim=1)


def _reduce_huber(diff, metric_param):
    delta = metric_param
    a = diff.abs()
    quad = 0.5 * diff.pow(2)
    lin = delta * (a - 0.5 * delta)
    return torch.where(a <= delta, quad, lin).sum(dim=1)


def _reduce_logsum(diff, metric_param):
    a = diff.abs()
    exps = torch.where(
        a > 0,
        torch.floor(torch.log2(a.clamp(min=1e-30))),
        torch.full_like(a, -126.0),
    )
    return exps.sum(dim=1)


# Codes MUST match the SearchMetric enum in
# runspace/src/quantization/cuda/ops_search.cu.
DYNAMIC_INPUT_METRICS = {
    "l2": DynamicInputMetricSpec("l2", "l2", 0, _reduce_l2),
    "l1": DynamicInputMetricSpec("l1", "l1", 1, _reduce_l1),
    "linf": DynamicInputMetricSpec("linf", "linf", 2, _reduce_linf),
    "bias": DynamicInputMetricSpec("bias", "bias", 3, _reduce_bias),
    "l0": DynamicInputMetricSpec("l0", "l0", 4, _reduce_l0),
    "huber": DynamicInputMetricSpec("huber", "huber", 5, _reduce_huber),
    "logsum": DynamicInputMetricSpec("logsum", "logsum", 6, _reduce_logsum),
    PSEUDO_MSE_CANONICAL_NAME: DynamicInputMetricSpec(
        PSEUDO_MSE_CANONICAL_NAME,
        PSEUDO_MSE_DISPLAY_NAME,
        7,
        None,
        pairwise=True,
    ),
    PSEUDO_MSE2_CANONICAL_NAME: DynamicInputMetricSpec(
        PSEUDO_MSE2_CANONICAL_NAME,
        PSEUDO_MSE2_DISPLAY_NAME,
        8,
        None,
        pairwise=True,
    ),
    PSEUDO_MSE3_CANONICAL_NAME: DynamicInputMetricSpec(
        PSEUDO_MSE3_CANONICAL_NAME,
        PSEUDO_MSE3_DISPLAY_NAME,
        9,
        None,
        pairwise=True,
    ),
}

DYNAMIC_INPUT_METRIC_ALIASES = {
    "mse": "l2",
    "mae": "l1",
    "max": "linf",
    "chebyshev": "linf",
    "sad": "l1",
    "mean_bias": "bias",
    "count": "l0",
    "logl1": "logsum",
    "logsum_exp": "logsum",
    "pseudo_mse": PSEUDO_MSE_CANONICAL_NAME,
    "pseudo_mse2": PSEUDO_MSE2_CANONICAL_NAME,
    "pseudo_mse3": PSEUDO_MSE3_CANONICAL_NAME,
}

IMPLEMENTED_DYNAMIC_INPUT_METRIC_CODES = {
    name: spec.cuda_code
    for name, spec in DYNAMIC_INPUT_METRICS.items()
    if spec.cuda_implemented
}


def normalize_dynamic_input_metric(metric):
    normalized = str(metric or "l2").strip().lower()
    normalized = DYNAMIC_INPUT_METRIC_ALIASES.get(normalized, normalized)
    if normalized not in DYNAMIC_INPUT_METRICS:
        raise ValueError(
            f"Unsupported dynamic input quantization metric: {metric!r}. "
            f"Expected one of {sorted(DYNAMIC_INPUT_METRICS)} "
            f"(or aliases {sorted(DYNAMIC_INPUT_METRIC_ALIASES)})."
        )
    return normalized


def get_dynamic_input_metric_spec(metric):
    return DYNAMIC_INPUT_METRICS[normalize_dynamic_input_metric(metric)]


def assert_dynamic_input_metric_implemented(metric):
    spec = get_dynamic_input_metric_spec(metric)
    if not spec.implemented:
        raise NotImplementedError(f"{spec.display_name} metric is not implemented yet")
    return spec


def dynamic_input_metric_code(metric):
    spec = assert_dynamic_input_metric_implemented(metric)
    if spec.cuda_code is None:
        raise NotImplementedError(f"{spec.display_name} metric does not have a CUDA search code")
    return spec.cuda_code


def reduce_dynamic_input_metric_python(metric, diff, metric_param):
    spec = assert_dynamic_input_metric_implemented(metric)
    if spec.reducer is None:
        raise ValueError(f"{spec.display_name} uses a pairwise selector, not a scalar reducer")
    return spec.reducer(diff, metric_param)


def is_pseudo_mse_metric(metric):
    return normalize_dynamic_input_metric(metric) == PSEUDO_MSE_CANONICAL_NAME


def is_pseudo_mse2_metric(metric):
    return normalize_dynamic_input_metric(metric) == PSEUDO_MSE2_CANONICAL_NAME


def is_pseudo_mse3_metric(metric):
    return normalize_dynamic_input_metric(metric) == PSEUDO_MSE3_CANONICAL_NAME


def is_pseudo_mse_family_metric(metric):
    return normalize_dynamic_input_metric(metric) in (
        PSEUDO_MSE_CANONICAL_NAME,
        PSEUDO_MSE2_CANONICAL_NAME,
        PSEUDO_MSE3_CANONICAL_NAME,
    )


def normalize_pseudo_mse3_fixed_rounding(mode=None):
    normalized = str(mode or PSEUDO_MSE3_FIXED_ROUNDING_FLOOR).strip().lower()
    normalized = PSEUDO_MSE3_FIXED_ROUNDING_ALIASES.get(normalized, normalized)
    if normalized not in PSEUDO_MSE3_FIXED_ROUNDING_CODES:
        raise ValueError(
            f"Unsupported pseudo_MSE3 fixed rounding mode: {mode!r}. "
            f"Expected one of {sorted(PSEUDO_MSE3_FIXED_ROUNDING_CODES)}."
        )
    return normalized


def pseudo_mse3_fixed_rounding_code(mode=None):
    return PSEUDO_MSE3_FIXED_ROUNDING_CODES[
        normalize_pseudo_mse3_fixed_rounding(mode)
    ]


def normalize_pseudo_mse3_tie_break(mode=None):
    normalized = str(mode or PSEUDO_MSE3_TIE_BREAK_EXP1).strip().lower()
    normalized = PSEUDO_MSE3_TIE_BREAK_ALIASES.get(normalized, normalized)
    if normalized not in PSEUDO_MSE3_TIE_BREAK_CODES:
        raise ValueError(
            f"Unsupported pseudo_MSE3 tie break: {mode!r}. "
            f"Expected one of {sorted(PSEUDO_MSE3_TIE_BREAK_CODES)}."
        )
    return normalized


def pseudo_mse3_tie_break_code(mode=None):
    return PSEUDO_MSE3_TIE_BREAK_CODES[normalize_pseudo_mse3_tie_break(mode)]


_FP_RE = re.compile(r"^(?P<prefix>u?fp)(?P<bits>\d+)_e(?P<exp>\d+)m(?P<mant>\d+)$")


def _parse_fp_format(fmt):
    match = _FP_RE.match(str(fmt))
    if match is None:
        raise ValueError(f"pseudo_MSE supports only fp/ufp formats; got {fmt!r}")
    prefix = match.group("prefix")
    bit_width = int(match.group("bits"))
    exp_bits = int(match.group("exp"))
    mant_bits = int(match.group("mant"))
    is_signed = prefix == "fp"
    expected_width = (1 if is_signed else 0) + exp_bits + mant_bits
    if bit_width != expected_width:
        raise ValueError(
            f"Invalid format width for {fmt!r}: name says {bit_width} bits, "
            f"but sign+exp+mant is {expected_width}"
        )
    return bit_width, exp_bits, mant_bits, is_signed


def validate_pseudo_mse_candidate_pairs(candidate_formats):
    groups = {}
    for idx, fmt in enumerate(candidate_formats or []):
        bit_width, exp_bits, mant_bits, is_signed = _parse_fp_format(fmt)
        if exp_bits not in (1, 2):
            raise ValueError(
                f"pseudo_MSE supports only exp=1 or exp=2 formats; got {fmt!r}"
            )
        group = groups.setdefault((bit_width, is_signed), {})
        if exp_bits in group:
            raise ValueError(
                f"pseudo_MSE requires one exp={exp_bits} candidate per bit width; "
                f"got duplicate for {bit_width}-bit formats"
            )
        group[exp_bits] = (idx, str(fmt), mant_bits, is_signed)

    if not groups:
        raise ValueError("pseudo_MSE requires one exp=1/exp=2 candidate pair")
    if len(groups) != 1:
        raise ValueError(
            "pseudo_MSE currently supports one bit-width pair per run; "
            f"got pairs for bit widths/signedness {sorted(groups)}"
        )

    (bit_width, is_signed), entries = next(iter(groups.items()))
    if set(entries) != {1, 2}:
        raise ValueError(
            f"pseudo_MSE requires both exp=1 and exp=2 candidates for {bit_width}-bit formats"
        )

    e1_index, e1_format, e1_m, _ = entries[1]
    e2_index, e2_format, e2_m, _ = entries[2]
    if e1_m != e2_m + 1:
        raise ValueError(
            "pseudo_MSE expects same-width e1/e2 pairs where exp=1 has one "
            f"more mantissa bit than exp=2; got {e1_format!r} and {e2_format!r}"
        )

    return [
        PseudoMseCandidatePair(
            bit_width=bit_width,
            e1_index=e1_index,
            e2_index=e2_index,
            e1_format=e1_format,
            e2_format=e2_format,
            exp1_mantissa_width=e1_m,
            is_signed=is_signed,
        )
    ]


def _as_uint32_i64(values):
    return torch.bitwise_and(values.to(torch.int64), 0xFFFFFFFF)


def _uint32_to_float32(bits):
    bits = _as_uint32_i64(bits)
    signed_bits = torch.where(bits >= 0x80000000, bits - 0x100000000, bits)
    return signed_bits.to(torch.int32).view(torch.float32)


def pseudo_mse_encode_emb_python(values, exp_bits, mantissa_bits, is_signed):
    """Vectorized Python equivalent of CUDA round-to-nearest encode_emb."""
    e = int(exp_bits)
    m = int(mantissa_bits)
    sgn = bool(is_signed)
    values = values.to(torch.float32).contiguous()

    u = _as_uint32_i64(values.view(torch.int32))
    sign = torch.bitwise_and(torch.bitwise_right_shift(u, 31), 1)
    mag = torch.bitwise_and(u, 0x7FFFFFFF)

    b = (1 << e) - 1
    max_exp = b
    max_mant = 0 if m == 0 else ((1 << m) - 1)

    zero_field = sign << (e + m) if sgn else torch.zeros_like(sign)
    if not sgn:
        zero_field = torch.zeros_like(sign)

    result = torch.zeros_like(u)

    active = mag != 0
    if not sgn:
        active = active & (sign == 0)

    if active.any():
        exp_f = (
            torch.bitwise_and(torch.bitwise_right_shift(mag, 23), 0xFF)
            .to(torch.int64)
            - 127
        )
        mant_full = torch.bitwise_or(
            torch.bitwise_and(mag, 0x7FFFFF),
            1 << 23,
        )

        m_mask = torch.clamp((1 - b) - exp_f, min=0)
        shift = (23 - m) + m_mask

        safe_shift = torch.clamp(shift, min=0, max=63)
        mant_shifted = torch.bitwise_left_shift(
            torch.bitwise_right_shift(mant_full, safe_shift),
            safe_shift,
        )
        mant_trunc = torch.where(
            shift >= 24,
            torch.zeros_like(mant_shifted),
            mant_shifted,
        )
        safe_round_shift = torch.clamp(shift - 1, min=0, max=63)
        round_bit = torch.where(
            (shift >= 1) & (shift <= 24),
            torch.bitwise_and(torch.bitwise_right_shift(mant_full, safe_round_shift), 1),
            torch.zeros_like(mant_full),
        )
        round_add = torch.where(
            round_bit != 0,
            torch.bitwise_left_shift(torch.ones_like(mant_full), safe_shift),
            torch.zeros_like(mant_full),
        )
        mant_trunc = mant_trunc + round_add

        overflow = torch.bitwise_and(torch.bitwise_right_shift(mant_trunc, 24), 1)
        exp_f = exp_f + overflow

        bits_23_24 = torch.bitwise_and(torch.bitwise_right_shift(mant_trunc, 23), 3)
        sign_field = sign << (e + m) if sgn else torch.zeros_like(sign)

        saturated = sign_field | (max_exp << m) | max_mant
        exp_t = torch.where(exp_f >= 1 - b, exp_f + b, torch.zeros_like(exp_f))
        mant_t = (
            torch.zeros_like(mant_trunc)
            if m == 0
            else torch.bitwise_and(
                torch.bitwise_right_shift(mant_trunc, safe_shift),
                max_mant,
            )
        )
        normal_or_subnormal = sign_field | torch.bitwise_left_shift(exp_t, m) | mant_t

        encoded_active = torch.where(exp_f > 0, saturated, normal_or_subnormal)
        encoded_active = torch.where(bits_23_24 == 0, zero_field, encoded_active)
        result = torch.where(active, encoded_active, result)

    result = torch.where((mag == 0) & sgn, zero_field, result)
    return result


def pseudo_mse_encode_emb_trunc_python(values, exp_bits, mantissa_bits, is_signed):
    """Vectorized Python equivalent of CUDA ``encode_emb_trunc``."""
    e = int(exp_bits)
    m = int(mantissa_bits)
    sgn = bool(is_signed)
    values = values.to(torch.float32).contiguous()

    u = _as_uint32_i64(values.view(torch.int32))
    sign = torch.bitwise_and(torch.bitwise_right_shift(u, 31), 1)
    mag = torch.bitwise_and(u, 0x7FFFFFFF)

    b = (1 << e) - 1
    max_exp = b
    max_mant = 0 if m == 0 else ((1 << m) - 1)
    zero_field = sign << (e + m) if sgn else torch.zeros_like(sign)
    result = torch.zeros_like(u)

    active = mag != 0
    if not sgn:
        active = active & (sign == 0)

    if active.any():
        exp_f = (
            torch.bitwise_and(torch.bitwise_right_shift(mag, 23), 0xFF)
            .to(torch.int64)
            - 127
        )
        mant_full = torch.bitwise_or(
            torch.bitwise_and(mag, 0x7FFFFF),
            1 << 23,
        )
        m_mask = torch.clamp((1 - b) - exp_f, min=0)
        shift = (23 - m) + m_mask
        safe_shift = torch.clamp(shift, min=0, max=63)
        mant_trunc = torch.bitwise_left_shift(
            torch.bitwise_right_shift(mant_full, safe_shift),
            safe_shift,
        )
        mant_trunc = torch.where(
            shift >= 24,
            torch.zeros_like(mant_trunc),
            mant_trunc,
        )

        bits_23_24 = torch.bitwise_and(
            torch.bitwise_right_shift(mant_trunc, 23),
            3,
        )
        sign_field = sign << (e + m) if sgn else torch.zeros_like(sign)
        saturated = sign_field | (max_exp << m) | max_mant
        exp_t = torch.where(exp_f >= 1 - b, exp_f + b, torch.zeros_like(exp_f))
        mant_t = (
            torch.zeros_like(mant_trunc)
            if m == 0
            else torch.bitwise_and(
                torch.bitwise_right_shift(mant_trunc, safe_shift),
                max_mant,
            )
        )
        encoded = sign_field | torch.bitwise_left_shift(exp_t, m) | mant_t
        encoded = torch.where(exp_f > 0, saturated, encoded)
        encoded = torch.where(bits_23_24 == 0, zero_field, encoded)
        result = torch.where(active, encoded, result)

    return torch.where((mag == 0) & sgn, zero_field, result)


def pseudo_mse_decode_emb_python(fields, exp_bits, mantissa_bits, is_signed):
    """Vectorized Python equivalent of CUDA decode_emb for pseudo_MSE."""
    e = int(exp_bits)
    m = int(mantissa_bits)
    sgn = bool(is_signed)
    fields = fields.to(torch.int64)

    sign = (
        torch.bitwise_and(torch.bitwise_right_shift(fields, e + m), 1)
        if sgn
        else torch.zeros_like(fields)
    )
    exp_t = torch.bitwise_and(torch.bitwise_right_shift(fields, m), (1 << e) - 1)
    mant = torch.zeros_like(fields) if m == 0 else torch.bitwise_and(fields, (1 << m) - 1)
    b = (1 << e) - 1

    subnormal = mant.to(torch.float32) * float(2.0 ** (1 - b - m))
    subnormal = torch.where(sign != 0, -subnormal, subnormal)

    exp_f = exp_t - b + 127
    bits = (
        torch.bitwise_left_shift(sign, 31)
        | torch.bitwise_left_shift(exp_f, 23)
        | (torch.zeros_like(mant) if m == 0 else torch.bitwise_left_shift(mant, 23 - m))
    )
    normal = _uint32_to_float32(bits)

    zero = torch.where(sign != 0, -torch.zeros_like(subnormal), torch.zeros_like(subnormal))
    subnormal = torch.where(mant == 0, zero, subnormal)
    return torch.where(exp_t == 0, subnormal, normal)


def pseudo_mse_reconstruct_scaled_python(
    scaled_values,
    exp_bits,
    mantissa_bits,
    is_signed,
):
    packed = pseudo_mse_encode_emb_python(
        scaled_values,
        exp_bits,
        mantissa_bits,
        is_signed,
    )
    return pseudo_mse_decode_emb_python(
        packed,
        exp_bits,
        mantissa_bits,
        is_signed,
    )


def pseudo_mse_reconstruct_scaled_trunc_python(
    scaled_values,
    exp_bits,
    mantissa_bits,
    is_signed,
):
    """Reconstruct values through truncation for metric selection only."""
    packed = pseudo_mse_encode_emb_trunc_python(
        scaled_values,
        exp_bits,
        mantissa_bits,
        is_signed,
    )
    return pseudo_mse_decode_emb_python(
        packed,
        exp_bits,
        mantissa_bits,
        is_signed,
    )


def pseudo_mse_err2_minus_err1_from_scaled(
    scaled_values,
    exp1_mantissa_width,
    exp2_mantissa_width,
    is_signed,
):
    """Return the bit-level pseudo err2-err1 signal.

    Values are already chunk-scaled into [-2, 2).  Let e be the exponent depth
    of abs(x), where e=0 for [1, 2), e=1 for [0.5, 1), etc.  Let M be the
    higher mantissa width from the exp=1 candidate.  The pseudo signal is:

      e == 0       -> +X_M
      e == 1       -> 0
      1 < e < M+1  -> -X_(M+1-e)
      e == M+1     -> -1  (the hidden leading 1)
      e > M+1      -> 0

    X_k is the kth normalized mantissa bit after the hidden leading 1.  The
    sign convention is positive for exp=1 wins and negative for exp=2 wins.
    """
    m1 = int(exp1_mantissa_width)
    m2 = int(exp2_mantissa_width)
    if m2 != m1 - 1:
        raise ValueError(f"pseudo_MSE requires m2 == m1 - 1; got m1={m1}, m2={m2}")

    values = scaled_values.to(torch.float32).contiguous()
    mag = torch.bitwise_and(_as_uint32_i64(values.view(torch.int32)), 0x7FFFFFFF)
    exp_field = torch.bitwise_and(torch.bitwise_right_shift(mag, 23), 0xFF)
    mant = torch.bitwise_and(mag, 0x7FFFFF)

    nonzero_normal = exp_field != 0
    e_depth = 127 - exp_field.to(torch.int64)
    result = torch.zeros_like(values)

    bit_m = torch.bitwise_and(torch.bitwise_right_shift(mant, 23 - m1), 1).to(torch.float32)
    result = torch.where(nonzero_normal & (e_depth == 0), bit_m, result)

    shifted_bit_valid = nonzero_normal & (e_depth > 1) & (e_depth < m1 + 1)
    bit_pos = 23 - (m1 + 1 - e_depth)
    safe_bit_pos = torch.clamp(bit_pos, min=0, max=63)
    shifted_bit = torch.bitwise_and(
        torch.bitwise_right_shift(mant, safe_bit_pos),
        1,
    ).to(torch.float32)
    result = torch.where(shifted_bit_valid, -shifted_bit, result)
    result = torch.where(nonzero_normal & (e_depth == m1 + 1), -torch.ones_like(result), result)
    return result


def _pseudo_mse_mantissa_bit(mant, bit_index):
    """Return normalized mantissa bit X_bit_index, where X_1 is just after the hidden 1."""
    valid = (bit_index >= 1) & (bit_index <= 23)
    bit_pos = 23 - bit_index
    safe_bit_pos = torch.clamp(bit_pos, min=0, max=63)
    bit = torch.bitwise_and(
        torch.bitwise_right_shift(mant, safe_bit_pos),
        1,
    ).to(torch.float32)
    return torch.where(valid, bit, torch.zeros_like(bit))


def _pseudo_mse_mantissa_tail_value(mant, start_bit, tail_bits=None):
    """Return X_(start+1) + X_(start+2)/2 + ... through the requested tail."""
    if tail_bits is not None:
        tail_bits = int(tail_bits)
        if tail_bits < 0:
            raise ValueError(f"tail_bits must be non-negative; got {tail_bits}")
        if tail_bits == 0:
            return torch.zeros_like(mant, dtype=torch.float32)
        max_offset = min(tail_bits, 23)
    else:
        max_offset = 23

    offsets = torch.arange(1, max_offset + 1, device=mant.device, dtype=start_bit.dtype)
    bit_index = start_bit.unsqueeze(-1) + offsets
    valid = (bit_index >= 1) & (bit_index <= 23)
    bit_pos = 23 - bit_index
    safe_bit_pos = torch.clamp(bit_pos, min=0, max=63)
    bit = torch.bitwise_and(
        torch.bitwise_right_shift(mant.unsqueeze(-1), safe_bit_pos),
        1,
    ).to(torch.float32)
    weights = torch.pow(
        torch.full((max_offset,), 0.5, device=mant.device, dtype=torch.float32),
        (offsets - 1).to(torch.float32),
    )
    return (torch.where(valid, bit, torch.zeros_like(bit)) * weights).sum(dim=-1)


def _pseudo_mse2_window_bits(mantissa_window_bits):
    if mantissa_window_bits is None:
        return None
    window_bits = int(mantissa_window_bits)
    if window_bits < 1:
        raise ValueError(f"mantissa_window_bits must be at least 1; got {mantissa_window_bits!r}")
    if window_bits > 24:
        raise ValueError(f"mantissa_window_bits must be at most 24; got {mantissa_window_bits!r}")
    return window_bits


def _pseudo_mse2_window_size(window_bits):
    return 24 if window_bits is None else int(window_bits)


def _pseudo_mse2_mantissa_window_int(mant, start_bit, window_bits):
    """Return the selected window as old weighted value scaled by 2^24."""
    window_size = _pseudo_mse2_window_size(window_bits)
    offsets = torch.arange(0, window_size, device=mant.device, dtype=torch.int64)
    bit_index = start_bit.unsqueeze(-1) + offsets
    valid = (bit_index >= 1) & (bit_index <= 23)
    bit_pos = 23 - bit_index
    safe_bit_pos = torch.clamp(bit_pos, min=0, max=63)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(mant.unsqueeze(-1), safe_bit_pos),
        1,
    ).to(torch.int64)
    weight_exponents = torch.where(
        offsets <= 1,
        torch.full_like(offsets, 24),
        25 - offsets,
    )
    weights = torch.bitwise_left_shift(
        torch.ones((window_size,), device=mant.device, dtype=torch.int64),
        weight_exponents,
    )
    return (torch.where(valid, bits, torch.zeros_like(bits)) * weights).sum(dim=-1)


def _pseudo_mse2_hidden_window_int(mant, window_bits):
    """Return the hidden-1/X_1... window as old weighted value scaled by 2^24."""
    window_size = _pseudo_mse2_window_size(window_bits)
    offsets = torch.arange(0, window_size, device=mant.device, dtype=torch.int64)
    hidden = offsets == 0
    bit_index = offsets
    bit_pos = 23 - bit_index
    safe_bit_pos = torch.clamp(bit_pos, min=0, max=63)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(mant.unsqueeze(-1), safe_bit_pos),
        1,
    ).to(torch.int64)
    bits = torch.where(hidden, torch.ones_like(bits), bits)
    weight_exponents = torch.where(
        offsets <= 1,
        torch.full_like(offsets, 24),
        25 - offsets,
    )
    weights = torch.bitwise_left_shift(
        torch.ones((window_size,), device=mant.device, dtype=torch.int64),
        weight_exponents,
    )
    return (bits * weights).sum(dim=-1)


def pseudo_mse2_err2_minus_err1_from_scaled(
    scaled_values,
    exp1_mantissa_width,
    exp2_mantissa_width,
    is_signed,
    mantissa_window_bits=None,
):
    """Return the pseudo_MSE2 fixed-point bit-level err2-err1 signal.

    The selected window is represented as the old weighted value scaled by
    2^24.  The default window size is 24, covering the full FP32 significand
    window.  Exp=2 winner windows are shifted right by two before accumulation.

      e == 0       -> +X_M * window_int(X_M...)
      e == 1       -> 0
      1 < e < M+1  -> -X_k * (window_int(X_k...) >> 2), k=M+1-e
      e == M+1     -> -(window_int(1, X_1, X_2, ...) >> 2)
      e > M+1      -> 0

    When mantissa_window_bits=N is set, the M/K cases pack X_n through
    X_(n+N-1), and the hidden case packs the same total window size: hidden 1
    plus N-1 explicit bits.
    """
    m1 = int(exp1_mantissa_width)
    m2 = int(exp2_mantissa_width)
    if m2 != m1 - 1:
        raise ValueError(f"pseudo_MSE2 requires m2 == m1 - 1; got m1={m1}, m2={m2}")
    window_bits = _pseudo_mse2_window_bits(mantissa_window_bits)

    values = scaled_values.to(torch.float32).contiguous()
    mag = torch.bitwise_and(_as_uint32_i64(values.view(torch.int32)), 0x7FFFFFFF)
    exp_field = torch.bitwise_and(torch.bitwise_right_shift(mag, 23), 0xFF)
    mant = torch.bitwise_and(mag, 0x7FFFFF)

    nonzero_normal = exp_field != 0
    e_depth = 127 - exp_field.to(torch.int64)
    result = torch.zeros_like(values, dtype=torch.int32)

    m_bit_index = torch.full_like(e_depth, m1)
    bit_m = _pseudo_mse_mantissa_bit(mant, m_bit_index).to(torch.int64)
    positive_window = _pseudo_mse2_mantissa_window_int(mant, m_bit_index, window_bits)
    positive_weight = (bit_m * positive_window).to(torch.int32)
    result = torch.where(nonzero_normal & (e_depth == 0), positive_weight, result)

    shifted_bit_valid = nonzero_normal & (e_depth > 1) & (e_depth < m1 + 1)
    k = m1 + 1 - e_depth
    bit_k = _pseudo_mse_mantissa_bit(mant, k).to(torch.int64)
    negative_window = _pseudo_mse2_mantissa_window_int(mant, k, window_bits)
    negative_weight = (-(bit_k * torch.bitwise_right_shift(negative_window, 2))).to(torch.int32)
    result = torch.where(shifted_bit_valid, negative_weight, result)
    hidden_window = _pseudo_mse2_hidden_window_int(mant, window_bits)
    hidden_weight = (-torch.bitwise_right_shift(hidden_window, 2)).to(torch.int32)
    result = torch.where(nonzero_normal & (e_depth == m1 + 1), hidden_weight, result)
    return result


def _assert_pseudo_mse3_scaled_diff_ranges(scaled_diff):
    ok = (scaled_diff >= -0.75) & (scaled_diff < 3.0)
    bad = ~ok
    if bool(bad.any()):
        raise AssertionError(
            "pseudo_MSE3 scaled diff outside expected ranges: "
            f"{scaled_diff[bad].detach().cpu().tolist()}"
        )


def pseudo_mse3_fixed_point_from_diff(diff, bits_to_take, fixed_rounding="floor"):
    """Convert signed squared-error differences to pseudo_MSE3 fixed point.

    ``nearest`` matches activation ``encode_emb`` rounding: round magnitude to
    nearest with exact half cases away from zero. ``floor`` preserves the
    original pseudo_MSE3 behavior.
    """
    bits_to_take_param = float(bits_to_take or 0)
    bits_to_take = int(bits_to_take_param)
    if bits_to_take_param != float(bits_to_take):
        raise ValueError(f"bits_to_take must be an integer; got {bits_to_take_param!r}")
    if bits_to_take < 0:
        raise ValueError(f"bits_to_take must be non-negative; got {bits_to_take}")
    fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
    scaled = diff * float(2.0**bits_to_take)
    scaled_pos = torch.where(scaled>0,scaled,0)
    scaled_neg = torch.where(scaled<0,scaled,0)
    scaled_neg = scaled_neg * 4.0

    if fixed_rounding == PSEUDO_MSE3_FIXED_ROUNDING_NEAREST:
        rounded_magnitude_pos = torch.floor(scaled_pos.abs() + 0.5)
        rounded_magnitude_neg = torch.floor(scaled_neg.abs() + 0.5)

        rounded_magnitude_pos = rounded_magnitude_pos * 4.0

        fixed = torch.where(scaled < 0, -rounded_magnitude_neg, rounded_magnitude_pos)
    else:
        fixed = torch.floor(scaled)

    int32_info = torch.iinfo(torch.int32)
    invalid = (
        ~torch.isfinite(fixed)
        | (fixed < int32_info.min)
        | (fixed > int32_info.max)
    )
    if bool(invalid.any()):
        raise OverflowError(
            f"bits_to_take={bits_to_take} produces contributions outside int32"
        )
    return fixed.to(torch.int32)


def pseudo_mse3_err2_minus_err1_from_scaled(
    scaled_values,
    exp1_mantissa_width,
    exp2_mantissa_width,
    is_signed,
    bits_to_take=0,
    fixed_rounding="floor",
):
    """Return normalized per-value squared-error diff err2^2 - err1^2.

    Values are already chunk-scaled into [-2, 2). The squared-error difference
    is normalized by 2^(2*M) before it is returned or converted to fixed point.
    The normalized value must be in the truncating-selection range [-3/4, 3). When
    bits_to_take is non-negative, it is converted to int32 fixed point using
    ``fixed_rounding`` before accumulation; zero uses the normal 2^0 scale.
    """
    m1 = int(exp1_mantissa_width)
    m2 = int(exp2_mantissa_width)
    if m2 != m1 - 1:
        raise ValueError(f"pseudo_MSE3 requires m2 == m1 - 1; got m1={m1}, m2={m2}")
    values = scaled_values.to(torch.float32).contiguous()
    # Format selection intentionally evaluates truncating candidate encodings.
    # The dynamic quantizer reconstructs the selected output separately through
    # pseudo_mse_reconstruct_scaled_python, which remains round-to-nearest.
    q1_scaled = pseudo_mse_reconstruct_scaled_trunc_python(
        values,
        exp_bits=1,
        mantissa_bits=m1,
        is_signed=is_signed,
    )
    q2_scaled = pseudo_mse_reconstruct_scaled_trunc_python(
        values,
        exp_bits=2,
        mantissa_bits=m2,
        is_signed=is_signed,
    )

    err1_sq = (values - q1_scaled).pow(2)
    err2_sq = (values - q2_scaled).pow(2)
    diff = err2_sq - err1_sq
    diff = diff * float(2.0 ** (2 * m1))
    _assert_pseudo_mse3_scaled_diff_ranges(diff)
    return pseudo_mse3_fixed_point_from_diff(
        diff,
        bits_to_take,
        fixed_rounding=fixed_rounding,
    )


def pseudo_mse3_choose_exp2_from_diff(diff, tie_break="exp1", fixed_rounding="floor"):
    """Choose exp=2 from the chunk sum under the requested exact-tie policy."""
    # if fixed_rounding == PSEUDO_MSE3_FIXED_ROUNDING_NEAREST:
    #     diff = torch.where(diff<0,diff/4,diff)
    chunk_sum = diff.sum(dim=1)
    if normalize_pseudo_mse3_tie_break(tie_break) == PSEUDO_MSE3_TIE_BREAK_EXP2:
        return chunk_sum <= 0
    return chunk_sum < 0


def pseudo_mse_sqerr_diff_from_scaled(
    scaled_values,
    exp1_mantissa_width,
    exp2_mantissa_width,
    is_signed,
):
    """Backward-compatible alias for the bit-level pseudo err2-err1 signal."""
    return pseudo_mse_err2_minus_err1_from_scaled(
        scaled_values,
        exp1_mantissa_width,
        exp2_mantissa_width,
        is_signed,
    )


def pseudo_mse_win_counts_from_diff(diff):
    """Return per-chunk (exp1_wins, exp2_wins), excluding exact ties."""
    exp1_wins = (diff > 0).sum(dim=1)
    exp2_wins = (diff < 0).sum(dim=1)
    return exp1_wins, exp2_wins


def pseudo_mse_weighted_win_counts_from_diff(diff):
    """Return per-chunk weighted (exp1_wins, exp2_wins), excluding exact ties."""
    exp1_wins = torch.clamp(diff, min=0).sum(dim=1)
    exp2_wins = torch.clamp(-diff, min=0).sum(dim=1)
    return exp1_wins, exp2_wins


def pseudo_mse_e2_win_divisor_from_param(metric_param=None):
    if metric_param is None:
        return PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR
    divisor_param = float(metric_param)
    divisor = int(divisor_param)
    if divisor_param != float(divisor) or divisor not in PSEUDO_MSE_SUPPORTED_E2_WIN_DIVISORS:
        raise ValueError(
            "pseudo_MSE e2 win divisor must be one of "
            f"{PSEUDO_MSE_SUPPORTED_E2_WIN_DIVISORS}; got {metric_param!r}"
        )
    return divisor


def pseudo_mse_shifted_e2_wins(exp2_wins, e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR):
    """Return the exp=2 decision count after division by the divisor."""
    divisor = pseudo_mse_e2_win_divisor_from_param(e2_win_divisor)
    return torch.div(exp2_wins, divisor)


def pseudo_mse_choose_exp2_from_diff(
    diff,
    e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
    weighted=False,
):
    """Return True where exp2's decision sum is greater than exp1's decision sum."""
    if weighted:
        exp1_wins, exp2_wins = pseudo_mse_weighted_win_counts_from_diff(diff)
        return exp2_wins > exp1_wins
    else:
        exp1_wins, exp2_wins = pseudo_mse_win_counts_from_diff(diff)
    return pseudo_mse_shifted_e2_wins(exp2_wins, e2_win_divisor) >= exp1_wins
