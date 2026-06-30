from dataclasses import dataclass
import re
from typing import Callable

import torch


PSEUDO_MSE_DISPLAY_NAME = "pseudo_MSE"
PSEUDO_MSE_CANONICAL_NAME = "pseudo_mse"


@dataclass(frozen=True)
class DynamicInputMetricSpec:
    name: str
    display_name: str
    cuda_code: int | None
    reducer: Callable[[torch.Tensor, float], torch.Tensor] | None
    pairwise: bool = False

    @property
    def implemented(self):
        return self.cuda_code is not None and (self.reducer is not None or self.pairwise)


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
}

IMPLEMENTED_DYNAMIC_INPUT_METRIC_CODES = {
    name: spec.cuda_code
    for name, spec in DYNAMIC_INPUT_METRICS.items()
    if spec.implemented
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
    return spec.cuda_code


def reduce_dynamic_input_metric_python(metric, diff, metric_param):
    spec = assert_dynamic_input_metric_implemented(metric)
    if spec.reducer is None:
        raise ValueError(f"{spec.display_name} uses a pairwise selector, not a scalar reducer")
    return spec.reducer(diff, metric_param)


def is_pseudo_mse_metric(metric):
    return normalize_dynamic_input_metric(metric) == PSEUDO_MSE_CANONICAL_NAME


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
    """Vectorized Python equivalent of CUDA encode_emb for pseudo_MSE."""
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

        round_shift = torch.clamp(shift - 1, min=0, max=63)
        round_bit_raw = torch.bitwise_and(
            torch.bitwise_right_shift(mant_full, round_shift),
            1,
        )
        round_bit = torch.where(
            (shift >= 1) & (shift <= 24),
            round_bit_raw,
            torch.zeros_like(round_bit_raw),
        )

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
        round_add = torch.bitwise_left_shift(
            torch.ones_like(mant_trunc),
            torch.clamp(shift, min=0, max=24),
        )
        mant_trunc = torch.where(round_bit != 0, mant_trunc + round_add, mant_trunc)

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


def pseudo_mse_sqerr_diff_from_scaled(
    scaled_values,
    exp1_mantissa_width,
    exp2_mantissa_width,
    is_signed,
):
    """Return exact (scaled_v - q_exp2)^2 - (scaled_v - q_exp1)^2."""
    m1 = int(exp1_mantissa_width)
    m2 = int(exp2_mantissa_width)
    if m2 != m1 - 1:
        raise ValueError(f"pseudo_MSE requires m2 == m1 - 1; got m1={m1}, m2={m2}")

    q1 = pseudo_mse_reconstruct_scaled_python(scaled_values, 1, m1, is_signed)
    q2 = pseudo_mse_reconstruct_scaled_python(scaled_values, 2, m2, is_signed)

    d1 = scaled_values.to(torch.float32) - q1
    d2 = scaled_values.to(torch.float32) - q2
    return d2 * d2 - d1 * d1
