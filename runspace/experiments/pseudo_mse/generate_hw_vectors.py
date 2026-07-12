import argparse
import csv
import os
import struct
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.quantization.dynamic_input_metrics import (
    PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
    is_pseudo_mse_metric,
    normalize_dynamic_input_metric,
    pseudo_mse_decode_emb_python,
    pseudo_mse_e2_win_divisor_from_param,
    pseudo_mse_encode_emb_python,
    pseudo_mse_err2_minus_err1_from_scaled,
    pseudo_mse_shifted_e2_wins,
    pseudo_mse_win_counts_from_diff,
    reduce_dynamic_input_metric_python,
)


BIT_WIDTHS = [8, 7, 6, 5, 4]
CHUNK_SIZE = 128
NUM_CHUNKS = 50
SEED = 42
ACTUAL_EXPONENTS = list(range(0, -11, -1))


def fp32_bits(value):
    return struct.unpack("!I", struct.pack("!f", float(value)))[0]


def fp32_hex(value):
    return f"0x{fp32_bits(value):08x}"


def fmt_float(value):
    return f"{float(value):.9g}"


def fmt_bits(value, width):
    return format(int(value), f"0{int(width)}b")


def packed_exp_field(value, exp_bits, mantissa_bits):
    return (int(value) >> int(mantissa_bits)) & ((1 << int(exp_bits)) - 1)


def packed_mant_field(value, mantissa_bits):
    return int(value) & ((1 << int(mantissa_bits)) - 1)


def packed_mant_bits(value, mantissa_bits):
    return fmt_bits(packed_mant_field(value, mantissa_bits), mantissa_bits)


def _alternating_signs(n):
    signs = torch.ones(n, dtype=torch.float32)
    signs[1::2] = -1.0
    return signs


def _midpoint_bucket(exp, *, signed=True, phase=0.0):
    """128 values with actual scaled exponent `exp`, spread across [1, 2)."""
    idx = torch.arange(CHUNK_SIZE, dtype=torch.float32)
    mant = 1.0 + torch.remainder(idx + 0.5 + float(phase), CHUNK_SIZE) / CHUNK_SIZE
    values = mant * float(2.0 ** exp)
    if signed:
        values = values * _alternating_signs(CHUNK_SIZE)
    return values.to(torch.float32).contiguous()


def _repeat_to_chunk(values):
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError("Cannot build a chunk from an empty value list")
    repeats = (CHUNK_SIZE + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:CHUNK_SIZE].contiguous()


def _mixed_exponent_chunk(gen, *, high_exponent_bias=False):
    """A deterministic random chunk with different actual exponents per value."""
    if high_exponent_bias:
        exp_pool = torch.tensor(
            [0] * 32 + [-1] * 32 + [-2] * 16 + [-3] * 12 + [-4] * 8 +
            [-5] * 8 + [-6] * 6 + [-7] * 6 + [-8] * 4 + [-9] * 2 + [-10] * 2,
            dtype=torch.int64,
        )
        perm = torch.randperm(exp_pool.numel(), generator=gen)
        exps = exp_pool[perm][:CHUNK_SIZE]
    else:
        exp_pool = torch.tensor(ACTUAL_EXPONENTS, dtype=torch.int64)
        repeats = (CHUNK_SIZE + exp_pool.numel() - 1) // exp_pool.numel()
        exps = exp_pool.repeat(repeats)[:CHUNK_SIZE]
        exps = exps[torch.randperm(CHUNK_SIZE, generator=gen)]

    mant = 1.0 + torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32)
    values = mant * torch.pow(torch.tensor(2.0, dtype=torch.float32), exps.to(torch.float32))
    signs = torch.where(
        torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32) < 0.5,
        torch.full((CHUNK_SIZE,), -1.0, dtype=torch.float32),
        torch.ones(CHUNK_SIZE, dtype=torch.float32),
    )
    return (values * signs).contiguous()


def _exact_grid_chunk():
    mantissas = [
        -1.5,
        -1.25,
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
    ]
    values = []
    for mant in mantissas:
        for exp in ACTUAL_EXPONENTS:
            values.append(mant * float(2.0 ** exp))
    return _repeat_to_chunk(values)


def _rounding_boundary_chunk():
    values = []
    # Around e1/e2 mantissa midpoints for several actual exponents. These are
    # useful for catching off-by-one rounding and tie handling in hardware.
    for k in (0, 1, 2, 3):
        for denom in (8, 16, 32, 64):
            for exp in [0, -1, -2, -5, -8, -10]:
                scale = float(2.0 ** exp)
                center = 1.0 + (k + 0.5) / denom
                eps = 1.0 / (denom * 256.0)
                values.extend([
                    -(center - eps) * scale,
                    -(center + eps) * scale,
                    (center - eps) * scale,
                    (center + eps) * scale,
                ])
    return _repeat_to_chunk(values)


def _underflow_boundary_chunk():
    values = []
    for exp in ACTUAL_EXPONENTS:
        if exp > -7:
            continue
        scale = float(2.0 ** exp)
        for mant in torch.linspace(1.0, 1.984375, 16, dtype=torch.float32):
            values.extend([-float(mant) * scale, float(mant) * scale])
    return _repeat_to_chunk(values)


def _random_boundary_chunk(gen):
    idx = torch.arange(CHUNK_SIZE, dtype=torch.float32)
    denom_choices = torch.tensor([8, 16, 32, 64, 128, 256], dtype=torch.int64)
    exp_choices = torch.tensor(ACTUAL_EXPONENTS, dtype=torch.int64)
    denom = denom_choices[torch.randint(0, denom_choices.numel(), (CHUNK_SIZE,), generator=gen)]
    exps = exp_choices[torch.randint(0, exp_choices.numel(), (CHUNK_SIZE,), generator=gen)]
    k = torch.remainder(idx.to(torch.int64), denom)
    offsets = torch.where(
        torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32) < 0.5,
        torch.full((CHUNK_SIZE,), -1.0, dtype=torch.float32),
        torch.ones(CHUNK_SIZE, dtype=torch.float32),
    )
    eps = 1.0 / (denom.to(torch.float32) * 512.0)
    mant = 1.0 + (k.to(torch.float32) + 0.5) / denom.to(torch.float32)
    values = (mant + offsets * eps) * torch.pow(torch.tensor(2.0, dtype=torch.float32), exps.to(torch.float32))
    signs = torch.where(
        torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32) < 0.5,
        torch.full((CHUNK_SIZE,), -1.0, dtype=torch.float32),
        torch.ones(CHUNK_SIZE, dtype=torch.float32),
    )
    return (values * signs).contiguous()


def _random_sparse_chunk(gen):
    values = _mixed_exponent_chunk(gen, high_exponent_bias=False)
    zero_mask = torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32) < 0.75
    tiny_exps = torch.randint(-24, -10, (CHUNK_SIZE,), generator=gen, dtype=torch.int64)
    tiny_mant = torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32)
    tiny = tiny_mant * torch.pow(torch.tensor(2.0, dtype=torch.float32), tiny_exps.to(torch.float32))
    signs = torch.where(
        torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32) < 0.5,
        torch.full((CHUNK_SIZE,), -1.0, dtype=torch.float32),
        torch.ones(CHUNK_SIZE, dtype=torch.float32),
    )
    tiny = tiny * signs
    return torch.where(zero_mask, torch.zeros_like(values), values + tiny).contiguous()


def _random_near_two_chunk(gen):
    exps = torch.randint(-10, 1, (CHUNK_SIZE,), generator=gen, dtype=torch.int64)
    eps = (1.0 + torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32)) / 512.0
    mant = 2.0 - eps
    values = mant * torch.pow(torch.tensor(2.0, dtype=torch.float32), exps.to(torch.float32))
    signs = torch.where(
        torch.rand(CHUNK_SIZE, generator=gen, dtype=torch.float32) < 0.5,
        torch.full((CHUNK_SIZE,), -1.0, dtype=torch.float32),
        torch.ones(CHUNK_SIZE, dtype=torch.float32),
    )
    return (values * signs).contiguous()


def _random_exact_grid_chunk(gen):
    mant_steps = torch.randint(-128, 128, (CHUNK_SIZE,), generator=gen, dtype=torch.int64)
    exps = torch.randint(-10, 1, (CHUNK_SIZE,), generator=gen, dtype=torch.int64)
    mant = mant_steps.to(torch.float32) / 64.0
    values = mant * torch.pow(torch.tensor(2.0, dtype=torch.float32), exps.to(torch.float32))
    return values.contiguous()


def _random_chunk(gen, chunk_idx):
    pattern = chunk_idx % 6
    if pattern == 0:
        return _mixed_exponent_chunk(gen, high_exponent_bias=False)
    if pattern == 1:
        return _mixed_exponent_chunk(gen, high_exponent_bias=True)
    if pattern == 2:
        return _random_boundary_chunk(gen)
    if pattern == 3:
        return _random_sparse_chunk(gen)
    if pattern == 4:
        return _random_near_two_chunk(gen)
    return _random_exact_grid_chunk(gen)


def make_raw_chunks(num_chunks=NUM_CHUNKS, seed=SEED):
    gen = torch.Generator().manual_seed(int(seed))

    chunks = [
        _midpoint_bucket(0, signed=True),
        _midpoint_bucket(-1, signed=True, phase=17.0),
        _midpoint_bucket(-2, signed=True, phase=31.0),
        _midpoint_bucket(-5, signed=True, phase=7.0),
        _underflow_boundary_chunk(),
        _mixed_exponent_chunk(gen, high_exponent_bias=False),
        _mixed_exponent_chunk(gen, high_exponent_bias=True),
        _exact_grid_chunk(),
        _rounding_boundary_chunk(),
        _repeat_to_chunk([0.0]),
    ]
    while len(chunks) < int(num_chunks):
        chunks.append(_random_chunk(gen, len(chunks)))
    chunks = chunks[:int(num_chunks)]

    raw_chunks = torch.stack(chunks).contiguous()
    if raw_chunks.shape != (int(num_chunks), CHUNK_SIZE):
        raise RuntimeError(f"Expected chunks shape {(int(num_chunks), CHUNK_SIZE)}, got {tuple(raw_chunks.shape)}")
    if not torch.isfinite(raw_chunks).all():
        raise RuntimeError("Generated non-finite raw HW vector value")
    return raw_chunks


def scale_raw_chunks(raw_chunks):
    amax = raw_chunks.abs().max(dim=1, keepdim=True).values
    scales = torch.ones_like(amax)
    nonzero = amax != 0
    if nonzero.any():
        values = amax[nonzero].contiguous()
        bits = values.view(torch.int32)
        mask = torch.tensor(-8388608, dtype=torch.int32, device=raw_chunks.device)
        scales[nonzero] = torch.bitwise_and(bits, mask).view(torch.float32)
    scales = scales.to(torch.float32).contiguous()
    scaled_chunks = raw_chunks / scales
    if not torch.isfinite(scales).all():
        raise RuntimeError("Generated non-finite source-code chunk scale")
    if not torch.isfinite(scaled_chunks).all():
        raise RuntimeError("Generated non-finite scaled HW vector value")
    if scaled_chunks.abs().max() >= 2.0:
        raise RuntimeError("Scaled HW vector values must stay in [-2, 2)")
    return scales, scaled_chunks.contiguous()


def pseudo_mse_encode_emb_export(values, exp_bits, mantissa_bits, is_signed):
    return pseudo_mse_encode_emb_python(values, exp_bits, mantissa_bits, is_signed)


def pseudo_mse_reconstruct_scaled_export(scaled_values, exp_bits, mantissa_bits, is_signed):
    packed = pseudo_mse_encode_emb_export(
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


def decision_for_bit_width(
    scaled_chunks,
    bit_width,
    e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
):
    e2_win_divisor = pseudo_mse_e2_win_divisor_from_param(e2_win_divisor)
    m1 = bit_width - 2
    m2 = bit_width - 3
    if m2 != m1 - 1 or m2 < 1:
        raise ValueError(f"Unsupported bit width for pseudo_MSE e1/e2 pair: {bit_width}")

    q1_bits = pseudo_mse_encode_emb_export(
        scaled_chunks,
        exp_bits=1,
        mantissa_bits=m1,
        is_signed=True,
    )
    q2_bits = pseudo_mse_encode_emb_export(
        scaled_chunks,
        exp_bits=2,
        mantissa_bits=m2,
        is_signed=True,
    )
    q1 = pseudo_mse_reconstruct_scaled_export(
        scaled_chunks,
        exp_bits=1,
        mantissa_bits=m1,
        is_signed=True,
    )
    q2 = pseudo_mse_reconstruct_scaled_export(
        scaled_chunks,
        exp_bits=2,
        mantissa_bits=m2,
        is_signed=True,
    )
    err_exp1_pre_square = scaled_chunks - q1
    err_exp2_pre_square = scaled_chunks - q2
    pseudo_diff = pseudo_mse_err2_minus_err1_from_scaled(
        scaled_chunks,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m2,
        is_signed=True,
    )
    err1 = err_exp1_pre_square.pow(2).sum(dim=1)
    err2 = err_exp2_pre_square.pow(2).sum(dim=1)
    chunk_diff = pseudo_diff.sum(dim=1)
    expected_e1_wins, expected_e2_wins = pseudo_mse_win_counts_from_diff(pseudo_diff)
    expected_e2_wins_shifted = pseudo_mse_shifted_e2_wins(
        expected_e2_wins,
        e2_win_divisor,
    )
    choose_exp2 = expected_e2_wins_shifted >= expected_e1_wins
    expected_error = torch.where(choose_exp2, err2, err1)
    return (
        err1,
        err2,
        chunk_diff,
        expected_e1_wins,
        expected_e2_wins,
        expected_e2_wins_shifted,
        choose_exp2,
        expected_error,
        q1_bits,
        q2_bits,
        err_exp1_pre_square,
        err_exp2_pre_square,
        pseudo_diff,
    )


def write_value_header(f, value_mode):
    if value_mode == "raw-and-scaled":
        f.write("# value mode: raw-and-scaled\n")
        f.write("# chunk_scale rows: chunk_scale chunk_scale_fp32_hex chunk_scale_fp32_dec\n")
        f.write("# chunk max rows use max absolute value; *_value is the signed value at that index\n")
        f.write(
            "# chunk max rows: chunk_max_abs_index value_index, "
            "chunk_max_abs_raw_value raw_fp32_hex raw_fp32_dec, "
            "chunk_max_abs_raw_abs raw_abs_fp32_hex raw_abs_fp32_dec, "
            "chunk_max_abs_scaled_value scaled_fp32_hex scaled_fp32_dec, "
            "chunk_max_abs_scaled_abs scaled_abs_fp32_hex scaled_abs_fp32_dec\n"
        )
        f.write("# raw max alias rows: raw_max_abs_index, raw_max_abs_value, raw_max_abs_abs\n")
        f.write(
            "# raw-and-scaled max summary rows: raw_and_scaled_max_abs "
            "value_index scale_fp32_hex scale_fp32_dec raw_value_fp32_hex raw_value_fp32_dec "
            "raw_abs_fp32_hex raw_abs_fp32_dec scaled_value_fp32_hex scaled_value_fp32_dec "
            "scaled_abs_fp32_hex scaled_abs_fp32_dec\n"
        )
        f.write(
            "# value rows: value_index raw_fp32_hex raw_fp32_dec "
            "scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec\n"
        )
        f.write("# scaled_fp32_dec = raw_fp32_dec / scale_fp32_dec\n\n")
    else:
        f.write("# chunk max rows use max absolute value; *_value is the signed value at that index\n")
        f.write(
            "# chunk max rows: chunk_max_abs_index value_index, "
            "chunk_max_abs_scaled_value scaled_fp32_hex scaled_fp32_dec, "
            "chunk_max_abs_scaled_abs scaled_abs_fp32_hex scaled_abs_fp32_dec\n"
        )
        f.write(
            "# value rows: value_index scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec\n\n"
        )


def write_chunk_max_rows(f, raw_chunks, scaled_chunks, scales, chunk_idx, value_mode):
    scale = scales[chunk_idx, 0]
    raw_values = raw_chunks[chunk_idx]
    scaled_values = scaled_chunks[chunk_idx]
    max_abs_idx = int(torch.argmax(raw_values.abs()).item())
    max_abs_raw_value = raw_values[max_abs_idx]
    max_abs_raw_abs = max_abs_raw_value.abs()
    max_abs_scaled_value = scaled_values[max_abs_idx]
    max_abs_scaled_abs = max_abs_scaled_value.abs()

    f.write(f"chunk_max_abs_index {max_abs_idx}\n")
    if value_mode == "raw-and-scaled":
        f.write(f"raw_max_abs_index {max_abs_idx}\n")
        f.write(f"raw_max_abs_value {fp32_hex(max_abs_raw_value)} {fmt_float(max_abs_raw_value)}\n")
        f.write(f"raw_max_abs_abs {fp32_hex(max_abs_raw_abs)} {fmt_float(max_abs_raw_abs)}\n")
        f.write(f"chunk_max_abs_raw_value {fp32_hex(max_abs_raw_value)} {fmt_float(max_abs_raw_value)}\n")
        f.write(f"chunk_max_abs_raw_abs {fp32_hex(max_abs_raw_abs)} {fmt_float(max_abs_raw_abs)}\n")
        f.write(
            f"raw_and_scaled_max_abs {max_abs_idx} "
            f"{fp32_hex(scale)} {fmt_float(scale)} "
            f"{fp32_hex(max_abs_raw_value)} {fmt_float(max_abs_raw_value)} "
            f"{fp32_hex(max_abs_raw_abs)} {fmt_float(max_abs_raw_abs)} "
            f"{fp32_hex(max_abs_scaled_value)} {fmt_float(max_abs_scaled_value)} "
            f"{fp32_hex(max_abs_scaled_abs)} {fmt_float(max_abs_scaled_abs)}\n"
        )
    f.write(f"chunk_max_abs_scaled_value {fp32_hex(max_abs_scaled_value)} {fmt_float(max_abs_scaled_value)}\n")
    f.write(f"chunk_max_abs_scaled_abs {fp32_hex(max_abs_scaled_abs)} {fmt_float(max_abs_scaled_abs)}\n")


def write_value_rows(
    f,
    raw_chunks,
    scaled_chunks,
    scales,
    q1_bits_chunks,
    q2_bits_chunks,
    err_exp1_pre_square_chunks,
    err_exp2_pre_square_chunks,
    pseudo_diff_chunks,
    bit_width,
    chunk_idx,
    value_mode,
):
    scale = scales[chunk_idx, 0]
    m1 = bit_width - 2
    m2 = bit_width - 3
    if value_mode == "raw-and-scaled":
        f.write(f"chunk_scale {fp32_hex(scale)} {fmt_float(scale)}\n")
        f.write(
            "# value_index raw_fp32_hex raw_fp32_dec scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec\n"
        )
    else:
        f.write(
            "# value_index scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec\n"
        )

    f.write("VALUES_BEGIN\n")
    for value_idx in range(CHUNK_SIZE):
        raw_v = raw_chunks[chunk_idx, value_idx]
        scaled_v = scaled_chunks[chunk_idx, value_idx]
        q1_bits_v = q1_bits_chunks[chunk_idx, value_idx]
        q2_bits_v = q2_bits_chunks[chunk_idx, value_idx]
        q1_exp_v = packed_exp_field(q1_bits_v, 1, m1)
        q1_mant_v = packed_mant_field(q1_bits_v, m1)
        q1_mant_bits_v = packed_mant_bits(q1_bits_v, m1)
        q2_exp_v = packed_exp_field(q2_bits_v, 2, m2)
        q2_mant_v = packed_mant_field(q2_bits_v, m2)
        q2_mant_bits_v = packed_mant_bits(q2_bits_v, m2)
        err_exp1_pre_square_v = err_exp1_pre_square_chunks[chunk_idx, value_idx]
        err_exp2_pre_square_v = err_exp2_pre_square_chunks[chunk_idx, value_idx]
        pseudo_diff_v = pseudo_diff_chunks[chunk_idx, value_idx]
        if value_mode == "raw-and-scaled":
            f.write(
                f"{value_idx:03d} "
                f"{fp32_hex(raw_v)} {fmt_float(raw_v)} "
                f"{fp32_hex(scaled_v)} {fmt_float(scaled_v)} "
                f"{fmt_bits(q1_bits_v, bit_width)} {q1_exp_v} {q1_mant_v} {q1_mant_bits_v} "
                f"{fmt_bits(q2_bits_v, bit_width)} {q2_exp_v} {q2_mant_v} {q2_mant_bits_v} "
                f"{fp32_hex(err_exp1_pre_square_v)} {fmt_float(err_exp1_pre_square_v)} "
                f"{fp32_hex(err_exp2_pre_square_v)} {fmt_float(err_exp2_pre_square_v)} "
                f"{fp32_hex(pseudo_diff_v)} {fmt_float(pseudo_diff_v)}\n"
            )
        else:
            f.write(
                f"{value_idx:03d} "
                f"{fp32_hex(scaled_v)} {fmt_float(scaled_v)} "
                f"{fmt_bits(q1_bits_v, bit_width)} {q1_exp_v} {q1_mant_v} {q1_mant_bits_v} "
                f"{fmt_bits(q2_bits_v, bit_width)} {q2_exp_v} {q2_mant_v} {q2_mant_bits_v} "
                f"{fp32_hex(err_exp1_pre_square_v)} {fmt_float(err_exp1_pre_square_v)} "
                f"{fp32_hex(err_exp2_pre_square_v)} {fmt_float(err_exp2_pre_square_v)} "
                f"{fp32_hex(pseudo_diff_v)} {fmt_float(pseudo_diff_v)}\n"
            )
    f.write("VALUES_END\n")


def write_vectors(
    path,
    value_mode="scaled",
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
):
    num_chunks = int(num_chunks)
    seed = int(seed)
    e2_win_divisor = pseudo_mse_e2_win_divisor_from_param(e2_win_divisor)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        f.write("# pseudo_MSE hardware test vectors\n")
        f.write(f"# seed={seed}\n")
        f.write(f"# bit_widths={','.join(str(b) for b in BIT_WIDTHS)}\n")
        f.write(f"# num_chunks_per_bit_width={num_chunks}\n")
        f.write(f"# chunk_size={CHUNK_SIZE}\n")
        f.write(f"# e2_win_divisor={e2_win_divisor}\n")
        f.write("# signedness: sgn=1 for every section in this file\n")
        f.write("# chunk scale source: DynamicInputQuantizer._chunk_scale(raw_chunk)\n")
        f.write("# chunk scale definition: pow2_floor(max(abs(raw_chunk))) with 1.0 for all-zero chunks\n")
        f.write("# metric input: values are already scaled FP32 numbers (scaled_v)\n")
        if value_mode == "raw-and-scaled":
            f.write("# raw_fp32 values are provided only to test the pre-scale path\n")
            f.write("# feed scaled_fp32 values to the pseudo_MSE block, or compute scaled_fp32 = raw_fp32 / chunk_scale\n")
        else:
            f.write("# do not apply a chunk scale before feeding these values to the pseudo_MSE block\n")
        f.write("# decision rule: choose_exp2 if expected_e2_wins / e2_win_divisor >= expected_e1_wins else choose_exp1\n")
        f.write("# expected_error rows: selected chunk error matching expected_decision, as fp32_hex fp32_dec\n")
        f.write("# expected_e1_wins/expected_e2_wins rows: per-value pseudo_diff winners; ties are not counted\n")
        f.write("# expected_e2_wins_shifted row: expected_e2_wins / e2_win_divisor, used only for expected_decision\n")
        f.write("# expected_e2_wins_shift2 row: legacy expected_e2_wins / 4 diagnostic\n")
        f.write("# mantissa mode: round-to-nearest\n")
        f.write("# q_exp*_bits are packed sign/exponent/mantissa fields after pseudo_MSE quantization\n")
        f.write("# q_exp*_exp_field and q_exp*_mant_field are the stored exponent and mantissa fields as integers\n")
        f.write("# q_exp*_mant_bits are the stored mantissa fields as zero-padded binary strings\n")
        f.write("# err_exp*_pre_square = scaled-q_exp* for each value, before squaring\n")
        f.write("# chunk_diff_exp2_minus_exp1 is the signed sum of pseudo_diff values for debug; it does not drive expected_decision\n")
        f.write("# pseudo_diff_exp2_minus_exp1 is the bit-level pseudo err2-err1 signal for each value\n")
        write_value_header(f, value_mode)

        for bit_width in BIT_WIDTHS:
            m1 = bit_width - 2
            m2 = bit_width - 3
            exp1_format = f"fp{bit_width}_e1m{m1}"
            exp2_format = f"fp{bit_width}_e2m{m2}"
            (
                err1,
                err2,
                chunk_diff,
                expected_e1_wins,
                expected_e2_wins,
                expected_e2_wins_shifted,
                choose_exp2,
                expected_error,
                q1_bits,
                q2_bits,
                err_exp1_pre_square,
                err_exp2_pre_square,
                pseudo_diff,
            ) = decision_for_bit_width(
                scaled_chunks,
                bit_width,
                e2_win_divisor=e2_win_divisor,
            )
            expected_e2_wins_shift2 = pseudo_mse_shifted_e2_wins(
                expected_e2_wins,
                PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
            )

            f.write(f"BEGIN_BIT_WIDTH {bit_width}\n")
            f.write(f"sgn 1\n")
            f.write(f"exp1_format {exp1_format}\n")
            f.write(f"exp2_format {exp2_format}\n")
            f.write(f"m1 {m1}\n")
            f.write(f"m2 {m2}\n")
            for chunk_idx in range(num_chunks):
                decision = "exp2" if bool(choose_exp2[chunk_idx]) else "exp1"
                f.write(f"BEGIN_CHUNK {chunk_idx}\n")
                f.write(f"err_exp1 {fmt_float(err1[chunk_idx])}\n")
                f.write(f"err_exp2 {fmt_float(err2[chunk_idx])}\n")
                f.write(f"chunk_diff_exp2_minus_exp1 {fmt_float(chunk_diff[chunk_idx])}\n")
                f.write(f"expected_e1_wins {int(expected_e1_wins[chunk_idx])}\n")
                f.write(f"expected_e2_wins {int(expected_e2_wins[chunk_idx])}\n")
                f.write(f"expected_e2_wins_shifted {int(expected_e2_wins_shifted[chunk_idx])}\n")
                f.write(f"expected_e2_wins_shift2 {int(expected_e2_wins_shift2[chunk_idx])}\n")
                f.write(f"expected_decision {decision}\n")
                f.write(
                    f"expected_error {fp32_hex(expected_error[chunk_idx])} "
                    f"{fmt_float(expected_error[chunk_idx])}\n"
                )
                write_chunk_max_rows(f, raw_chunks, scaled_chunks, scales, chunk_idx, value_mode)
                write_value_rows(
                    f,
                    raw_chunks,
                    scaled_chunks,
                    scales,
                    q1_bits,
                    q2_bits,
                    err_exp1_pre_square,
                    err_exp2_pre_square,
                    pseudo_diff,
                    bit_width,
                    chunk_idx,
                    value_mode,
                )
                f.write(f"END_CHUNK {chunk_idx}\n")
            f.write(f"END_BIT_WIDTH {bit_width}\n\n")


def _csv_fieldnames(value_mode):
    fields = [
        "value_mode",
        "bit_width",
        "sgn",
        "exp1_format",
        "exp2_format",
        "m1",
        "m2",
        "e2_win_divisor",
        "chunk_idx",
        "value_idx",
        "expected_decision",
        "expected_e1_wins",
        "expected_e2_wins",
        "expected_e2_wins_shifted",
        "expected_e2_wins_shift2",
        "err_exp1_dec",
        "err_exp2_dec",
        "chunk_diff_exp2_minus_exp1_dec",
        "expected_error_fp32_hex",
        "expected_error_fp32_dec",
        "chunk_scale_fp32_hex",
        "chunk_scale_fp32_dec",
    ]
    if value_mode == "raw-and-scaled":
        fields.extend(["raw_fp32_hex", "raw_fp32_dec"])
    fields.extend([
        "scaled_fp32_hex",
        "scaled_fp32_dec",
        "q_exp1_bits",
        "q_exp1_exp_field",
        "q_exp1_mant_field",
        "q_exp1_mant_bits",
        "q_exp2_bits",
        "q_exp2_exp_field",
        "q_exp2_mant_field",
        "q_exp2_mant_bits",
        "err_exp1_pre_square_fp32_hex",
        "err_exp1_pre_square_fp32_dec",
        "err_exp2_pre_square_fp32_hex",
        "err_exp2_pre_square_fp32_dec",
        "pseudo_diff_exp2_minus_exp1_fp32_hex",
        "pseudo_diff_exp2_minus_exp1_fp32_dec",
    ])
    return fields


def _value_metadata_row(
    value_mode,
    bit_width,
    m1,
    m2,
    raw_chunks,
    scaled_chunks,
    q1_bits,
    q2_bits,
    err_exp1_pre_square,
    err_exp2_pre_square,
    pseudo_diff,
    chunk_idx,
    value_idx,
):
    raw_v = raw_chunks[chunk_idx, value_idx]
    scaled_v = scaled_chunks[chunk_idx, value_idx]
    q1_bits_v = q1_bits[chunk_idx, value_idx]
    q2_bits_v = q2_bits[chunk_idx, value_idx]
    err_exp1_pre_square_v = err_exp1_pre_square[chunk_idx, value_idx]
    err_exp2_pre_square_v = err_exp2_pre_square[chunk_idx, value_idx]
    pseudo_diff_v = pseudo_diff[chunk_idx, value_idx]
    row = {
        "value_idx": value_idx,
        "value_index": value_idx,
        "scaled_fp32_hex": fp32_hex(scaled_v),
        "scaled_fp32_dec": fmt_float(scaled_v),
        "q_exp1_bits": fmt_bits(q1_bits_v, bit_width),
        "q_exp1_exp_field": packed_exp_field(q1_bits_v, 1, m1),
        "q_exp1_mant_field": packed_mant_field(q1_bits_v, m1),
        "q_exp1_mant_bits": packed_mant_bits(q1_bits_v, m1),
        "q_exp2_bits": fmt_bits(q2_bits_v, bit_width),
        "q_exp2_exp_field": packed_exp_field(q2_bits_v, 2, m2),
        "q_exp2_mant_field": packed_mant_field(q2_bits_v, m2),
        "q_exp2_mant_bits": packed_mant_bits(q2_bits_v, m2),
        "err_exp1_pre_square_fp32_hex": fp32_hex(err_exp1_pre_square_v),
        "err_exp1_pre_square_fp32_dec": fmt_float(err_exp1_pre_square_v),
        "err_exp2_pre_square_fp32_hex": fp32_hex(err_exp2_pre_square_v),
        "err_exp2_pre_square_fp32_dec": fmt_float(err_exp2_pre_square_v),
        "pseudo_diff_exp2_minus_exp1_fp32_hex": fp32_hex(pseudo_diff_v),
        "pseudo_diff_exp2_minus_exp1_fp32_dec": fmt_float(pseudo_diff_v),
    }
    if value_mode == "raw-and-scaled":
        row.update({
            "raw_fp32_hex": fp32_hex(raw_v),
            "raw_fp32_dec": fmt_float(raw_v),
        })
    return row


def write_csv_vectors(
    path,
    value_mode="scaled",
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
):
    num_chunks = int(num_chunks)
    seed = int(seed)
    e2_win_divisor = pseudo_mse_e2_win_divisor_from_param(e2_win_divisor)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = _csv_fieldnames(value_mode)
    with open(path, "w", newline="") as f:
        f.write(",".join(fieldnames) + "\n")

        for bit_width in BIT_WIDTHS:
            m1 = bit_width - 2
            m2 = bit_width - 3
            exp1_format = f"fp{bit_width}_e1m{m1}"
            exp2_format = f"fp{bit_width}_e2m{m2}"
            (
                err1,
                err2,
                chunk_diff,
                expected_e1_wins,
                expected_e2_wins,
                expected_e2_wins_shifted,
                choose_exp2,
                expected_error,
                q1_bits,
                q2_bits,
                err_exp1_pre_square,
                err_exp2_pre_square,
                pseudo_diff,
            ) = decision_for_bit_width(
                scaled_chunks,
                bit_width,
                e2_win_divisor=e2_win_divisor,
            )
            expected_e2_wins_shift2 = pseudo_mse_shifted_e2_wins(
                expected_e2_wins,
                PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
            )

            for chunk_idx in range(num_chunks):
                decision = "exp2" if bool(choose_exp2[chunk_idx]) else "exp1"
                scale = scales[chunk_idx, 0]
                chunk_fields = {
                    "value_mode": value_mode,
                    "bit_width": bit_width,
                    "sgn": 1,
                    "exp1_format": exp1_format,
                    "exp2_format": exp2_format,
                    "m1": m1,
                    "m2": m2,
                    "e2_win_divisor": e2_win_divisor,
                    "chunk_idx": chunk_idx,
                    "expected_decision": decision,
                    "expected_e1_wins": int(expected_e1_wins[chunk_idx]),
                    "expected_e2_wins": int(expected_e2_wins[chunk_idx]),
                    "expected_e2_wins_shifted": int(expected_e2_wins_shifted[chunk_idx]),
                    "expected_e2_wins_shift2": int(expected_e2_wins_shift2[chunk_idx]),
                    "err_exp1_dec": fmt_float(err1[chunk_idx]),
                    "err_exp2_dec": fmt_float(err2[chunk_idx]),
                    "chunk_diff_exp2_minus_exp1_dec": fmt_float(chunk_diff[chunk_idx]),
                    "expected_error_fp32_hex": fp32_hex(expected_error[chunk_idx]),
                    "expected_error_fp32_dec": fmt_float(expected_error[chunk_idx]),
                    "chunk_scale_fp32_hex": fp32_hex(scale),
                    "chunk_scale_fp32_dec": fmt_float(scale),
                }
                for value_idx in range(CHUNK_SIZE):
                    row = {
                        **chunk_fields,
                        **_value_metadata_row(
                            value_mode,
                            bit_width,
                            m1,
                            m2,
                            raw_chunks,
                            scaled_chunks,
                            q1_bits,
                            q2_bits,
                            err_exp1_pre_square,
                            err_exp2_pre_square,
                            pseudo_diff,
                            chunk_idx,
                            value_idx,
                        ),
                    }
                    f.write(",".join(str(row[field]).strip() for field in fieldnames) + "\n")


def _normalize_compare_metric(metric):
    normalized = normalize_dynamic_input_metric(metric)
    if is_pseudo_mse_metric(normalized):
        raise ValueError("Use a scalar metric such as l1/l2/linf/bias/l0/huber/logsum for comparison")
    return normalized


def _comparison_csv_fieldnames(value_mode):
    return [
        "compare_metric",
        "compare_metric_param",
        "compare_atol",
        "compare_tie_policy",
        "mismatch_kind",
        "metric_decision",
        "metric_tie",
        "metric_min_mismatch",
        "decision_mismatch",
        "metric_exp1_error_dec",
        "metric_exp2_error_dec",
        "metric_min_error_dec",
        "pseudo_selected_metric_error_dec",
        "metric_error_delta_dec",
        "value_index",
    ] + _csv_fieldnames(value_mode)


def _metric_choose_exp2(metric_exp1_error, metric_exp2_error, tie_policy, compare_atol):
    if tie_policy == "exp2":
        return metric_exp2_error <= metric_exp1_error + compare_atol
    return metric_exp2_error < metric_exp1_error - compare_atol


def _metric_decision_label(is_tie, choose_exp2, tie_policy):
    if bool(is_tie) and tie_policy == "min-error":
        return "tie"
    return "exp2" if bool(choose_exp2) else "exp1"


def _mismatch_kind(metric_min_bad, metric_tie):
    if bool(metric_min_bad):
        return "metric_min"
    if bool(metric_tie):
        return "tie_decision"
    return "decision"


def compare_pseudo_mse_with_metric(
    csv_path,
    compare_metric="l1",
    compare_metric_param=0.0625,
    compare_atol=0.0,
    compare_tie_policy="min-error",
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
    max_mismatches=20,
):
    compare_metric = _normalize_compare_metric(compare_metric)
    compare_metric_param = float(compare_metric_param)
    compare_atol = float(compare_atol)
    if compare_atol < 0.0:
        raise ValueError("--compare-atol must be non-negative")
    e2_win_divisor = pseudo_mse_e2_win_divisor_from_param(e2_win_divisor)
    if compare_tie_policy not in ("min-error", "exp1", "exp2"):
        raise ValueError("--compare-tie-policy must be one of: min-error, exp1, exp2")

    num_chunks = int(num_chunks)
    seed = int(seed)
    max_mismatches = int(max_mismatches)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    value_mode = "raw-and-scaled"
    fieldnames = _comparison_csv_fieldnames(value_mode)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    totals = {
        "reported_mismatched_chunks": 0,
        "metric_min_mismatched_chunks": 0,
        "decision_disagreements": 0,
        "metric_tie_chunks": 0,
        "rows_written": 0,
    }

    print("pseudo_MSE metric comparison")
    print(f"seed={seed} num_chunks={num_chunks} chunk_size={CHUNK_SIZE}")
    print(f"e2_win_divisor={e2_win_divisor}")
    print(f"compare_metric={compare_metric} compare_metric_param={fmt_float(compare_metric_param)}")
    print(f"compare_atol={fmt_float(compare_atol)}")
    print(f"compare_tie_policy={compare_tie_policy}")
    print(f"mismatch_csv={csv_path}")

    printed = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for bit_width in BIT_WIDTHS:
            m1 = bit_width - 2
            m2 = bit_width - 3
            exp1_format = f"fp{bit_width}_e1m{m1}"
            exp2_format = f"fp{bit_width}_e2m{m2}"
            (
                err1,
                err2,
                chunk_diff,
                expected_e1_wins,
                expected_e2_wins,
                expected_e2_wins_shifted,
                choose_exp2,
                expected_error,
                q1_bits,
                q2_bits,
                err_exp1_pre_square,
                err_exp2_pre_square,
                pseudo_diff,
            ) = decision_for_bit_width(
                scaled_chunks,
                bit_width,
                e2_win_divisor=e2_win_divisor,
            )
            expected_e2_wins_shift2 = pseudo_mse_shifted_e2_wins(
                expected_e2_wins,
                PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
            )

            metric_exp1_error = reduce_dynamic_input_metric_python(
                compare_metric,
                err_exp1_pre_square,
                compare_metric_param,
            )
            metric_exp2_error = reduce_dynamic_input_metric_python(
                compare_metric,
                err_exp2_pre_square,
                compare_metric_param,
            )
            metric_tie = (metric_exp1_error - metric_exp2_error).abs() <= compare_atol
            metric_choose_exp2 = _metric_choose_exp2(
                metric_exp1_error,
                metric_exp2_error,
                compare_tie_policy,
                compare_atol,
            )
            metric_min_error = torch.minimum(metric_exp1_error, metric_exp2_error)
            pseudo_selected_metric_error = torch.where(
                choose_exp2,
                metric_exp2_error,
                metric_exp1_error,
            )
            metric_min_bad = pseudo_selected_metric_error > metric_min_error + compare_atol
            if compare_tie_policy == "min-error":
                decision_bad = (choose_exp2 != metric_choose_exp2) & (~metric_tie)
                report_bad = metric_min_bad
            else:
                decision_bad = choose_exp2 != metric_choose_exp2
                report_bad = decision_bad

            reported_count = int(report_bad.sum().item())
            metric_min_count = int(metric_min_bad.sum().item())
            decision_count = int(decision_bad.sum().item())
            tie_count = int(metric_tie.sum().item())
            totals["reported_mismatched_chunks"] += reported_count
            totals["metric_min_mismatched_chunks"] += metric_min_count
            totals["decision_disagreements"] += decision_count
            totals["metric_tie_chunks"] += tie_count

            print(
                f"bit_width={bit_width} "
                f"reported_mismatched_chunks={reported_count}/{num_chunks} "
                f"metric_min_mismatches={metric_min_count}/{num_chunks} "
                f"decision_disagreements={decision_count}/{num_chunks} "
                f"metric_ties={tie_count}/{num_chunks}"
            )

            bad_indices = torch.nonzero(report_bad, as_tuple=False).flatten().tolist()
            for chunk_idx in bad_indices:
                pseudo_decision = "exp2" if bool(choose_exp2[chunk_idx]) else "exp1"
                metric_decision = _metric_decision_label(
                    metric_tie[chunk_idx],
                    metric_choose_exp2[chunk_idx],
                    compare_tie_policy,
                )
                mismatch_kind = _mismatch_kind(
                    metric_min_bad[chunk_idx],
                    metric_tie[chunk_idx],
                )
                metric_delta = pseudo_selected_metric_error[chunk_idx] - metric_min_error[chunk_idx]

                if printed < max_mismatches:
                    print(
                        "  mismatch "
                        f"bit_width={bit_width} "
                        f"chunk={chunk_idx} "
                        f"kind={mismatch_kind} "
                        f"pseudo={pseudo_decision} "
                        f"metric={metric_decision} "
                        f"metric_exp1={fmt_float(metric_exp1_error[chunk_idx])} "
                        f"metric_exp2={fmt_float(metric_exp2_error[chunk_idx])} "
                        f"delta={fmt_float(metric_delta)} "
                        f"expected_e1_wins={int(expected_e1_wins[chunk_idx])} "
                        f"expected_e2_wins={int(expected_e2_wins[chunk_idx])} "
                        f"expected_e2_wins_shifted={int(expected_e2_wins_shifted[chunk_idx])}"
                    )
                    printed += 1

                scale = scales[chunk_idx, 0]
                chunk_fields = {
                    "value_mode": value_mode,
                    "bit_width": bit_width,
                    "sgn": 1,
                    "exp1_format": exp1_format,
                    "exp2_format": exp2_format,
                    "m1": m1,
                    "m2": m2,
                    "e2_win_divisor": e2_win_divisor,
                    "chunk_idx": chunk_idx,
                    "expected_decision": pseudo_decision,
                    "expected_e1_wins": int(expected_e1_wins[chunk_idx]),
                    "expected_e2_wins": int(expected_e2_wins[chunk_idx]),
                    "expected_e2_wins_shifted": int(expected_e2_wins_shifted[chunk_idx]),
                    "expected_e2_wins_shift2": int(expected_e2_wins_shift2[chunk_idx]),
                    "err_exp1_dec": fmt_float(err1[chunk_idx]),
                    "err_exp2_dec": fmt_float(err2[chunk_idx]),
                    "chunk_diff_exp2_minus_exp1_dec": fmt_float(chunk_diff[chunk_idx]),
                    "expected_error_fp32_hex": fp32_hex(expected_error[chunk_idx]),
                    "expected_error_fp32_dec": fmt_float(expected_error[chunk_idx]),
                    "chunk_scale_fp32_hex": fp32_hex(scale),
                    "chunk_scale_fp32_dec": fmt_float(scale),
                }
                compare_fields = {
                    "compare_metric": compare_metric,
                    "compare_metric_param": fmt_float(compare_metric_param),
                    "compare_atol": fmt_float(compare_atol),
                    "compare_tie_policy": compare_tie_policy,
                    "mismatch_kind": mismatch_kind,
                    "metric_decision": metric_decision,
                    "metric_tie": int(bool(metric_tie[chunk_idx])),
                    "metric_min_mismatch": int(bool(metric_min_bad[chunk_idx])),
                    "decision_mismatch": int(bool(decision_bad[chunk_idx])),
                    "metric_exp1_error_dec": fmt_float(metric_exp1_error[chunk_idx]),
                    "metric_exp2_error_dec": fmt_float(metric_exp2_error[chunk_idx]),
                    "metric_min_error_dec": fmt_float(metric_min_error[chunk_idx]),
                    "pseudo_selected_metric_error_dec": fmt_float(pseudo_selected_metric_error[chunk_idx]),
                    "metric_error_delta_dec": fmt_float(metric_delta),
                }
                for value_idx in range(CHUNK_SIZE):
                    writer.writerow({
                        **compare_fields,
                        **chunk_fields,
                        **_value_metadata_row(
                            value_mode,
                            bit_width,
                            m1,
                            m2,
                            raw_chunks,
                            scaled_chunks,
                            q1_bits,
                            q2_bits,
                            err_exp1_pre_square,
                            err_exp2_pre_square,
                            pseudo_diff,
                            chunk_idx,
                            value_idx,
                        ),
                    })
                    totals["rows_written"] += 1

    print(
        "TOTAL "
        f"reported_mismatched_chunks={totals['reported_mismatched_chunks']} "
        f"metric_min_mismatches={totals['metric_min_mismatched_chunks']} "
        f"decision_disagreements={totals['decision_disagreements']} "
        f"metric_ties={totals['metric_tie_chunks']} "
        f"rows_written={totals['rows_written']}"
    )
    return totals


def _format_candidate(bit_width, index):
    m1 = bit_width - 2
    m2 = bit_width - 3
    return f"fp{bit_width}_e1m{m1}" if int(index) == 0 else f"fp{bit_width}_e2m{m2}"


def _first_abs_mismatch(a, b):
    diff = (a - b).abs()
    flat_idx = int(torch.argmax(diff).item())
    return flat_idx, float(diff.reshape(-1)[flat_idx].item())


def verify_cuda_vectors(
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    max_mismatches=20,
    e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA verifier requires torch.cuda.is_available()")

    from runspace.src.quantization.cuda import search_best_chunk_format

    num_chunks = int(num_chunks)
    seed = int(seed)
    max_mismatches = int(max_mismatches)
    e2_win_divisor = pseudo_mse_e2_win_divisor_from_param(e2_win_divisor)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)

    raw_cuda = raw_chunks.reshape(-1).contiguous().cuda()
    total_mismatches = 0

    print("CUDA pseudo_MSE verification")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"seed={seed} num_chunks={num_chunks} chunk_size={CHUNK_SIZE}")
    print(f"e2_win_divisor={e2_win_divisor}")
    print("mantissa_mode=round-to-nearest")

    for bit_width in BIT_WIDTHS:
        m1 = bit_width - 2
        m2 = bit_width - 3
        (
            _err1,
            _err2,
            chunk_diff,
            expected_e1_wins,
            expected_e2_wins,
            expected_e2_wins_shifted,
            choose_exp2,
            _expected_error,
            _q1_bits,
            _q2_bits,
            _err_exp1_pre_square,
            _err_exp2_pre_square,
            _pseudo_diff,
        ) = decision_for_bit_width(
            scaled_chunks,
            bit_width,
            e2_win_divisor=e2_win_divisor,
        )

        q1_scaled = pseudo_mse_reconstruct_scaled_export(
            scaled_chunks,
            exp_bits=1,
            mantissa_bits=m1,
            is_signed=True,
        )
        q2_scaled = pseudo_mse_reconstruct_scaled_export(
            scaled_chunks,
            exp_bits=2,
            mantissa_bits=m2,
            is_signed=True,
        )
        expected_indices = torch.where(
            choose_exp2,
            torch.ones_like(choose_exp2, dtype=torch.long),
            torch.zeros_like(choose_exp2, dtype=torch.long),
        )
        expected_unscaled = torch.where(choose_exp2.unsqueeze(1), q2_scaled, q1_scaled)
        expected_q = expected_unscaled * scales

        cands_e = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        cands_m = torch.tensor([m1, m2], dtype=torch.int32, device="cuda")
        cands_sgn = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
        best_indices, best_scales, best_q_flat, best_unscaled_flat = search_best_chunk_format(
            raw_cuda,
            cands_e,
            cands_m,
            cands_sgn,
            True,
            7,
            float(e2_win_divisor),
        )
        torch.cuda.synchronize()

        cuda_indices = best_indices.cpu()
        cuda_scales = best_scales.cpu()
        cuda_q = best_q_flat.cpu().reshape(num_chunks, CHUNK_SIZE)
        cuda_unscaled = best_unscaled_flat.cpu().reshape(num_chunks, CHUNK_SIZE)

        index_bad = cuda_indices != expected_indices
        scale_bad = cuda_scales.view(-1, 1) != scales
        q_bad = cuda_q != expected_q
        unscaled_bad = cuda_unscaled != expected_unscaled
        bad_chunks = (
            index_bad
            | scale_bad.view(-1)
            | q_bad.reshape(num_chunks, CHUNK_SIZE).any(dim=1)
            | unscaled_bad.reshape(num_chunks, CHUNK_SIZE).any(dim=1)
        )
        bad_count = int(bad_chunks.sum().item())
        total_mismatches += bad_count

        max_q_err = float((cuda_q - expected_q).abs().max().item())
        max_unscaled_err = float((cuda_unscaled - expected_unscaled).abs().max().item())
        max_scale_err = float((cuda_scales.view(-1, 1) - scales).abs().max().item())
        print(
            f"bit_width={bit_width} "
            f"mismatched_chunks={bad_count}/{num_chunks} "
            f"max_q_err={max_q_err:.9g} "
            f"max_unscaled_err={max_unscaled_err:.9g} "
            f"max_scale_err={max_scale_err:.9g}"
        )

        if bad_count == 0:
            continue

        bad_indices = torch.nonzero(bad_chunks, as_tuple=False).flatten().tolist()
        remaining_budget = max(0, max_mismatches - (total_mismatches - bad_count))
        for chunk_idx in bad_indices[:remaining_budget]:
            ref_idx = int(expected_indices[chunk_idx].item())
            cuda_idx = int(cuda_indices[chunk_idx].item())
            print(
                "  mismatch "
                f"chunk={chunk_idx} "
                f"ref={ref_idx}:{_format_candidate(bit_width, ref_idx)} "
                f"cuda={cuda_idx}:{_format_candidate(bit_width, cuda_idx)} "
                f"chunk_diff={fmt_float(chunk_diff[chunk_idx])} "
                f"expected_e1_wins={int(expected_e1_wins[chunk_idx])} "
                f"expected_e2_wins={int(expected_e2_wins[chunk_idx])} "
                f"expected_e2_wins_shifted={int(expected_e2_wins_shifted[chunk_idx])}"
            )
            if bool(scale_bad.view(-1)[chunk_idx]):
                print(
                    "    scale "
                    f"ref={fmt_float(scales[chunk_idx, 0])} "
                    f"cuda={fmt_float(cuda_scales[chunk_idx])}"
                )
            if bool(q_bad[chunk_idx].any()):
                value_idx, err = _first_abs_mismatch(cuda_q[chunk_idx], expected_q[chunk_idx])
                print(
                    "    q "
                    f"value={value_idx} "
                    f"ref={fmt_float(expected_q[chunk_idx, value_idx])} "
                    f"cuda={fmt_float(cuda_q[chunk_idx, value_idx])} "
                    f"abs_err={err:.9g}"
                )
            if bool(unscaled_bad[chunk_idx].any()):
                value_idx, err = _first_abs_mismatch(
                    cuda_unscaled[chunk_idx],
                    expected_unscaled[chunk_idx],
                )
                print(
                    "    unscaled_q "
                    f"value={value_idx} "
                    f"ref={fmt_float(expected_unscaled[chunk_idx, value_idx])} "
                    f"cuda={fmt_float(cuda_unscaled[chunk_idx, value_idx])} "
                    f"abs_err={err:.9g}"
                )

    print(f"TOTAL mismatched_chunks={total_mismatches}")
    return total_mismatches


def verify_python_vectors(
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    max_mismatches=20,
    e2_win_divisor=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
):
    from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

    num_chunks = int(num_chunks)
    seed = int(seed)
    max_mismatches = int(max_mismatches)
    e2_win_divisor = pseudo_mse_e2_win_divisor_from_param(e2_win_divisor)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    quantizer = object.__new__(DynamicInputQuantizer)
    quantizer.metric_param = float(e2_win_divisor)
    total_mismatches = 0

    print("Python pseudo_MSE verification")
    print(f"seed={seed} num_chunks={num_chunks} chunk_size={CHUNK_SIZE}")
    print(f"e2_win_divisor={e2_win_divisor}")
    print("mantissa_mode=round-to-nearest")

    for bit_width in BIT_WIDTHS:
        m1 = bit_width - 2
        m2 = bit_width - 3
        candidates = [f"fp{bit_width}_e1m{m1}", f"fp{bit_width}_e2m{m2}"]
        (
            _err1,
            _err2,
            chunk_diff,
            expected_e1_wins,
            expected_e2_wins,
            expected_e2_wins_shifted,
            choose_exp2,
            _expected_error,
            _q1_bits,
            _q2_bits,
            _err_exp1_pre_square,
            _err_exp2_pre_square,
            _pseudo_diff,
        ) = decision_for_bit_width(
            scaled_chunks,
            bit_width,
            e2_win_divisor=e2_win_divisor,
        )

        q1_scaled = pseudo_mse_reconstruct_scaled_export(
            scaled_chunks,
            exp_bits=1,
            mantissa_bits=m1,
            is_signed=True,
        )
        q2_scaled = pseudo_mse_reconstruct_scaled_export(
            scaled_chunks,
            exp_bits=2,
            mantissa_bits=m2,
            is_signed=True,
        )
        expected_indices = torch.where(
            choose_exp2,
            torch.ones_like(choose_exp2, dtype=torch.long),
            torch.zeros_like(choose_exp2, dtype=torch.long),
        )
        expected_unscaled = torch.where(choose_exp2.unsqueeze(1), q2_scaled, q1_scaled)
        expected_q = expected_unscaled * scales

        py_indices, py_scales, py_q, py_unscaled = quantizer._select_best_format_pseudo_mse_python(
            raw_chunks.contiguous(),
            candidates,
            CHUNK_SIZE,
            True,
        )

        index_bad = py_indices != expected_indices
        scale_bad = py_scales.view(-1, 1) != scales
        q_bad = py_q != expected_q
        unscaled_bad = py_unscaled != expected_unscaled
        bad_chunks = (
            index_bad
            | scale_bad.view(-1)
            | q_bad.reshape(num_chunks, CHUNK_SIZE).any(dim=1)
            | unscaled_bad.reshape(num_chunks, CHUNK_SIZE).any(dim=1)
        )
        bad_count = int(bad_chunks.sum().item())
        total_mismatches += bad_count

        max_q_err = float((py_q - expected_q).abs().max().item())
        max_unscaled_err = float((py_unscaled - expected_unscaled).abs().max().item())
        max_scale_err = float((py_scales.view(-1, 1) - scales).abs().max().item())
        print(
            f"bit_width={bit_width} "
            f"mismatched_chunks={bad_count}/{num_chunks} "
            f"max_q_err={max_q_err:.9g} "
            f"max_unscaled_err={max_unscaled_err:.9g} "
            f"max_scale_err={max_scale_err:.9g}"
        )

        if bad_count == 0:
            continue

        bad_indices = torch.nonzero(bad_chunks, as_tuple=False).flatten().tolist()
        remaining_budget = max(0, max_mismatches - (total_mismatches - bad_count))
        for chunk_idx in bad_indices[:remaining_budget]:
            ref_idx = int(expected_indices[chunk_idx].item())
            py_idx = int(py_indices[chunk_idx].item())
            print(
                "  mismatch "
                f"chunk={chunk_idx} "
                f"ref={ref_idx}:{_format_candidate(bit_width, ref_idx)} "
                f"python={py_idx}:{_format_candidate(bit_width, py_idx)} "
                f"chunk_diff={fmt_float(chunk_diff[chunk_idx])} "
                f"expected_e1_wins={int(expected_e1_wins[chunk_idx])} "
                f"expected_e2_wins={int(expected_e2_wins[chunk_idx])} "
                f"expected_e2_wins_shifted={int(expected_e2_wins_shifted[chunk_idx])}"
            )
            if bool(scale_bad.view(-1)[chunk_idx]):
                print(
                    "    scale "
                    f"ref={fmt_float(scales[chunk_idx, 0])} "
                    f"python={fmt_float(py_scales[chunk_idx])}"
                )
            if bool(q_bad[chunk_idx].any()):
                value_idx, err = _first_abs_mismatch(py_q[chunk_idx], expected_q[chunk_idx])
                print(
                    "    q "
                    f"value={value_idx} "
                    f"ref={fmt_float(expected_q[chunk_idx, value_idx])} "
                    f"python={fmt_float(py_q[chunk_idx, value_idx])} "
                    f"abs_err={err:.9g}"
                )
            if bool(unscaled_bad[chunk_idx].any()):
                value_idx, err = _first_abs_mismatch(
                    py_unscaled[chunk_idx],
                    expected_unscaled[chunk_idx],
                )
                print(
                    "    unscaled_q "
                    f"value={value_idx} "
                    f"ref={fmt_float(expected_unscaled[chunk_idx, value_idx])} "
                    f"python={fmt_float(py_unscaled[chunk_idx, value_idx])} "
                    f"abs_err={err:.9g}"
                )

    print(f"TOTAL mismatched_chunks={total_mismatches}")
    return total_mismatches


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo_MSE hardware vectors.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "pseudo_mse_hw_vectors.txt"),
    )
    parser.add_argument(
        "--value-mode",
        choices=("scaled", "raw-and-scaled"),
        default="scaled",
        help="Output only scaled values, or both pre-scale raw values and post-scale values.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path for a row-per-value CSV export.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=NUM_CHUNKS,
        help="Number of chunks to emit per bit width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed for deterministic random extra chunks.",
    )
    parser.add_argument(
        "--e2-win-divisor",
        type=int,
        choices=(2, 4),
        default=PSEUDO_MSE_DEFAULT_E2_WIN_DIVISOR,
        help="Divisor for exp=2 wins before decision: 4 is default, 2 matches L1 selection.",
    )
    parser.add_argument(
        "--compare-metric",
        default=None,
        help="Compare pseudo_MSE decisions against a scalar metric such as l1, l2/mse, linf, bias, l0, huber, or logsum.",
    )
    parser.add_argument(
        "--compare-metric-param",
        type=float,
        default=0.0625,
        help="Metric parameter for --compare-metric, currently used by huber as delta.",
    )
    parser.add_argument(
        "--compare-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for metric-error ties/mismatch checks. Default 0 means exact FP32 comparison.",
    )
    parser.add_argument(
        "--compare-csv-output",
        default=None,
        help="CSV path for per-value mismatch metadata when --compare-metric is set.",
    )
    parser.add_argument(
        "--compare-tie-policy",
        choices=("min-error", "exp1", "exp2"),
        default="min-error",
        help=(
            "How to treat exact metric ties. min-error treats either pseudo choice as a match; "
            "exp1/exp2 force a decision tie-break and report tie-decision disagreements."
        ),
    )
    parser.add_argument(
        "--compare-fail-on-decision",
        action="store_true",
        help="Exit nonzero on reported decision disagreements, including tie-policy disagreements.",
    )
    parser.add_argument(
        "--verify-cuda",
        action="store_true",
        help="Verify CUDA pseudo_MSE search against the same Python reference chunks.",
    )
    parser.add_argument(
        "--verify-python",
        action="store_true",
        help="Verify DynamicInputQuantizer's Python pseudo_MSE path against the same reference chunks.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run verification/comparison without writing regular TXT/CSV vector outputs.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=20,
        help="Maximum detailed CUDA mismatch examples to print.",
    )
    parser.add_argument(
        "--cuda-build-dir",
        default=None,
        help="Optional fresh CUDA extension build directory for verification runs.",
    )
    args = parser.parse_args()
    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")
    if args.cuda_build_dir:
        os.environ["QBENCH_CUDA_BUILD_DIR"] = args.cuda_build_dir
    if args.verify_only and not (args.verify_cuda or args.verify_python or args.compare_metric):
        raise ValueError("--verify-only requires --verify-cuda, --verify-python, or --compare-metric")
    if not args.verify_only:
        write_vectors(
            args.output,
            value_mode=args.value_mode,
            num_chunks=args.num_chunks,
            seed=args.seed,
            e2_win_divisor=args.e2_win_divisor,
        )
        print(args.output)
        if args.csv_output is not None:
            write_csv_vectors(
                args.csv_output,
                value_mode=args.value_mode,
                num_chunks=args.num_chunks,
                seed=args.seed,
                e2_win_divisor=args.e2_win_divisor,
            )
            print(args.csv_output)
    verify_mismatches = 0
    comparison_totals = None
    if args.compare_metric:
        compare_csv_output = args.compare_csv_output
        if compare_csv_output is None:
            compare_csv_output = os.path.join(
                os.path.dirname(__file__),
                "pseudo_mse_compare_mismatches.csv",
            )
        comparison_totals = compare_pseudo_mse_with_metric(
            compare_csv_output,
            compare_metric=args.compare_metric,
            compare_metric_param=args.compare_metric_param,
            compare_atol=args.compare_atol,
            compare_tie_policy=args.compare_tie_policy,
            num_chunks=args.num_chunks,
            seed=args.seed,
            e2_win_divisor=args.e2_win_divisor,
            max_mismatches=args.max_mismatches,
        )
    if args.verify_python:
        verify_mismatches += verify_python_vectors(
            num_chunks=args.num_chunks,
            seed=args.seed,
            max_mismatches=args.max_mismatches,
            e2_win_divisor=args.e2_win_divisor,
        )
    if args.verify_cuda:
        verify_mismatches += verify_cuda_vectors(
            num_chunks=args.num_chunks,
            seed=args.seed,
            max_mismatches=args.max_mismatches,
            e2_win_divisor=args.e2_win_divisor,
        )
    if verify_mismatches:
        raise SystemExit(1)
    if comparison_totals is not None:
        if comparison_totals["metric_min_mismatched_chunks"]:
            raise SystemExit(1)
        if args.compare_fail_on_decision and comparison_totals["reported_mismatched_chunks"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
