import argparse
import os
import struct
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.quantization.dynamic_input_metrics import (
    pseudo_mse_reconstruct_scaled_python,
    pseudo_mse_sqerr_diff_from_scaled,
)
from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer


BIT_WIDTHS = [8, 7, 6, 5, 4]
CHUNK_SIZE = 128
NUM_CHUNKS = 10
SEED = 20260625
ACTUAL_EXPONENTS = list(range(0, -11, -1))


def fp32_bits(value):
    return struct.unpack("!I", struct.pack("!f", float(value)))[0]


def fp32_hex(value):
    return f"0x{fp32_bits(value):08x}"


def fmt_float(value):
    return f"{float(value):.9g}"


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


def make_raw_chunks():
    gen = torch.Generator().manual_seed(SEED)

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

    raw_chunks = torch.stack(chunks).contiguous()
    if raw_chunks.shape != (NUM_CHUNKS, CHUNK_SIZE):
        raise RuntimeError(f"Expected chunks shape {(NUM_CHUNKS, CHUNK_SIZE)}, got {tuple(raw_chunks.shape)}")
    if not torch.isfinite(raw_chunks).all():
        raise RuntimeError("Generated non-finite raw HW vector value")
    return raw_chunks


def scale_raw_chunks(raw_chunks):
    scales = DynamicInputQuantizer._chunk_scale(raw_chunks).to(torch.float32).contiguous()
    scaled_chunks = raw_chunks / scales
    if not torch.isfinite(scales).all():
        raise RuntimeError("Generated non-finite source-code chunk scale")
    if not torch.isfinite(scaled_chunks).all():
        raise RuntimeError("Generated non-finite scaled HW vector value")
    if scaled_chunks.abs().max() >= 2.0:
        raise RuntimeError("Scaled HW vector values must stay in [-2, 2)")
    return scales, scaled_chunks.contiguous()


def decision_for_bit_width(scaled_chunks, bit_width):
    m1 = bit_width - 2
    m2 = bit_width - 3
    if m2 != m1 - 1 or m2 < 1:
        raise ValueError(f"Unsupported bit width for pseudo_MSE e1/e2 pair: {bit_width}")

    q1 = pseudo_mse_reconstruct_scaled_python(
        scaled_chunks,
        exp_bits=1,
        mantissa_bits=m1,
        is_signed=True,
    )
    q2 = pseudo_mse_reconstruct_scaled_python(
        scaled_chunks,
        exp_bits=2,
        mantissa_bits=m2,
        is_signed=True,
    )
    diff = pseudo_mse_sqerr_diff_from_scaled(
        scaled_chunks,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m2,
        is_signed=True,
    )
    err1 = (scaled_chunks - q1).pow(2).sum(dim=1)
    err2 = (scaled_chunks - q2).pow(2).sum(dim=1)
    chunk_diff = diff.sum(dim=1)
    choose_exp2 = chunk_diff < 0
    return err1, err2, chunk_diff, choose_exp2


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
            "scaled_fp32_hex scaled_fp32_dec\n"
        )
        f.write("# scaled_fp32_dec = raw_fp32_dec / scale_fp32_dec\n\n")
    else:
        f.write("# chunk max rows use max absolute value; *_value is the signed value at that index\n")
        f.write(
            "# chunk max rows: chunk_max_abs_index value_index, "
            "chunk_max_abs_scaled_value scaled_fp32_hex scaled_fp32_dec, "
            "chunk_max_abs_scaled_abs scaled_abs_fp32_hex scaled_abs_fp32_dec\n"
        )
        f.write("# value rows: value_index scaled_fp32_hex scaled_fp32_dec\n\n")


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


def write_value_rows(f, raw_chunks, scaled_chunks, scales, chunk_idx, value_mode):
    scale = scales[chunk_idx, 0]
    if value_mode == "raw-and-scaled":
        f.write(f"chunk_scale {fp32_hex(scale)} {fmt_float(scale)}\n")

    f.write("VALUES_BEGIN\n")
    for value_idx in range(CHUNK_SIZE):
        raw_v = raw_chunks[chunk_idx, value_idx]
        scaled_v = scaled_chunks[chunk_idx, value_idx]
        if value_mode == "raw-and-scaled":
            f.write(
                f"{value_idx:03d} "
                f"{fp32_hex(raw_v)} {fmt_float(raw_v)} "
                f"{fp32_hex(scaled_v)} {fmt_float(scaled_v)}\n"
            )
        else:
            f.write(
                f"{value_idx:03d} "
                f"{fp32_hex(scaled_v)} {fmt_float(scaled_v)}\n"
            )
    f.write("VALUES_END\n")


def write_vectors(path, value_mode="scaled"):
    raw_chunks = make_raw_chunks()
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        f.write("# pseudo_MSE hardware test vectors\n")
        f.write(f"# seed={SEED}\n")
        f.write(f"# bit_widths={','.join(str(b) for b in BIT_WIDTHS)}\n")
        f.write(f"# num_chunks_per_bit_width={NUM_CHUNKS}\n")
        f.write(f"# chunk_size={CHUNK_SIZE}\n")
        f.write("# signedness: sgn=1 for every section in this file\n")
        f.write("# chunk scale source: DynamicInputQuantizer._chunk_scale(raw_chunk)\n")
        f.write("# chunk scale definition: pow2_floor(max(abs(raw_chunk))) with 1.0 for all-zero chunks\n")
        f.write("# metric input: values are already scaled FP32 numbers (scaled_v)\n")
        if value_mode == "raw-and-scaled":
            f.write("# raw_fp32 values are provided only to test the pre-scale path\n")
            f.write("# feed scaled_fp32 values to the pseudo_MSE block, or compute scaled_fp32 = raw_fp32 / chunk_scale\n")
        else:
            f.write("# do not apply a chunk scale before feeding these values to the pseudo_MSE block\n")
        f.write("# decision rule: choose_exp2 if sum((scaled-q_exp2)^2 - (scaled-q_exp1)^2) < 0 else choose_exp1\n")
        write_value_header(f, value_mode)

        for bit_width in BIT_WIDTHS:
            m1 = bit_width - 2
            m2 = bit_width - 3
            exp1_format = f"fp{bit_width}_e1m{m1}"
            exp2_format = f"fp{bit_width}_e2m{m2}"
            err1, err2, chunk_diff, choose_exp2 = decision_for_bit_width(
                scaled_chunks,
                bit_width,
            )

            f.write(f"BEGIN_BIT_WIDTH {bit_width}\n")
            f.write(f"sgn 1\n")
            f.write(f"exp1_format {exp1_format}\n")
            f.write(f"exp2_format {exp2_format}\n")
            f.write(f"m1 {m1}\n")
            f.write(f"m2 {m2}\n")
            for chunk_idx in range(NUM_CHUNKS):
                decision = "exp2" if bool(choose_exp2[chunk_idx]) else "exp1"
                f.write(f"BEGIN_CHUNK {chunk_idx}\n")
                f.write(f"err_exp1 {fmt_float(err1[chunk_idx])}\n")
                f.write(f"err_exp2 {fmt_float(err2[chunk_idx])}\n")
                f.write(f"chunk_diff_exp2_minus_exp1 {fmt_float(chunk_diff[chunk_idx])}\n")
                f.write(f"expected_decision {decision}\n")
                write_chunk_max_rows(f, raw_chunks, scaled_chunks, scales, chunk_idx, value_mode)
                write_value_rows(f, raw_chunks, scaled_chunks, scales, chunk_idx, value_mode)
                f.write(f"END_CHUNK {chunk_idx}\n")
            f.write(f"END_BIT_WIDTH {bit_width}\n\n")


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
    args = parser.parse_args()
    write_vectors(args.output, value_mode=args.value_mode)
    print(args.output)


if __name__ == "__main__":
    main()
