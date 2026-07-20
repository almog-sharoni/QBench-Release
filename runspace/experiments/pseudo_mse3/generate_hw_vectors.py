import argparse
import csv
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.pseudo_mse2.generate_hw_vectors import (  # noqa: E402
    BIT_WIDTHS,
    CHUNK_SIZE,
    NUM_CHUNKS,
    SEED,
    fmt_bits,
    fmt_float,
    fp32_hex,
    make_raw_chunks,
    packed_exp_field,
    packed_mant_bits,
    packed_mant_field,
    pseudo_mse_encode_emb_export,
    pseudo_mse_reconstruct_scaled_export,
    scale_raw_chunks,
    write_chunk_max_rows,
)
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    is_pseudo_mse_family_metric,
    normalize_dynamic_input_metric,
    normalize_pseudo_mse3_fixed_rounding,
    normalize_pseudo_mse3_tie_break,
    pseudo_mse_encode_emb_trunc_python,
    pseudo_mse3_choose_exp2_from_diff,
    pseudo_mse3_err2_minus_err1_from_scaled,
    pseudo_mse3_fixed_rounding_code,
    pseudo_mse3_tie_break_code,
    pseudo_mse_reconstruct_scaled_trunc_python,
    reduce_dynamic_input_metric_python,
)


def _candidate_formats(bit_width):
    m1 = int(bit_width) - 2
    m2 = int(bit_width) - 3
    return f"fp{int(bit_width)}_e1m{m1}", f"fp{int(bit_width)}_e2m{m2}"


def _normalize_bits_to_take(bits_to_take):
    value = int(bits_to_take)
    if float(bits_to_take) != float(value) or value < 0:
        raise ValueError(
            f"bits_to_take must be a non-negative integer; got {bits_to_take!r}"
        )
    return value


def decision_for_bit_width(
    scaled_chunks,
    bit_width,
    bits_to_take=0,
    fixed_rounding="floor",
    tie_break="exp1",
):
    bits_to_take = _normalize_bits_to_take(bits_to_take)
    fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
    tie_break = normalize_pseudo_mse3_tie_break(tie_break)
    m1 = int(bit_width) - 2
    m2 = int(bit_width) - 3
    if m2 != m1 - 1 or m2 < 1:
        raise ValueError(f"Unsupported bit width for pseudo_MSE3 e1/e2 pair: {bit_width}")

    q1_bits = pseudo_mse_encode_emb_trunc_python(
        scaled_chunks,
        exp_bits=1,
        mantissa_bits=m1,
        is_signed=True,
    )
    q2_bits = pseudo_mse_encode_emb_trunc_python(
        scaled_chunks,
        exp_bits=2,
        mantissa_bits=m2,
        is_signed=True,
    )
    q1_for_decision = pseudo_mse_reconstruct_scaled_trunc_python(
        scaled_chunks,
        exp_bits=1,
        mantissa_bits=m1,
        is_signed=True,
    )
    q2_for_decision = pseudo_mse_reconstruct_scaled_trunc_python(
        scaled_chunks,
        exp_bits=2,
        mantissa_bits=m2,
        is_signed=True,
    )
    err_exp1_pre_square = scaled_chunks - q1_for_decision
    err_exp2_pre_square = scaled_chunks - q2_for_decision
    err1 = err_exp1_pre_square.pow(2).sum(dim=1)
    err2 = err_exp2_pre_square.pow(2).sum(dim=1)
    pseudo_diff = pseudo_mse3_err2_minus_err1_from_scaled(
        scaled_chunks,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m2,
        is_signed=True,
        bits_to_take=bits_to_take,
        fixed_rounding=fixed_rounding,
    )
    chunk_diff = pseudo_diff.sum(dim=1, dtype=torch.int64)
    choose_exp2 = pseudo_mse3_choose_exp2_from_diff(
        pseudo_diff,
        tie_break=tie_break,
        fixed_rounding=fixed_rounding,
    )
    expected_error = torch.where(choose_exp2, err2, err1)
    return (
        err1,
        err2,
        chunk_diff,
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
            "# value rows: value_index raw_fp32_hex raw_fp32_dec "
            "scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec "
            "pseudo_diff_times_2_2m_dec\n"
        )
        f.write("# scaled_fp32_dec = raw_fp32_dec / scale_fp32_dec\n\n")
    else:
        f.write("# value mode: scaled\n")
        f.write("# chunk max rows use max absolute value; *_value is the signed value at that index\n")
        f.write(
            "# value rows: value_index scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec "
            "pseudo_diff_times_2_2m_dec\n\n"
        )


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
    m1 = int(bit_width) - 2
    m2 = int(bit_width) - 3
    if value_mode == "raw-and-scaled":
        f.write(f"chunk_scale {fp32_hex(scale)} {fmt_float(scale)}\n")
        f.write(
            "# value_index raw_fp32_hex raw_fp32_dec scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec "
            "pseudo_diff_times_2_2m_dec\n"
        )
    else:
        f.write(
            "# value_index scaled_fp32_hex scaled_fp32_dec "
            "q_exp1_bits q_exp1_exp_field q_exp1_mant_field q_exp1_mant_bits "
            "q_exp2_bits q_exp2_exp_field q_exp2_mant_field q_exp2_mant_bits "
            "err_exp1_pre_square_fp32_hex err_exp1_pre_square_fp32_dec "
            "err_exp2_pre_square_fp32_hex err_exp2_pre_square_fp32_dec "
            "pseudo_diff_exp2_minus_exp1_fp32_hex pseudo_diff_exp2_minus_exp1_fp32_dec "
            "pseudo_diff_times_2_2m_dec\n"
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
        scaled_pseudo_diff_v = pseudo_diff_v * float(2.0 ** (2 * m1))
        if value_mode == "raw-and-scaled":
            f.write(
                f"{value_idx:03d} "
                f"{fp32_hex(raw_v)} {fmt_float(raw_v)} "
                f"{fp32_hex(scaled_v)} {fmt_float(scaled_v)} "
                f"{fmt_bits(q1_bits_v, bit_width)} {q1_exp_v} {q1_mant_v} {q1_mant_bits_v} "
                f"{fmt_bits(q2_bits_v, bit_width)} {q2_exp_v} {q2_mant_v} {q2_mant_bits_v} "
                f"{fp32_hex(err_exp1_pre_square_v)} {fmt_float(err_exp1_pre_square_v)} "
                f"{fp32_hex(err_exp2_pre_square_v)} {fmt_float(err_exp2_pre_square_v)} "
                f"{fp32_hex(pseudo_diff_v)} {fmt_float(pseudo_diff_v)} "
                f"{fmt_float(scaled_pseudo_diff_v)}\n"
            )
        else:
            f.write(
                f"{value_idx:03d} "
                f"{fp32_hex(scaled_v)} {fmt_float(scaled_v)} "
                f"{fmt_bits(q1_bits_v, bit_width)} {q1_exp_v} {q1_mant_v} {q1_mant_bits_v} "
                f"{fmt_bits(q2_bits_v, bit_width)} {q2_exp_v} {q2_mant_v} {q2_mant_bits_v} "
                f"{fp32_hex(err_exp1_pre_square_v)} {fmt_float(err_exp1_pre_square_v)} "
                f"{fp32_hex(err_exp2_pre_square_v)} {fmt_float(err_exp2_pre_square_v)} "
                f"{fp32_hex(pseudo_diff_v)} {fmt_float(pseudo_diff_v)} "
                f"{fmt_float(scaled_pseudo_diff_v)}\n"
            )
    f.write("VALUES_END\n")


def write_vectors(
    path,
    value_mode="scaled",
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    bits_to_take=0,
    fixed_rounding="floor",
    tie_break="exp1",
):
    num_chunks = int(num_chunks)
    seed = int(seed)
    bits_to_take = _normalize_bits_to_take(bits_to_take)
    fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
    tie_break = normalize_pseudo_mse3_tie_break(tie_break)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        f.write("# pseudo_MSE3 PyTorch hardware test vectors\n")
        f.write(f"# seed={seed}\n")
        f.write(f"# bit_widths={','.join(str(b) for b in BIT_WIDTHS)}\n")
        f.write(f"# num_chunks_per_bit_width={num_chunks}\n")
        f.write(f"# chunk_size={CHUNK_SIZE}\n")
        f.write("# signedness: sgn=1 for every section in this file\n")
        f.write("# implementation: PyTorch reference; optional CUDA verification uses the same quantization path\n")
        f.write("# chunk scale source: DynamicInputQuantizer._chunk_scale(raw_chunk)\n")
        f.write("# chunk scale definition: pow2_floor(max(abs(raw_chunk))) with 1.0 for all-zero chunks\n")
        f.write("# metric input: values are already scaled FP32 numbers (scaled_v)\n")
        if value_mode == "raw-and-scaled":
            f.write("# raw_fp32 values are provided only to test the pre-scale path\n")
            f.write("# feed scaled_fp32 values to the pseudo_MSE3 block, or compute scaled_fp32 = raw_fp32 / chunk_scale\n")
        else:
            f.write("# do not apply a chunk scale before feeding these values to the pseudo_MSE3 block\n")
        comparison = "<=" if tie_break == "exp2" else "<"
        f.write(
            "# decision rule: choose_exp2 if sum(err2^2 - err1^2) < 0 "
            "else choose_exp1 (floating-point baseline)\n"
        )
        f.write(
            "# configured decision rule: choose_exp2 if summed fixed-point contribution "
            f"{comparison} 0 else choose_exp1\n"
        )
        f.write("# expected_error rows: selected chunk squared error, as fp32_hex fp32_dec\n")
        f.write("# output mantissa mode: round-to-nearest\n")
        f.write("# format-selection candidate mode: truncate\n")
        f.write("# selected dynamic-quantizer output mode: round-to-nearest\n")
        f.write(f"# fixed-point bits_to_take: {bits_to_take}\n")
        f.write(f"# fixed-point rounding: {fixed_rounding}\n")
        f.write(f"# exact-tie decision: {tie_break}\n")
        f.write("# q_exp*_bits are truncated candidate fields used by format selection\n")
        f.write("# err_exp*_pre_square = scaled-q_exp* for each truncated candidate\n")
        f.write("# pseudo_diff_exp2_minus_exp1 = err_exp2_pre_square^2 - err_exp1_pre_square^2\n")
        f.write("# pseudo_diff_times_2_2m must be in the truncating-selection range [-3/4,3)\n")
        write_value_header(f, value_mode)

        for bit_width in BIT_WIDTHS:
            m1 = bit_width - 2
            m2 = bit_width - 3
            exp1_format, exp2_format = _candidate_formats(bit_width)
            (
                err1,
                err2,
                chunk_diff,
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
                bits_to_take=bits_to_take,
                fixed_rounding=fixed_rounding,
                tie_break=tie_break,
            )

            f.write(f"BEGIN_BIT_WIDTH {bit_width}\n")
            f.write("sgn 1\n")
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
        "bits_to_take",
        "fixed_rounding",
        "tie_break",
        "bit_width",
        "sgn",
        "exp1_format",
        "exp2_format",
        "m1",
        "m2",
        "chunk_idx",
        "value_idx",
        "expected_decision",
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
    fields.extend(
        [
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
            "pseudo_diff_times_2_2m_dec",
        ]
    )
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
        "pseudo_diff_times_2_2m_dec": fmt_float(pseudo_diff_v * float(2.0 ** (2 * m1))),
    }
    if value_mode == "raw-and-scaled":
        row.update(
            {
                "raw_fp32_hex": fp32_hex(raw_v),
                "raw_fp32_dec": fmt_float(raw_v),
            }
        )
    return row


def write_csv_vectors(
    path,
    value_mode="scaled",
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    bits_to_take=0,
    fixed_rounding="floor",
    tie_break="exp1",
):
    num_chunks = int(num_chunks)
    seed = int(seed)
    bits_to_take = _normalize_bits_to_take(bits_to_take)
    fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
    tie_break = normalize_pseudo_mse3_tie_break(tie_break)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    fieldnames = _csv_fieldnames(value_mode)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bit_width in BIT_WIDTHS:
            m1 = bit_width - 2
            m2 = bit_width - 3
            exp1_format, exp2_format = _candidate_formats(bit_width)
            (
                err1,
                err2,
                chunk_diff,
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
                bits_to_take=bits_to_take,
                fixed_rounding=fixed_rounding,
                tie_break=tie_break,
            )
            for chunk_idx in range(num_chunks):
                decision = "exp2" if bool(choose_exp2[chunk_idx]) else "exp1"
                for value_idx in range(CHUNK_SIZE):
                    row = {
                        "value_mode": value_mode,
                        "bits_to_take": bits_to_take,
                        "fixed_rounding": fixed_rounding,
                        "tie_break": tie_break,
                        "bit_width": bit_width,
                        "sgn": 1,
                        "exp1_format": exp1_format,
                        "exp2_format": exp2_format,
                        "m1": m1,
                        "m2": m2,
                        "chunk_idx": chunk_idx,
                        "value_idx": value_idx,
                        "expected_decision": decision,
                        "err_exp1_dec": fmt_float(err1[chunk_idx]),
                        "err_exp2_dec": fmt_float(err2[chunk_idx]),
                        "chunk_diff_exp2_minus_exp1_dec": fmt_float(chunk_diff[chunk_idx]),
                        "expected_error_fp32_hex": fp32_hex(expected_error[chunk_idx]),
                        "expected_error_fp32_dec": fmt_float(expected_error[chunk_idx]),
                        "chunk_scale_fp32_hex": fp32_hex(scales[chunk_idx, 0]),
                        "chunk_scale_fp32_dec": fmt_float(scales[chunk_idx, 0]),
                    }
                    row.update(
                        _value_metadata_row(
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
                        )
                    )
                    writer.writerow(row)


def _first_abs_mismatch(actual, expected):
    diff = (actual - expected).abs().flatten()
    idx = int(torch.argmax(diff).item())
    return idx, float(diff[idx].item())


def verify_python_vectors(
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    max_mismatches=20,
    bits_to_take=0,
    fixed_rounding="floor",
    tie_break="exp1",
):
    from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

    num_chunks = int(num_chunks)
    seed = int(seed)
    max_mismatches = int(max_mismatches)
    bits_to_take = _normalize_bits_to_take(bits_to_take)
    fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
    tie_break = normalize_pseudo_mse3_tie_break(tie_break)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    quantizer = object.__new__(DynamicInputQuantizer)
    quantizer.metric = "pseudo_mse3"
    quantizer.metric_param = 0.0
    quantizer.pseudo_mse2_mantissa_window_bits = 0
    quantizer.pseudo_mse3_bits_to_take = bits_to_take
    quantizer.pseudo_mse3_fixed_rounding = fixed_rounding
    quantizer.pseudo_mse3_tie_break = tie_break
    total_mismatches = 0

    print("Python pseudo_MSE3 verification")
    print(f"seed={seed} num_chunks={num_chunks} chunk_size={CHUNK_SIZE}")
    print("mantissa_mode=round-to-nearest")
    print(
        f"bits_to_take={bits_to_take} "
        f"fixed_rounding={fixed_rounding} tie_break={tie_break}"
    )

    for bit_width in BIT_WIDTHS:
        m1 = bit_width - 2
        m2 = bit_width - 3
        candidates = [f"fp{bit_width}_e1m{m1}", f"fp{bit_width}_e2m{m2}"]
        (
            _err1,
            _err2,
            chunk_diff,
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
            bits_to_take=bits_to_take,
            fixed_rounding=fixed_rounding,
            tie_break=tie_break,
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
                f"  mismatch bit_width={bit_width} chunk={chunk_idx} "
                f"expected_idx={ref_idx} python_idx={py_idx} "
                f"chunk_diff={fmt_float(chunk_diff[chunk_idx])}"
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


def verify_cuda_vectors(
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    max_mismatches=20,
    bits_to_take=0,
    fixed_rounding="floor",
    tie_break="exp1",
):
    from runspace.src.quantization.cuda import search_best_chunk_format
    from runspace.src.quantization.dynamic_input_metrics import dynamic_input_metric_code

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for --verify-cuda")

    num_chunks = int(num_chunks)
    seed = int(seed)
    max_mismatches = int(max_mismatches)
    bits_to_take = _normalize_bits_to_take(bits_to_take)
    fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
    tie_break = normalize_pseudo_mse3_tie_break(tie_break)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    raw_cuda = raw_chunks.cuda().contiguous()
    metric_code = dynamic_input_metric_code("pseudo_mse3")
    total_mismatches = 0

    print("CUDA pseudo_MSE3 verification")
    print(f"seed={seed} num_chunks={num_chunks} chunk_size={CHUNK_SIZE}")
    print(f"metric_code={metric_code}")
    print("mantissa_mode=round-to-nearest")
    print(
        f"bits_to_take={bits_to_take} "
        f"fixed_rounding={fixed_rounding} tie_break={tie_break}"
    )

    for bit_width in BIT_WIDTHS:
        m1 = bit_width - 2
        m2 = bit_width - 3
        (
            _err1,
            _err2,
            chunk_diff,
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
            bits_to_take=bits_to_take,
            fixed_rounding=fixed_rounding,
            tie_break=tie_break,
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
        cuda_indices, cuda_scales, cuda_q_flat, cuda_unscaled_flat = search_best_chunk_format(
            raw_cuda.reshape(-1).contiguous(),
            cands_e,
            cands_m,
            cands_sgn,
            True,
            metric_code,
            float(bits_to_take),
            0,
            pseudo_mse3_fixed_rounding_code(fixed_rounding),
            pseudo_mse3_tie_break_code(tie_break),
        )
        cuda_indices = cuda_indices.cpu()
        cuda_scales = cuda_scales.cpu()
        cuda_q = cuda_q_flat.view(num_chunks, CHUNK_SIZE).cpu()
        cuda_unscaled = cuda_unscaled_flat.view(num_chunks, CHUNK_SIZE).cpu()

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
                f"  mismatch bit_width={bit_width} chunk={chunk_idx} "
                f"expected_idx={ref_idx} cuda_idx={cuda_idx} "
                f"chunk_diff={fmt_float(chunk_diff[chunk_idx])}"
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


def compare_pseudo_mse3_with_metric(
    mismatch_csv,
    compare_metric="l2",
    compare_metric_param=0.0625,
    compare_atol=0.0,
    compare_tie_policy="min-error",
    num_chunks=NUM_CHUNKS,
    seed=SEED,
    max_mismatches=20,
    bits_to_take=0,
    fixed_rounding="floor",
    tie_break="exp1",
):
    normalized_metric = normalize_dynamic_input_metric(compare_metric)
    if is_pseudo_mse_family_metric(normalized_metric):
        raise ValueError("--compare-metric must be a scalar metric such as mse/l2 or l1")

    num_chunks = int(num_chunks)
    seed = int(seed)
    bits_to_take = _normalize_bits_to_take(bits_to_take)
    fixed_rounding = normalize_pseudo_mse3_fixed_rounding(fixed_rounding)
    tie_break = normalize_pseudo_mse3_tie_break(tie_break)
    raw_chunks = make_raw_chunks(num_chunks=num_chunks, seed=seed)
    _scales, scaled_chunks = scale_raw_chunks(raw_chunks)
    os.makedirs(os.path.dirname(os.path.abspath(mismatch_csv)), exist_ok=True)

    totals = {
        "reported_mismatched_chunks": 0,
        "metric_min_mismatched_chunks": 0,
        "decision_disagreements": 0,
        "metric_ties": 0,
        "rows_written": 0,
    }

    print("pseudo_MSE3 metric comparison")
    print(f"seed={seed} num_chunks={num_chunks} chunk_size={CHUNK_SIZE}")
    print(f"compare_metric={normalized_metric} compare_metric_param={compare_metric_param}")
    print(f"compare_atol={compare_atol}")
    print(f"compare_tie_policy={compare_tie_policy}")
    print(f"mismatch_csv={mismatch_csv}")
    print(
        f"bits_to_take={bits_to_take} "
        f"fixed_rounding={fixed_rounding} tie_break={tie_break}"
    )

    fields = [
        "bits_to_take",
        "fixed_rounding",
        "tie_break",
        "bit_width",
        "chunk_idx",
        "mismatch_kind",
        "pseudo_decision",
        "metric_decision",
        "metric_exp1",
        "metric_exp2",
        "metric_delta_exp2_minus_exp1",
        "chunk_diff_exp2_minus_exp1",
    ]
    with open(mismatch_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for bit_width in BIT_WIDTHS:
            (
                _err1,
                _err2,
                chunk_diff,
                choose_exp2,
                _expected_error,
                _q1_bits,
                _q2_bits,
                err_exp1_pre_square,
                err_exp2_pre_square,
                _pseudo_diff,
            ) = decision_for_bit_width(
                scaled_chunks,
                bit_width,
                bits_to_take=bits_to_take,
                fixed_rounding=fixed_rounding,
                tie_break=tie_break,
            )

            metric_exp1 = reduce_dynamic_input_metric_python(
                normalized_metric,
                err_exp1_pre_square,
                float(compare_metric_param),
            )
            metric_exp2 = reduce_dynamic_input_metric_python(
                normalized_metric,
                err_exp2_pre_square,
                float(compare_metric_param),
            )
            metric_delta = metric_exp2 - metric_exp1
            metric_tie = metric_delta.abs() <= float(compare_atol)
            metric_exp2_better = metric_delta < -float(compare_atol)
            metric_exp1_better = metric_delta > float(compare_atol)
            if compare_tie_policy == "exp2":
                metric_choose_exp2 = metric_exp2_better | metric_tie
            else:
                metric_choose_exp2 = metric_exp2_better

            if compare_tie_policy == "min-error":
                metric_min_mismatch = (
                    (choose_exp2 & metric_exp1_better)
                    | (~choose_exp2 & metric_exp2_better)
                )
                decision_disagreement = metric_min_mismatch
            else:
                metric_min_mismatch = (
                    (choose_exp2 & metric_exp1_better)
                    | (~choose_exp2 & metric_exp2_better)
                )
                decision_disagreement = choose_exp2 != metric_choose_exp2

            reported = metric_min_mismatch | decision_disagreement
            reported_count = int(reported.sum().item())
            metric_min_count = int(metric_min_mismatch.sum().item())
            decision_count = int(decision_disagreement.sum().item())
            tie_count = int(metric_tie.sum().item())
            totals["reported_mismatched_chunks"] += reported_count
            totals["metric_min_mismatched_chunks"] += metric_min_count
            totals["decision_disagreements"] += decision_count
            totals["metric_ties"] += tie_count

            print(
                f"bit_width={bit_width} "
                f"reported_mismatched_chunks={reported_count}/{num_chunks} "
                f"metric_min_mismatches={metric_min_count}/{num_chunks} "
                f"decision_disagreements={decision_count}/{num_chunks} "
                f"metric_ties={tie_count}/{num_chunks}"
            )

            bad_indices = torch.nonzero(reported, as_tuple=False).flatten().tolist()
            remaining_budget = max(0, int(max_mismatches) - totals["rows_written"])
            for chunk_idx in bad_indices[:remaining_budget]:
                pseudo_decision = "exp2" if bool(choose_exp2[chunk_idx]) else "exp1"
                metric_decision = "exp2" if bool(metric_choose_exp2[chunk_idx]) else "exp1"
                if bool(metric_min_mismatch[chunk_idx]):
                    mismatch_kind = "metric_min"
                else:
                    mismatch_kind = "decision"
                row = {
                    "bits_to_take": bits_to_take,
                    "fixed_rounding": fixed_rounding,
                    "tie_break": tie_break,
                    "bit_width": bit_width,
                    "chunk_idx": chunk_idx,
                    "mismatch_kind": mismatch_kind,
                    "pseudo_decision": pseudo_decision,
                    "metric_decision": metric_decision,
                    "metric_exp1": fmt_float(metric_exp1[chunk_idx]),
                    "metric_exp2": fmt_float(metric_exp2[chunk_idx]),
                    "metric_delta_exp2_minus_exp1": fmt_float(metric_delta[chunk_idx]),
                    "chunk_diff_exp2_minus_exp1": fmt_float(chunk_diff[chunk_idx]),
                }
                writer.writerow(row)
                totals["rows_written"] += 1

    print(
        "TOTAL "
        f"reported_mismatched_chunks={totals['reported_mismatched_chunks']} "
        f"metric_min_mismatches={totals['metric_min_mismatched_chunks']} "
        f"decision_disagreements={totals['decision_disagreements']} "
        f"metric_ties={totals['metric_ties']} "
        f"rows_written={totals['rows_written']}"
    )
    return totals


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo_MSE3 PyTorch hardware vectors.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "pseudo_mse3_hw_vectors.txt"),
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
        "--bits-to-take",
        "--bits_to_take",
        type=int,
        default=0,
        help="Number of fractional bits used for pseudo_MSE3 fixed-point contributions.",
    )
    parser.add_argument(
        "--fixed-rounding",
        "--fixed_rounding",
        type=normalize_pseudo_mse3_fixed_rounding,
        choices=("floor", "nearest"),
        default="floor",
        help=(
            "Fixed-point contribution rounding. nearest uses round-to-nearest "
            "with exact half cases away from zero."
        ),
    )
    parser.add_argument(
        "--tie-break",
        "--tie_break",
        type=normalize_pseudo_mse3_tie_break,
        choices=("exp1", "exp2"),
        default="exp1",
        help="Exact chunk-sum tie policy: exp1 uses < 0; exp2 uses <= 0.",
    )
    parser.add_argument(
        "--compare-metric",
        default=None,
        help="Compare pseudo_MSE3 decisions against a scalar metric such as l2/mse, l1, linf, bias, l0, huber, or logsum.",
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
        help="CSV path for per-chunk mismatch metadata when --compare-metric is set.",
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
        help="Verify CUDA pseudo_MSE3 search against the same PyTorch reference chunks.",
    )
    parser.add_argument(
        "--verify-python",
        action="store_true",
        help="Verify DynamicInputQuantizer's Python pseudo_MSE3 path against the same reference chunks.",
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
        help="Maximum detailed mismatch examples to print.",
    )
    parser.add_argument(
        "--cuda-build-dir",
        default=None,
        help="Optional fresh CUDA extension build directory for verification runs.",
    )
    args = parser.parse_args()
    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")
    if args.bits_to_take < 0:
        raise ValueError("--bits-to-take must be non-negative")
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
            bits_to_take=args.bits_to_take,
            fixed_rounding=args.fixed_rounding,
            tie_break=args.tie_break,
        )
        print(args.output)
        if args.csv_output is not None:
            write_csv_vectors(
                args.csv_output,
                value_mode=args.value_mode,
                num_chunks=args.num_chunks,
                seed=args.seed,
                bits_to_take=args.bits_to_take,
                fixed_rounding=args.fixed_rounding,
                tie_break=args.tie_break,
            )
            print(args.csv_output)

    verify_mismatches = 0
    comparison_totals = None
    if args.compare_metric:
        compare_csv_output = args.compare_csv_output
        if compare_csv_output is None:
            compare_csv_output = os.path.join(
                os.path.dirname(__file__),
                "pseudo_mse3_compare_mismatches.csv",
            )
        comparison_totals = compare_pseudo_mse3_with_metric(
            compare_csv_output,
            compare_metric=args.compare_metric,
            compare_metric_param=args.compare_metric_param,
            compare_atol=args.compare_atol,
            compare_tie_policy=args.compare_tie_policy,
            num_chunks=args.num_chunks,
            seed=args.seed,
            max_mismatches=args.max_mismatches,
            bits_to_take=args.bits_to_take,
            fixed_rounding=args.fixed_rounding,
            tie_break=args.tie_break,
        )
    if args.verify_python:
        verify_mismatches += verify_python_vectors(
            num_chunks=args.num_chunks,
            seed=args.seed,
            max_mismatches=args.max_mismatches,
            bits_to_take=args.bits_to_take,
            fixed_rounding=args.fixed_rounding,
            tie_break=args.tie_break,
        )
    if args.verify_cuda:
        verify_mismatches += verify_cuda_vectors(
            num_chunks=args.num_chunks,
            seed=args.seed,
            max_mismatches=args.max_mismatches,
            bits_to_take=args.bits_to_take,
            fixed_rounding=args.fixed_rounding,
            tie_break=args.tie_break,
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
