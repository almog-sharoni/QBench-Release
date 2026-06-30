#!/usr/bin/env python3
"""Print pseudo_MSE e1/e2 reconstruction examples.

The values printed here are already in the scaled domain used by dynamic input
quantization. For each FP32 value, the script applies the repository's direct
qtype simulator (`runspace.src.quantization.quantizer.quantize`) to the e1/e2
formats and prints the resulting squared-error difference.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.quantization.quantizer import quantize  # noqa: E402
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    pseudo_mse_sqerr_diff_from_scaled,
)


def parse_csv_floats(raw: str) -> list[float]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def mantissa_widths(bit_width: int, unsigned: bool) -> tuple[int, int]:
    sign_bits = 0 if unsigned else 1
    m1 = bit_width - sign_bits - 1
    m2 = bit_width - sign_bits - 2
    if m1 < 1 or m2 < 0:
        kind = "unsigned" if unsigned else "signed"
        raise ValueError(f"{kind} {bit_width}-bit e1/e2 pair is invalid: m1={m1}, m2={m2}")
    return m1, m2


def format_name(bit_width: int, exp_bits: int, mantissa_bits: int, unsigned: bool) -> str:
    prefix = "ufp" if unsigned else "fp"
    return f"{prefix}{bit_width}_e{exp_bits}m{mantissa_bits}"


def format_pair(bit_width: int, unsigned: bool) -> tuple[str, str]:
    m1, m2 = mantissa_widths(bit_width, unsigned)
    return (
        format_name(bit_width, 1, m1, unsigned),
        format_name(bit_width, 2, m2, unsigned),
    )


def chunk_values_for_exp(args: argparse.Namespace, exp: int) -> torch.Tensor:
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    if args.mantissas is None:
        # Midpoints cover the whole [1, 2) significand interval without landing
        # exactly on common grid points.
        mantissas = 1.0 + (torch.arange(chunk_size, dtype=torch.float32) + 0.5) / chunk_size
    else:
        base = torch.tensor(args.mantissas, dtype=torch.float32)
        repeats = math.ceil(chunk_size / int(base.numel()))
        mantissas = base.repeat(repeats)[:chunk_size]

    values = mantissas * float(2.0 ** exp)
    if args.include_negative and not args.unsigned:
        signs = torch.ones_like(values)
        signs[::2] = -1.0
        values = values * signs
    return values.contiguous()


def mixed_chunk_values(args: argparse.Namespace, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    exp_choices = torch.randint(
        low=int(args.min_exp),
        high=int(args.max_exp) + 1,
        size=(chunk_size,),
        generator=generator,
        dtype=torch.int64,
    )
    exp_range = torch.arange(int(args.min_exp), int(args.max_exp) + 1, dtype=torch.int64)
    if chunk_size >= int(exp_range.numel()):
        exp_choices[: int(exp_range.numel())] = exp_range
        exp_choices = exp_choices[torch.randperm(chunk_size, generator=generator)]
    mantissas = 1.0 + torch.rand(chunk_size, generator=generator, dtype=torch.float32)
    values = mantissas * torch.pow(torch.tensor(2.0, dtype=torch.float32), exp_choices.to(torch.float32))

    if args.include_negative and not args.unsigned:
        signs = torch.where(
            torch.rand(chunk_size, generator=generator, dtype=torch.float32) < 0.5,
            torch.full((chunk_size,), -1.0, dtype=torch.float32),
            torch.ones(chunk_size, dtype=torch.float32),
        )
        values = values * signs

    return values.contiguous(), exp_choices


def row_for_values(
    args: argparse.Namespace,
    values: torch.Tensor,
    actual_e=None,
    exp_min=None,
    exp_max=None,
    chunk_index=None,
) -> dict[str, object]:
    e1_name, e2_name = format_pair(args.bit_width, args.unsigned)
    m1, m2 = mantissa_widths(args.bit_width, args.unsigned)
    q1 = quantize(values, q_type=e1_name)
    q2 = quantize(values, q_type=e2_name)

    err1 = (values - q1).pow(2)
    err2 = (values - q2).pow(2)
    diff = err2 - err1
    sum_err1 = float(err1.sum().item())
    sum_err2 = float(err2.sum().item())
    sum_diff = float(diff.sum().item())
    choice = "e2" if sum_diff < 0.0 else "e1"
    pseudo_diff = pseudo_mse_sqerr_diff_from_scaled(
        values,
        exp1_mantissa_width=m1,
        exp2_mantissa_width=m2,
        is_signed=not args.unsigned,
    )
    pseudo_sum_diff = float(pseudo_diff.sum().item())
    pseudo_choice = "e2" if pseudo_sum_diff < 0.0 else "e1"
    abs_delta = abs(sum_diff - pseudo_sum_diff)
    choices_match = choice == pseudo_choice
    values_match = abs_delta <= args.tolerance
    check = "OK" if choices_match and values_match else "DIFF"

    return {
        "chunk_index": chunk_index,
        "actual_e": actual_e,
        "exp_min": exp_min,
        "exp_max": exp_max,
        "values": values,
        "q1": q1,
        "q2": q2,
        "err1": err1,
        "err2": err2,
        "diff": diff,
        "x_min": float(values.min().item()),
        "x_max": float(values.max().item()),
        "sum_err1": sum_err1,
        "sum_err2": sum_err2,
        "sum_diff": sum_diff,
        "choice": choice,
        "pseudo_diff": pseudo_diff,
        "pseudo_sum_diff": pseudo_sum_diff,
        "pseudo_choice": pseudo_choice,
        "abs_delta": abs_delta,
        "check": check,
        "e2_element_wins": int((diff < 0).sum().item()),
    }


def row_for_chunk(args: argparse.Namespace, exp: int) -> dict[str, object]:
    values = chunk_values_for_exp(args, exp)
    return row_for_values(args, values, actual_e=exp, exp_min=exp, exp_max=exp)


def row_for_mixed_chunk(args: argparse.Namespace, chunk_index: int, generator: torch.Generator) -> dict[str, object]:
    values, exp_choices = mixed_chunk_values(args, generator)
    return row_for_values(
        args,
        values,
        exp_min=int(exp_choices.min().item()),
        exp_max=int(exp_choices.max().item()),
        chunk_index=chunk_index,
    )


def print_chunk_elements(row: dict[str, object]) -> None:
    values = row["values"]
    q1 = row["q1"]
    q2 = row["q2"]
    err1 = row["err1"]
    err2 = row["err2"]
    diff = row["diff"]
    pseudo_diff = row["pseudo_diff"]

    print(
        "    idx  scaled_fp32      e1_value      err_e1        "
        "e2_value      err_e2        direct_diff   pseudo_diff"
    )
    for idx in range(int(values.numel())):
        print(
            f"    {idx:3d}  "
            f"{float(values[idx].item()):>12.8g}  "
            f"{float(q1[idx].item()):>12.8g}  {float(err1[idx].item()):>11.4e}  "
            f"{float(q2[idx].item()):>12.8g}  {float(err2[idx].item()):>11.4e}  "
            f"{float(diff[idx].item()):>11.4e}  {float(pseudo_diff[idx].item()):>11.4e}"
        )


def print_examples(args: argparse.Namespace) -> None:
    unsigned = bool(args.unsigned)
    e1_name, e2_name = format_pair(args.bit_width, unsigned)

    print(f"pseudo_MSE examples for {e1_name} vs {e2_name}")
    print("Inputs are scaled FP32 values. Errors are squared reconstruction errors.")
    print("Quantized values are produced by runspace.src.quantization.quantizer.quantize.")
    print(f"Each row is one chunk of {args.chunk_size} values in the actual_e bucket.")
    print("sum_diff = sum((x - q_e2)^2) - sum((x - q_e1)^2); sum_diff < 0 chooses e2.")
    print("pseudo_sum_diff is computed by pseudo_mse_sqerr_diff_from_scaled on the same chunk.")
    print()
    print(
        "actual_e  "
        "x_min         x_max         "
        "sum_err_e1    sum_err_e2    direct_diff  direct  "
        "pseudo_diff  pseudo  abs_delta    check"
    )
    print("-" * 132)

    failed = []
    for exp in range(args.max_exp, args.min_exp - 1, -1):
        row = row_for_chunk(args, exp)
        print(
            f"{row['actual_e']:>8}  "
            f"{row['x_min']:>12.8g}  {row['x_max']:>12.8g}  "
            f"{row['sum_err1']:>11.4e}  {row['sum_err2']:>11.4e}  "
            f"{row['sum_diff']:>11.4e}  {row['choice']:>6}  "
            f"{row['pseudo_sum_diff']:>11.4e}  {row['pseudo_choice']:>6}  "
            f"{row['abs_delta']:>9.2e}  {row['check']}"
        )
        if row["check"] != "OK":
            failed.append(row)
        if args.print_elements:
            print_chunk_elements(row)

    if failed and not args.no_strict:
        raise SystemExit(
            f"pseudo_MSE comparison failed for {len(failed)} exponent bucket(s); "
            "rerun with --no-strict to print without failing"
        )


def print_mixed_examples(args: argparse.Namespace) -> None:
    unsigned = bool(args.unsigned)
    e1_name, e2_name = format_pair(args.bit_width, unsigned)
    generator = torch.Generator().manual_seed(int(args.seed))

    print(f"pseudo_MSE mixed-chunk check for {e1_name} vs {e2_name}")
    print("Inputs are scaled FP32 values. Errors are squared reconstruction errors.")
    print("Quantized values are produced by runspace.src.quantization.quantizer.quantize.")
    print(
        f"Each chunk has {args.chunk_size} values with per-element actual_e sampled "
        f"uniformly from [{args.min_exp}, {args.max_exp}]."
    )
    print("sum_diff = sum((x - q_e2)^2) - sum((x - q_e1)^2); sum_diff < 0 chooses e2.")
    print("pseudo_sum_diff is computed by pseudo_mse_sqerr_diff_from_scaled on the same chunk.")
    print()
    print(
        "chunk  e_min  e_max  x_min         x_max         "
        "direct_diff  direct  pseudo_diff  pseudo  abs_delta    check"
    )
    print("-" * 128)

    failed = []
    ok_count = 0
    direct_e2 = 0
    pseudo_e2 = 0
    max_abs_delta = 0.0
    printed = 0
    print_limit = int(args.print_limit)

    for chunk_index in range(int(args.num_chunks)):
        row = row_for_mixed_chunk(args, chunk_index, generator)
        ok = row["check"] == "OK"
        ok_count += int(ok)
        direct_e2 += int(row["choice"] == "e2")
        pseudo_e2 += int(row["pseudo_choice"] == "e2")
        max_abs_delta = max(max_abs_delta, float(row["abs_delta"]))
        if not ok:
            failed.append(row)

        should_print = print_limit < 0 or printed < print_limit or not ok
        if should_print:
            print(
                f"{chunk_index:5d}  "
                f"{row['exp_min']:>5}  {row['exp_max']:>5}  "
                f"{row['x_min']:>12.8g}  {row['x_max']:>12.8g}  "
                f"{row['sum_diff']:>11.4e}  {row['choice']:>6}  "
                f"{row['pseudo_sum_diff']:>11.4e}  {row['pseudo_choice']:>6}  "
                f"{row['abs_delta']:>9.2e}  {row['check']}"
            )
            printed += 1
            if args.print_elements:
                print_chunk_elements(row)

    print()
    print(
        "Summary: "
        f"chunks={args.num_chunks}, ok={ok_count}, diff={len(failed)}, "
        f"direct_e2={direct_e2}, pseudo_e2={pseudo_e2}, "
        f"max_abs_delta={max_abs_delta:.3e}, seed={args.seed}"
    )

    if failed and not args.no_strict:
        raise SystemExit(
            f"pseudo_MSE mixed-chunk comparison failed for {len(failed)} chunk(s); "
            "rerun with --no-strict to print without failing"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("by-exp", "mixed"),
        default="by-exp",
        help="'by-exp' prints one chunk per actual exponent; 'mixed' tests many chunks with mixed exponents.",
    )
    parser.add_argument("--bit-width", type=int, default=4, help="Total bit-width for the e1/e2 pair.")
    parser.add_argument("--unsigned", action="store_true", help="Use ufp formats instead of signed fp formats.")
    parser.add_argument(
        "--min-exp",
        type=int,
        default=-10,
        help="Smallest actual FP32 log2 exponent after scaling.",
    )
    parser.add_argument(
        "--max-exp",
        type=int,
        default=0,
        help="Largest actual FP32 log2 exponent after scaling.",
    )
    parser.add_argument(
        "--mantissas",
        type=parse_csv_floats,
        default=None,
        help=(
            "Optional comma-separated significands multiplied by 2^exp, "
            "for example '1.0,1.25,1.5'. If omitted, the script "
            "uses midpoint significands that cover [1, 2). If fewer than "
            "--chunk-size values are provided, they are repeated to fill the chunk."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of scaled FP32 values per exponent bucket.",
    )
    parser.add_argument(
        "--print-elements",
        action="store_true",
        help="Print every element in each chunk after the chunk-level summary.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1000,
        help="Number of mixed-exponent chunks to test when --mode mixed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for --mode mixed.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="Number of mixed chunks to print. Use -1 to print all. Mismatches are always printed.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for direct sum_diff vs pseudo_MSE sum_diff.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not exit nonzero when direct and pseudo_MSE comparisons differ.",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Also print negative examples for signed formats.",
    )
    args = parser.parse_args()
    if args.max_exp > 0:
        parser.error("scaled-domain examples should use --max-exp <= 0")
    if args.min_exp > args.max_exp:
        parser.error("--min-exp must be <= --max-exp")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.num_chunks <= 0:
        parser.error("--num-chunks must be positive")
    if args.mode == "mixed":
        print_mixed_examples(args)
    else:
        print_examples(args)


if __name__ == "__main__":
    main()
