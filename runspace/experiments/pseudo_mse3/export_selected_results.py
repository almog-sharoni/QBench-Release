#!/usr/bin/env python3
"""Export MSE and pseudo-MSE comparison results to JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "mse_vs_pseudo_mse.json"
SELECTED_METRICS = ("mse", "pseudo_mse")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export only MSE and pseudo-MSE comparison results."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def _metric_name(value):
    normalized = str(value or "").strip().lower()
    if normalized == "mse":
        return "mse"
    if normalized.startswith("pseudo_mse"):
        return "pseudo_mse"
    return None


def _optional_float(value):
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _optional_int(value):
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _candidate_formats(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _public_activation_dt(value):
    value = str(value or "")
    marker = "_pseudo_mse"
    if marker not in value:
        return value
    prefix, _separator, _suffix = value.partition(marker)
    return f"{prefix}_pseudo_mse"


def _result_from_row(row, metric):
    return OrderedDict(
        (
            ("metric", metric),
            ("bit_width", int(row["bit_width"])),
            ("activation_dt", _public_activation_dt(row["activation_dt"])),
            ("candidate_formats", _candidate_formats(row["candidate_formats"])),
            ("bits_to_take", _optional_int(row.get("bits_to_take"))),
            ("fixed_rounding", str(row.get("fixed_rounding") or "")),
            ("tie_break", str(row.get("tie_break") or "")),
            ("dataset_size", _optional_int(row.get("dataset_size"))),
            ("random_seed", _optional_int(row.get("random_seed"))),
            ("limit_batches", _optional_int(row.get("limit_batches"))),
            ("status", str(row.get("status") or "")),
            ("acc1", _optional_float(row.get("acc1"))),
            ("acc5", _optional_float(row.get("acc5"))),
            ("certainty", _optional_float(row.get("certainty"))),
            ("norm_mse", _optional_float(row.get("norm_mse"))),
        )
    )


def load_selected_results(results_dir):
    """Load the newest summary row for each model, width, and selected metric."""
    summary_paths = sorted(
        Path(results_dir).glob("*/*_summary.csv"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    latest_rows = {}
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                metric = _metric_name(row.get("metric"))
                if metric not in SELECTED_METRICS:
                    continue
                identity = (
                    str(row["model"]),
                    int(row["bit_width"]),
                    metric,
                )
                latest_rows[identity] = row

    models = sorted({identity[0] for identity in latest_rows})
    exported = []
    for model_name in models:
        activation_results = OrderedDict()
        bit_widths = sorted(
            {
                bit_width
                for model, bit_width, _metric in latest_rows
                if model == model_name
            },
            reverse=True,
        )
        for bit_width in bit_widths:
            metric_results = OrderedDict()
            for metric in SELECTED_METRICS:
                row = latest_rows.get((model_name, bit_width, metric))
                if row is None:
                    continue
                metric_results[metric] = _result_from_row(row, metric)
            if metric_results:
                activation_results[f"{bit_width}_bit"] = metric_results

        exported.append(
            OrderedDict(
                (
                    ("model_name", model_name),
                    ("weights", "fp32"),
                    ("activation_results", activation_results),
                )
            )
        )
    return exported


def export_selected_results(results_dir, output_path):
    results = load_selected_results(results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return results


def main(argv=None):
    args = parse_args(argv)
    results = export_selected_results(args.results_dir, args.output)
    result_count = sum(
        len(metrics)
        for model in results
        for metrics in model["activation_results"].values()
    )
    print(
        f"Exported {result_count} MSE/pseudo-MSE results for "
        f"{len(results)} models to {args.output}"
    )


if __name__ == "__main__":
    main()
