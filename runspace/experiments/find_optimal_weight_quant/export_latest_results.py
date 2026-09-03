#!/usr/bin/env python3
"""Export the latest successful weight-quantization results from the run DB."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import OrderedDict
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = EXPERIMENT_DIR.parents[1] / "database" / "runs.db"
DEFAULT_OUTPUT_PATH = EXPERIMENT_DIR / "results" / "latest_db_results.json"
_OPTIMIZED_EXPERIMENT_RE = re.compile(r"^weight_quant_optimized_(?P<bits>\d+)$")
_FORMAT_RE = re.compile(r"^(?:u?fp)(?P<bits>\d+)_e(?P<exponent>\d+)m(?P<mantissa>\d+)$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export the newest successful FP32-input result for every model and "
            "weight-quantization type."
        )
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def _weight_type(row):
    experiment_type = str(row["experiment_type"])
    weight_dt = str(row["weight_dt"])
    if experiment_type == "fp32_ref":
        return "fp32"
    if experiment_type == "weight_quant_baseline":
        return weight_dt
    if experiment_type == "weight_quant_optimized":
        return f"optimized_{weight_dt.removeprefix('opt_')}"

    match = _OPTIMIZED_EXPERIMENT_RE.fullmatch(experiment_type)
    if match is None:
        raise ValueError(f"Unsupported weight-quant experiment type: {experiment_type!r}")
    optimizer = weight_dt.removeprefix("opt_")
    return f"optimized_{match.group('bits')}_{optimizer}"


def _weight_type_sort_key(weight_type):
    if weight_type == "fp32":
        return (0, 0, 0, "")
    format_match = _FORMAT_RE.fullmatch(weight_type)
    if format_match is not None:
        return (
            1,
            -int(format_match.group("bits")),
            int(format_match.group("exponent")),
            weight_type,
        )
    optimized_match = re.match(r"^optimized_(\d+)_(.+)$", weight_type)
    if optimized_match is not None:
        return (2, -int(optimized_match.group(1)), 0, optimized_match.group(2))
    if weight_type.startswith("optimized_"):
        return (3, 0, 0, weight_type)
    return (4, 0, 0, weight_type)


def load_latest_results(db_path):
    """Return one ordered model record per model, newest run per weight identity."""
    db_uri = f"file:{Path(db_path).resolve()}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id, model_name, weight_dt, activation_dt, experiment_type,
                acc1, acc5, mse, l1, certainty, run_date
            FROM runs
            WHERE status = 'SUCCESS'
              AND activation_dt = 'fp32'
              AND (
                    experiment_type = 'fp32_ref'
                 OR experiment_type = 'weight_quant_baseline'
                 OR experiment_type = 'weight_quant_optimized'
                 OR experiment_type GLOB 'weight_quant_optimized_[0-9]'
              )
            ORDER BY id DESC
            """
        ).fetchall()

    # Rows are newest-first, so the first occurrence is the requested latest run.
    latest_rows = {}
    for row in rows:
        identity = (
            row["model_name"],
            row["experiment_type"],
            row["weight_dt"],
        )
        latest_rows.setdefault(identity, row)

    rows_by_model = {}
    for row in latest_rows.values():
        rows_by_model.setdefault(row["model_name"], []).append(row)

    exported = []
    for model_name in sorted(rows_by_model):
        weight_results = OrderedDict()
        typed_rows = [(_weight_type(row), row) for row in rows_by_model[model_name]]
        for weight_type, row in sorted(typed_rows, key=lambda item: _weight_type_sort_key(item[0])):
            weight_results[weight_type] = OrderedDict(
                (
                    ("run_id", row["id"]),
                    ("run_date", row["run_date"]),
                    ("experiment_type", row["experiment_type"]),
                    ("weight_dt", row["weight_dt"]),
                    ("acc1", row["acc1"]),
                    ("acc5", row["acc5"]),
                    ("mse", row["mse"]),
                    ("l1", row["l1"]),
                    ("certainty", row["certainty"]),
                )
            )

        exported.append(
            OrderedDict(
                (
                    ("model_name", model_name),
                    ("inputs", "fp32"),
                    ("weight_results", weight_results),
                )
            )
        )
    return exported


def export_latest_results(db_path, output_path):
    results = load_latest_results(db_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return results


def main(argv=None):
    args = parse_args(argv)
    results = export_latest_results(args.db_path, args.output)
    result_count = sum(len(model["weight_results"]) for model in results)
    print(f"Exported {result_count} latest results for {len(results)} models to {args.output}")


if __name__ == "__main__":
    main()
