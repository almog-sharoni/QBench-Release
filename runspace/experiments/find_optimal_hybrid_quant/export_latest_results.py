#!/usr/bin/env python3
"""Export the latest successful hybrid-quantization results from the run DB."""

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
HYBRID_EXPERIMENT_TYPE = "hybrid_quant_optimal"
_FORMAT_RE = re.compile(r"^(?:u?fp)(?P<bits>\d+)_e\d+m\d+$")
_DYNAMIC_BITS_RE = re.compile(r"_(?P<bits>\d+)bit(?:_|$)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export the newest successful result for every hybrid weight/input "
            "configuration."
        )
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def _format_bit_width(value):
    value = str(value or "")
    match = _FORMAT_RE.fullmatch(value)
    if match is not None:
        return int(match.group("bits"))
    match = _DYNAMIC_BITS_RE.search(value)
    if match is not None:
        return int(match.group("bits"))
    return None


def _mode(value, *, optimized_prefix, dynamic_prefix=None):
    value = str(value or "")
    if value == "fp32":
        return "fp32"
    if value.startswith(optimized_prefix):
        return "optimized"
    if dynamic_prefix is not None and value.startswith(dynamic_prefix):
        return "dynamic"
    return "baseline"


def _selection_value(row, key, fallback=None):
    value = row[key]
    return fallback if value is None else value


def _row_sort_key(row):
    if row["experiment_type"] == "fp32_ref":
        return (0, 0, "", "")
    bit_width = row["selection_bit_width"]
    if bit_width is None:
        bit_width = _format_bit_width(row["activation_dt"])
    if bit_width is None:
        bit_width = _format_bit_width(row["weight_dt"])
    return (
        1,
        -int(bit_width or 0),
        str(row["weight_dt"]),
        str(row["activation_dt"]),
    )


def load_latest_results(db_path):
    """Return FP32 references and the newest row per hybrid semantic identity."""
    db_uri = f"file:{Path(db_path).resolve()}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id, model_name, weight_dt, activation_dt, experiment_type,
                acc1, acc5, mse, l1, certainty, run_date, run_identity,
                json_extract(config_json, '$.meta.selection.direction')
                    AS selection_direction,
                json_extract(config_json, '$.meta.selection.bit_width')
                    AS selection_bit_width,
                json_extract(config_json, '$.meta.selection.weight.mode')
                    AS selection_weight_mode,
                json_extract(config_json, '$.meta.selection.weight.source_run_id')
                    AS selection_weight_source_run_id,
                json_extract(config_json, '$.meta.selection.weight.source_acc1')
                    AS selection_weight_source_acc1,
                json_extract(config_json, '$.meta.selection.input.mode')
                    AS selection_input_mode,
                json_extract(config_json, '$.meta.selection.input.source_run_id')
                    AS selection_input_source_run_id,
                json_extract(config_json, '$.meta.selection.input.source_acc1')
                    AS selection_input_source_acc1
            FROM runs
            WHERE status = 'SUCCESS'
              AND (
                    experiment_type = ?
                 OR experiment_type = 'fp32_ref'
              )
            ORDER BY id DESC
            """,
            (HYBRID_EXPERIMENT_TYPE,),
        ).fetchall()

    hybrid_models = {
        row["model_name"]
        for row in rows
        if row["experiment_type"] == HYBRID_EXPERIMENT_TYPE
    }

    # Hybrid identities preserve distinct optimized selections even when their
    # display weight/input labels are equal. References are newest per model.
    latest_rows = {}
    for row in rows:
        if row["model_name"] not in hybrid_models:
            continue
        if row["experiment_type"] == "fp32_ref":
            identity = (row["model_name"], "fp32_ref")
        else:
            identity = (
                row["model_name"],
                row["run_identity"]
                or f"{row['weight_dt']}::{row['activation_dt']}",
            )
        latest_rows.setdefault(identity, row)

    rows_by_model = {}
    for row in latest_rows.values():
        rows_by_model.setdefault(row["model_name"], []).append(row)

    exported = []
    for model_name in sorted(rows_by_model):
        hybrid_results = []
        model_rows = rows_by_model[model_name]
        row_views = [
            (row, row["selection_direction"], False)
            for row in model_rows
        ]
        suppress_undirected_row_ids = set()

        # The sweep deliberately stores the shared (best-weight, best-input)
        # pair once. Reconstruct its missing directional view for reporting so
        # both fixed-input and fixed-weight sweeps remain complete in the JSON.
        selection_rows_by_width = {}
        for row in model_rows:
            if row["experiment_type"] != HYBRID_EXPERIMENT_TYPE:
                continue
            bit_width = row["selection_bit_width"]
            if bit_width is None:
                continue
            selection_rows_by_width.setdefault(int(bit_width), []).append(row)

        for width_rows in selection_rows_by_width.values():
            weight_options = {
                int(row["selection_weight_source_run_id"]): float(
                    row["selection_weight_source_acc1"]
                )
                for row in width_rows
                if row["selection_weight_source_run_id"] is not None
                and row["selection_weight_source_acc1"] is not None
            }
            input_options = {
                int(row["selection_input_source_run_id"]): float(
                    row["selection_input_source_acc1"]
                )
                for row in width_rows
                if row["selection_input_source_run_id"] is not None
                and row["selection_input_source_acc1"] is not None
            }
            if not weight_options or not input_options:
                continue
            best_weight_source = max(
                weight_options,
                key=lambda run_id: (weight_options[run_id], run_id),
            )
            best_input_source = max(
                input_options,
                key=lambda run_id: (input_options[run_id], run_id),
            )
            shared_rows = [
                row
                for row in width_rows
                if row["selection_weight_source_run_id"] == best_weight_source
                and row["selection_input_source_run_id"] == best_input_source
            ]
            if not shared_rows:
                best_weight_label = next(
                    row["weight_dt"]
                    for row in width_rows
                    if row["selection_weight_source_run_id"]
                    == best_weight_source
                )
                best_input_label = next(
                    row["activation_dt"]
                    for row in width_rows
                    if row["selection_input_source_run_id"]
                    == best_input_source
                )
                shared_rows = [
                    row
                    for row in model_rows
                    if row["experiment_type"] == HYBRID_EXPERIMENT_TYPE
                    and row["weight_dt"] == best_weight_label
                    and row["activation_dt"] == best_input_label
                ]
            if not shared_rows:
                continue
            present_directions = {
                row["selection_direction"]
                for row in shared_rows
                if row["selection_direction"] is not None
            }
            source_row = max(shared_rows, key=lambda row: row["id"])
            if source_row["selection_direction"] is None:
                suppress_undirected_row_ids.add(source_row["id"])
            for missing_direction in (
                {"weight_fixed", "input_fixed"} - present_directions
            ):
                row_views.append((source_row, missing_direction, True))

        row_views = [
            view
            for view in row_views
            if not (
                view[0]["id"] in suppress_undirected_row_ids
                and view[1] is None
            )
        ]

        for row, selection_direction, reconstructed_duplicate in sorted(
            row_views,
            key=lambda view: (_row_sort_key(view[0]), str(view[1] or "")),
        ):
            is_reference = row["experiment_type"] == "fp32_ref"
            weight_mode = _selection_value(
                row,
                "selection_weight_mode",
                _mode(row["weight_dt"], optimized_prefix="opt_"),
            )
            input_mode = _selection_value(
                row,
                "selection_input_mode",
                _mode(
                    row["activation_dt"],
                    optimized_prefix="opt_",
                    dynamic_prefix="dyn_input_",
                ),
            )
            bit_width = row["selection_bit_width"]
            if bit_width is None and not is_reference:
                bit_width = _format_bit_width(row["activation_dt"])
            if bit_width is None and not is_reference:
                bit_width = _format_bit_width(row["weight_dt"])

            hybrid_results.append(
                OrderedDict(
                    (
                        ("weight_type", row["weight_dt"]),
                        ("weight_mode", weight_mode),
                        ("input_type", row["activation_dt"]),
                        ("input_mode", input_mode),
                        ("bit_width", 32 if is_reference else bit_width),
                        (
                            "selection_direction",
                            "reference" if is_reference else selection_direction,
                        ),
                        ("reconstructed_duplicate", reconstructed_duplicate),
                        ("run_id", row["id"]),
                        ("run_date", row["run_date"]),
                        ("experiment_type", row["experiment_type"]),
                        ("acc1", row["acc1"]),
                        ("acc5", row["acc5"]),
                        ("mse", row["mse"]),
                        ("l1", row["l1"]),
                        ("certainty", row["certainty"]),
                    )
                )
            )

        exported.append(
            OrderedDict(
                (
                    ("model_name", model_name),
                    ("hybrid_results", hybrid_results),
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
    result_count = sum(len(model["hybrid_results"]) for model in results)
    print(f"Exported {result_count} latest results for {len(results)} models to {args.output}")


if __name__ == "__main__":
    main()
