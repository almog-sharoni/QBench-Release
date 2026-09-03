#!/usr/bin/env python3
"""Export selected activation candidate-pool sweep results to JSON."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_PATH = EXPERIMENT_DIR / "results" / "summary.csv"
DEFAULT_OUTPUT_PATH = EXPERIMENT_DIR / "results" / "selected_candidate_pools.json"
SELECTED_POLICIES = ("exp1", "only_e2", "exp2", "all")
POLICY_DESCRIPTIONS = {
    "exp1": "e1 only",
    "only_e2": "e2 only",
    "exp2": "e1 and e2",
    "all": "full pool",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export only the e1-only, e2-only, e1/e2, and full activation "
            "candidate pools from an activation_candidate_sweep summary."
        )
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def _optional_float(value):
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _candidate_formats(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _format_counts(value):
    if value is None or str(value).strip() == "":
        return {}
    parsed = json.loads(value)
    return {str(fmt): int(count) for fmt, count in parsed.items()}


def _result_from_row(row, *, policy=None, reconstructed_duplicate=False):
    selected_policy = str(policy or row["exp_cap"])
    bit_width = int(row["bit_width"])
    source_policy = str(row["exp_cap"])
    activation_dt = str(row["activation_dt"])
    if reconstructed_duplicate:
        activation_dt = f"dyn_a{bit_width}_{selected_policy}_l2"

    return OrderedDict(
        (
            ("policy", selected_policy),
            ("policy_description", POLICY_DESCRIPTIONS[selected_policy]),
            ("bit_width", bit_width),
            ("activation_dt", activation_dt),
            ("candidate_count", int(row["candidate_count"])),
            ("candidate_formats", _candidate_formats(row["candidate_formats"])),
            ("format_counts", _format_counts(row.get("format_counts_json"))),
            ("acc1", _optional_float(row.get("acc1"))),
            ("acc5", _optional_float(row.get("acc5"))),
            (
                "delta_acc1_vs_fp32",
                _optional_float(row.get("delta_acc1_vs_fp32")),
            ),
            ("certainty", _optional_float(row.get("certainty"))),
            ("norm_mse", _optional_float(row.get("norm_mse"))),
            ("norm_l1", _optional_float(row.get("norm_l1"))),
            ("source_policy", source_policy),
            ("reconstructed_duplicate", bool(reconstructed_duplicate)),
        )
    )


def load_selected_results(summary_path):
    """Return selected policy results, reconstructing deduplicated exp2 pools."""
    with Path(summary_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    latest_rows = {}
    for row in rows:
        if row.get("exp_cap") not in SELECTED_POLICIES:
            continue
        identity = (
            str(row["model"]),
            int(row["bit_width"]),
            str(row["exp_cap"]),
        )
        latest_rows[identity] = row

    models = sorted({identity[0] for identity in latest_rows})
    exported = []
    for model_name in models:
        activation_results = OrderedDict()
        bit_widths = sorted(
            {
                bit_width
                for model, bit_width, _ in latest_rows
                if model == model_name
            },
            reverse=True,
        )
        for bit_width in bit_widths:
            policy_results = OrderedDict()
            for policy in SELECTED_POLICIES:
                row = latest_rows.get((model_name, bit_width, policy))
                reconstructed = False
                if row is None and policy == "exp2":
                    # At 4 bits the full signed pool is exactly e1/e2, so the
                    # sweep deduplicates exp2 against all. Keep both report views.
                    row = latest_rows.get((model_name, bit_width, "all"))
                    reconstructed = row is not None
                if row is None:
                    continue
                policy_results[policy] = _result_from_row(
                    copy.deepcopy(row),
                    policy=policy,
                    reconstructed_duplicate=reconstructed,
                )
            if policy_results:
                activation_results[f"{bit_width}_bit"] = policy_results

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


def export_selected_results(summary_path, output_path):
    results = load_selected_results(summary_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return results


def main(argv=None):
    args = parse_args(argv)
    results = export_selected_results(args.summary_csv, args.output)
    result_count = sum(
        len(policies)
        for model in results
        for policies in model["activation_results"].values()
    )
    print(
        f"Exported {result_count} selected candidate-pool results for "
        f"{len(results)} models to {args.output}"
    )


if __name__ == "__main__":
    main()
