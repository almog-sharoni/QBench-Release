#!/usr/bin/env python3
"""Summarize whether a dynamic-input run mostly behaves like a baseline format."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple


FORMAT_RE = re.compile(r"^(u?fp)(\d+)_e(\d+)m(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a dynamic-input experiment by comparing its accuracy to a baseline "
            "and summarizing which formats the dynamic chooser actually selected."
        )
    )
    parser.add_argument("--model", required=True, help="Model name, for example mobilevit_xxs")
    parser.add_argument("--metric", default="l1", help="Dynamic metric suffix, for example l1 or mse")
    parser.add_argument(
        "--baseline-format",
        default="fp8_e1m6",
        help="Baseline activation format to compare against",
    )
    parser.add_argument(
        "--db-path",
        default="runspace/database/runs.db",
        help="SQLite DB path used by the experiments",
    )
    parser.add_argument(
        "--results-root",
        default="runspace/experiments/find_optimal_input_quant/results",
        help="Root directory for saved comparison_results.json and layer_stats.json files",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=10,
        help="How many layers with the highest non-baseline share to print",
    )
    parser.add_argument(
        "--top-formats",
        type=int,
        default=10,
        help="How many aggregated formats to print",
    )
    parser.add_argument(
        "--min-acc-gap",
        type=float,
        default=10.0,
        help="Flag if dynamic acc1 exceeds baseline acc1 by at least this amount",
    )
    parser.add_argument(
        "--min-baseline-share",
        type=float,
        default=0.95,
        help="Flag if the exact baseline format still covers at least this fraction of chunks",
    )
    parser.add_argument(
        "--min-family-share",
        type=float,
        default=0.99,
        help="Flag if the baseline family still covers at least this fraction of chunks",
    )
    return parser.parse_args()


def load_best_run(
    db_path: str,
    *,
    model: str,
    experiment_type: str,
    activation_dt: str,
    weight_dt: str = "fp32",
) -> Optional[dict]:
    if not os.path.exists(db_path):
        return None

    query = """
        SELECT *
        FROM runs
        WHERE model_name = ?
          AND experiment_type = ?
          AND activation_dt = ?
          AND weight_dt = ?
          AND status = 'SUCCESS'
        ORDER BY acc1 DESC, id DESC
        LIMIT 1
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query, (model, experiment_type, activation_dt, weight_dt)).fetchone()
    return dict(row) if row else None


def safe_json_load(raw_value: Optional[str]) -> Optional[dict]:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return None
    return None


def load_comparison_results(results_root: str, model: str) -> Optional[Iterable[dict]]:
    path = os.path.join(results_root, model, "comparison_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_layer_stats_file(results_root: str, model: str, metric: str) -> Optional[dict]:
    path = os.path.join(results_root, model, metric, "layer_stats.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_baseline_acc1_from_results(
    comparison_results: Optional[Iterable[dict]],
    baseline_format: str,
) -> Optional[float]:
    if comparison_results is None:
        return None
    expected_name = f"Base_{baseline_format}"
    for row in comparison_results:
        if row.get("output_name") == expected_name:
            return float(row.get("acc1", 0.0))
    return None


def parse_format(fmt: str) -> Optional[Tuple[str, int, int, int]]:
    match = FORMAT_RE.match(fmt or "")
    if not match:
        return None
    prefix, bits, exp_bits, mant_bits = match.groups()
    return prefix, int(bits), int(exp_bits), int(mant_bits)


def is_same_or_lower_mantissa_family(fmt: str, baseline_format: str) -> bool:
    fmt_info = parse_format(fmt)
    baseline_info = parse_format(baseline_format)
    if fmt_info is None or baseline_info is None:
        return fmt == baseline_format
    prefix, _, exp_bits, mant_bits = fmt_info
    base_prefix, _, base_exp_bits, base_mant_bits = baseline_info
    return prefix == base_prefix and exp_bits == base_exp_bits and mant_bits <= base_mant_bits


def normalize_layer_map(raw_map: dict) -> Dict[str, dict]:
    normalized = {}
    for layer_name, entry in raw_map.items():
        if layer_name == "accuracy":
            continue
        if not isinstance(entry, dict):
            continue
        counts = entry.get("format_counts", {})
        if not isinstance(counts, dict):
            continue
        clean_counts = {}
        for fmt, count in counts.items():
            try:
                count_int = int(count)
            except (TypeError, ValueError):
                continue
            if count_int > 0:
                clean_counts[str(fmt)] = count_int
        if not clean_counts:
            continue
        dominant_format = entry.get("dominant_format")
        if dominant_format is None:
            dominant_format = max(clean_counts.items(), key=lambda item: (item[1], item[0]))[0]
        normalized[layer_name] = {
            "format_counts": clean_counts,
            "dominant_format": str(dominant_format),
            "type": entry.get("type", "unknown"),
        }
    return normalized


def aggregate_counts(layer_map: Dict[str, dict]) -> Counter:
    totals: Counter = Counter()
    for entry in layer_map.values():
        totals.update(entry["format_counts"])
    return totals


def summarize_layers(layer_map: Dict[str, dict], baseline_format: str) -> Tuple[int, int, list]:
    dominant_baseline_layers = 0
    details = []
    for layer_name, entry in layer_map.items():
        counts = entry["format_counts"]
        total = sum(counts.values())
        if total <= 0:
            continue
        baseline_count = counts.get(baseline_format, 0)
        dominant = entry["dominant_format"]
        if dominant == baseline_format:
            dominant_baseline_layers += 1
        details.append(
            {
                "layer": layer_name,
                "type": entry.get("type", "unknown"),
                "baseline_share": baseline_count / total,
                "non_baseline_share": 1.0 - (baseline_count / total),
                "dominant_format": dominant,
                "total_chunks": total,
            }
        )
    return dominant_baseline_layers, len(details), sorted(
        details,
        key=lambda item: (item["non_baseline_share"], item["total_chunks"], item["layer"]),
        reverse=True,
    )


def load_dynamic_payload(
    args: argparse.Namespace,
) -> Tuple[Optional[float], Optional[Dict[str, dict]], str]:
    activation_dt = f"dyn_input_{args.metric}"
    dynamic_run = load_best_run(
        args.db_path,
        model=args.model,
        experiment_type="input_quant_dynamic",
        activation_dt=activation_dt,
    )
    if dynamic_run:
        layer_map = normalize_layer_map(safe_json_load(dynamic_run.get("input_map_json")) or {})
        if layer_map:
            return float(dynamic_run.get("acc1", 0.0)), layer_map, f"db:{args.db_path}"

    layer_stats = load_layer_stats_file(args.results_root, args.model, args.metric)
    if layer_stats:
        accuracy = layer_stats.get("accuracy", {})
        layer_map = normalize_layer_map(layer_stats)
        if layer_map:
            return float(accuracy.get("top1", 0.0)), layer_map, f"file:{args.results_root}"

    return None, None, "unavailable"


def load_baseline_acc1(args: argparse.Namespace) -> Tuple[Optional[float], str]:
    baseline_run = load_best_run(
        args.db_path,
        model=args.model,
        experiment_type="input_quant_baseline",
        activation_dt=args.baseline_format,
    )
    if baseline_run:
        return float(baseline_run.get("acc1", 0.0)), f"db:{args.db_path}"

    comparison_results = load_comparison_results(args.results_root, args.model)
    acc1 = extract_baseline_acc1_from_results(comparison_results, args.baseline_format)
    if acc1 is not None:
        return acc1, f"file:{args.results_root}"

    return None, "unavailable"


def print_summary(args: argparse.Namespace) -> int:
    dynamic_acc1, layer_map, dynamic_source = load_dynamic_payload(args)
    baseline_acc1, baseline_source = load_baseline_acc1(args)

    if dynamic_acc1 is None or layer_map is None:
        print("Could not find dynamic-input accuracy and format counts.")
        return 1
    if baseline_acc1 is None:
        print(f"Could not find baseline accuracy for {args.baseline_format}.")
        return 1

    aggregated = aggregate_counts(layer_map)
    total_chunks = sum(aggregated.values())
    if total_chunks <= 0:
        print("Dynamic-input run was found, but it has no chunk counts.")
        return 1

    baseline_chunks = aggregated.get(args.baseline_format, 0)
    baseline_share = baseline_chunks / total_chunks
    family_chunks = sum(
        count for fmt, count in aggregated.items() if is_same_or_lower_mantissa_family(fmt, args.baseline_format)
    )
    family_share = family_chunks / total_chunks
    acc_gap = dynamic_acc1 - baseline_acc1

    dominant_baseline_layers, total_layers, layer_details = summarize_layers(layer_map, args.baseline_format)

    print(f"Model: {args.model}")
    print(f"Metric: {args.metric}")
    print(f"Baseline format: {args.baseline_format}")
    print(f"Dynamic acc1:  {dynamic_acc1:.3f}  ({dynamic_source})")
    print(f"Baseline acc1: {baseline_acc1:.3f}  ({baseline_source})")
    print(f"Acc1 gap:      {acc_gap:+.3f}")
    print()
    print(f"Total dynamic chunks counted: {total_chunks}")
    print(f"Exact {args.baseline_format} share: {baseline_share:.4%}")
    print(f"Same-family share (same exp, <= mantissa): {family_share:.4%}")
    print(f"Layers dominated by {args.baseline_format}: {dominant_baseline_layers}/{total_layers}")
    print()
    print("Top selected formats:")
    for fmt, count in aggregated.most_common(args.top_formats):
        print(f"  {fmt:>10}  {count:>12}  {count / total_chunks:.4%}")
    print()
    print("Layers with the highest non-baseline share:")
    for row in layer_details[: args.top_layers]:
        print(
            f"  {row['layer']:<40}  non-baseline={row['non_baseline_share']:.4%}  "
            f"{args.baseline_format}={row['baseline_share']:.4%}  dominant={row['dominant_format']}"
        )
    print()

    suspicious = (
        acc_gap >= args.min_acc_gap
        and baseline_share >= args.min_baseline_share
        and family_share >= args.min_family_share
    )
    if suspicious:
        print(
            "Flag: dynamic accuracy is much higher even though the chooser stayed almost entirely on the "
            "baseline format family. This is worth a runtime or logging sanity-check."
        )
    else:
        print("Flag: no automatic inconsistency triggered by the configured thresholds.")

    return 0


def main() -> None:
    raise SystemExit(print_summary(parse_args()))


if __name__ == "__main__":
    main()
