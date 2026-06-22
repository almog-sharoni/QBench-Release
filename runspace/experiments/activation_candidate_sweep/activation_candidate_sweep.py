import argparse
import copy
import csv
import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import yaml


os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner
from runspace.experiments.utils.common import (
    build_dynamic_input_quant_cfg,
    build_loader,
    build_runtime_config,
    get_or_run_fp32_ref,
    run_exists,
    run_inference,
    serialize_config,
)


EXPERIMENT_TYPE = "activation_candidate_sweep"
WEIGHT_DT = "fp32"
METRIC = "l2"
DEFAULT_BIT_WIDTHS = [8, 7, 6, 5, 4]
DEFAULT_EXP_CAPS = [None, 4, 3, 2, 1]
UNSIGNED_INPUT_SOURCES = ["relu", "relu6", "softmax"]


@dataclass(frozen=True)
class CandidateSweepSpec:
    bit_width: int
    exp_cap: Optional[int]
    exp_cap_label: str
    candidate_formats: List[str]
    activation_dt: str


def _parse_csv(value, default):
    if value is None:
        return list(default)
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    return parsed or list(default)


def _parse_int_csv(value, default):
    parsed = _parse_csv(value, default)
    return [int(item) for item in parsed]


def _parse_exp_caps(value, default=DEFAULT_EXP_CAPS):
    if value is None:
        return list(default)

    caps = []
    for raw in _parse_csv(value, []):
        item = str(raw).strip().lower()
        if item == "all":
            caps.append(None)
            continue
        if item.startswith("exp"):
            item = item[3:]
        caps.append(int(item))
    return caps or list(default)


def exp_cap_label(exp_cap):
    return "all" if exp_cap is None else f"exp{int(exp_cap)}"


def activation_dt_for_spec(bit_width, exp_cap):
    return f"dyn_a{int(bit_width)}_{exp_cap_label(exp_cap)}_{METRIC}"


def candidate_formats_for_bit_width(bit_width, exp_cap=None):
    """
    Return signed fp candidates for a total activation width.

    Signed formats spend one bit on sign, so m = bit_width - 1 - exp_bits.
    Formats with m=0 are intentionally excluded.
    """
    bit_width = int(bit_width)
    if bit_width < 3:
        return []

    candidates = []
    max_exp = bit_width - 2
    if exp_cap is not None:
        max_exp = min(max_exp, int(exp_cap))

    for exp_bits in range(1, max_exp + 1):
        mant_bits = bit_width - 1 - exp_bits
        if mant_bits <= 0:
            continue
        candidates.append(f"fp{bit_width}_e{exp_bits}m{mant_bits}")
    return candidates


def build_sweep_specs(bit_widths, exp_caps, skip_duplicate_pools=True):
    specs = []
    for bit_width in bit_widths:
        seen_for_width = set()
        for exp_cap in exp_caps:
            candidates = candidate_formats_for_bit_width(bit_width, exp_cap)
            if not candidates:
                continue

            key = tuple(candidates)
            if skip_duplicate_pools and key in seen_for_width:
                continue
            seen_for_width.add(key)

            specs.append(
                CandidateSweepSpec(
                    bit_width=int(bit_width),
                    exp_cap=None if exp_cap is None else int(exp_cap),
                    exp_cap_label=exp_cap_label(exp_cap),
                    candidate_formats=candidates,
                    activation_dt=activation_dt_for_spec(bit_width, exp_cap),
                )
            )
    return specs


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep dynamic activation candidate pools for fp32-weight ImageNet "
            "evaluation using L2 selection."
        )
    )
    parser.add_argument("--models_file", type=str, default="runspace/inputs/models.yaml")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--model_source", type=str, default="auto")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit_batches", type=int, default=-1)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--bit_widths", type=str, default="8,7,6,5,4")
    parser.add_argument("--exp_caps", type=str, default="all,4,3,2,1")
    parser.add_argument(
        "--unsigned_input_sources",
        type=str,
        default=",".join(UNSIGNED_INPUT_SOURCES),
        help="Comma-separated activation sources that should use derived unsigned candidates.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--no_plot", action="store_true", help="Skip result plots.")
    args = parser.parse_args()
    args.bit_widths = _parse_int_csv(args.bit_widths, DEFAULT_BIT_WIDTHS)
    args.exp_caps = _parse_exp_caps(args.exp_caps)
    args.unsigned_input_sources = _parse_csv(
        args.unsigned_input_sources, UNSIGNED_INPUT_SOURCES
    )
    return args


def _load_models(models_file):
    if not models_file:
        return []
    with open(models_file, "r") as f:
        models_cfg = yaml.safe_load(f)

    entries = models_cfg if isinstance(models_cfg, list) else models_cfg.get("models", [])
    models = []
    for entry in entries:
        if isinstance(entry, dict):
            models.append(
                {
                    "name": entry.get("name"),
                    "weights": entry.get("weights", "DEFAULT"),
                    "source": entry.get("source"),
                }
            )
        else:
            models.append({"name": str(entry), "weights": "DEFAULT", "source": None})
    return [model for model in models if model.get("name")]


def _build_w32_dynamic_runtime_config(
    args,
    model_name=None,
    weights=None,
    candidate_formats=None,
):
    candidates = list(candidate_formats or candidate_formats_for_bit_width(8))
    default_format = candidates[0] if candidates else "fp8_e4m3"
    cfg = build_runtime_config(args, model_name=model_name, weights=weights)
    cfg.setdefault("adapter", {}).update(
        {
            "quantized_ops": ["all"],
            "build_quantized": True,
            "weight_quantization": False,
            "input_quantization": False,
            "output_quantization": False,
            "unsigned_input_sources": list(args.unsigned_input_sources),
        }
    )
    cfg.setdefault("quantization", {}).update(
        {
            "format": default_format,
            "input_format": default_format,
            "mode": "chunk",
            "chunk_size": int(args.chunk_size),
            "weight_mode": "tensor",
            "weight_chunk_size": int(args.chunk_size),
            "weight_source": "fp32",
            "unsigned_input_sources": list(args.unsigned_input_sources),
        }
    )
    return cfg


def _build_sweep_input_quant_cfg(args, spec, model_name):
    return build_dynamic_input_quant_cfg(
        metric=METRIC,
        chunk_size=args.chunk_size,
        candidate_formats=spec.candidate_formats,
        restrict_post_relu_ufp=False,
        unsigned_input_sources=args.unsigned_input_sources,
        dynamic_unsigned_input_candidates=True,
        model_name=model_name,
    )


def _config_json_for_run(config, input_quant_cfg, args, spec):
    cfg = copy.deepcopy(config)
    cfg.setdefault("evaluation", {})["input_quant"] = copy.deepcopy(input_quant_cfg)
    cfg["evaluation"]["dynamic_input_quant"] = copy.deepcopy(input_quant_cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "name": EXPERIMENT_TYPE,
            "type": EXPERIMENT_TYPE,
            "metric": METRIC,
            "weight_dt": WEIGHT_DT,
            "activation_dt": spec.activation_dt,
            "bit_width": int(spec.bit_width),
            "exp_cap": spec.exp_cap,
            "exp_cap_label": spec.exp_cap_label,
            "candidate_formats": list(spec.candidate_formats),
        }
    )
    cfg.setdefault("dataset", {})["limit_batches"] = args.limit_batches
    return serialize_config(
        cfg,
        activation_fmt=spec.activation_dt,
        chunk_size=args.chunk_size,
        weight_source="fp32",
    )


def _serialize_ref_config(config, limit_batches):
    cfg = copy.deepcopy(config)
    cfg.setdefault("dataset", {})["limit_batches"] = limit_batches
    return serialize_config(cfg, weight_source="fp32")


def _safe_json_load(config_json):
    if not config_json:
        return None
    if isinstance(config_json, dict):
        return config_json
    try:
        return json.loads(config_json)
    except Exception:
        return None


def _normalize_metric(metric):
    aliases = {
        "mse": "l2",
        "mae": "l1",
        "max": "linf",
        "chebyshev": "linf",
        "sad": "l1",
        "mean_bias": "bias",
        "count": "l0",
        "logl1": "logsum",
    }
    normalized = str(metric or METRIC).strip().lower()
    return aliases.get(normalized, normalized)


def _metric_from_config_json(config_json):
    cfg = _safe_json_load(config_json)
    if not isinstance(cfg, dict):
        return None
    for path in (
        ("evaluation", "input_quant", "metric"),
        ("evaluation", "dynamic_input_quant", "metric"),
        ("experiment", "metric"),
    ):
        node = cfg
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if node:
            return _normalize_metric(node)
    return None


def _limit_from_config_json(config_json):
    cfg = _safe_json_load(config_json)
    if not isinstance(cfg, dict):
        return None
    return cfg.get("dataset", {}).get("limit_batches")


def _limit_matches(config_json, limit_batches):
    logged = _limit_from_config_json(config_json)
    if logged is None:
        return limit_batches in (None, -1)
    try:
        return int(logged) == int(limit_batches)
    except Exception:
        return str(logged) == str(limit_batches)


def _input_quant_from_config_json(config_json):
    cfg = _safe_json_load(config_json)
    if not isinstance(cfg, dict):
        return {}
    eval_cfg = cfg.get("evaluation", {})
    input_quant = eval_cfg.get("input_quant") or eval_cfg.get("dynamic_input_quant") or {}
    return input_quant if isinstance(input_quant, dict) else {}


def _experiment_from_config_json(config_json):
    cfg = _safe_json_load(config_json)
    if not isinstance(cfg, dict):
        return {}
    experiment = cfg.get("experiment", {})
    return experiment if isinstance(experiment, dict) else {}


def _list_matches(actual, expected):
    return [str(item) for item in (actual or [])] == [str(item) for item in (expected or [])]


def _run_config_matches(config_json, spec, limit_batches, chunk_size):
    if _metric_from_config_json(config_json) != METRIC:
        return False
    if not _limit_matches(config_json, limit_batches):
        return False

    input_quant = _input_quant_from_config_json(config_json)
    logged_chunk_size = input_quant.get("chunk_size")
    if logged_chunk_size is not None:
        try:
            if int(logged_chunk_size) != int(chunk_size):
                return False
        except Exception:
            return False

    logged_candidates = input_quant.get("candidate_formats")
    if not _list_matches(logged_candidates, spec.candidate_formats):
        return False

    experiment = _experiment_from_config_json(config_json)
    try:
        if int(experiment.get("bit_width")) != int(spec.bit_width):
            return False
    except Exception:
        return False
    if str(experiment.get("exp_cap_label")) != str(spec.exp_cap_label):
        return False

    return True


def _limit_matches_strict(config_json, limit_batches):
    logged = _limit_from_config_json(config_json)
    if logged is None:
        return False
    try:
        return int(logged) == int(limit_batches)
    except Exception:
        return str(logged) == str(limit_batches)


def _latest_fp32_ref_for_limit(db, model_name, limit_batches):
    runs = db.get_runs()
    if runs.empty:
        return None
    subset = runs[
        (runs["model_name"] == model_name)
        & (runs["weight_dt"] == "fp32")
        & (runs["activation_dt"] == "fp32")
        & (runs["status"] == "SUCCESS")
        & (runs["acc1"].notna())
    ]
    if subset.empty:
        return None
    rows = [
        row
        for _, row in subset.iterrows()
        if _limit_matches_strict(row.get("config_json"), limit_batches)
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: int(row.get("id", 0) or 0))
    return rows[-1]


def _get_or_run_fp32_ref_for_limit(runner, args, device, db, model_name):
    if not args.force_rerun:
        row = _latest_fp32_ref_for_limit(db, model_name, args.limit_batches)
        if row is not None:
            ref_acc1 = float(row["acc1"])
            ref_acc5 = float(row["acc5"])
            ref_certainty = (
                float(row["certainty"]) if row["certainty"] is not None else 0.0
            )
            print(
                f"[FP32 ref] Found matching limit_batches={args.limit_batches}: "
                f"Top1={ref_acc1:.2f}%, Top5={ref_acc5:.2f}%"
            )
            return ref_acc1, ref_acc5, ref_certainty

    ref_args = copy.copy(args)
    ref_args.force_rerun = True
    return get_or_run_fp32_ref(
        runner,
        ref_args,
        device,
        db,
        model_name,
        experiment_name=EXPERIMENT_TYPE,
        config_builder=build_runtime_config,
        config_json_builder=lambda cfg: _serialize_ref_config(cfg, args.limit_batches),
        respect_force_rerun=True,
    )


def _sweep_run_rows(db, model_name, spec, limit_batches=None, chunk_size=128):
    runs = db.get_runs()
    if runs.empty:
        return []
    subset = runs[
        (runs["model_name"] == model_name)
        & (runs["experiment_type"] == EXPERIMENT_TYPE)
        & (runs["weight_dt"] == WEIGHT_DT)
        & (runs["activation_dt"] == spec.activation_dt)
        & (runs["status"] == "SUCCESS")
    ]
    if subset.empty:
        return []

    rows = []
    for _, row in subset.iterrows():
        if _run_config_matches(
            row.get("config_json"),
            spec=spec,
            limit_batches=limit_batches,
            chunk_size=chunk_size,
        ):
            rows.append(row)
    rows.sort(key=lambda row: int(row.get("id", 0) or 0))
    return rows


def _sweep_run_exists(db, model_name, spec, limit_batches=None, chunk_size=128):
    if not run_exists(db, model_name, EXPERIMENT_TYPE, WEIGHT_DT, spec.activation_dt):
        return False
    return bool(
        _sweep_run_rows(
            db,
            model_name,
            spec,
            limit_batches=limit_batches,
            chunk_size=chunk_size,
        )
    )


def _latest_sweep_result(db, model_name, spec, limit_batches=None, chunk_size=128):
    rows = _sweep_run_rows(
        db,
        model_name,
        spec,
        limit_batches=limit_batches,
        chunk_size=chunk_size,
    )
    return rows[-1] if rows else None


def _log_sweep_run(
    runner,
    config,
    input_quant_cfg,
    args,
    model_name,
    spec,
    acc1,
    acc5,
    certainty,
    input_quant_stats,
    ref_acc1,
    ref_acc5,
    ref_certainty,
    status="SUCCESS",
):
    norm_mse = (input_quant_stats or {}).get("norm_mse")
    norm_l1 = (input_quant_stats or {}).get("norm_l1")
    log_cfg = copy.deepcopy(config)
    log_cfg["experiment"] = {
        "name": EXPERIMENT_TYPE,
        "type": EXPERIMENT_TYPE,
        "metric": METRIC,
        "weight_dt": WEIGHT_DT,
        "activation_dt": spec.activation_dt,
        "bit_width": int(spec.bit_width),
        "exp_cap": spec.exp_cap,
        "exp_cap_label": spec.exp_cap_label,
        "candidate_formats": list(spec.candidate_formats),
        "ref_acc1": ref_acc1,
        "ref_acc5": ref_acc5,
        "ref_certainty": ref_certainty,
        "metrics": {
            "mse": norm_mse,
            "l1": norm_l1,
            "certainty": certainty,
        },
        "config_json": _config_json_for_run(config, input_quant_cfg, args, spec),
    }
    runner.log_experiment_result(
        config=log_cfg,
        result={
            "model_name": model_name,
            "status": status,
            "acc1": acc1,
            "acc5": acc5,
            "certainty": certainty,
            "input_quant": input_quant_stats or {},
        },
    )


def _summary_rows_for_model(
    db,
    model_name,
    specs,
    ref_acc1,
    limit_batches=None,
    chunk_size=128,
):
    rows = []
    for spec in specs:
        row = _latest_sweep_result(
            db,
            model_name,
            spec,
            limit_batches=limit_batches,
            chunk_size=chunk_size,
        )
        base = {
            "bit_width": int(spec.bit_width),
            "exp_cap": spec.exp_cap_label,
            "activation_dt": spec.activation_dt,
            "candidate_count": len(spec.candidate_formats),
            "candidate_formats": ",".join(spec.candidate_formats),
            "format_counts_json": None,
        }
        if row is None:
            base.update(
                {
                    "acc1": None,
                    "acc5": None,
                    "delta_acc1_vs_fp32": None,
                    "certainty": None,
                    "norm_mse": None,
                    "norm_l1": None,
                }
            )
            rows.append(base)
            continue

        acc1 = float(row.get("acc1") or 0.0)
        format_counts = _format_counts_from_input_map(row.get("input_map_json"))
        base.update(
            {
                "acc1": acc1,
                "acc5": float(row.get("acc5") or 0.0),
                "delta_acc1_vs_fp32": acc1 - float(ref_acc1 or 0.0),
                "certainty": float(row.get("certainty") or 0.0),
                "norm_mse": float(row.get("mse") or 0.0),
                "norm_l1": float(row.get("l1") or 0.0),
                "format_counts_json": (
                    json.dumps(format_counts, sort_keys=True) if format_counts else None
                ),
            }
        )
        rows.append(base)
    return rows


def _format_value(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _format_sort_key(fmt):
    text = str(fmt or "").strip().lower()
    match = re.match(r"^(uefp|ufp|efp|fp)(\d+)(?:_e(\d+)(?:m(\d+))?)?", text)
    if not match:
        return (999, 999, 999, 999, text)

    prefix, bits, exp, mant = match.groups()
    prefix_order = {"fp": 0, "efp": 1, "ufp": 2, "uefp": 3}.get(prefix, 9)
    return (
        int(bits),
        prefix_order,
        int(exp) if exp is not None else -1,
        int(mant) if mant is not None else -1,
        text,
    )


def _sort_quant_formats(formats):
    return sorted(
        {str(fmt) for fmt in formats if str(fmt or "").strip()},
        key=_format_sort_key,
    )


def _exp_cap_sort_key(exp_cap):
    text = str(exp_cap or "").strip().lower()
    if text == "all":
        return (0, 0, text)
    if text.startswith("exp"):
        try:
            return (1, -int(text[3:]), text)
        except Exception:
            pass
    return (2, 0, text)


def _add_format_count(totals: Dict[str, int], fmt, count=1):
    if fmt is None:
        return
    key = str(fmt).strip()
    if not key:
        return
    try:
        count_i = int(count)
    except Exception:
        return
    if count_i <= 0:
        return
    totals[key] = totals.get(key, 0) + count_i


def _format_counts_from_input_map(input_map_json):
    input_map = _safe_json_load(input_map_json)
    if not isinstance(input_map, dict):
        return {}

    totals = {}
    for entry in input_map.values():
        if isinstance(entry, dict):
            raw_counts = entry.get("format_counts")
            if isinstance(raw_counts, dict) and raw_counts:
                for fmt, count in raw_counts.items():
                    _add_format_count(totals, fmt, count)
                continue

            fmt_spec = entry.get("format")
            if isinstance(fmt_spec, list):
                for fmt in fmt_spec:
                    _add_format_count(totals, fmt)
            else:
                _add_format_count(totals, fmt_spec)
            continue

        if isinstance(entry, list):
            for fmt in entry:
                _add_format_count(totals, fmt)
        else:
            _add_format_count(totals, entry)

    return totals


def _format_counts_from_summary_row(row):
    counts = _safe_json_load(row.get("format_counts_json"))
    if not isinstance(counts, dict):
        return {}

    totals = {}
    for fmt, count in counts.items():
        _add_format_count(totals, fmt, count)
    return totals


def _format_choice_counts_by_exp_cap(rows):
    series_counts = {}
    for row in rows:
        counts = _format_counts_from_summary_row(row)
        if not counts:
            continue

        exp_cap = str(row.get("exp_cap") or "unknown")
        target = series_counts.setdefault(exp_cap, {})
        for fmt, count in counts.items():
            _add_format_count(target, fmt, count)

    return {
        exp_cap: dict(sorted(counts.items(), key=lambda item: _format_sort_key(item[0])))
        for exp_cap, counts in sorted(
            series_counts.items(), key=lambda item: _exp_cap_sort_key(item[0])
        )
    }


def _write_model_summary(output_dir, model_name, rows):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_summary.csv")
    txt_path = os.path.join(output_dir, f"{model_name}_summary.txt")
    fields = [
        "bit_width",
        "exp_cap",
        "activation_dt",
        "candidate_count",
        "candidate_formats",
        "format_counts_json",
        "acc1",
        "acc5",
        "delta_acc1_vs_fp32",
        "certainty",
        "norm_mse",
        "norm_l1",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"SUMMARY: {model_name}",
        "bits | exp_cap | candidates | acc1 | acc5 | delta_acc1_vs_fp32 | norm_mse",
        "-" * 86,
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["bit_width"]),
                    str(row["exp_cap"]),
                    str(row["candidate_count"]),
                    _format_value(row["acc1"], 2),
                    _format_value(row["acc5"], 2),
                    _format_value(row["delta_acc1_vs_fp32"], 2),
                    _format_value(row["norm_mse"], 6),
                ]
            )
        )
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, txt_path


def _write_combined_summary(output_dir, all_rows):
    if not all_rows:
        return None
    path = os.path.join(output_dir, "summary.csv")
    fields = ["model"] + [
        "bit_width",
        "exp_cap",
        "activation_dt",
        "candidate_count",
        "candidate_formats",
        "format_counts_json",
        "acc1",
        "acc5",
        "delta_acc1_vs_fp32",
        "certainty",
        "norm_mse",
        "norm_l1",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    return path


def _plot_model_summary(output_dir, model_name, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped for {model_name} ({exc})")
        return []

    paths = []
    series = {}
    for row in rows:
        if row.get("acc1") is None:
            continue
        series.setdefault(row["exp_cap"], []).append(row)

    if not series:
        return []

    plot_specs = [
        ("acc1", "Top-1 Accuracy", f"{model_name}_accuracy_vs_bits.png"),
        ("norm_mse", "Normalized L2 Error", f"{model_name}_norm_mse_vs_bits.png"),
    ]
    for key, ylabel, filename in plot_specs:
        plt.figure(figsize=(8, 5))
        has_values = False
        for exp_cap, exp_rows in sorted(
            series.items(), key=lambda item: _exp_cap_sort_key(item[0])
        ):
            points = [
                (int(row["bit_width"]), float(row[key]))
                for row in exp_rows
                if row.get(key) is not None
            ]
            if not points:
                continue
            points.sort()
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            plt.plot(xs, ys, marker="o", label=str(exp_cap))
            has_values = True

        if not has_values:
            plt.close()
            continue
        plt.xlabel("Activation Bit Width")
        plt.ylabel(ylabel)
        plt.title(f"{model_name}: {ylabel} by Candidate Pool")
        plt.grid(alpha=0.3)
        plt.legend(title="Exp Cap")
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths


def _plot_format_choice_counts(output_dir, rows, filename, title):
    path = os.path.join(output_dir, filename)
    try:
        from runspace.experiments.activation_candidate_sweep.plot_format_choices import (
            plot_format_choice_counts_from_rows,
        )
    except Exception as exc:
        print(f"[plot] skipped format choices ({exc})")
        return None

    return plot_format_choice_counts_from_rows(
        rows=rows,
        output_path=path,
        title=title,
    )


def _plot_model_format_choices(output_dir, model_name, rows):
    return _plot_format_choice_counts(
        output_dir=output_dir,
        rows=rows,
        filename=f"{model_name}_format_choices_by_exp_cap.png",
        title=f"{model_name}: Dynamic Input Format Selections by Exp Cap",
    )


def _plot_combined_format_choices(output_dir, rows):
    return _plot_format_choice_counts(
        output_dir=output_dir,
        rows=rows,
        filename="format_choices_by_exp_cap.png",
        title="Dynamic Input Format Selections by Exp Cap",
    )


def _print_specs(specs: Iterable[CandidateSweepSpec]):
    print("Candidate pools:")
    for spec in specs:
        print(
            f"  {spec.activation_dt}: "
            f"{len(spec.candidate_formats)} candidates = {spec.candidate_formats}"
        )


def process_single_model(args, device, specs):
    model_name = args.model_name
    runner = Runner(device)
    db = runner._get_db()
    model_out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"ACTIVATION CANDIDATE SWEEP: {model_name}")
    print(f"Metric: {METRIC.upper()}")
    print(f"{'=' * 72}")
    _print_specs(specs)

    ref_acc1, ref_acc5, ref_certainty = _get_or_run_fp32_ref_for_limit(
        runner, args, device, db, model_name
    )

    first_spec = specs[0]

    def config_builder(current_args):
        return _build_w32_dynamic_runtime_config(
            current_args,
            model_name=current_args.model_name,
            weights=current_args.weights,
            candidate_formats=first_spec.candidate_formats,
        )

    loader = build_loader(args, device, runner, config_builder=config_builder)

    try:
        for spec in specs:
            if not args.force_rerun and _sweep_run_exists(
                db,
                model_name,
                spec,
                limit_batches=args.limit_batches,
                chunk_size=args.chunk_size,
            ):
                print(f"[{model_name}/{spec.activation_dt}] already complete; skipping.")
                continue

            print(
                f"\n[{model_name}/{spec.activation_dt}] running W32A{spec.bit_width} "
                f"dynamic activation quantization"
            )
            print(f"Candidates: {spec.candidate_formats}")
            config = _build_w32_dynamic_runtime_config(
                args,
                model_name=model_name,
                weights=args.weights,
                candidate_formats=spec.candidate_formats,
            )
            input_quant_cfg = _build_sweep_input_quant_cfg(args, spec, model_name)
            model = None
            adapter = None
            try:
                model, adapter, _ = runner.prepare_model_with_materialized_weights(
                    config=config,
                    output_dir=model_out_dir,
                )
                acc1, acc5, certainty, input_quant_stats = run_inference(
                    runner,
                    model,
                    adapter,
                    loader,
                    args,
                    input_quant_cfg=input_quant_cfg,
                    desc=f"{model_name}/{spec.activation_dt}",
                )
                norm_mse = (input_quant_stats or {}).get("norm_mse", 0.0)
                norm_l1 = (input_quant_stats or {}).get("norm_l1", 0.0)
                print(
                    f"[{model_name}/{spec.activation_dt}] Top1={acc1:.2f} "
                    f"Top5={acc5:.2f} Certainty={certainty:.4f} "
                    f"NormMSE={norm_mse:.4e} NormL1={norm_l1:.4e}"
                )
                _log_sweep_run(
                    runner=runner,
                    config=config,
                    input_quant_cfg=input_quant_cfg,
                    args=args,
                    model_name=model_name,
                    spec=spec,
                    acc1=acc1,
                    acc5=acc5,
                    certainty=certainty,
                    input_quant_stats=input_quant_stats,
                    ref_acc1=ref_acc1,
                    ref_acc5=ref_acc5,
                    ref_certainty=ref_certainty,
                )
            except Exception as exc:
                print(f"[{model_name}/{spec.activation_dt}] ERROR: {exc}")
                _log_sweep_run(
                    runner=runner,
                    config=config,
                    input_quant_cfg=input_quant_cfg,
                    args=args,
                    model_name=model_name,
                    spec=spec,
                    acc1=0.0,
                    acc5=0.0,
                    certainty=0.0,
                    input_quant_stats={"mode": "dynamic", "metric": METRIC},
                    ref_acc1=ref_acc1,
                    ref_acc5=ref_acc5,
                    ref_certainty=ref_certainty,
                    status="ERROR",
                )
            finally:
                if model is not None:
                    del model
                if adapter is not None:
                    del adapter
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        runner._shutdown_dataloader_workers(loader)
        del loader

    rows = _summary_rows_for_model(
        db,
        model_name,
        specs,
        ref_acc1,
        limit_batches=args.limit_batches,
        chunk_size=args.chunk_size,
    )
    _csv_path, txt_path = _write_model_summary(args.output_dir, model_name, rows)
    print(f"Summary written to {txt_path}")
    if not args.no_plot:
        for path in _plot_model_summary(args.output_dir, model_name, rows):
            print(f"Plot written to {path}")
        path = _plot_model_format_choices(args.output_dir, model_name, rows)
        if path:
            print(f"Plot written to {path}")
    return rows


def main():
    args = get_args()
    specs = build_sweep_specs(args.bit_widths, args.exp_caps)
    if not specs:
        raise SystemExit("No candidate pools to run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Unsigned input sources: {args.unsigned_input_sources}")

    models = _load_models(args.models_file)
    if not models:
        models = [
            {
                "name": args.model_name,
                "weights": args.weights,
                "source": args.model_source,
            }
        ]

    os.makedirs(args.output_dir, exist_ok=True)
    combined_rows = []
    for entry in models:
        args.model_name = entry["name"]
        args.weights = entry.get("weights", "DEFAULT")
        args.model_source = entry.get("source") or "auto"
        rows = process_single_model(args, device, specs)
        for row in rows:
            combined = {"model": args.model_name}
            combined.update(row)
            combined_rows.append(combined)

    combined_path = _write_combined_summary(args.output_dir, combined_rows)
    if combined_path:
        print(f"\nCombined summary written to {combined_path}")
    if not args.no_plot:
        path = _plot_combined_format_choices(args.output_dir, combined_rows)
        if path:
            print(f"Combined plot written to {path}")


if __name__ == "__main__":
    main()
