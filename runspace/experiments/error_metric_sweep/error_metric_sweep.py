import argparse
import copy
import csv
import gc
import json
import os
import sys

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


EXPERIMENT_TYPE = "error_metric_sweep"
WEIGHT_DT = "fp32"
ACTIVATION_DT = "dyn5b"
SIGNED_5BIT_FORMATS = [
    "fp5_e1m3",
    "fp5_e2m2",
    "fp5_e3m1",
    "fp5_e4m0",
]
UNSIGNED_INPUT_SOURCES = ["relu", "relu6", "softmax"]


def _parse_csv(value, default):
    if value is None:
        return list(default)
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    return parsed or list(default)


def _normalize_metric(metric):
    normalized = str(metric or "l2").strip().lower()
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
    return aliases.get(normalized, normalized)


def get_args():
    parser = argparse.ArgumentParser(
        description="Sweep dynamic activation quantizer error metrics for W32A5 ImageNet eval."
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
    parser.add_argument(
        "--metrics",
        type=str,
        default="l2,l1,linf,bias,huber,l0,logsum",
        help="Comma-separated per-chunk selection metrics to sweep.",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=0.0625,
        help="Huber transition point in the scaled domain (used by metric=huber).",
    )
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--no_plot", action="store_true", help="Skip rendering result plots.")
    return parser.parse_args()


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
    return [m for m in models if m.get("name")]


def _build_w32a5_runtime_config(args, model_name=None, weights=None):
    cfg = build_runtime_config(args, model_name=model_name, weights=weights)
    cfg.setdefault("adapter", {}).update(
        {
            "quantized_ops": ["all"],
            "build_quantized": True,
            "weight_quantization": False,
            "input_quantization": False,
            "output_quantization": False,
            "unsigned_input_sources": list(UNSIGNED_INPUT_SOURCES),
        }
    )
    cfg.setdefault("quantization", {}).update(
        {
            "format": SIGNED_5BIT_FORMATS[0],
            "input_format": SIGNED_5BIT_FORMATS[0],
            "mode": "chunk",
            "chunk_size": int(args.chunk_size),
            "weight_mode": "tensor",
            "weight_chunk_size": int(args.chunk_size),
            "weight_source": "fp32",
            "unsigned_input_sources": list(UNSIGNED_INPUT_SOURCES),
        }
    )
    return cfg


def _build_metric_input_quant_cfg(args, metric, model_name):
    cfg = build_dynamic_input_quant_cfg(
        metric=metric,
        chunk_size=args.chunk_size,
        candidate_formats=SIGNED_5BIT_FORMATS,
        restrict_post_relu_ufp=True,
        unsigned_input_sources=UNSIGNED_INPUT_SOURCES,
        dynamic_unsigned_input_candidates=True,
        model_name=model_name,
    )
    # Huber transition point (ignored by the other metrics).
    cfg["metric_param"] = float(getattr(args, "huber_delta", 0.0625))
    return cfg


def _config_json_for_run(config, input_quant_cfg, args, metric):
    cfg = copy.deepcopy(config)
    cfg.setdefault("evaluation", {})["input_quant"] = copy.deepcopy(input_quant_cfg)
    cfg["evaluation"]["dynamic_input_quant"] = copy.deepcopy(input_quant_cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "name": EXPERIMENT_TYPE,
            "type": EXPERIMENT_TYPE,
            "metric": metric,
            "weight_dt": WEIGHT_DT,
            "activation_dt": ACTIVATION_DT,
            "candidate_formats": list(SIGNED_5BIT_FORMATS),
        }
    )
    if args.limit_batches is not None:
        cfg.setdefault("dataset", {})["limit_batches"] = args.limit_batches
    return serialize_config(
        cfg,
        activation_fmt=ACTIVATION_DT,
        chunk_size=args.chunk_size,
        weight_source="fp32",
    )


def _serialize_ref_config(config, limit_batches):
    cfg = copy.deepcopy(config)
    cfg.setdefault("dataset", {})["limit_batches"] = limit_batches
    return serialize_config(cfg, weight_source="fp32")


def _metric_from_config_json(config_json):
    if not config_json:
        return None
    try:
        cfg = json.loads(config_json)
    except Exception:
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
    if not config_json:
        return None
    try:
        cfg = json.loads(config_json)
    except Exception:
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
    if not config_json:
        return {}
    try:
        cfg = json.loads(config_json)
    except Exception:
        return {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    input_quant = eval_cfg.get("input_quant") or eval_cfg.get("dynamic_input_quant") or {}
    return input_quant if isinstance(input_quant, dict) else {}


def _list_matches(actual, expected):
    return [str(item) for item in (actual or [])] == [str(item) for item in (expected or [])]


def _float_matches(actual, expected, atol=1e-12):
    try:
        return abs(float(actual) - float(expected)) <= atol
    except Exception:
        return False


def _run_config_matches(config_json, metric, limit_batches, chunk_size, metric_param, candidate_formats):
    if _metric_from_config_json(config_json) != _normalize_metric(metric):
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
    if logged_candidates is not None and not _list_matches(logged_candidates, candidate_formats):
        return False

    if _normalize_metric(metric) == "huber":
        logged_param = input_quant.get("metric_param")
        if logged_param is None or not _float_matches(logged_param, metric_param):
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
            ref_certainty = float(row["certainty"]) if row["certainty"] is not None else 0.0
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


def _metric_run_rows(
    db,
    model_name,
    metric,
    limit_batches=None,
    chunk_size=128,
    metric_param=0.0625,
    candidate_formats=None,
):
    runs = db.get_runs()
    if runs.empty:
        return []
    subset = runs[
        (runs["model_name"] == model_name)
        & (runs["experiment_type"] == EXPERIMENT_TYPE)
        & (runs["weight_dt"] == WEIGHT_DT)
        & (runs["activation_dt"] == ACTIVATION_DT)
        & (runs["status"] == "SUCCESS")
    ]
    if subset.empty:
        return []

    metric = _normalize_metric(metric)
    rows = []
    for _, row in subset.iterrows():
        config_json = row.get("config_json")
        if _run_config_matches(
            config_json,
            metric=metric,
            limit_batches=limit_batches,
            chunk_size=chunk_size,
            metric_param=metric_param,
            candidate_formats=candidate_formats or SIGNED_5BIT_FORMATS,
        ):
            rows.append(row)
    rows.sort(key=lambda row: int(row.get("id", 0) or 0))
    return rows


def _metric_run_exists(
    db,
    model_name,
    metric,
    limit_batches=None,
    chunk_size=128,
    metric_param=0.0625,
    candidate_formats=None,
):
    if not run_exists(db, model_name, EXPERIMENT_TYPE, WEIGHT_DT, ACTIVATION_DT):
        return False
    return bool(
        _metric_run_rows(
            db,
            model_name,
            metric,
            limit_batches=limit_batches,
            chunk_size=chunk_size,
            metric_param=metric_param,
            candidate_formats=candidate_formats or SIGNED_5BIT_FORMATS,
        )
    )


def _latest_metric_result(
    db,
    model_name,
    metric,
    limit_batches=None,
    chunk_size=128,
    metric_param=0.0625,
    candidate_formats=None,
):
    rows = _metric_run_rows(
        db,
        model_name,
        metric,
        limit_batches=limit_batches,
        chunk_size=chunk_size,
        metric_param=metric_param,
        candidate_formats=candidate_formats or SIGNED_5BIT_FORMATS,
    )
    return rows[-1] if rows else None


def _log_metric_run(
    runner,
    config,
    input_quant_cfg,
    args,
    model_name,
    metric,
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
        "metric": metric,
        "weight_dt": WEIGHT_DT,
        "activation_dt": ACTIVATION_DT,
        "ref_acc1": ref_acc1,
        "ref_acc5": ref_acc5,
        "ref_certainty": ref_certainty,
        "metrics": {
            "mse": norm_mse,
            "l1": norm_l1,
            "certainty": certainty,
        },
        "config_json": _config_json_for_run(config, input_quant_cfg, args, metric),
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
    metrics,
    ref_acc1,
    limit_batches=None,
    chunk_size=128,
    metric_param=0.0625,
    candidate_formats=None,
):
    rows = []
    for metric in metrics:
        row = _latest_metric_result(
            db,
            model_name,
            metric,
            limit_batches=limit_batches,
            chunk_size=chunk_size,
            metric_param=metric_param,
            candidate_formats=candidate_formats or SIGNED_5BIT_FORMATS,
        )
        if row is None:
            rows.append(
                {
                    "metric": metric,
                    "acc1": None,
                    "acc5": None,
                    "delta_acc1_vs_fp32": None,
                    "certainty": None,
                    "norm_mse": None,
                    "norm_l1": None,
                }
            )
            continue
        acc1 = float(row.get("acc1") or 0.0)
        rows.append(
            {
                "metric": metric,
                "acc1": acc1,
                "acc5": float(row.get("acc5") or 0.0),
                "delta_acc1_vs_fp32": acc1 - float(ref_acc1 or 0.0),
                "certainty": float(row.get("certainty") or 0.0),
                "norm_mse": float(row.get("mse") or 0.0),
                "norm_l1": float(row.get("l1") or 0.0),
            }
        )
    return rows


def _format_value(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _write_model_summary(output_dir, model_name, rows):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_summary.csv")
    txt_path = os.path.join(output_dir, f"{model_name}_summary.txt")
    fields = [
        "metric",
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

    complete = [row for row in rows if row["acc1"] is not None]
    winner = max(complete, key=lambda row: row["acc1"]) if complete else None
    lines = _summary_lines(model_name, rows, winner)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, txt_path, winner


def _summary_lines(model_name, rows, winner):
    lines = [
        f"SUMMARY: {model_name}",
        "metric | acc1 | acc5 | delta_acc1_vs_fp32 | certainty | norm_mse | norm_l1",
        "-" * 82,
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["metric"]),
                    _format_value(row["acc1"], 2),
                    _format_value(row["acc5"], 2),
                    _format_value(row["delta_acc1_vs_fp32"], 2),
                    _format_value(row["certainty"], 4),
                    _format_value(row["norm_mse"], 6),
                    _format_value(row["norm_l1"], 6),
                ]
            )
        )
    if winner is not None:
        lines.append(f"Winning metric: {winner['metric']} (acc1={winner['acc1']:.2f})")
    else:
        lines.append("Winning metric: none")
    return lines


def _write_combined_summary(output_dir, all_rows):
    if not all_rows:
        return None
    path = os.path.join(output_dir, "summary.csv")
    fields = ["model", "winning_metric"] + [
        "metric",
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


def process_single_model(args, device, metrics):
    model_name = args.model_name
    runner = Runner(device)
    db = runner._get_db()
    model_out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"ERROR METRIC SWEEP: {model_name}")
    print(f"Metrics: {metrics}")
    print(f"Candidates: {SIGNED_5BIT_FORMATS}")
    print(f"{'=' * 72}")

    ref_acc1, ref_acc5, ref_certainty = _get_or_run_fp32_ref_for_limit(
        runner, args, device, db, model_name
    )

    def config_builder(current_args):
        return _build_w32a5_runtime_config(
            current_args,
            model_name=current_args.model_name,
            weights=current_args.weights,
        )

    loader = build_loader(args, device, runner, config_builder=config_builder)

    for metric in metrics:
        metric = _normalize_metric(metric)
        if not args.force_rerun and _metric_run_exists(
            db,
            model_name,
            metric,
            limit_batches=args.limit_batches,
            chunk_size=args.chunk_size,
            metric_param=args.huber_delta,
            candidate_formats=SIGNED_5BIT_FORMATS,
        ):
            print(f"[{model_name}/{metric}] already complete; skipping.")
            continue

        print(f"\n[{model_name}/{metric}] running W32A5 dynamic input quantization")
        config = _build_w32a5_runtime_config(args, model_name=model_name, weights=args.weights)
        input_quant_cfg = _build_metric_input_quant_cfg(args, metric, model_name)
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
                desc=f"{model_name}/{metric}",
            )
            norm_mse = (input_quant_stats or {}).get("norm_mse", 0.0)
            norm_l1 = (input_quant_stats or {}).get("norm_l1", 0.0)
            print(
                f"[{model_name}/{metric}] Top1={acc1:.2f} Top5={acc5:.2f} "
                f"Certainty={certainty:.4f} NormMSE={norm_mse:.4e} NormL1={norm_l1:.4e}"
            )
            _log_metric_run(
                runner=runner,
                config=config,
                input_quant_cfg=input_quant_cfg,
                args=args,
                model_name=model_name,
                metric=metric,
                acc1=acc1,
                acc5=acc5,
                certainty=certainty,
                input_quant_stats=input_quant_stats,
                ref_acc1=ref_acc1,
                ref_acc5=ref_acc5,
                ref_certainty=ref_certainty,
            )
        except Exception as exc:
            print(f"[{model_name}/{metric}] ERROR: {exc}")
            config = _build_w32a5_runtime_config(args, model_name=model_name, weights=args.weights)
            input_quant_cfg = _build_metric_input_quant_cfg(args, metric, model_name)
            _log_metric_run(
                runner=runner,
                config=config,
                input_quant_cfg=input_quant_cfg,
                args=args,
                model_name=model_name,
                metric=metric,
                acc1=0.0,
                acc5=0.0,
                certainty=0.0,
                input_quant_stats={"mode": "dynamic", "metric": metric},
                ref_acc1=ref_acc1,
                ref_acc5=ref_acc5,
                ref_certainty=ref_certainty,
                status="ERROR",
            )
        finally:
            if "model" in locals():
                del model
            if "adapter" in locals():
                del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = _summary_rows_for_model(
        db,
        model_name,
        metrics,
        ref_acc1,
        limit_batches=args.limit_batches,
        chunk_size=args.chunk_size,
        metric_param=args.huber_delta,
        candidate_formats=SIGNED_5BIT_FORMATS,
    )
    _csv_path, txt_path, winner = _write_model_summary(args.output_dir, model_name, rows)
    print("")
    for line in _summary_lines(model_name, rows, winner):
        print(line)
    print(f"Summary written to {txt_path}")
    return rows, winner


def main():
    args = get_args()
    args.metrics = _parse_csv(args.metrics, ["l2", "l1", "linf"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = _load_models(args.models_file)
    if not models:
        models = [{"name": args.model_name, "weights": args.weights, "source": args.model_source}]

    combined_rows = []
    for entry in models:
        args.model_name = entry["name"]
        args.weights = entry.get("weights", "DEFAULT")
        args.model_source = entry.get("source") or "auto"
        rows, winner = process_single_model(args, device, args.metrics)
        winning_metric = winner["metric"] if winner is not None else None
        for row in rows:
            combined = {"model": args.model_name, "winning_metric": winning_metric}
            combined.update(row)
            combined_rows.append(combined)

    combined_path = _write_combined_summary(args.output_dir, combined_rows)
    if combined_path:
        print(f"\nCombined summary written to {combined_path}")

    if not args.no_plot:
        try:
            from plot_results import generate_plots
            generate_plots(args.output_dir)
        except Exception as exc:  # plotting must never fail the sweep
            print(f"[plot] skipped ({exc})")


if __name__ == "__main__":
    main()
