#!/usr/bin/env python3
"""
Run an FP32 reference evaluation and stamp matching DB rows with its metrics.

Example:
    python runspace/tools/run_fp32_ref_and_update_db.py \
        --model_name vit_b_16 \
        --weights DEFAULT \
        --dataset_path /data/imagenet/val \
        --batch_size 128 \
        --limit_batches 10

For a full run, omit --limit_batches or set it to -1.
"""

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner
from runspace.src.database.handler import RunDatabase


DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "runspace/database/runs.db")
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "runspace/outputs/fp32_refs")


def positive_float(value) -> bool:
    try:
        return float(value) > 0.0
    except Exception:
        return False


def build_fp32_config(args, model_name: str, weights: str) -> Dict:
    dataset_cfg = {
        "name": args.dataset_name,
        "path": args.dataset_path,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    eval_cfg = {"mode": "evaluate"}
    if args.limit_batches is not None and args.limit_batches >= 0:
        eval_cfg["max_batches"] = args.limit_batches

    return {
        "model": {
            "name": model_name,
            "weights": weights,
        },
        "adapter": {
            "type": "generic",
            "quantized_ops": [],
            "input_quantization": False,
            "weight_quantization": False,
            "quantize_first_layer": False,
            "input_size": args.input_size,
        },
        "quantization": {
            "format": "fp32",
            "input_format": "fp32",
        },
        "dataset": dataset_cfg,
        "evaluation": eval_cfg,
        "experiment": {
            "name": "fp32_reference_refresh",
            "type": "fp32_ref",
            "weight_dt": "fp32",
            "activation_dt": "fp32",
            "resolve_ref_from_db": False,
        },
        "output_name": f"{model_name}_fp32_ref",
    }


def find_existing_ref(db_path: str, model_name: str) -> Optional[Dict]:
    query = """
        SELECT *
        FROM runs
        WHERE model_name = ?
          AND weight_dt = 'fp32'
          AND activation_dt = 'fp32'
          AND status = 'SUCCESS'
          AND acc1 IS NOT NULL
          AND acc1 > 0.0
        ORDER BY
          CASE WHEN experiment_type = 'fp32_ref' THEN 1 ELSE 0 END DESC,
          id DESC
        LIMIT 1
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query, (model_name,)).fetchone()
        return dict(row) if row is not None else None


def latest_ref_row(db_path: str, model_name: str) -> Optional[Dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT *
            FROM runs
            WHERE model_name = ?
              AND experiment_type = 'fp32_ref'
              AND weight_dt = 'fp32'
              AND activation_dt = 'fp32'
              AND status = 'SUCCESS'
            ORDER BY id DESC
            LIMIT 1
            """,
            (model_name,),
        ).fetchone()
        return dict(row) if row is not None else None


def run_and_log_fp32_ref(args, model_name: str, weights: str) -> Dict:
    config = build_fp32_config(args, model_name, weights)
    runner = Runner()

    print(f"\n[FP32 ref] Running {model_name} weights={weights}")
    result = runner.run_single(config, output_root=args.output_root)
    status = str(result.get("status", "SUCCESS"))
    acc1 = float(result.get("acc1", 0.0) or 0.0)
    acc5 = float(result.get("acc5", 0.0) or 0.0)
    certainty = float(result.get("certainty", 0.0) or 0.0)

    if status not in ("SUCCESS", "NO_QUANT") or not positive_float(acc1):
        raise RuntimeError(
            f"FP32 reference run failed for {model_name}: "
            f"status={status}, acc1={acc1}, acc5={acc5}"
        )

    log_config = dict(config)
    log_config["experiment"] = dict(config["experiment"])
    log_config["experiment"].update(
        {
            "ref_acc1": acc1,
            "ref_acc5": acc5,
            "ref_certainty": certainty,
            "metrics": {"certainty": certainty},
            "config_json": json.dumps(config, default=str),
        }
    )
    runner.log_experiment_result(
        config=log_config,
        result={
            "model_name": model_name,
            "status": "SUCCESS",
            "acc1": acc1,
            "acc5": acc5,
            "certainty": certainty,
        },
        db_path=args.db_path,
    )

    ref_row = latest_ref_row(args.db_path, model_name)
    if ref_row is None:
        raise RuntimeError(f"Logged FP32 reference for {model_name}, but could not read it back.")

    print(
        f"[FP32 ref] Logged id={ref_row['id']} "
        f"Top1={acc1:.4f}% Top5={acc5:.4f}% Certainty={certainty:.6f}"
    )
    return ref_row


def update_reference_metrics(
    db_path: str,
    model_name: str,
    ref_acc1: float,
    ref_acc5: float,
    ref_certainty: float,
    ref_run_id: Optional[int],
    only_missing: bool,
) -> int:
    where = ["model_name = ?"]
    params: List = [model_name]

    if ref_run_id is not None:
        where.append("id != ?")
        params.append(int(ref_run_id))

    if only_missing:
        where.append("(ref_acc1 IS NULL OR ref_acc1 <= 0.0 OR ref_acc5 IS NULL OR ref_acc5 <= 0.0)")

    query = f"""
        UPDATE runs
        SET ref_acc1 = ?, ref_acc5 = ?, ref_certainty = ?
        WHERE {' AND '.join(where)}
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, [ref_acc1, ref_acc5, ref_certainty, *params])
        conn.commit()
        return int(cursor.rowcount or 0)


def distinct_models_from_db(db_path: str, only_missing: bool) -> List[str]:
    missing_filter = ""
    if only_missing:
        missing_filter = "AND (ref_acc1 IS NULL OR ref_acc1 <= 0.0 OR ref_acc5 IS NULL OR ref_acc5 <= 0.0)"

    query = f"""
        SELECT DISTINCT model_name
        FROM runs
        WHERE model_name IS NOT NULL
          AND model_name != ''
          {missing_filter}
        ORDER BY model_name
    """
    with sqlite3.connect(db_path) as conn:
        return [str(row[0]) for row in conn.execute(query).fetchall()]


def infer_weights_for_model(db_path: str, model_name: str, fallback: str) -> str:
    query = """
        SELECT config_json
        FROM runs
        WHERE model_name = ?
          AND config_json IS NOT NULL
          AND config_json != ''
        ORDER BY id DESC
        LIMIT 20
    """
    with sqlite3.connect(db_path) as conn:
        for (raw_config,) in conn.execute(query, (model_name,)).fetchall():
            try:
                cfg = json.loads(raw_config)
            except Exception:
                continue
            weights = (cfg.get("model") or {}).get("weights")
            if weights:
                return str(weights)
    return fallback


def selected_models(args) -> List[Tuple[str, str]]:
    if args.all_models_from_db:
        model_names = distinct_models_from_db(args.db_path, only_missing=args.only_missing)
        return [
            (model_name, infer_weights_for_model(args.db_path, model_name, args.weights))
            for model_name in model_names
        ]

    if not args.model_name:
        raise SystemExit("Provide --model_name or use --all_models_from_db.")

    return [(args.model_name, args.weights)]


def process_model(args, model_name: str, weights: str) -> None:
    db = RunDatabase(db_path=args.db_path)
    _ = db  # Initializes/migrates schema before direct sqlite updates.

    ref_row = None
    if args.reuse_existing_ref:
        ref_row = find_existing_ref(args.db_path, model_name)
        if ref_row is not None:
            print(
                f"\n[FP32 ref] Reusing id={ref_row['id']} for {model_name}: "
                f"Top1={float(ref_row['acc1']):.4f}% Top5={float(ref_row['acc5']):.4f}%"
            )

    if ref_row is None:
        ref_row = run_and_log_fp32_ref(args, model_name, weights)

    ref_acc1 = float(ref_row.get("acc1", 0.0) or 0.0)
    ref_acc5 = float(ref_row.get("acc5", 0.0) or 0.0)
    ref_certainty = float(ref_row.get("certainty", 0.0) or 0.0)
    ref_run_id = int(ref_row["id"]) if ref_row.get("id") is not None else None

    updated = update_reference_metrics(
        db_path=args.db_path,
        model_name=model_name,
        ref_acc1=ref_acc1,
        ref_acc5=ref_acc5,
        ref_certainty=ref_certainty,
        ref_run_id=ref_run_id,
        only_missing=args.only_missing,
    )
    print(
        f"[DB] Updated {updated} other rows for {model_name} "
        f"with ref_acc1={ref_acc1:.4f}, ref_acc5={ref_acc5:.4f}, "
        f"ref_certainty={ref_certainty:.6f}"
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Run/log FP32 reference metrics and propagate them to matching DB rows."
    )
    parser.add_argument("--model_name", type=str, default=None, help="Single model to evaluate.")
    parser.add_argument(
        "--all_models_from_db",
        action="store_true",
        help="Process every model present in the DB. Weights are inferred from config_json when possible.",
    )
    parser.add_argument("--weights", type=str, default="DEFAULT", help="Model weights spec fallback.")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH, help="SQLite runs.db path.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Runner output root.")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name.")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val", help="Dataset path.")
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--input_size", type=int, default=224, help="Model input size.")
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=-1,
        help="Limit evaluation batches. Use -1 for the full dataset.",
    )
    parser.add_argument(
        "--reuse_existing_ref",
        action="store_true",
        help="Reuse a successful fp32/fp32 DB row instead of running a new reference.",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        help="Only update rows whose ref_acc1/ref_acc5 are missing or non-positive.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.db_path = os.path.abspath(args.db_path)
    args.output_root = os.path.abspath(args.output_root)

    models = selected_models(args)
    if not models:
        print("No models selected.")
        return

    print(f"DB: {args.db_path}")
    print(f"Output root: {args.output_root}")
    print(f"Models: {', '.join(name for name, _ in models)}")

    for model_name, weights in models:
        process_model(args, model_name, weights)


if __name__ == "__main__":
    main()
