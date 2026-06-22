"""Plot error-metric-sweep results.

Reads the per-model summary CSVs (or the combined ``summary.csv``) written by
``error_metric_sweep.py`` and renders, per model, a grouped comparison of the
selection metrics, plus a model x metric accuracy heatmap overview.

Standalone:
    ./apptainer.sh runspace/experiments/error_metric_sweep/plot_results.py \
        --results_dir runspace/experiments/error_metric_sweep/results
"""

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_ORDER = ["l2", "l1", "linf", "bias", "huber", "l0", "logsum"]
NUMERIC_FIELDS = ("acc1", "acc5", "delta_acc1_vs_fp32", "certainty", "norm_mse", "norm_l1")


def _to_float(value):
    if value is None or value == "" or value == "-":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _coerce_row(row):
    out = {"metric": str(row.get("metric", "")).strip().lower()}
    for field in NUMERIC_FIELDS:
        out[field] = _to_float(row.get(field))
    if "model" in row:
        out["model"] = row.get("model")
    return out


def _load_results(results_dir):
    """Return {model_name: [coerced rows...]} from combined or per-model CSVs."""
    combined = os.path.join(results_dir, "summary.csv")
    models = {}
    if os.path.exists(combined):
        for raw in _read_csv(combined):
            row = _coerce_row(raw)
            models.setdefault(row.get("model") or "model", []).append(row)
        if models:
            return models

    # Fall back to per-model <name>_summary.csv files.
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_summary.csv"):
            continue
        model_name = fname[: -len("_summary.csv")]
        rows = [_coerce_row(r) for r in _read_csv(os.path.join(results_dir, fname))]
        if rows:
            models[model_name] = rows
    return models


def _ordered_metrics(rows):
    present = [r["metric"] for r in rows]
    ordered = [m for m in METRIC_ORDER if m in present]
    ordered += [m for m in present if m not in ordered]
    return ordered


def _row_by_metric(rows):
    return {r["metric"]: r for r in rows}


def plot_model(model_name, rows, out_dir):
    metrics = _ordered_metrics(rows)
    by_metric = _row_by_metric(rows)
    delta = [by_metric[m]["delta_acc1_vs_fp32"] for m in metrics]
    nmse = [by_metric[m]["norm_mse"] for m in metrics]
    nl1 = [by_metric[m]["norm_l1"] for m in metrics]
    x = np.arange(len(metrics))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(7, 1.4 * len(metrics)), 7))

    colors = ["#4c72b0" if not np.isnan(d) and d == np.nanmax(delta) else "#a7b8d4" for d in delta]
    ax_top.bar(x, delta, color=colors)
    ax_top.axhline(0.0, color="black", linewidth=0.8)
    ax_top.set_ylabel("Δ Top-1 vs FP32 (pts)")
    ax_top.set_title(f"{model_name}: selection-metric comparison (W32A5 dynamic)")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(metrics)
    for xi, d in zip(x, delta):
        if not np.isnan(d):
            ax_top.annotate(f"{d:.2f}", (xi, d), ha="center",
                            va="bottom" if d >= 0 else "top", fontsize=8)

    width = 0.4
    ax_bot.bar(x - width / 2, nmse, width, label="norm MSE", color="#dd8452")
    ax_bot.bar(x + width / 2, nl1, width, label="norm L1", color="#55a868")
    ax_bot.set_ylabel("realized quant error")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(metrics)
    ax_bot.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_metrics.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_overview(models, out_dir):
    model_names = sorted(models)
    metric_set = []
    for rows in models.values():
        for m in _ordered_metrics(rows):
            if m not in metric_set:
                metric_set.append(m)
    metric_set = [m for m in METRIC_ORDER if m in metric_set] + \
                 [m for m in metric_set if m not in METRIC_ORDER]

    grid = np.full((len(model_names), len(metric_set)), np.nan)
    for i, name in enumerate(model_names):
        by_metric = _row_by_metric(models[name])
        for j, metric in enumerate(metric_set):
            if metric in by_metric:
                grid[i, j] = by_metric[metric]["delta_acc1_vs_fp32"]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(metric_set)), max(3, 0.8 * len(model_names))))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(metric_set)))
    ax.set_xticklabels(metric_set)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_title("Δ Top-1 vs FP32 (pts) by model × metric")
    for i in range(len(model_names)):
        for j in range(len(metric_set)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "overview_delta_acc1.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def generate_plots(results_dir):
    """Render all plots for a results directory. Returns list of written paths."""
    models = _load_results(results_dir)
    if not models:
        print(f"[plot] no summary CSVs found under {results_dir}")
        return []
    written = []
    for model_name, rows in models.items():
        written.append(plot_model(model_name, rows, results_dir))
    if len(models) > 1:
        written.append(plot_overview(models, results_dir))
    for path in written:
        print(f"[plot] wrote {path}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Plot error-metric-sweep results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    args = parser.parse_args()
    generate_plots(args.results_dir)


if __name__ == "__main__":
    main()
