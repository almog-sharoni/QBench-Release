import argparse
import csv
from dataclasses import dataclass
import gc
import os
import sys

import torch
import yaml


os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.experiments.utils.common import (  # noqa: E402
    build_dynamic_input_quant_cfg,
    build_runtime_config,
    run_inference,
)
from runspace.src.quantization.dynamic_input_metrics import (  # noqa: E402
    assert_dynamic_input_metric_implemented,
    normalize_dynamic_input_metric,
)


EXPERIMENT_TYPE = "pseudo_mse2"
WEIGHT_DT = "fp32"
METRIC_NAME = "pseudo_MSE2"
BASELINE_METRIC_NAME = "mse"
L1_METRIC_NAME = "l1"
L1_METRIC_LABEL = "L1"
DEFAULT_BIT_WIDTHS = [8, 7, 6, 5, 4]
DEFAULT_EXP_BITS = [1, 2]
DEFAULT_RANDOM_SUBSET_SIZE = -1
DEFAULT_RANDOM_SEED = 42
UNSIGNED_INPUT_SOURCES = ["relu", "relu6", "softmax"]
DEFAULT_METRICS = [BASELINE_METRIC_NAME, L1_METRIC_NAME, METRIC_NAME]


@dataclass(frozen=True)
class PseudoMseSweepSpec:
    bit_width: int
    activation_dt: str
    candidate_formats: list[str]


@dataclass(frozen=True)
class MetricComparisonSpec:
    bit_width: int
    activation_dt: str
    candidate_formats: list[str]
    metric: str
    metric_label: str
    metric_slug: str


def _parse_int_csv(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    return parsed or list(default)


def _parse_csv(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        parsed = [str(item).strip() for item in value if str(item).strip()]
    else:
        parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    return parsed or list(default)


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a future pseudo_MSE2 dynamic activation metric on ImageNet."
    )
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument(
        "--models_file",
        type=str,
        default=None,
        help=(
            "Optional path to a models.yaml file. Also supported by passing a "
            ".yaml/.yml path as --model_name."
        ),
    )
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--model_source", type=str, default="auto")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit_batches", type=int, default=-1)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bit_widths", type=str, default=",".join(str(b) for b in DEFAULT_BIT_WIDTHS))
    parser.add_argument("--exp_bits", type=str, default=",".join(str(e) for e in DEFAULT_EXP_BITS))
    parser.add_argument("--random_subset_size", type=int, default=DEFAULT_RANDOM_SUBSET_SIZE)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args(argv)
    args.bit_widths = _parse_int_csv(args.bit_widths, DEFAULT_BIT_WIDTHS)
    args.exp_bits = _parse_int_csv(args.exp_bits, DEFAULT_EXP_BITS)
    args.metrics = _parse_csv(args.metrics, DEFAULT_METRICS)
    return args


def _is_yaml_path(value):
    return isinstance(value, str) and value.lower().endswith((".yaml", ".yml"))


def _model_entry_from_yaml(entry):
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("model_name")
        return {
            "name": name,
            "weights": entry.get("weights", "DEFAULT"),
            "source": entry.get("source") or entry.get("model_source"),
        }
    return {"name": str(entry), "weights": "DEFAULT", "source": None}


def _load_models(models_file):
    if not models_file:
        return []
    with open(models_file, "r") as f:
        models_cfg = yaml.safe_load(f)

    if models_cfg is None:
        return []
    if isinstance(models_cfg, list):
        entries = models_cfg
    elif isinstance(models_cfg, dict):
        entries = models_cfg.get("models")
        if entries is None:
            entries = [models_cfg]
    else:
        raise ValueError(
            f"models.yaml must contain a model entry, a list, or a 'models' list; got {type(models_cfg).__name__}"
        )

    if not isinstance(entries, list):
        raise ValueError("models.yaml 'models' field must be a list")

    models = []
    for entry in entries:
        model = _model_entry_from_yaml(entry)
        if model.get("name"):
            models.append(model)
    return models


def _models_from_args(args):
    models_file = args.models_file
    if not models_file and _is_yaml_path(args.model_name):
        models_file = args.model_name

    if models_file:
        models = _load_models(models_file)
        if not models:
            raise ValueError(f"No model entries found in {models_file}")
        return models, models_file

    return [
        {
            "name": args.model_name,
            "weights": args.weights,
            "source": args.model_source,
        }
    ], None


def candidate_formats_for_bit_width(bit_width, exp_bits=None):
    formats = []
    for exp in _parse_int_csv(exp_bits, DEFAULT_EXP_BITS):
        mantissa_bits = int(bit_width) - int(exp) - 1
        if mantissa_bits < 1:
            continue
        formats.append(f"fp{int(bit_width)}_e{int(exp)}m{mantissa_bits}")
    return formats


def build_pseudo_mse_sweep_specs(args):
    specs = []
    for bit_width in args.bit_widths:
        candidates = candidate_formats_for_bit_width(bit_width, args.exp_bits)
        if not candidates:
            continue
        exp_label = "e" + "e".join(str(exp) for exp in args.exp_bits)
        specs.append(
            PseudoMseSweepSpec(
                bit_width=int(bit_width),
                activation_dt=f"dyn_a{int(bit_width)}_{exp_label}",
                candidate_formats=candidates,
            )
        )
    return specs


def _metric_label(metric):
    normalized = normalize_dynamic_input_metric(metric)
    if normalized == "l2":
        return "MSE"
    if normalized == "l1":
        return L1_METRIC_LABEL
    if normalized == "pseudo_mse2":
        return METRIC_NAME
    raise ValueError(
        f"pseudo_MSE2 comparison supports only mse, l1, and {METRIC_NAME}; got {metric!r}"
    )


def _metric_slug(metric):
    return _metric_label(metric).lower()


def _metric_config_value(metric):
    normalized = normalize_dynamic_input_metric(metric)
    if normalized == "l2":
        return BASELINE_METRIC_NAME
    if normalized == "l1":
        return L1_METRIC_NAME
    if normalized == "pseudo_mse2":
        return METRIC_NAME
    raise ValueError(f"Unsupported comparison metric: {metric!r}")


def build_metric_comparison_specs(args):
    specs = []
    for base_spec in build_pseudo_mse_sweep_specs(args):
        for metric in args.metrics:
            assert_dynamic_input_metric_implemented(metric)
            metric_value = _metric_config_value(metric)
            metric_slug = _metric_slug(metric)
            specs.append(
                MetricComparisonSpec(
                    bit_width=base_spec.bit_width,
                    activation_dt=f"{base_spec.activation_dt}_{metric_slug}",
                    candidate_formats=list(base_spec.candidate_formats),
                    metric=metric_value,
                    metric_label=_metric_label(metric),
                    metric_slug=metric_slug,
                )
            )
    return specs


def _assert_comparison_metrics_ready(metrics):
    for metric in metrics:
        _metric_label(metric)
        assert_dynamic_input_metric_implemented(metric)


def build_pseudo_mse_runtime_config(args, spec, model_name=None, weights=None):
    cfg = build_runtime_config(args, model_name=model_name, weights=weights)
    cfg.setdefault("dataset", {}).update(
        {
            "random_subset_size": int(args.random_subset_size),
            "random_seed": int(args.random_seed),
        }
    )
    cfg.setdefault("adapter", {}).update(
        {
            "quantized_ops": ["all"],
            "build_quantized": True,
            "weight_quantization": False,
            # Match activation_candidate_sweep: build the graph with input
            # quantization enabled, then let DynamicInputQuantizer hooks provide
            # the actual runtime inputs and disable module input quant per call.
            "input_quantization": True,
            "output_quantization": False,
            "unsigned_input_sources": list(UNSIGNED_INPUT_SOURCES),
        }
    )
    cfg.setdefault("quantization", {}).update(
        {
            "format": spec.candidate_formats[0],
            "input_format": spec.candidate_formats[0],
            "mode": "chunk",
            "chunk_size": int(args.chunk_size),
            "weight_mode": "tensor",
            "weight_chunk_size": int(args.chunk_size),
            "weight_source": "fp32",
            "unsigned_input_sources": list(UNSIGNED_INPUT_SOURCES),
        }
    )
    cfg["experiment"].update(
        {
            "name": EXPERIMENT_TYPE,
            "type": EXPERIMENT_TYPE,
            "metric": spec.metric,
            "metric_label": spec.metric_label,
            "weight_dt": WEIGHT_DT,
            "activation_dt": spec.activation_dt,
            "bit_width": int(spec.bit_width),
            "candidate_formats": list(spec.candidate_formats),
            "random_subset_size": int(args.random_subset_size),
            "random_seed": int(args.random_seed),
        }
    )
    return cfg


def build_pseudo_mse_input_quant_cfg(args, spec, model_name=None):
    return build_dynamic_input_quant_cfg(
        metric=spec.metric,
        chunk_size=args.chunk_size,
        candidate_formats=spec.candidate_formats,
        restrict_post_relu_ufp=False,
        unsigned_input_sources=UNSIGNED_INPUT_SOURCES,
        dynamic_unsigned_input_candidates=True,
        model_name=model_name or args.model_name,
    )


def _effective_dataset_size(args, loader):
    loader_size = len(loader.dataset)
    if args.limit_batches is None or int(args.limit_batches) < 0:
        return loader_size
    return min(loader_size, int(args.limit_batches) * int(args.batch_size))


def _dataset_label(args, dataset_size):
    configured = int(args.random_subset_size)
    if configured > 0:
        source = f"ImageNet random subset={configured}, seed={int(args.random_seed)}"
    else:
        source = "ImageNet full validation set"
    if args.limit_batches is not None and int(args.limit_batches) >= 0:
        source += f", limit_batches={int(args.limit_batches)}"
    return f"{source}; evaluated samples={int(dataset_size)}"


def _format_value(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _metric_plot_order(rows):
    preferred = ["MSE", L1_METRIC_LABEL, METRIC_NAME]
    labels = []
    for row in rows:
        label = row.get("metric")
        if label and label not in labels:
            labels.append(label)
    return [label for label in preferred if label in labels] + [
        label for label in labels if label not in preferred
    ]


def _write_model_summary(output_dir, model_name, rows, dataset_label):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_summary.csv")
    txt_path = os.path.join(output_dir, f"{model_name}_summary.txt")
    fields = [
        "model",
        "metric",
        "bit_width",
        "activation_dt",
        "candidate_formats",
        "dataset_size",
        "random_seed",
        "limit_batches",
        "status",
        "acc1",
        "acc5",
        "certainty",
        "norm_mse",
        "norm_l1",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"SUMMARY: {model_name}",
        f"Dataset: {dataset_label}",
        "metric | bits | acc1 | acc5 | certainty | norm_mse | norm_l1 | status",
        "-" * 86,
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["metric"]),
                    str(row["bit_width"]),
                    _format_value(row["acc1"], 2),
                    _format_value(row["acc5"], 2),
                    _format_value(row["certainty"], 4),
                    _format_value(row["norm_mse"], 6),
                    _format_value(row["norm_l1"], 6),
                    str(row["status"]),
                ]
            )
        )
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, txt_path


def _write_combined_summary(output_dir, rows):
    if not rows:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.csv")
    fields = [
        "model",
        "metric",
        "bit_width",
        "activation_dt",
        "candidate_formats",
        "dataset_size",
        "random_seed",
        "limit_batches",
        "status",
        "acc1",
        "acc5",
        "certainty",
        "norm_mse",
        "norm_l1",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_model_summary(output_dir, model_name, rows, dataset_label):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped for {model_name} ({exc})")
        return []

    complete_rows = [row for row in rows if row.get("status") == "SUCCESS"]
    if not complete_rows:
        return []

    paths = []
    plot_specs = [
        ("acc1", "Top-1 Accuracy", f"{model_name}_accuracy_vs_bits.png"),
        ("norm_mse", "Normalized L2 Error", f"{model_name}_norm_mse_vs_bits.png"),
        ("norm_l1", "Normalized L1 Error", f"{model_name}_norm_l1_vs_bits.png"),
    ]
    for key, ylabel, filename in plot_specs:
        plt.figure(figsize=(8, 5))
        has_values = False
        for metric in _metric_plot_order(complete_rows):
            points = [
                (int(row["bit_width"]), float(row[key]))
                for row in complete_rows
                if row.get("metric") == metric and row.get(key) is not None
            ]
            if not points:
                continue
            points.sort()
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            plt.plot(xs, ys, marker="o", label=metric)
            has_values = True

        if not has_values:
            plt.close()
            continue
        plt.xlabel("Activation Bit Width")
        plt.ylabel(ylabel)
        plt.title(f"{model_name}: {ylabel} by Metric\n{dataset_label}")
        plt.grid(alpha=0.3)
        plt.legend(title="Metric")
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths


def process_single_model(args, device=None):
    _assert_comparison_metrics_ready(args.metrics)

    from runspace.core.runner import Runner

    runner = Runner(device)
    model_name = args.model_name
    model_out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    specs = build_metric_comparison_specs(args)
    if not specs:
        raise ValueError("No metric comparison specs were generated")

    print(f"\n{'=' * 72}")
    print(f"PSEUDO_MSE2 METRIC COMPARISON: {model_name}")
    print(f"Metrics: {[spec.metric_label for spec in specs[:len(args.metrics)]]}")
    print(f"{'=' * 72}")

    def loader_config_builder(current_args):
        return build_pseudo_mse_runtime_config(
            current_args,
            specs[0],
            model_name=current_args.model_name,
            weights=current_args.weights,
        )

    loader = runner.setup_data_loader(loader_config_builder(args))
    dataset_size = _effective_dataset_size(args, loader)
    dataset_label = _dataset_label(args, dataset_size)
    print(f"Dataset: {dataset_label}")

    rows = []
    for spec in specs:
        print(
            f"\n[{model_name}/{spec.activation_dt}] "
            f"metric={spec.metric_label} candidates={spec.candidate_formats}"
        )
        config = build_pseudo_mse_runtime_config(
            args,
            spec,
            model_name=model_name,
            weights=args.weights,
        )
        input_quant_cfg = build_pseudo_mse_input_quant_cfg(args, spec, model_name=model_name)
        model = None
        adapter = None
        row = {
            "model": model_name,
            "metric": spec.metric_label,
            "bit_width": int(spec.bit_width),
            "activation_dt": spec.activation_dt,
            "candidate_formats": ",".join(spec.candidate_formats),
            "dataset_size": int(dataset_size),
            "random_seed": int(args.random_seed),
            "limit_batches": int(args.limit_batches),
            "status": "SUCCESS",
            "acc1": None,
            "acc5": None,
            "certainty": None,
            "norm_mse": None,
            "norm_l1": None,
            "error": None,
        }
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
            row.update(
                {
                    "acc1": acc1,
                    "acc5": acc5,
                    "certainty": certainty,
                    "norm_mse": (input_quant_stats or {}).get("norm_mse"),
                    "norm_l1": (input_quant_stats or {}).get("norm_l1"),
                }
            )
            print(
                f"[{model_name}/{spec.activation_dt}] Top1={acc1:.2f} "
                f"Top5={acc5:.2f} Certainty={certainty:.4f} "
                f"NormMSE={_format_value(row['norm_mse'], 6)}"
            )
        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = str(exc)
            print(f"[{model_name}/{spec.activation_dt}] ERROR: {exc}")
        finally:
            if model is not None:
                del model
            if adapter is not None:
                del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)

    _csv_path, txt_path = _write_model_summary(model_out_dir, model_name, rows, dataset_label)
    print(f"\nSummary written to {txt_path}")
    if not args.no_plot:
        for path in _plot_model_summary(model_out_dir, model_name, rows, dataset_label):
            print(f"Plot written to {path}")
    return rows


def main(argv=None):
    args = get_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    models, models_file = _models_from_args(args)
    if models_file:
        print(f"Loaded {len(models)} model(s) from {models_file}")

    combined_rows = []
    for entry in models:
        args.model_name = entry["name"]
        args.weights = entry.get("weights", "DEFAULT")
        args.model_source = entry.get("source") or "auto"
        rows = process_single_model(args, device=device)
        combined_rows.extend(rows)

    if models_file:
        path = _write_combined_summary(args.output_dir, combined_rows)
        if path:
            print(f"\nCombined summary written to {path}")


if __name__ == "__main__":
    main()
