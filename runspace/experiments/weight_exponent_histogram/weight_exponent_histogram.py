import os

os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.core.runner import Runner


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot FP32 weight exponent histograms per model"
    )
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--models_config", type=str, default=None,
                        help="Path to YAML with a list of models (same format as inputs/models.yaml)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of largest exponent values to show as individual bins")
    return parser.parse_args()


def _param_to_exponents(param: torch.Tensor) -> np.ndarray:
    """Extract unbiased FP32 exponents from a single parameter tensor."""
    w = param.detach().float().cpu()
    bits = w.view(torch.int32)
    biased = ((bits >> 23) & 0xFF).numpy().astype(np.int32)
    unbiased = biased - 127
    mask = (biased > 0) & (biased < 255)
    return unbiased[mask]


def extract_exponents(model) -> np.ndarray:
    """Return unbiased FP32 exponents for all weight params."""
    all_exps = [
        _param_to_exponents(p)
        for _, p in model.named_parameters()
        if p.requires_grad and p.numel() > 0
    ]
    return np.concatenate(all_exps) if all_exps else np.array([], dtype=np.int32)


def extract_exponents_per_linear(model) -> list[tuple[str, np.ndarray]]:
    """Return [(layer_name, exponents)] for each nn.Linear module in the model."""
    import torch.nn as nn
    results = []
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        all_exps = [
            _param_to_exponents(p)
            for p in mod.parameters(recurse=False)
            if p.requires_grad and p.numel() > 0
        ]
        if all_exps:
            results.append((mod_name, np.concatenate(all_exps)))
    return results


def plot_exponent_histogram(exponents: np.ndarray, model_name: str, top_n: int, output_path: str):
    """
    Create a bar chart with individual bars for the top_n largest exponent values
    and one aggregated 'other' bar for the rest.
    """
    if exponents.size == 0:
        print(f"  No valid weights found for {model_name}, skipping plot.")
        return

    # Count occurrences per unique exponent value
    unique_exps, counts = np.unique(exponents, return_counts=True)

    # Identify the top_n largest exponent *values* (not counts)
    sorted_by_value = unique_exps[np.argsort(unique_exps)[::-1]]
    top_exps = sorted_by_value[:top_n]
    top_exps_set = set(top_exps.tolist())

    # Build bar data: individual bars for top_n, one aggregated bar for others
    bar_labels = []
    bar_values = []

    # Add individual bars in descending exponent order
    for exp in top_exps:
        idx = np.where(unique_exps == exp)[0]
        count = int(counts[idx[0]]) if len(idx) > 0 else 0
        bar_labels.append(f"e={exp}")
        bar_values.append(count)

    # Aggregate the rest
    other_mask = np.array([e not in top_exps_set for e in unique_exps])
    other_count = int(counts[other_mask].sum()) if other_mask.any() else 0
    bar_labels.append("other")
    bar_values.append(other_count)

    x = np.arange(len(bar_labels))
    colors = ['#4C72B0'] * top_n + ['#DD8452']

    fig, ax = plt.subplots(figsize=(max(10, top_n + 3), 6))
    bars = ax.bar(x, bar_values, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=0, fontsize=11)
    ax.set_xlabel("FP32 Unbiased Exponent", fontsize=12)
    ax.set_ylabel("Weight Element Count", fontsize=12)
    ax.set_title(f"FP32 Weight Exponent Distribution — {model_name}", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis='y', alpha=0.3)

    # Annotate bars with counts
    for bar, val in zip(bars, bar_values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:,}",
                ha='center', va='bottom', fontsize=8, rotation=45
            )

    total = exponents.size
    ax.text(
        0.99, 0.97,
        f"Total weights: {total:,}",
        transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color='gray'
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_model(model_name: str, weights: str, top_n: int, output_dir: str):
    device = torch.device('cpu')
    runner = Runner(device)

    config = {
        'model': {'name': model_name, 'weights': weights},
        'adapter': {
            'type': 'generic',
            'quantized_ops': [],
            'build_quantized': False,
            'weight_quantization': False,
            'input_quantization': False,
        },
        'quantization': {'weight_source': 'fp32'},
    }

    model_dir = os.path.join(output_dir, model_name, "fp32_materialized")
    print(f"[{model_name}] Loading fp32 weights...")
    try:
        model, _adapter, _info = runner.prepare_model_with_materialized_weights(
            config=config,
            output_dir=model_dir,
        )
    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}")
        import traceback; traceback.print_exc()
        return

    model.eval()
    print(f"[{model_name}] Extracting exponents...")
    model_out_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    exponents = extract_exponents(model)
    print(f"[{model_name}] all weights: {exponents.size:,} elements, "
          f"exponent range [{exponents.min() if exponents.size else 'N/A'}, "
          f"{exponents.max() if exponents.size else 'N/A'}]")
    plot_exponent_histogram(
        exponents, model_name, top_n,
        os.path.join(model_out_dir, "exponent_histogram.png"),
    )

    linear_layers = extract_exponents_per_linear(model)
    print(f"[{model_name}] Found {len(linear_layers)} Linear layer(s).")
    linear_out_dir = os.path.join(model_out_dir, "linear_layers")
    os.makedirs(linear_out_dir, exist_ok=True)
    for layer_name, exps in linear_layers:
        safe_name = layer_name.replace("/", "_").replace(".", "_")
        title = f"{model_name} — {layer_name}"
        out_path = os.path.join(linear_out_dir, f"{safe_name}.png")
        plot_exponent_histogram(exps, title, top_n, out_path)

    del model
    import gc; gc.collect()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    models_to_run = []
    if args.models_config:
        with open(args.models_config, 'r') as f:
            raw = yaml.safe_load(f)
        raw_list = raw['models'] if isinstance(raw, dict) and 'models' in raw else raw
        for m in raw_list:
            if isinstance(m, str):
                models_to_run.append({'name': m, 'weights': 'DEFAULT'})
            elif isinstance(m, dict):
                models_to_run.append({'name': m['name'], 'weights': m.get('weights', 'DEFAULT')})
    else:
        models_to_run.append({'name': args.model_name, 'weights': args.weights})

    print(f"Processing {len(models_to_run)} model(s).")
    for m in models_to_run:
        process_model(m['name'], m['weights'], args.top_n, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
