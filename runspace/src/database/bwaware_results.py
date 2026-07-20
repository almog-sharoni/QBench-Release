"""Discovery helpers for bandwidth-aware dashboard result files."""

import json
import os


RESULT_FILENAME = "bandwidth_aware_quant_results.json"


def normalize_cache_cycles(cycles_per_cache_size):
    """Normalize JSON cache-size keys so ``2`` and ``2.0`` are equivalent."""
    normalized = {}
    for cache_size, cycles in (cycles_per_cache_size or {}).items():
        try:
            normalized[float(cache_size)] = cycles
        except (TypeError, ValueError):
            continue
    return normalized


def is_greedy_descent_result(data):
    """Return whether a bandwidth-aware result was produced by greedy descent."""
    if not isinstance(data, dict):
        return False
    experiment_config = data.get("experiment_config") or {}
    return bool(
        data.get("used_descent")
        or (
            isinstance(experiment_config, dict)
            and experiment_config.get("descent")
        )
    )


def load_greedy_descent_results(project_root):
    """Load only greedy-descent results from bandwidth-aware output trees."""
    experiment_root = os.path.join(
        project_root, "runspace/experiments/bandwidth_aware_quant"
    )
    results_roots = []
    if os.path.isdir(experiment_root):
        for entry in sorted(os.scandir(experiment_root), key=lambda item: item.name):
            if entry.is_dir() and entry.name.startswith("results"):
                results_roots.append(
                    (f"bandwidth_aware_quant/{entry.name}", entry.path)
                )

    legacy_root = os.path.join(
        project_root,
        "runspace/experiments/baselines_vs_dynamic_runs/results/bandwidth_aware",
    )
    if os.path.isdir(legacy_root):
        results_roots.append(("baselines_vs_dynamic_runs", legacy_root))

    runs = []
    for source_label, results_root in results_roots:
        for dirpath, _, filenames in os.walk(results_root):
            if RESULT_FILENAME not in filenames:
                continue
            json_path = os.path.join(dirpath, RESULT_FILENAME)
            try:
                with open(json_path, "r", encoding="utf-8") as result_file:
                    data = json.load(result_file)
            except (OSError, ValueError, TypeError):
                continue
            if not is_greedy_descent_result(data):
                continue

            rel_dir = os.path.relpath(dirpath, results_root)
            runs.append({
                "label": os.path.join(source_label, rel_dir),
                "path": json_path,
                "dir": dirpath,
                "data": data,
                "model_name": data.get("model_name", rel_dir),
            })

    return sorted(runs, key=lambda run: run["label"])
