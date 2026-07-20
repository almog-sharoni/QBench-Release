
import os
import sys
import gc
import json
import yaml
import argparse
import torch
import copy
import types
import csv
import math
import re
import shlex
from tqdm import tqdm

# Fix for container permission issues
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.registry.op_registry import OpRegistry  # noqa: E402
from runspace.core.runner import Runner  # noqa: E402
from runspace.experiments.utils.common import (  # noqa: E402
    build_dynamic_input_quant_cfg as _build_dynamic_input_quant_cfg,
    build_loader as _build_loader,
    build_runtime_config as _base_runtime_config,
    build_uniform_input_quant_cfg as _build_uniform_input_quant_cfg,
    build_weight_map_json as _build_weight_map_json,
    get_or_run_fp32_ref as _get_or_run_fp32_ref_common,
    layer_types_from_model as _layer_types_from_model,
    load_fp32_model as _load_fp32_model,
    row_uses_encoded_activation_transport as _row_uses_encoded_activation_transport,
    run_inference as _run_inference,
)

from runspace.experiments.find_optimal_weight_quant.find_optimal_weight_quant import (  # noqa: E402
    get_quantized_tensor_sim,
    create_quantized_state_dict,
    run_weight_quantization_analysis,
    baseline_formats as weight_baseline_formats,
)

from runspace.experiments.find_optimal_input_quant.find_optimal_input_quant import (  # noqa: E402
    candidate_formats as input_candidate_formats,
)

HYBRID_WEIGHT_FORMATS = list(weight_baseline_formats)
HYBRID_INPUT_CANDIDATE_FORMATS = [
    fmt for fmt in input_candidate_formats if str(fmt).strip().lower() != 'fp32'
]
DEFAULT_HYBRID_EXPERIMENT_TYPE = "hybrid_quant_optimal"
DEFAULT_WEIGHT_BASELINE_EXPERIMENT_TYPE = "weight_quant_baseline"
DEFAULT_INPUT_BASELINE_EXPERIMENT_TYPE = "input_quant_baseline"
DEFAULT_WEIGHT_OPTIMIZED_EXPERIMENT_TYPE = "weight_quant_optimized"
DEFAULT_INPUT_DYNAMIC_EXPERIMENT_TYPE = "input_quant_dynamic"

_FORMAT_BIT_WIDTH_RE = re.compile(
    r"^(?:fp|ufp|efp|uefp)(?P<bit_width>\d+)(?:_|$)",
    re.IGNORECASE,
)


def _parse_csv_arg(value, fallback):
    if value is None:
        return list(fallback)
    parsed = [item.strip() for item in str(value).split(',') if item.strip()]
    return parsed if parsed else list(fallback)


def _parse_bit_widths(value):
    if value is None:
        return None
    widths = []
    for item in str(value).split(','):
        item = item.strip()
        if not item:
            continue
        width = int(item)
        if width <= 0:
            raise ValueError(f"Bit widths must be positive; got {width}.")
        if width not in widths:
            widths.append(width)
    return widths or None


def _format_bit_width(fmt):
    match = _FORMAT_BIT_WIDTH_RE.match(str(fmt).strip())
    return int(match.group('bit_width')) if match is not None else None


def _candidate_formats_by_bit_width(formats):
    grouped = {}
    for candidate in formats:
        candidate = str(candidate).strip()
        if candidate.lower() == 'fp32':
            raise ValueError(
                "FP32 cannot be a dynamic input candidate; remove it from "
                "--input_candidate_formats."
            )
        bit_width = _format_bit_width(candidate)
        if bit_width is None:
            raise ValueError(
                "Input candidate formats must include a bit width in their name "
                f"(for example, fp8_e4m3); got {candidate!r}."
            )
        grouped.setdefault(bit_width, []).append(candidate)
    return grouped


def _experiment_bit_width(experiment_type, base_experiment_type):
    value = str(experiment_type or '').strip()
    prefix = f"{str(base_experiment_type).strip()}_"
    if not value.startswith(prefix):
        return None
    suffix = value[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _optimized_weight_bit_width(row, base_experiment_type):
    """Resolve the width of current and legacy optimized-weight runs."""
    bit_width = _experiment_bit_width(
        row.get('experiment_type'), base_experiment_type
    )
    if bit_width is not None:
        return bit_width
    if str(row.get('experiment_type') or '').strip() != str(
        base_experiment_type
    ).strip():
        return None

    # Older runs used the unsuffixed experiment type. Their per-layer/chunk
    # quantization map is still authoritative about the candidate width.
    widths = set()
    for layer_spec in _safe_json_dict(row.get('quant_map_json')).values():
        formats = (
            layer_spec.get('format')
            if isinstance(layer_spec, dict)
            else layer_spec
        )
        if not isinstance(formats, list):
            formats = [formats]
        widths.update(
            width
            for width in map(_format_bit_width, formats)
            if width is not None
        )
        if len(widths) > 1:
            return None
    return next(iter(widths)) if widths else None


def _safe_json_dict(value):
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dynamic_input_cfg_from_row(row):
    config = _row_config(row)
    evaluation = config.get('evaluation', {})
    candidate = evaluation.get('dynamic_input_quant') or evaluation.get(
        'input_quant'
    )
    return copy.deepcopy(candidate) if isinstance(candidate, dict) else None


def _dynamic_activation_label(row, bit_width):
    label = str(row.get('activation_dt', '') or '').strip()
    if not label:
        label = 'dyn_input_mse'
    width_suffix = f"_{int(bit_width)}bit"
    return label if label.endswith(width_suffix) else f"{label}{width_suffix}"


def _finite_number(value, fallback=float('-inf')):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _best_accuracy_row(rows, description):
    if rows.empty:
        raise ValueError(f"No successful database rows found for {description}.")

    def rank(item):
        _, row = item
        return (
            _finite_number(row.get('acc1')),
            _finite_number(row.get('acc5')),
            _finite_number(row.get('certainty')),
            _finite_number(row.get('id')),
        )

    _, best = max(rows.iterrows(), key=rank)
    if _finite_number(best.get('acc1')) == float('-inf'):
        raise ValueError(f"All database rows for {description} have invalid Top-1 accuracy.")
    return best


def _successful_experiment_rows(runs, model_name, experiment_type):
    if runs is None or runs.empty:
        raise ValueError("The run database is empty; run the weight/input baselines first.")
    return runs[
        (runs['model_name'] == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['status'] == 'SUCCESS')
    ]


def _row_config(row):
    """Return a logged runtime config as a dictionary, or an empty dict."""
    raw = row.get('config_json')
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalised_string_list(value):
    if isinstance(value, str):
        value = value.split(',')
    return sorted(
        str(item).strip().lower()
        for item in (value or [])
        if str(item).strip()
    )


def _same_dataset_path(left, right):
    if left is None or right is None:
        return left == right
    return os.path.normpath(str(left)) == os.path.normpath(str(right))


def _cli_option_value(row, option):
    command = row.get('cli_command')
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    prefix = f"{option}="
    for index, token in enumerate(tokens):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _row_is_full_evaluation(row, config):
    dataset = config.get('dataset', {})
    evaluation = config.get('evaluation', {})
    value = dataset.get('limit_batches')
    if value is None:
        value = evaluation.get('max_batches', evaluation.get('limit_batches'))
    if value is None:
        value = _cli_option_value(row, '--limit_batches')
    if value is None:
        # Both source experiments default to -1. Their older weight-baseline
        # configs omitted the field, so an absent CLI override means full eval.
        return True
    try:
        return int(value) < 0
    except (TypeError, ValueError):
        return False


def _row_fold_input_norm(row, config):
    adapter = config.get('adapter', {})
    if 'fold_input_norm' in adapter:
        return bool(adapter['fold_input_norm'])
    command = row.get('cli_command')
    if isinstance(command, str):
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        if '--no_fold_input_norm' in tokens:
            return False
        if '--fold_input_norm' in tokens:
            return True
    return True


def _row_matches_source_requirements(row, kind, requirements):
    """Check that a baseline score was measured under reproducible semantics."""
    if not requirements:
        return True
    config = _row_config(row)
    if not config:
        return False

    model = config.get('model', {})
    dataset = config.get('dataset', {})
    quantization = config.get('quantization', {})
    if model.get('weights') != requirements.get('model_weights'):
        return False
    if dataset.get('name') != requirements.get('dataset_name'):
        return False
    if not _same_dataset_path(
        dataset.get('path'), requirements.get('dataset_path')
    ):
        return False
    if requirements.get('require_full_evaluation', True):
        if not _row_is_full_evaluation(row, config):
            return False
    if _row_fold_input_norm(row, config) != bool(
        requirements.get('fold_input_norm', True)
    ):
        return False

    if kind == 'weight':
        try:
            chunk_size = int(quantization.get('weight_chunk_size'))
        except (TypeError, ValueError):
            return False
        return (
            chunk_size == int(requirements['weight_chunk_size'])
            and quantization.get('weight_mode') == 'chunk'
            and quantization.get('weight_source') == 'prequantized_state_dict'
            and quantization.get('weight_format') == row.get('weight_dt')
        )

    adapter = config.get('adapter', {})
    evaluation = config.get('evaluation', {})
    input_quant = evaluation.get('input_quant') or evaluation.get(
        'dynamic_input_quant'
    )
    if not isinstance(input_quant, dict):
        return False
    try:
        input_chunk_size = int(input_quant.get('chunk_size'))
        input_size = int(adapter.get('input_size', 224))
    except (TypeError, ValueError):
        return False
    return (
        input_quant.get('mode') == 'uniform'
        and str(input_quant.get('transport')).strip().lower() == 'encoded'
        and input_quant.get('format') == row.get('activation_dt')
        and input_chunk_size == int(requirements['input_chunk_size'])
        and input_size == int(requirements.get('input_size', 224))
        and _normalised_string_list(adapter.get('excluded_ops'))
        == _normalised_string_list(requirements.get('excluded_ops'))
        and _normalised_string_list(input_quant.get('unsigned_input_sources'))
        == _normalised_string_list(requirements.get('unsigned_input_sources'))
        and bool(input_quant.get('uniform_unsigned_input_candidates', True))
        == bool(requirements.get('uniform_unsigned_input_candidates', True))
    )


def _compatible_source_rows(rows, kind, requirements):
    if rows.empty or not requirements:
        return rows
    mask = rows.apply(
        lambda row: _row_matches_source_requirements(row, kind, requirements),
        axis=1,
    ).astype(bool)
    return rows.loc[mask].copy()


def _row_matches_dynamic_source_requirements(row, requirements):
    if not requirements:
        return True
    config = _row_config(row)
    if not config:
        return False
    model = config.get('model', {})
    dataset = config.get('dataset', {})
    adapter = config.get('adapter', {})
    input_quant = _dynamic_input_cfg_from_row(row)
    if not isinstance(input_quant, dict):
        return False
    try:
        chunk_size = int(input_quant.get('chunk_size'))
        input_size = int(adapter.get('input_size', 224))
    except (TypeError, ValueError):
        return False
    return (
        model.get('weights') == requirements.get('model_weights')
        and dataset.get('name') == requirements.get('dataset_name')
        and _same_dataset_path(
            dataset.get('path'), requirements.get('dataset_path')
        )
        and (
            not requirements.get('require_full_evaluation', True)
            or _row_is_full_evaluation(row, config)
        )
        and _row_fold_input_norm(row, config)
        == bool(requirements.get('fold_input_norm', True))
        and str(input_quant.get('mode')).strip().lower() == 'dynamic'
        and str(input_quant.get('transport')).strip().lower() == 'encoded'
        and chunk_size == int(requirements['input_chunk_size'])
        and input_size == int(requirements.get('input_size', 224))
        and _normalised_string_list(adapter.get('excluded_ops'))
        == _normalised_string_list(requirements.get('excluded_ops'))
        and _normalised_string_list(input_quant.get('unsigned_input_sources'))
        == _normalised_string_list(requirements.get('unsigned_input_sources'))
    )


def _build_best_db_sweep_plan(
    runs,
    model_name,
    dynamic_candidates,
    requested_bit_widths=None,
    weight_experiment_type=DEFAULT_WEIGHT_BASELINE_EXPERIMENT_TYPE,
    input_experiment_type=DEFAULT_INPUT_BASELINE_EXPERIMENT_TYPE,
    source_requirements=None,
):
    """Resolve one best weight format and fixed/dynamic inputs for each width."""
    candidate_groups = _candidate_formats_by_bit_width(dynamic_candidates)
    if requested_bit_widths is None:
        bit_widths = list(candidate_groups)
    else:
        bit_widths = list(requested_bit_widths)

    missing_candidate_widths = [
        width for width in bit_widths if width not in candidate_groups
    ]
    if missing_candidate_widths:
        raise ValueError(
            "No dynamic input candidates were provided for bit width(s): "
            + ', '.join(str(width) for width in missing_candidate_widths)
        )

    weight_rows = _successful_experiment_rows(
        runs, model_name, weight_experiment_type
    )
    weight_rows = weight_rows[
        (weight_rows['activation_dt'] == 'fp32') &
        (weight_rows['weight_dt'] != 'fp32') &
        (weight_rows['weight_dt'].map(_format_bit_width).notna())
    ]
    weight_rows = _compatible_source_rows(
        weight_rows, 'weight', source_requirements
    )
    best_weight = _best_accuracy_row(
        weight_rows,
        f"{model_name!r} weight baselines ({weight_experiment_type})",
    )

    input_rows = _successful_experiment_rows(
        runs, model_name, input_experiment_type
    )
    input_rows = input_rows[
        (input_rows['weight_dt'] == 'fp32') &
        (input_rows['activation_dt'].map(_format_bit_width).notna())
    ].copy()
    if not input_rows.empty:
        encoded_mask = input_rows.apply(
            _row_uses_encoded_activation_transport, axis=1
        ).astype(bool)
        input_rows = input_rows.loc[encoded_mask].copy()
    input_rows = _compatible_source_rows(
        input_rows, 'input', source_requirements
    )
    if input_rows.empty:
        qualifier = " compatible" if source_requirements else ""
        raise ValueError(
            f"Missing successful{qualifier} input baselines for {model_name!r} "
            f"({input_experiment_type})."
        )
    input_rows['_bit_width'] = input_rows['activation_dt'].map(_format_bit_width)

    entries = []
    missing_baseline_widths = []
    for bit_width in bit_widths:
        width_rows = input_rows[input_rows['_bit_width'] == bit_width]
        if width_rows.empty:
            missing_baseline_widths.append(bit_width)
            continue
        best_input = _best_accuracy_row(
            width_rows,
            f"{model_name!r} {bit_width}-bit input baselines ({input_experiment_type})",
        )
        entries.append({
            'mode': 'fixed',
            'bit_width': bit_width,
            'format': str(best_input['activation_dt']),
            'source_acc1': float(best_input['acc1']),
            'source_run_id': int(best_input['id']),
        })
        entries.append({
            'mode': 'dynamic',
            'bit_width': bit_width,
            'candidate_formats': list(candidate_groups[bit_width]),
        })

    if missing_baseline_widths:
        available = sorted(int(width) for width in input_rows['_bit_width'].dropna().unique())
        raise ValueError(
            f"Missing successful input baselines for {model_name!r} at bit width(s) "
            f"{missing_baseline_widths}; available widths: {available}."
        )

    return {
        'weight_format': str(best_weight['weight_dt']),
        'weight_source_acc1': float(best_weight['acc1']),
        'weight_source_run_id': int(best_weight['id']),
        'entries': entries,
    }


def _build_bidirectional_db_sweep_plan(
    runs,
    model_name,
    dynamic_candidates,
    requested_bit_widths=None,
    weight_baseline_experiment_type=DEFAULT_WEIGHT_BASELINE_EXPERIMENT_TYPE,
    weight_optimized_experiment_type=DEFAULT_WEIGHT_OPTIMIZED_EXPERIMENT_TYPE,
    input_baseline_experiment_type=DEFAULT_INPUT_BASELINE_EXPERIMENT_TYPE,
    input_dynamic_experiment_type=DEFAULT_INPUT_DYNAMIC_EXPERIMENT_TYPE,
    source_requirements=None,
):
    """Build both fixed-best directional sweeps for every requested width."""
    candidate_groups = _candidate_formats_by_bit_width(dynamic_candidates)
    bit_widths = (
        list(candidate_groups)
        if requested_bit_widths is None
        else list(requested_bit_widths)
    )
    missing_candidates = [
        width for width in bit_widths if width not in candidate_groups
    ]
    if missing_candidates:
        raise ValueError(
            "No dynamic input candidates were provided for bit width(s): "
            + ', '.join(str(width) for width in missing_candidates)
        )

    weight_baselines = _successful_experiment_rows(
        runs, model_name, weight_baseline_experiment_type
    )
    weight_baselines = weight_baselines[
        (weight_baselines['activation_dt'] == 'fp32')
        & (weight_baselines['weight_dt'] != 'fp32')
    ].copy()
    weight_baselines['_bit_width'] = weight_baselines['weight_dt'].map(
        _format_bit_width
    )
    weight_baselines = _compatible_source_rows(
        weight_baselines, 'weight', source_requirements
    )

    weight_optimized = runs[
        (runs['model_name'] == model_name)
        & (runs['status'] == 'SUCCESS')
        & (runs['activation_dt'] == 'fp32')
        & runs.apply(
            lambda row: _optimized_weight_bit_width(
                row, weight_optimized_experiment_type
            ),
            axis=1,
        ).notna()
    ].copy()
    weight_optimized['_bit_width'] = weight_optimized.apply(
        lambda row: _optimized_weight_bit_width(
            row, weight_optimized_experiment_type
        ),
        axis=1,
    )
    weight_optimized = _compatible_source_rows(
        weight_optimized, 'weight', source_requirements
    )

    input_baselines = _successful_experiment_rows(
        runs, model_name, input_baseline_experiment_type
    )
    input_baselines = input_baselines[
        input_baselines['weight_dt'].eq('fp32')
    ].copy()
    input_baselines['_bit_width'] = input_baselines['activation_dt'].map(
        _format_bit_width
    )
    if not input_baselines.empty:
        input_baselines = input_baselines.loc[
            input_baselines.apply(
                _row_uses_encoded_activation_transport, axis=1
            ).astype(bool)
        ].copy()
    input_baselines = _compatible_source_rows(
        input_baselines, 'input', source_requirements
    )

    input_dynamic = runs[
        (runs['model_name'] == model_name)
        & (runs['status'] == 'SUCCESS')
        & (runs['weight_dt'] == 'fp32')
        & runs['experiment_type'].map(
            lambda value: _experiment_bit_width(
                value, input_dynamic_experiment_type
            )
        ).notna()
    ].copy()
    input_dynamic['_bit_width'] = input_dynamic['experiment_type'].map(
        lambda value: _experiment_bit_width(
            value, input_dynamic_experiment_type
        )
    )
    if source_requirements and not input_dynamic.empty:
        input_dynamic = input_dynamic.loc[
            input_dynamic.apply(
                lambda row: _row_matches_dynamic_source_requirements(
                    row, source_requirements
                ),
                axis=1,
            ).astype(bool)
        ].copy()

    def weight_option(row, mode):
        return {
            'kind': 'weight',
            'mode': mode,
            'bit_width': int(row['_bit_width']),
            'label': str(row['weight_dt']),
            'source_acc1': float(row['acc1']),
            'source_run_id': int(row['id']),
            'source_config': _row_config(row),
            'quant_map': _safe_json_dict(row.get('quant_map_json')),
        }

    def input_option(row, mode, bit_width):
        option = {
            'kind': 'input',
            'mode': mode,
            'bit_width': int(bit_width),
            'label': str(row['activation_dt']),
            'source_acc1': float(row['acc1']),
            'source_run_id': int(row['id']),
            'source_config': _row_config(row),
        }
        if mode == 'dynamic':
            option['label'] = _dynamic_activation_label(row, bit_width)
            option['input_quant_cfg'] = _dynamic_input_cfg_from_row(row)
        return option

    def best_rows_by_label(rows, column, description):
        return [
            _best_accuracy_row(group, f"{description} {label!r}")
            for label, group in rows.groupby(column, sort=False)
        ]

    widths = []
    missing = []
    for bit_width in bit_widths:
        width_weight_baselines = weight_baselines[
            weight_baselines['_bit_width'] == bit_width
        ]
        width_weight_optimized = weight_optimized[
            weight_optimized['_bit_width'] == bit_width
        ]
        width_input_baselines = input_baselines[
            input_baselines['_bit_width'] == bit_width
        ]
        width_input_dynamic = input_dynamic[
            input_dynamic['_bit_width'] == bit_width
        ]
        absent = []
        if width_weight_baselines.empty:
            absent.append('weight baselines')
        if width_input_baselines.empty:
            absent.append('input baselines')
        if width_input_dynamic.empty:
            absent.append('dynamic inputs')
        if absent:
            missing.append(f"{bit_width}-bit: {', '.join(absent)}")
            continue

        weight_options = [
            weight_option(row, 'baseline')
            for row in best_rows_by_label(
                width_weight_baselines,
                'weight_dt',
                f"{model_name!r} {bit_width}-bit weight baseline",
            )
        ]
        if not width_weight_optimized.empty:
            weight_options.append(
                weight_option(
                    _best_accuracy_row(
                        width_weight_optimized,
                        f"{model_name!r} {bit_width}-bit optimal weights",
                    ),
                    'optimized',
                )
            )
        input_options = [
            input_option(row, 'baseline', bit_width)
            for row in best_rows_by_label(
                width_input_baselines,
                'activation_dt',
                f"{model_name!r} {bit_width}-bit input baseline",
            )
        ]
        input_options.append(
            input_option(
                _best_accuracy_row(
                    width_input_dynamic,
                    f"{model_name!r} {bit_width}-bit dynamic inputs",
                ),
                'dynamic',
                bit_width,
            )
        )
        best_weight = max(
            weight_options,
            key=lambda option: (
                option['source_acc1'], option['source_run_id']
            ),
        )
        best_input = max(
            input_options,
            key=lambda option: (
                option['source_acc1'], option['source_run_id']
            ),
        )
        entries = []
        seen = set()
        for direction, fixed_weight, fixed_input, options in (
            ('weight_fixed', best_weight, None, input_options),
            ('input_fixed', None, best_input, weight_options),
        ):
            for option in options:
                weight = fixed_weight if fixed_weight is not None else option
                input_option_value = fixed_input if fixed_input is not None else option
                key = (weight['source_run_id'], input_option_value['source_run_id'])
                if key in seen:
                    continue
                seen.add(key)
                entries.append({
                    'direction': direction,
                    'bit_width': bit_width,
                    'weight': copy.deepcopy(weight),
                    'input': copy.deepcopy(input_option_value),
                })
        widths.append({
            'bit_width': bit_width,
            'best_weight': best_weight,
            'best_input': best_input,
            'weight_options': weight_options,
            'input_options': input_options,
            'entries': entries,
        })

    if missing:
        raise ValueError(
            "Missing successful source experiments required by the hybrid sweep: "
            + '; '.join(missing)
        )
    return {'widths': widths, 'entries': [
        entry for width in widths for entry in width['entries']
    ]}


def _hybrid_run_exists(
    runs,
    model_name,
    experiment_type,
    weight_dt,
    activation_dt,
    run_identity=None,
):
    if runs is None or runs.empty:
        return False
    matching = runs[
        (runs['model_name'] == model_name) &
        (runs['experiment_type'] == experiment_type) &
        (runs['weight_dt'] == weight_dt) &
        (runs['activation_dt'] == activation_dt) &
        (runs['status'] == 'SUCCESS')
    ]
    if run_identity is not None:
        if 'run_identity' not in matching.columns:
            return False
        matching = matching[matching['run_identity'] == run_identity]
    return not matching.empty


def _pending_hybrid_entries(
    entries,
    runs,
    model_name,
    experiment_type,
    force_rerun=False,
):
    """Remove entries already logged with the same runtime identity."""
    if force_rerun:
        return list(entries)
    return [
        entry
        for entry in entries
        if not _hybrid_run_exists(
            runs,
            model_name,
            experiment_type,
            entry['weight_dt'],
            entry['activation_dt'],
            run_identity=entry.get('run_identity'),
        )
    ]


def get_args():
    parser = argparse.ArgumentParser(
        description="Hybrid experiment: configure specific weight and input quantizations directly."
    )
    # Model
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--weights", type=str, default="DEFAULT")
    parser.add_argument("--models_file", type=str, default=None,
                        help="YAML file with list of models to run on multiple models")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--dataset_path", type=str, default="/data/imagenet/val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--limit_batches", type=int, default=-1,
                        help="Limit batches (-1 = all)")

    # Weight quantization
    parser.add_argument("--weight_mode", type=str, choices=["fixed", "optimized", "best"], required=True,
                        help=("Mode for weight quantization. Together with input_mode=sweep, "
                              "'best' runs the bidirectional per-width database sweep."))
    parser.add_argument("--weight_format", type=str, default="fp8_e4m3",
                        help="Fixed weight format (e.g. 'fp8_e4m3') if weight_mode is 'fixed'.")
    parser.add_argument("--weight_metric", type=str, default="mse",
                        help="Metric (e.g. 'mse', 'l1') if weight_mode is 'optimized'.")
    parser.add_argument("--weight_candidate_formats", type=str, default=None,
                        help="Comma-separated candidates for optimized weights.")
    parser.add_argument("--weight_chunk_size", type=int, default=128,
                        help="Chunk size for weight quantization blocks")
    parser.add_argument("--weight_act_batches", type=int, default=10,
                        help="Calibration batches for activation-aware weight metrics such as act_mse")
    parser.add_argument("--per_chunk_format", action="store_true",
                        help="Enable per-chunk format for weights in optimized mode")
    parser.add_argument("--force_recalc", action="store_true",
                        help="Force recalculation of weight errors even if cached (for optimized mode)")
    parser.add_argument("--skip_weight_analysis", action="store_true",
                        help="Skip weight analysis and load cached quantized weights (for optimized mode)")
    parser.add_argument(
        "--weight_baseline_experiment_type",
        type=str,
        default=DEFAULT_WEIGHT_BASELINE_EXPERIMENT_TYPE,
        help="Database experiment type used to select --weight_mode best.",
    )
    parser.add_argument(
        "--weight_optimized_experiment_type",
        type=str,
        default=DEFAULT_WEIGHT_OPTIMIZED_EXPERIMENT_TYPE,
        help=(
            "Base database experiment type for per-width optimal weights "
            "(default: weight_quant_optimized)."
        ),
    )

    # Input quantization
    parser.add_argument("--input_mode", type=str, choices=["fixed", "dynamic", "sweep"], required=True,
                        help=("Mode for input quantization. 'sweep' fixes the best baseline/optimal "
                              "side and sweeps all options on the opposite side for every width."))
    parser.add_argument("--input_format", type=str, default="fp8_e4m3",
                        help="Fixed input format (e.g. 'fp8_e4m3') if input_mode is 'fixed'.")
    parser.add_argument("--input_metric", type=str, default="mse",
                        help="Metric if input_mode is 'dynamic'. Only 'mse' is supported.")
    parser.add_argument("--input_candidate_formats", type=str, default=None,
                        help="Comma-separated input candidate formats for dynamic input selection.")
    parser.add_argument("--input_chunk_size", type=int, default=128,
                        help="Chunk size for dynamic input quantization")
    parser.add_argument("--use_cache_sim_db", action="store_true",
                        help="Use cache simulation results from DB for residency-aware quantization")
    parser.add_argument("--unsigned_input_sources", type=str, default=None,
                        help="Comma-separated list of ops whose output is always unsigned (e.g. 'relu,softmax')")
    parser.add_argument(
        "--excluded_ops",
        type=str,
        default=None,
        help=(
            "Comma-separated ops excluded from input quantization. For a DB "
            "sweep, defaults to none."
        ),
    )
    parser.add_argument("--dynamic_unsigned_input_candidates", action="store_true", default=True,
                        help="Allow using unsigned formats (UFP) for layers with unsigned inputs")
    parser.add_argument("--no_dynamic_unsigned_input_candidates", action="store_false", dest="dynamic_unsigned_input_candidates",
                        help="Disable using unsigned formats (UFP) for layers with unsigned inputs")
    parser.add_argument("--skip_depthwise_input_quant", action="store_true",
                        help="Ablation: leave depthwise Conv2d inputs in FP32 while keeping other dynamic input quantization enabled")
    parser.add_argument("--fold_input_norm", action="store_true", default=True,
                        help="Fold input normalization into first layer weights and quantize first layer")
    parser.add_argument("--no_fold_input_norm", action="store_false", dest="fold_input_norm",
                        help="Disable input normalization folding and first layer quantization")
    parser.add_argument(
        "--input_baseline_experiment_type",
        type=str,
        default=DEFAULT_INPUT_BASELINE_EXPERIMENT_TYPE,
        help="Database experiment type used to select best per-width input baselines.",
    )
    parser.add_argument(
        "--input_dynamic_experiment_type",
        type=str,
        default=DEFAULT_INPUT_DYNAMIC_EXPERIMENT_TYPE,
        help=(
            "Base database experiment type for per-width dynamic inputs "
            "(default: input_quant_dynamic)."
        ),
    )
    parser.add_argument(
        "--input_bit_widths",
        type=str,
        default=None,
        help="Comma-separated widths for --input_mode sweep (default: all candidate widths).",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Re-run successful hybrid sweep entries already present in runs.db.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve and print the database-selected sweep without loading a model.",
    )

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--experiment_type", type=str, default=DEFAULT_HYBRID_EXPERIMENT_TYPE,
                        help="Experiment type label for database logging")

    args = parser.parse_args()
    args.weight_candidate_formats = _parse_csv_arg(args.weight_candidate_formats, HYBRID_WEIGHT_FORMATS)
    args.input_candidate_formats = _parse_csv_arg(args.input_candidate_formats, HYBRID_INPUT_CANDIDATE_FORMATS)
    try:
        args.input_bit_widths = _parse_bit_widths(args.input_bit_widths)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.weight_mode == 'best') != (args.input_mode == 'sweep'):
        parser.error(
            "--weight_mode best and --input_mode sweep must be used together."
        )
    # A DB-selected sweep reproduces the default semantic profile used by
    # find_optimal_input_quant unless the user explicitly overrides it.
    if args.input_mode == 'sweep':
        default_unsigned_sources = [
            'relu', 'softmax', 'quantrelu', 'quantsoftmax'
        ]
    else:
        default_unsigned_sources = []
    default_excluded_ops = []
    args.unsigned_input_sources = _parse_csv_arg(
        args.unsigned_input_sources, default_unsigned_sources
    )
    args.excluded_ops = _parse_csv_arg(args.excluded_ops, default_excluded_ops)
    
    return args


def _make_weight_args(args):
    """Build a namespace compatible with weight-quant helpers."""
    wa = types.SimpleNamespace()
    wa.model_name = args.model_name
    wa.weights = args.weights
    wa.weight_chunk_size = args.weight_chunk_size
    wa.per_chunk_format = args.per_chunk_format
    wa.dataset_name = args.dataset_name
    wa.dataset_path = args.dataset_path
    wa.batch_size = args.batch_size
    wa.num_workers = args.num_workers
    wa.limit_batches = args.limit_batches
    wa.force_recalc = args.force_recalc
    wa.plot_layers = False
    wa.skip_layer_wise = False
    wa.run_eval = False
    wa.include_fp32 = False
    wa.baseline_formats = ','.join(args.weight_candidate_formats)
    wa.metrics = args.weight_metric
    return wa


def run_weight_phase(runner, args, device, model_dir, base_config):
    """
    Analyse weights and build per-layer/chunk optimal quantized state dict.
    Returns: (quantized_weights_path, quant_map, layer_types)
    """
    m = args.weight_metric
    wa = _make_weight_args(args)

    analysis_dir = os.path.join(model_dir, "weight_phase_fp32")
    model, adapter, _ = runner.prepare_model_with_materialized_weights(
        config=base_config,
        output_dir=analysis_dir,
    )

    qt_options = list(args.weight_candidate_formats)
    supported_ops = tuple(OpRegistry.get_supported_ops().keys())
    layer_results_map = {}

    if not args.skip_weight_analysis:
        print(f"\n[Weight Phase] Analysing weight tensors for metric: {m} ...")
        run_weight_quantization_analysis(
            wa, model, [m], qt_options, layer_results_map, supported_ops
        )
    else:
        print("\n[Weight Phase] --skip_weight_analysis set; loading cached quant maps ...")
        csv_path = os.path.join(model_dir, m, "layer_errors.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lname = row['layer']
                    if lname not in layer_results_map:
                        layer_results_map[lname] = {
                            'layer': lname, 'shape': row.get('shape', ''),
                            'max_val': float(row.get('max_val', 0)),
                            'metrics': {}, 'chunk_wins': {}, 'chunk_winners': {},
                        }
                    errs = {}
                    for k, v in row.items():
                        if k.endswith('_error'):
                            fmt = k[:-6]
                            if fmt not in qt_options:
                                continue
                            try:
                                errs[fmt] = float(v) if v else float('inf')
                            except ValueError:
                                errs[fmt] = float('inf')
                    layer_results_map[lname]['metrics'][m] = errs
        else:
            print(f"  WARNING: no cached CSV at {csv_path} for metric {m}")

    metric_dir = os.path.join(model_dir, f"weights_{m}")
    os.makedirs(metric_dir, exist_ok=True)

    q_state_dict, q_map = create_quantized_state_dict(
        model, layer_results_map, wa, m, use_chunking=args.per_chunk_format
    )
    q_state_dict = _materialize_weight_buffers_from_map(model, q_state_dict, q_map, args)
    q_path = os.path.join(metric_dir, "quantized_weights.pt")
    torch.save(q_state_dict, q_path)

    with open(os.path.join(metric_dir, "quantization_map.json"), 'w') as f:
        json.dump(q_map, f, indent=4)

    print(f"[Weight Phase] Saved optimized quantized weights ({m}) → {q_path}")

    layer_types = {
        name: type(module).__name__
        for name, module in model.named_modules()
        if name
    }

    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()

    return q_path, q_map, layer_types


def _iter_weight_modules(model):
    supported = (torch.nn.Conv2d, torch.nn.Linear)
    for name, module in model.named_modules():
        if name and isinstance(module, supported) and getattr(module, 'weight', None) is not None:
            yield name, module


def _disable_runtime_io_quant(model):
    for module in model.modules():
        if hasattr(module, 'input_quantization'):
            module.input_quantization = False
        if hasattr(module, 'output_quantization'):
            module.output_quantization = False
        if hasattr(module, 'weight_fp8'):
            module.weight_fp8 = None
        if hasattr(module, 'weight_scale'):
            module.weight_scale = None
        if hasattr(module, 'weight_scale_packed'):
            module.weight_scale_packed = None


def _activation_weight_error_for_module(module, inputs, ref_output, quantized_weights):
    import torch.nn.functional as F

    x = inputs[0] if isinstance(inputs, tuple) else inputs
    if not isinstance(x, torch.Tensor) or not isinstance(ref_output, torch.Tensor):
        return {}

    errors = {}
    with torch.no_grad():
        for fmt, q_weight in quantized_weights.items():
            try:
                if isinstance(module, torch.nn.Conv2d):
                    q_out = F.conv2d(
                        x,
                        q_weight,
                        module.bias,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                    )
                elif isinstance(module, torch.nn.Linear):
                    q_out = F.linear(x, q_weight, module.bias)
                else:
                    continue
                diff = ref_output - q_out
                errors[fmt] = (diff.pow(2).sum().detach(), diff.numel())
            except Exception:
                errors[fmt] = (torch.tensor(float('inf'), device=x.device), 1)
    return errors


def _materialize_weight_buffers_from_map(model, q_state_dict, q_map, args):
    """
    Keep saved `weight`, `weight_fp8`, and `weight_scale` mutually consistent.

    The evaluation path prefers weight_fp8/weight_scale when present. A state dict
    with quantized `.weight` tensors but stale FP8 buffers can therefore evaluate a
    different model than the one described by the quantization map.
    """
    modules = dict(model.named_modules())
    for layer_name, selected_format in q_map.items():
        module = modules.get(layer_name)
        if module is None or not hasattr(module, 'calibrate_weights'):
            continue
        module.weight_quantization = True
        module.weight_chunk_size = args.weight_chunk_size
        if isinstance(selected_format, list):
            module.chunk_formats = selected_format
            module.weight_mode = 'chunk'
            module.q_type = selected_format[0] if selected_format else getattr(module, 'q_type', 'fp8_e1m6')
        else:
            # Standalone weight baselines use per-context chunks. Broadcast the
            # selected format through that same path so the hybrid checkpoint is
            # numerically identical to the run from which the format was chosen.
            module.chunk_formats = [selected_format]
            module.weight_mode = 'chunk'
            module.q_type = selected_format
        module.calibrate_weights()

    # Calibration above quantizes each original FP32 tensor exactly once. Keep
    # those freshly calibrated transport buffers, while taking the already
    # dequantized parameter tensors (and all unrelated state) from q_state_dict.
    # Loading q_state_dict before calibration would quantize those weights twice.
    materialized = model.state_dict()
    calibrated_buffer_suffixes = (
        '.weight_fp8',
        '.weight_scale',
        '.weight_scale_packed',
    )
    for key, value in q_state_dict.items():
        if key.endswith(calibrated_buffer_suffixes):
            continue
        materialized[key] = value
    return materialized


def run_activation_weight_phase(runner, args, device, model_dir, base_config):
    """
    Activation-aware layer-wise weight selection.

    For each Conv2d/Linear-like quantized module, run a small calibration window
    through the FP32-weight model, and choose the weight format that minimizes the
    module's local output MSE for the observed inputs. This intentionally ignores
    per-chunk format selection; it answers whether layer sensitivity, not raw
    weight reconstruction, is driving the MobileNetV3 drop.
    """
    metric = args.weight_metric
    analysis_dir = os.path.join(model_dir, f"weight_phase_{metric}")
    model, adapter, _ = runner.prepare_model_with_materialized_weights(
        config=base_config,
        output_dir=analysis_dir,
    )
    model.eval()
    _disable_runtime_io_quant(model)

    loader = _build_loader(args, device, runner, config_builder=lambda _: base_config)
    calib_batches = int(args.weight_act_batches if args.weight_act_batches > 0 else 10)
    if args.limit_batches and args.limit_batches > 0:
        calib_batches = min(calib_batches, int(args.limit_batches))

    qt_options = [fmt for fmt in args.weight_candidate_formats if str(fmt).lower() != 'fp32']
    layer_results_map = {}

    print(
        f"\n[Weight Phase] Activation-aware weight search ({metric}) "
        f"over {calib_batches} calibration batches ..."
    )

    for layer_name, module in tqdm(list(_iter_weight_modules(model)), desc="Analyzing Layers (ActMSE)"):
        q_weights = {}
        for fmt in qt_options:
            try:
                q_w, _ = get_quantized_tensor_sim(
                    module.weight.detach(),
                    fmt,
                    chunk_size=args.weight_chunk_size,
                )
                q_weights[fmt] = q_w.detach()
            except Exception:
                pass
        if not q_weights:
            continue

        sum_err = {fmt: 0.0 for fmt in q_weights}
        sum_numel = {fmt: 0 for fmt in q_weights}

        def hook_fn(mod, inputs, output):
            batch_errors = _activation_weight_error_for_module(mod, inputs, output, q_weights)
            for fmt, (err_sum, numel) in batch_errors.items():
                sum_err[fmt] += float(err_sum.item())
                sum_numel[fmt] += int(numel)

        handle = module.register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= calib_batches:
                        break
                    inputs, targets = adapter.prepare_batch(batch)
                    inputs = runner._to_device(inputs)
                    targets = runner._to_device(targets)
                    adapter.forward(model, (inputs, targets))
        finally:
            handle.remove()

        metrics = {
            fmt: (sum_err[fmt] / sum_numel[fmt] if sum_numel[fmt] > 0 else float('inf'))
            for fmt in q_weights
        }
        best_fmt = min(metrics, key=metrics.get)
        layer_results_map[layer_name] = {
            'layer': layer_name,
            'shape': tuple(module.weight.shape),
            'max_val': float(module.weight.detach().abs().max().item()),
            'numel': int(module.weight.numel()),
            'metrics': {metric: metrics},
            'best_error': metrics[best_fmt],
        }

    metric_dir = os.path.join(model_dir, f"weights_{metric}")
    os.makedirs(metric_dir, exist_ok=True)
    q_state_dict, q_map = create_quantized_state_dict(
        model, layer_results_map, args, metric, use_chunking=False
    )
    q_state_dict = _materialize_weight_buffers_from_map(model, q_state_dict, q_map, args)
    q_path = os.path.join(metric_dir, "quantized_weights.pt")
    torch.save(q_state_dict, q_path)
    with open(os.path.join(metric_dir, "quantization_map.json"), 'w') as f:
        json.dump(q_map, f, indent=4)

    print(f"[Weight Phase] Saved activation-aware quantized weights ({metric}) → {q_path}")

    layer_types = _layer_types_from_model(model)
    del model, adapter, loader
    gc.collect()
    torch.cuda.empty_cache()
    return q_path, q_map, layer_types


def _build_uniform_quant_state_dict(model, fmt, chunk_size):
    """
    Build a quantized state dict where every Conv2d/Linear layer weight is
    uniformly quantized to `fmt`.  Returns (state_dict, quant_map).
    """
    supported = (torch.nn.Conv2d, torch.nn.Linear)
    state_dict = model.state_dict()
    quant_map = {}
    for name, module in model.named_modules():
        if not isinstance(module, supported):
            continue
        weight_key = f"{name}.weight"
        if weight_key not in state_dict:
            continue
        w = state_dict[weight_key]
        w_dequant, _ = get_quantized_tensor_sim(w, fmt, chunk_size=chunk_size)
        state_dict[weight_key] = w_dequant
        quant_map[name] = fmt
    return state_dict, quant_map


def _plain_weight_quant_map(raw_map):
    """Strip dashboard enrichment and retain layer -> format/list mappings."""
    plain = {}
    for layer_name, spec in (raw_map or {}).items():
        if isinstance(spec, dict):
            spec = spec.get('format')
        if isinstance(spec, str) or (
            isinstance(spec, list) and all(isinstance(item, str) for item in spec)
        ):
            plain[str(layer_name)] = copy.deepcopy(spec)
    return plain


def _build_quant_state_dict_from_map(model, quant_map, chunk_size):
    """Recreate an optimal weight checkpoint from its logged quantization map."""
    quant_map = _plain_weight_quant_map(quant_map)
    if not quant_map:
        raise ValueError("The selected optimal weight run has no usable quantization map.")
    has_chunk_map = any(isinstance(spec, list) for spec in quant_map.values())
    layer_results = {}
    for layer_name, spec in quant_map.items():
        record = {
            'metrics': {'source_map': {}},
            'chunk_winners': {},
        }
        if isinstance(spec, list):
            record['chunk_winners']['source_map'] = list(spec)
        else:
            record['metrics']['source_map'][spec] = 0.0
        layer_results[layer_name] = record
    helper_args = types.SimpleNamespace(
        per_chunk_format=has_chunk_map,
        weight_chunk_size=chunk_size,
    )
    state_dict, rebuilt_map = create_quantized_state_dict(
        model,
        layer_results,
        helper_args,
        'source_map',
        use_chunking=has_chunk_map,
    )
    return state_dict, rebuilt_map


def _build_weight_materialization_source_config(weight_config):
    """Build quantized wrappers around FP32 weights without calibrating them.

    Hybrid weight reconstruction quantizes the source weights explicitly after
    loading them.  Database labels such as ``opt_chunk_mse`` describe that
    reconstruction and are not codec formats, so they must not reach adapter
    weight calibration as ``q_type``.
    """
    source_config = copy.deepcopy(weight_config)
    source_config.setdefault('adapter', {})['weight_quantization'] = False
    quantization = source_config.setdefault('quantization', {})
    quantization.pop('format', None)
    quantization['weight_source'] = 'fp32'
    return source_config


def _summarise_quant_map(quant_map, prefix="opt"):
    if not quant_map:
        return f"{prefix}_unknown"
    counts = {}
    for v in quant_map.values():
        key = str(v) if not isinstance(v, list) else "per_chunk"
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{fmt}x{cnt}" for fmt, cnt in sorted(counts.items(), key=lambda x: -x[1])]
    return prefix + "[" + ",".join(parts[:5]) + "]"


def _build_hybrid_log_config(
    base_config,
    *,
    experiment_name,
    experiment_type,
    weight_dt,
    activation_dt,
    ref_acc1=None,
    ref_acc5=None,
    ref_certainty=None,
    certainty=None,
    mse=None,
    quant_map_json=None,
    input_map_json=None,
    input_quant_cfg=None,
    selection_metadata=None,
):
    cfg = copy.deepcopy(base_config)
    if input_quant_cfg is not None:
        cfg.setdefault('evaluation', {})['input_quant'] = copy.deepcopy(
            input_quant_cfg
        )
    if selection_metadata is not None:
        # Provenance belongs in the stored config, but not in Runner's semantic
        # identity: a newer source row selecting the same formats must remain
        # resumable as the same hybrid configuration.
        cfg.setdefault('meta', {})['selection'] = copy.deepcopy(
            selection_metadata
        )
    cfg['experiment'] = {
        'name': experiment_name,
        'type': experiment_type,
        'weight_dt': weight_dt,
        'activation_dt': activation_dt,
        'ref_acc1': ref_acc1,
        'ref_acc5': ref_acc5,
        'ref_certainty': ref_certainty,
        'metrics': {
            'mse': mse,
            'certainty': certainty,
        },
        'quant_map_json': quant_map_json,
        'input_map_json': input_map_json,
    }
    return cfg


def _log_hybrid_run(
    runner,
    base_config,
    model_name,
    weight_dt,
    activation_dt,
    acc1,
    acc5,
    status,
    experiment_name='hybrid_quant',
    ref_acc1=None,
    ref_acc5=None,
    ref_certainty=None,
    certainty=None,
    mse=None,
    quant_map_json=None,
    input_map_json=None,
    input_quant_stats=None,
    input_quant_cfg=None,
    selection_metadata=None,
):
    experiment_type = (
        runner.args.experiment_type
        if hasattr(runner, 'args') and hasattr(runner.args, 'experiment_type')
        else 'hybrid_quant'
    )
    cfg = _build_hybrid_log_config(
        base_config,
        experiment_name=experiment_name,
        experiment_type=experiment_type,
        weight_dt=weight_dt,
        activation_dt=activation_dt,
        ref_acc1=ref_acc1,
        ref_acc5=ref_acc5,
        ref_certainty=ref_certainty,
        certainty=certainty,
        mse=mse,
        quant_map_json=quant_map_json,
        input_map_json=input_map_json,
        input_quant_cfg=input_quant_cfg,
        selection_metadata=selection_metadata,
    )
    result = {
        'model_name': model_name,
        'status': status,
        'acc1': acc1,
        'acc5': acc5,
        'certainty': certainty if certainty is not None else 0.0,
    }
    if input_quant_stats is not None:
        result['input_quant'] = input_quant_stats
    runner.log_experiment_result(cfg, result)


def _format_family(fmt):
    value = str(fmt).strip()
    return value.split('_', 1)[0] if value else 'unknown'


def _save_hybrid_layer_stats(
    model_dir,
    run_label,
    input_stats,
    quant_map,
    *,
    acc1,
    acc5,
    certainty,
    norm_mse,
    weight_mode,
    weight_dt,
    input_mode,
    input_dt,
    bit_width=None,
):
    out_dir = os.path.join(model_dir, run_label)
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "layer_stats.json")
    save_data = copy.deepcopy(
        input_stats.get('layer_stats', {}) if input_stats else {}
    )

    for layer_name, spec in quant_map.items():
        if layer_name not in save_data:
            save_data[layer_name] = {}
        if isinstance(spec, list):
            counts = {}
            for fmt in spec:
                counts[str(fmt)] = counts.get(str(fmt), 0) + 1
            save_data[layer_name]['weight_format_counts'] = counts
            save_data[layer_name]['weight_total_chunks'] = len(spec)
            if counts:
                save_data[layer_name]['weight_format'] = sorted(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )[0][0]
        else:
            save_data[layer_name]['weight_format'] = str(spec)
            save_data[layer_name]['weight_format_counts'] = {str(spec): 1}
            save_data[layer_name]['weight_total_chunks'] = 1

    save_data['accuracy'] = {
        'top1': acc1,
        'top5': acc5,
        'certainty': certainty,
        'norm_mse': norm_mse,
        'weight_mode': weight_mode,
        'weight_dt': weight_dt,
        'input_mode': input_mode,
        'input_dt': input_dt,
        'bit_width': bit_width,
    }
    with open(stats_path, 'w') as f:
        json.dump(save_data, f, indent=4)
    return stats_path


def _process_best_db_sweep(args, device):
    """Run one DB-selected weight format against fixed/dynamic inputs by width."""
    args.input_metric = "mse"
    model_name = args.model_name
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    base_config = _base_runtime_config(
        args, model_name=model_name, weights=args.weights
    )
    base_config.setdefault('adapter', {})
    base_config['adapter']['quantized_ops'] = ['all']
    base_config['adapter']['input_quantization'] = True
    base_config['adapter']['weight_quantization'] = True
    base_config['adapter']['fold_input_norm'] = args.fold_input_norm
    base_config['adapter']['quantize_first_layer'] = args.fold_input_norm
    base_config['adapter']['excluded_ops'] = list(args.excluded_ops)
    base_config['adapter']['input_size'] = args.input_size

    runner = Runner(device)
    runner.args = args
    db = runner._get_db()
    existing_runs = db.get_runs()
    plan = _build_best_db_sweep_plan(
        existing_runs,
        model_name,
        args.input_candidate_formats,
        requested_bit_widths=args.input_bit_widths,
        weight_experiment_type=args.weight_baseline_experiment_type,
        input_experiment_type=args.input_baseline_experiment_type,
        source_requirements={
            'model_weights': args.weights,
            'dataset_name': args.dataset_name,
            'dataset_path': args.dataset_path,
            'weight_chunk_size': args.weight_chunk_size,
            'input_chunk_size': args.input_chunk_size,
            'fold_input_norm': args.fold_input_norm,
            'input_size': args.input_size,
            'excluded_ops': args.excluded_ops,
            'unsigned_input_sources': args.unsigned_input_sources,
            'uniform_unsigned_input_candidates': (
                args.dynamic_unsigned_input_candidates
            ),
            'require_full_evaluation': True,
        },
    )

    dynamic_widths = [
        entry['bit_width'] for entry in plan['entries']
        if entry['mode'] == 'dynamic'
    ]
    if args.use_cache_sim_db and any(width != 8 for width in dynamic_widths):
        raise ValueError(
            "--use_cache_sim_db is compatible only with an 8-bit dynamic input "
            "sweep; pass --input_bit_widths 8 or disable cache simulation."
        )

    weight_dt_str = plan['weight_format']
    base_config.setdefault('evaluation', {})['max_batches'] = args.limit_batches
    base_config.setdefault('quantization', {}).update({
        'format': weight_dt_str,
        'weight_mode': 'chunk',
        'weight_chunk_size': args.weight_chunk_size,
        'weight_source': 'prequantized_state_dict',
        'unsigned_input_sources': list(args.unsigned_input_sources),
    })
    for entry in plan['entries']:
        bit_width = entry['bit_width']
        if entry['mode'] == 'fixed':
            entry['activation_dt'] = entry['format']
            entry['run_label'] = (
                f"hybrid_best_baseline_{bit_width}bit_{entry['format']}"
            )
            entry['input_quant_cfg'] = _build_uniform_input_quant_cfg(
                entry['format'],
                args.input_chunk_size,
                unsigned_input_sources=args.unsigned_input_sources,
                use_unsigned_input_candidates=(
                    args.dynamic_unsigned_input_candidates
                ),
            )
            input_selection = {
                'mode': 'best_baseline',
                'experiment_type': args.input_baseline_experiment_type,
                'source_run_id': entry['source_run_id'],
                'source_acc1': entry['source_acc1'],
                'format': entry['format'],
                'bit_width': bit_width,
            }
        else:
            entry['activation_dt'] = (
                f"dyn_input_{args.input_metric}_{bit_width}bit"
            )
            entry['run_label'] = (
                f"hybrid_best_dynamic_{bit_width}bit_{args.input_metric}"
            )
            entry['input_quant_cfg'] = _build_dynamic_input_quant_cfg(
                metric=args.input_metric,
                chunk_size=args.input_chunk_size,
                candidate_formats=entry['candidate_formats'],
                use_cache_sim_db=args.use_cache_sim_db,
                model_name=model_name,
                unsigned_input_sources=args.unsigned_input_sources,
                dynamic_unsigned_input_candidates=(
                    args.dynamic_unsigned_input_candidates
                ),
                skip_depthwise_input_quant=args.skip_depthwise_input_quant,
            )
            input_selection = {
                'mode': 'dynamic',
                'metric': args.input_metric,
                'candidate_formats': entry['candidate_formats'],
                'bit_width': bit_width,
            }

        entry['selection_metadata'] = {
            'weight': {
                'mode': 'best_baseline',
                'experiment_type': args.weight_baseline_experiment_type,
                'source_run_id': plan['weight_source_run_id'],
                'source_acc1': plan['weight_source_acc1'],
                'format': weight_dt_str,
                'chunk_size': args.weight_chunk_size,
            },
            'input': input_selection,
        }
        entry['experiment_name'] = (
            f"hybrid_quant_{_format_family(weight_dt_str)}/"
            f"fp{bit_width}_{entry['mode']}"
        )
        if args.use_cache_sim_db:
            entry['experiment_name'] += "_w_cache_sim"

        identity_cfg = _build_hybrid_log_config(
            base_config,
            experiment_name=entry['experiment_name'],
            experiment_type=args.experiment_type,
            weight_dt=weight_dt_str,
            activation_dt=entry['activation_dt'],
            input_quant_cfg=entry['input_quant_cfg'],
            selection_metadata=entry['selection_metadata'],
        )
        entry['run_identity'] = runner._run_identity(identity_cfg)

    print(f"\n{'=' * 72}")
    print(f" HYBRID BEST-FROM-DB SWEEP: {model_name}")
    print(
        f"  Weight: {weight_dt_str} "
        f"(source Top1={plan['weight_source_acc1']:.3f}%, "
        f"run_id={plan['weight_source_run_id']})"
    )
    for entry in plan['entries']:
        if entry['mode'] == 'fixed':
            detail = (
                f"best baseline={entry['format']} "
                f"(source Top1={entry['source_acc1']:.3f}%, "
                f"run_id={entry['source_run_id']})"
            )
        else:
            detail = f"dynamic candidates={entry['candidate_formats']}"
        already_done = _hybrid_run_exists(
            existing_runs,
            model_name,
            args.experiment_type,
            weight_dt_str,
            entry['activation_dt'],
            run_identity=entry['run_identity'],
        )
        disposition = " [already successful]" if already_done else ""
        print(
            f"  {entry['bit_width']}-bit {entry['mode']}: {detail}{disposition}"
        )
    print(f"{'=' * 72}")

    if args.dry_run:
        print("[Dry Run] Selection resolved; no model was loaded.")
        return

    for entry in plan['entries']:
        if (
            not args.force_rerun
            and _hybrid_run_exists(
                existing_runs,
                model_name,
                args.experiment_type,
                weight_dt_str,
                entry['activation_dt'],
                run_identity=entry['run_identity'],
            )
        ):
            print(
                f"[DB] Skipping {entry['activation_dt']} — successful hybrid "
                "result already exists."
            )
        else:
            pending_entries.append(entry)

    if not pending_entries:
        print(f"[Done] All requested hybrid runs already exist for {model_name}.")
        return

    if args.use_cache_sim_db:
        sim = db.get_latest_cache_simulation(model_name)
        if sim is None:
            print(
                f"\n[CacheSim] Model {model_name!r} is missing; "
                "running cache simulation."
            )
            from runspace.experiments.asic_cache_simulation.simulate_cache import (
                run_simulation as run_cache_sim,
            )
            sim_args = types.SimpleNamespace(
                model_name=model_name,
                cache_size=2.0,
                num_banks=16,
                metadata_bits=0,
                batch_size=1,
                device=str(device),
            )
            run_cache_sim(sim_args)

    ref_experiment_name = (
        f"hybrid_quant_{_format_family(weight_dt_str)}/input_sweep"
    )
    if args.use_cache_sim_db:
        ref_experiment_name += "_w_cache_sim"
    ref_acc1, ref_acc5, ref_certainty = _get_or_run_fp32_ref_common(
        runner,
        args,
        device,
        db,
        model_name,
        experiment_name=ref_experiment_name,
    )

    print(
        f"\n[Weights] Building DB-selected uniform weights ({weight_dt_str}) ..."
    )
    model_fp32, adapter_fp32 = _load_fp32_model(
        runner,
        args,
        device,
        config_builder=lambda _: base_config,
    )
    q_state_dict, quant_map = _build_uniform_quant_state_dict(
        model_fp32,
        weight_dt_str,
        chunk_size=args.weight_chunk_size,
    )
    layer_types = _layer_types_from_model(model_fp32)
    q_state_dict = _materialize_weight_buffers_from_map(
        model_fp32,
        q_state_dict,
        quant_map,
        args,
    )
    q_path = os.path.join(model_dir, f"best_weights_{weight_dt_str}.pt")
    torch.save(q_state_dict, q_path)
    quant_map_json = _build_weight_map_json(quant_map, layer_types)
    del model_fp32, adapter_fp32, q_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, adapter = runner.load_model_from_weight_file(
        config=base_config,
        weight_file_path=q_path,
        skip_calibration=True,
    )
    loader = _build_loader(args, device, runner)

    try:
        for entry in pending_entries:
            bit_width = entry['bit_width']
            activation_dt_str = entry['activation_dt']
            input_quant_cfg = entry['input_quant_cfg']
            selection_metadata = entry['selection_metadata']
            experiment_name = entry['experiment_name']

            print(
                f"\n[Inference] Weight={weight_dt_str}, "
                f"Input={activation_dt_str} ..."
            )
            try:
                acc1, acc5, certainty, input_stats = _run_inference(
                    runner,
                    model,
                    adapter,
                    loader,
                    args,
                    input_quant_cfg=input_quant_cfg,
                    desc=(
                        f"Hybrid {model_name} W={weight_dt_str} "
                        f"A={activation_dt_str}"
                    ),
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"  ERROR during {activation_dt_str}: {exc}")
                import traceback
                traceback.print_exc()
                _log_hybrid_run(
                    runner=runner,
                    base_config=base_config,
                    model_name=model_name,
                    weight_dt=weight_dt_str,
                    activation_dt=activation_dt_str,
                    acc1=0.0,
                    acc5=0.0,
                    status="ERROR",
                    experiment_name=experiment_name,
                    ref_acc1=ref_acc1,
                    ref_acc5=ref_acc5,
                    ref_certainty=ref_certainty,
                    quant_map_json=quant_map_json,
                    input_quant_cfg=input_quant_cfg,
                    selection_metadata=selection_metadata,
                )
                continue

            norm_mse = input_stats.get('norm_mse') if input_stats else None
            print(
                f"  Result: Top1={acc1:.2f}%, Top5={acc5:.2f}%, "
                f"Certainty={certainty:.4f}"
                + (
                    f", NormMSE={norm_mse:.4e}"
                    if norm_mse is not None else ""
                )
            )
            _log_hybrid_run(
                runner=runner,
                base_config=base_config,
                model_name=model_name,
                weight_dt=weight_dt_str,
                activation_dt=activation_dt_str,
                acc1=acc1,
                acc5=acc5,
                status="SUCCESS",
                experiment_name=experiment_name,
                ref_acc1=ref_acc1,
                ref_acc5=ref_acc5,
                ref_certainty=ref_certainty,
                certainty=certainty,
                mse=norm_mse,
                quant_map_json=quant_map_json,
                input_quant_stats=input_stats,
                input_quant_cfg=input_quant_cfg,
                selection_metadata=selection_metadata,
            )
            stats_path = _save_hybrid_layer_stats(
                model_dir,
                entry['run_label'],
                input_stats,
                quant_map,
                acc1=acc1,
                acc5=acc5,
                certainty=certainty,
                norm_mse=norm_mse,
                weight_mode='best',
                weight_dt=weight_dt_str,
                input_mode=entry['mode'],
                input_dt=activation_dt_str,
                bit_width=bit_width,
            )
            print(f"  Layer stats: {stats_path}")
    finally:
        del model, adapter, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[Done] Hybrid best-from-DB sweep finished for {model_name}.")


def _process_bidirectional_db_sweep(args, device):
    """Run both per-width fixed-best hybrid sweep directions."""
    args.input_metric = "mse"
    model_name = args.model_name
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    base_config = _base_runtime_config(
        args, model_name=model_name, weights=args.weights
    )
    base_config.setdefault('adapter', {})
    base_config['adapter'].update({
        'quantized_ops': ['all'],
        'input_quantization': True,
        'weight_quantization': True,
        'fold_input_norm': args.fold_input_norm,
        'quantize_first_layer': args.fold_input_norm,
        'excluded_ops': list(args.excluded_ops),
        'input_size': args.input_size,
    })
    base_config.setdefault('evaluation', {})['max_batches'] = args.limit_batches

    runner = Runner(device)
    runner.args = args
    db = runner._get_db()
    existing_runs = db.get_runs()
    source_requirements = {
        'model_weights': args.weights,
        'dataset_name': args.dataset_name,
        'dataset_path': args.dataset_path,
        'weight_chunk_size': args.weight_chunk_size,
        'input_chunk_size': args.input_chunk_size,
        'fold_input_norm': args.fold_input_norm,
        'input_size': args.input_size,
        'excluded_ops': args.excluded_ops,
        'unsigned_input_sources': args.unsigned_input_sources,
        'uniform_unsigned_input_candidates': (
            args.dynamic_unsigned_input_candidates
        ),
        'require_full_evaluation': True,
    }
    plan = _build_bidirectional_db_sweep_plan(
        existing_runs,
        model_name,
        args.input_candidate_formats,
        requested_bit_widths=args.input_bit_widths,
        weight_baseline_experiment_type=(
            args.weight_baseline_experiment_type
        ),
        weight_optimized_experiment_type=(
            args.weight_optimized_experiment_type
        ),
        input_baseline_experiment_type=args.input_baseline_experiment_type,
        input_dynamic_experiment_type=args.input_dynamic_experiment_type,
        source_requirements=source_requirements,
    )

    candidate_groups = _candidate_formats_by_bit_width(
        args.input_candidate_formats
    )
    if args.use_cache_sim_db and any(
        width['bit_width'] != 8 for width in plan['widths']
    ):
        raise ValueError(
            "--use_cache_sim_db is compatible only with an 8-bit dynamic input "
            "sweep; pass --input_bit_widths 8 or disable cache simulation."
        )

    for entry in plan['entries']:
        bit_width = entry['bit_width']
        weight = entry['weight']
        input_option = entry['input']
        if input_option['mode'] == 'baseline':
            input_quant_cfg = _build_uniform_input_quant_cfg(
                input_option['label'],
                args.input_chunk_size,
                unsigned_input_sources=args.unsigned_input_sources,
                use_unsigned_input_candidates=(
                    args.dynamic_unsigned_input_candidates
                ),
            )
        else:
            input_quant_cfg = copy.deepcopy(
                input_option.get('input_quant_cfg')
            )
            if not isinstance(input_quant_cfg, dict):
                input_quant_cfg = _build_dynamic_input_quant_cfg(
                    metric=args.input_metric,
                    chunk_size=args.input_chunk_size,
                    candidate_formats=candidate_groups[bit_width],
                    use_cache_sim_db=args.use_cache_sim_db,
                    model_name=model_name,
                    unsigned_input_sources=args.unsigned_input_sources,
                    dynamic_unsigned_input_candidates=(
                        args.dynamic_unsigned_input_candidates
                    ),
                    skip_depthwise_input_quant=(
                        args.skip_depthwise_input_quant
                    ),
                )
        entry['input_quant_cfg'] = input_quant_cfg
        entry['weight_dt'] = weight['label']
        entry['activation_dt'] = input_option['label']
        entry['selection_metadata'] = {
            'direction': entry['direction'],
            'bit_width': bit_width,
            'weight': {
                'mode': weight['mode'],
                'source_run_id': weight['source_run_id'],
                'source_acc1': weight['source_acc1'],
                'format': weight['label'],
                'chunk_size': args.weight_chunk_size,
            },
            'input': {
                'mode': input_option['mode'],
                'source_run_id': input_option['source_run_id'],
                'source_acc1': input_option['source_acc1'],
                'format': input_option['label'],
                'bit_width': bit_width,
            },
        }
        entry['experiment_name'] = (
            f"hybrid_quant_{_format_family(weight['label'])}/"
            f"fp{bit_width}_{input_option['mode']}"
        )
        if args.use_cache_sim_db:
            entry['experiment_name'] += "_w_cache_sim"
        identity_config = copy.deepcopy(base_config)
        identity_config.setdefault('quantization', {}).update({
            'format': weight['label'],
            'weight_mode': 'chunk',
            'weight_chunk_size': args.weight_chunk_size,
            'weight_source': 'prequantized_state_dict',
        })
        identity_cfg = _build_hybrid_log_config(
            identity_config,
            experiment_name=entry['experiment_name'],
            experiment_type=args.experiment_type,
            weight_dt=entry['weight_dt'],
            activation_dt=entry['activation_dt'],
            input_quant_cfg=input_quant_cfg,
            selection_metadata=entry['selection_metadata'],
        )
        entry['run_identity'] = runner._run_identity(identity_cfg)

    print(f"\n{'=' * 78}")
    print(f" HYBRID BIDIRECTIONAL PER-WIDTH SWEEP: {model_name}")
    for width in plan['widths']:
        print(
            f"  {width['bit_width']}-bit best weight: "
            f"{width['best_weight']['label']} "
            f"({width['best_weight']['mode']}, "
            f"Top1={width['best_weight']['source_acc1']:.3f}%)"
        )
        print(
            f"  {width['bit_width']}-bit best input : "
            f"{width['best_input']['label']} "
            f"({width['best_input']['mode']}, "
            f"Top1={width['best_input']['source_acc1']:.3f}%)"
        )
        print(
            f"    {len(width['input_options'])} input options + "
            f"{len(width['weight_options'])} weight options - 1 duplicate = "
            f"{len(width['entries'])} runs"
        )

    pending_entries = []
    for entry in plan['entries']:
        already_done = _hybrid_run_exists(
            existing_runs,
            model_name,
            args.experiment_type,
            entry['weight_dt'],
            entry['activation_dt'],
            run_identity=entry['run_identity'],
        )
        disposition = "already successful" if already_done else "pending"
        print(
            f"  [{disposition}] {entry['bit_width']}-bit "
            f"{entry['direction']}: W={entry['weight_dt']} "
            f"A={entry['activation_dt']}"
        )
    print(f"{'=' * 78}")

    pending_entries = _pending_hybrid_entries(
        plan['entries'],
        existing_runs,
        model_name,
        args.experiment_type,
        force_rerun=args.force_rerun,
    )

    if args.dry_run:
        print("[Dry Run] Sweep resolved; no model was loaded.")
        return
    if not pending_entries:
        print(f"[Done] All requested hybrid runs already exist for {model_name}.")
        return

    if args.use_cache_sim_db and db.get_latest_cache_simulation(model_name) is None:
        from runspace.experiments.asic_cache_simulation.simulate_cache import (
            run_simulation as run_cache_sim,
        )
        run_cache_sim(types.SimpleNamespace(
            model_name=model_name,
            cache_size=2.0,
            num_banks=16,
            metadata_bits=0,
            batch_size=1,
            device=str(device),
        ))

    ref_acc1, ref_acc5, ref_certainty = _get_or_run_fp32_ref_common(
        runner,
        args,
        device,
        db,
        model_name,
        experiment_name="hybrid_quant_bidirectional_sweep",
    )

    entries_by_weight = {}
    for entry in pending_entries:
        entries_by_weight.setdefault(
            entry['weight']['source_run_id'], []
        ).append(entry)

    for weight_run_id, weight_entries in entries_by_weight.items():
        weight = weight_entries[0]['weight']
        weight_dt = weight['label']
        weight_config = copy.deepcopy(base_config)
        weight_config.setdefault('quantization', {}).update({
            'format': weight_dt,
            'weight_mode': 'chunk',
            'weight_chunk_size': args.weight_chunk_size,
            'weight_source': 'prequantized_state_dict',
            'unsigned_input_sources': list(args.unsigned_input_sources),
        })
        source_weight_config = _build_weight_materialization_source_config(
            weight_config
        )

        if weight['mode'] == 'baseline':
            print(f"\n[Weights] Materializing baseline {weight_dt} ...")
            model_fp32, adapter_fp32 = _load_fp32_model(
                runner,
                args,
                device,
                config_builder=lambda _: source_weight_config,
            )
            q_state_dict, quant_map = _build_uniform_quant_state_dict(
                model_fp32,
                weight_dt,
                chunk_size=args.weight_chunk_size,
            )
            layer_types = _layer_types_from_model(model_fp32)
            q_state_dict = _materialize_weight_buffers_from_map(
                model_fp32, q_state_dict, quant_map, args
            )
            q_path = os.path.join(
                model_dir,
                f"sweep_weight_{weight_run_id}_{weight_dt}.pt",
            )
            torch.save(q_state_dict, q_path)
            quant_map_json = _build_weight_map_json(quant_map, layer_types)
            del model_fp32, adapter_fp32, q_state_dict
        else:
            print(f"\n[Weights] Rebuilding optimal source run {weight_run_id} ...")
            source_quant_map = copy.deepcopy(weight.get('quant_map') or {})
            model_fp32, adapter_fp32 = _load_fp32_model(
                runner,
                args,
                device,
                config_builder=lambda _: source_weight_config,
            )
            q_state_dict, quant_map = _build_quant_state_dict_from_map(
                model_fp32,
                source_quant_map,
                args.weight_chunk_size,
            )
            q_state_dict = _materialize_weight_buffers_from_map(
                model_fp32, q_state_dict, quant_map, args
            )
            q_path = os.path.join(
                model_dir,
                f"sweep_weight_{weight_run_id}_optimal.pt",
            )
            torch.save(q_state_dict, q_path)
            layer_types = _layer_types_from_model(model_fp32)
            quant_map_json = _build_weight_map_json(quant_map, layer_types)
            del model_fp32, adapter_fp32, q_state_dict

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, adapter = runner.load_model_from_weight_file(
            config=weight_config,
            weight_file_path=q_path,
            skip_calibration=True,
        )
        loader = _build_loader(args, device, runner)
        try:
            for entry in weight_entries:
                print(
                    f"\n[Inference] {entry['bit_width']}-bit "
                    f"W={entry['weight_dt']} A={entry['activation_dt']}"
                )
                try:
                    acc1, acc5, certainty, input_stats = _run_inference(
                        runner,
                        model,
                        adapter,
                        loader,
                        args,
                        input_quant_cfg=entry['input_quant_cfg'],
                        desc=(
                            f"Hybrid {model_name} W={entry['weight_dt']} "
                            f"A={entry['activation_dt']}"
                        ),
                    )
                    status = 'SUCCESS'
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    acc1, acc5, certainty, input_stats = 0.0, 0.0, 0.0, {}
                    status = 'ERROR'
                norm_mse = input_stats.get('norm_mse') if input_stats else None
                _log_hybrid_run(
                    runner=runner,
                    base_config=weight_config,
                    model_name=model_name,
                    weight_dt=entry['weight_dt'],
                    activation_dt=entry['activation_dt'],
                    acc1=acc1,
                    acc5=acc5,
                    status=status,
                    experiment_name=entry['experiment_name'],
                    ref_acc1=ref_acc1,
                    ref_acc5=ref_acc5,
                    ref_certainty=ref_certainty,
                    certainty=certainty,
                    mse=norm_mse,
                    quant_map_json=quant_map_json,
                    input_quant_stats=input_stats,
                    input_quant_cfg=entry['input_quant_cfg'],
                    selection_metadata=entry['selection_metadata'],
                )
                if status == 'SUCCESS':
                    _save_hybrid_layer_stats(
                        model_dir,
                        (
                            f"sweep_{entry['bit_width']}bit_"
                            f"w{weight_run_id}_i"
                            f"{entry['input']['source_run_id']}"
                        ),
                        input_stats,
                        quant_map,
                        acc1=acc1,
                        acc5=acc5,
                        certainty=certainty,
                        norm_mse=norm_mse,
                        weight_mode=weight['mode'],
                        weight_dt=entry['weight_dt'],
                        input_mode=entry['input']['mode'],
                        input_dt=entry['activation_dt'],
                        bit_width=entry['bit_width'],
                    )
        finally:
            del model, adapter, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[Done] Hybrid bidirectional sweep finished for {model_name}.")


def process_single_model(args, device):
    if args.input_mode == 'sweep':
        return _process_bidirectional_db_sweep(args, device)

    args.input_metric = "mse"
    model_name = args.model_name
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    base_config = _base_runtime_config(args, model_name=model_name, weights=args.weights)

    # We must explicitly turn ON quantized_ops and input_quantization in the base config
    # so the adapter actually builds Quant layers (including functional replacements like QuantAdd)
    # AND decomposes complex layers like MHA into q_proj/k_proj/v_proj so the weight shapes match.
    base_config.setdefault('adapter', {})
    base_config['adapter']['quantized_ops'] = ['all']
    base_config['adapter']['input_quantization'] = True
    base_config['adapter']['weight_quantization'] = True
    base_config['adapter']['fold_input_norm'] = args.fold_input_norm
    base_config['adapter']['quantize_first_layer'] = args.fold_input_norm

    runner = Runner(device)
    runner.args = args # Store args in runner for logging access
    db = runner._get_db()

    print(f"\n{'='*60}")
    print(f" HYBRID EXPERIMENT: {model_name}")
    print(f"  Weight Mode  : {args.weight_mode}")
    if args.weight_mode == "fixed":
        print(f"  Weight Format: {args.weight_format}")
    else:
        print(f"  Weight Metric: {args.weight_metric}")
        print(f"  Weight Cands : {args.weight_candidate_formats}")

    print(f"  Input Mode   : {args.input_mode}")
    if args.input_mode == "fixed":
        print(f"  Input Format : {args.input_format}")
    else:
        print(f"  Input Metric : {args.input_metric}")
        print(f"  Input Cands  : {args.input_candidate_formats}")
    print(f"{'='*60}")

    # Determine experiment name: hybrid_quant_<w_bits>/<i_bits>[_w_cache_sim]
    # Based on the first (highest) candidate bit-widths.
    w_primary = args.weight_format if args.weight_mode == "fixed" else (args.weight_candidate_formats[0] if args.weight_candidate_formats else "unknown")
    i_primary = args.input_format if args.input_mode == "fixed" else (args.input_candidate_formats[0] if args.input_candidate_formats else "unknown")
    
    def _get_bits(fmt):
        s = str(fmt)
        return s.split('_')[0] if '_' in s else s

    experiment_name = f"hybrid_quant_{_get_bits(w_primary)}/{_get_bits(i_primary)}"
    if args.use_cache_sim_db:
        experiment_name += "_w_cache_sim"
        
        # Check if model exists in cache simulation DB
        sim = db.get_latest_cache_simulation(model_name)
        if sim is None:
            print(f"\n[CacheSim] Model '{model_name}' not found in cache simulation database.")
            print("[CacheSim] Triggering automatic cache simulation...")
            try:
                from runspace.experiments.asic_cache_simulation.simulate_cache import run_simulation as run_cache_sim
                # Create dummy args for simulation
                sim_args = types.SimpleNamespace()
                sim_args.model_name = model_name
                sim_args.cache_size = 2.0  # Default from simulate_cache.py
                sim_args.num_banks = 16    # Default
                sim_args.metadata_bits = 0 # Default
                sim_args.batch_size = 1    # Default for residency analysis
                sim_args.device = str(device)
                
                # Execute simulation (this will also upload to DB)
                run_cache_sim(sim_args)
                print("[CacheSim] Simulation completed and uploaded to DB.\n")
            except Exception as e:
                print(f"[CacheSim] Error during automatic simulation: {e}")
                import traceback
                traceback.print_exc()

    print(f"  Experiment Name: {experiment_name}")

    ref_acc1, ref_acc5, ref_certainty = _get_or_run_fp32_ref_common(
        runner, args, device, db, model_name, experiment_name=experiment_name
    )

    # Prepare Weights
    if args.weight_mode == "fixed":
        weight_dt_str = args.weight_format
        print(f"\n[Weights] Building uniformly quantized weights ({weight_dt_str}) ...")
        model_fp32, adapter_fp32 = _load_fp32_model(runner, args, device, config_builder=lambda _: base_config)
        q_state_dict, quant_map = _build_uniform_quant_state_dict(
            model_fp32, args.weight_format, chunk_size=args.weight_chunk_size
        )
        layer_types = _layer_types_from_model(model_fp32)
        q_state_dict = _materialize_weight_buffers_from_map(
            model_fp32,
            q_state_dict,
            quant_map,
            args,
        )
        q_path = os.path.join(
            model_dir,
            f"fixed_weights_{args.weight_format}.pt",
        )
        torch.save(q_state_dict, q_path)
        del model_fp32, adapter_fp32
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[Weights] Running optimized weight quantization (metric={args.weight_metric}) ...")
        if args.weight_metric == "act_mse":
            q_path, quant_map, layer_types = run_activation_weight_phase(
                runner, args, device, model_dir, base_config
            )
        else:
            q_path, quant_map, layer_types = run_weight_phase(runner, args, device, model_dir, base_config)
        weight_dt_str = _summarise_quant_map(quant_map, prefix=f"opt_w{args.weight_metric}")

    quant_map_json = _build_weight_map_json(quant_map, layer_types)

    # Load Model with Quantized Weights
    model, adapter = runner.load_model_from_weight_file(
        config=base_config,
        weight_file_path=q_path,
        skip_calibration=True
    )

    # Prepare Inputs
    if args.input_mode == "fixed":
        activation_dt_str = args.input_format
        input_quant_cfg = _build_uniform_input_quant_cfg(
            args.input_format,
            args.input_chunk_size,
            unsigned_input_sources=args.unsigned_input_sources,
            use_unsigned_input_candidates=args.dynamic_unsigned_input_candidates,
        )
    else:
        activation_dt_str = f"dyn_input_{args.input_metric}"
        input_quant_cfg = _build_dynamic_input_quant_cfg(
            metric=args.input_metric,
            chunk_size=args.input_chunk_size,
            candidate_formats=args.input_candidate_formats,
            use_cache_sim_db=args.use_cache_sim_db,
            model_name=args.model_name,
            unsigned_input_sources=args.unsigned_input_sources,
            dynamic_unsigned_input_candidates=args.dynamic_unsigned_input_candidates,
            skip_depthwise_input_quant=args.skip_depthwise_input_quant,
        )

    loader = _build_loader(args, device, runner)

    print(f"\n[Inference] Running Hybrid Setup: Weight={weight_dt_str}, Input={activation_dt_str} ...")
    try:
        acc1, acc5, certainty, input_stats = _run_inference(
            runner, model, adapter, loader, args,
            input_quant_cfg=input_quant_cfg,
            desc=f"Hybrid W={args.weight_mode} I={args.input_mode}"
        )
    except Exception as e:
        print(f"  ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        del model, adapter
        gc.collect()
        torch.cuda.empty_cache()
        _log_hybrid_run(
            runner=runner, base_config=base_config, model_name=model_name,
            weight_dt=weight_dt_str, activation_dt=activation_dt_str,
            acc1=0.0, acc5=0.0, status="ERROR",
            experiment_name=experiment_name,
            ref_acc1=ref_acc1, ref_acc5=ref_acc5, ref_certainty=ref_certainty
        )
        return

    norm_mse = input_stats['norm_mse'] if input_stats else None

    print(
        f"  Result: Top1={acc1:.2f}%, Top5={acc5:.2f}%, Certainty={certainty:.4f}"
        + (f", NormMSE={norm_mse:.4e}" if norm_mse is not None else "")
    )

    _log_hybrid_run(
        runner=runner,
        base_config=base_config,
        model_name=model_name,
        weight_dt=weight_dt_str,
        activation_dt=activation_dt_str,
        acc1=acc1,
        acc5=acc5,
        status="SUCCESS",
        experiment_name=experiment_name,
        ref_acc1=ref_acc1,
        ref_acc5=ref_acc5,
        ref_certainty=ref_certainty,
        certainty=certainty,
        mse=norm_mse,
        quant_map_json=quant_map_json,
        input_quant_stats=input_stats,
    )

    # Log layer stats (contains chunk win rates for dynamic inputs)
    run_label = f"hybrid_{args.weight_mode}_{args.input_mode}"
    out_dir = os.path.join(model_dir, run_label)
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "layer_stats.json")
    with open(stats_path, 'w') as f:
        save_data = dict(input_stats.get('layer_stats', {}) if input_stats else {})
        
        # Merge weight format counts into the layer stats for visibility
        for layer_name, spec in quant_map.items():
            if layer_name not in save_data:
                save_data[layer_name] = {}
            if isinstance(spec, list):
                counts = {}
                for fmt in spec:
                    counts[str(fmt)] = counts.get(str(fmt), 0) + 1
                save_data[layer_name]['weight_format_counts'] = counts
                save_data[layer_name]['weight_total_chunks'] = len(spec)
                if counts:
                    save_data[layer_name]['weight_format'] = sorted(counts.items(), key=lambda x: -x[1])[0][0]
            else:
                save_data[layer_name]['weight_format'] = str(spec)
                save_data[layer_name]['weight_format_counts'] = {str(spec): 1}
                save_data[layer_name]['weight_total_chunks'] = 1
                
        save_data['accuracy'] = {
            'top1': acc1, 'top5': acc5, 'certainty': certainty,
            'norm_mse': norm_mse,
            'weight_mode': args.weight_mode, 'weight_dt': weight_dt_str,
            'input_mode': args.input_mode, 'input_dt': activation_dt_str,
        }
        json.dump(save_data, f, indent=4)

    del model, adapter
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[Done] Hybrid experiment finished for {model_name}.")


def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine models to run
    if args.models_file:
        with open(args.models_file) as f:
            yaml_models = yaml.safe_load(f)
        if not isinstance(yaml_models, list):
            print("Error: models_file must be a YAML list.")
            sys.exit(1)
        models_to_run = yaml_models
    else:
        models_to_run = [{'name': args.model_name, 'weights': args.weights}]

    print(f"Processing {len(models_to_run)} model(s).")

    for model_cfg in models_to_run:
        if isinstance(model_cfg, str):
            args.model_name = model_cfg
            args.weights = 'DEFAULT'
        else:
            args.model_name = model_cfg.get('name', args.model_name)
            args.weights = model_cfg.get('weights', 'DEFAULT')

        process_single_model(args, device)


if __name__ == "__main__":
    main()
