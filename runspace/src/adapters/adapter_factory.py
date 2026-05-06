"""
Adapter Factory

Creates adapters from configuration dictionaries.
Provides a single entry point for adapter instantiation.
"""

import json
import os

from .base_adapter import BaseAdapter
from ..quantization.constants import DEFAULT_QUANTIZATION_TYPE


ADAPTER_SCHEMA = {
    'model': ['name', 'pipeline', 'source', 'weights', 'repo_path', 'sg_weights', 'sp_config', 'sg_config'],
    'adapter': ['type', 'quantize_first_layer', 'quantized_ops', 'excluded_ops', 'input_quantization', 'weight_quantization', 'output_quantization', 'quantization_type', 'layers', 'fold_layers', 'input_quantization_type', 'input_chunk_size', 'skip_calibration', 'build_quantized', 'quantize_components', 'input_size', 'unsigned_input_sources', 'enable_fx_quantization'],
    'quantization': ['format', 'bias', 'calib_method', 'layers', 'type', 'enabled', 'input_format', 'mode', 'chunk_size', 'weight_mode', 'weight_chunk_size', 'weight_source', 'act_mode', 'act_chunk_size', 'output_format', 'output_mode', 'output_chunk_size', 'simulate_tf32_accum', 'rounding', 'per_chunk_format', 'strict_format_check', 'cache_simulation_path', 'unsigned_input_sources'],
    'dataset': ['name', 'path', 'batch_size', 'num_workers', 'image_size', 'grayscale', 'pairs_file', 'max_pairs', 'resize_size', 'multiprocessing_context', 'persistent_workers', 'prefetch_factor'],
    'evaluation': ['mode', 'compare_batches', 'dataset', 'batch_size', 'max_samples', 'generate_graph_svg', 'save_histograms', 'max_batches', 'graph_only', 'dynamic_input_quant', 'input_quant', 'save_visualizations', 'num_viz_samples'],
}


def _should_print_adapter_config(config: dict) -> bool:
    """Return True if adapter config snapshots should be printed."""
    env_flag = os.environ.get("QBENCH_PRINT_ADAPTER_CONFIG", "").strip().lower()
    if env_flag in {"1", "true", "yes", "on"}:
        return True
    debug_cfg = config.get("debug", {})
    if isinstance(debug_cfg, dict) and bool(debug_cfg.get("print_adapter_config", False)):
        return True
    return False


def _print_adapter_config_snapshot(config: dict, adapter_type: str, resolved_kwargs: dict):
    """Print a compact, machine-readable snapshot of adapter inputs."""
    snapshot = {
        "model": config.get("model", {}),
        "adapter": config.get("adapter", {}),
        "quantization": config.get("quantization", {}),
        "dataset": config.get("dataset", {}),
        "evaluation": config.get("evaluation", {}),
        "output_name": config.get("output_name", ""),
        "adapter_type": adapter_type,
        "resolved_adapter_kwargs": resolved_kwargs,
    }
    print("[AdapterConfig] BEGIN")
    print(json.dumps(snapshot, indent=2, sort_keys=True, default=str))
    print("[AdapterConfig] END")


def _merge_cache_sim_overrides(layer_config, quantization_config, quantization_type, input_quantization_type):
    cache_sim_path = quantization_config.get('cache_simulation_path')
    if not cache_sim_path:
        return layer_config

    abs_path = os.path.abspath(cache_sim_path)
    if not os.path.isfile(abs_path):
        import warnings
        warnings.warn(f"cache_simulation_path '{abs_path}' not found — skipping FP8 overrides.")
        return layer_config

    with open(abs_path, 'r') as f:
        sim = json.load(f)
    off_chip = sim.get('off_chip_layers', [])
    if not off_chip:
        return layer_config

    overrides = {
        name: {
            'format': quantization_type,
            'input_format': input_quantization_type or quantization_type,
            'mode': quantization_config.get('mode', 'tensor'),
            'weight_mode': quantization_config.get('weight_mode', 'channel'),
        }
        for name in off_chip
    }
    print(
        f"[CacheSim] Applied {len(off_chip)} off-chip layer overrides "
        f"(format={quantization_type}, mode={quantization_config.get('mode', 'tensor')}) from {abs_path}"
    )
    return dict(overrides, **(layer_config or {}))


def _build_quantized_default(adapter_config: dict, quantized_ops, layer_config) -> bool:
    qops = quantized_ops if isinstance(quantized_ops, list) else [quantized_ops]
    has_qops = any(bool(op) for op in qops)
    has_layer_overrides = isinstance(layer_config, dict) and len(layer_config) > 0
    return bool(adapter_config.get('quantize_first_layer', False) or has_qops or has_layer_overrides)


def _resolve_adapter_inputs(config: dict) -> dict:
    model_config = config.get('model', {})
    adapter_config = config.get('adapter', {})
    quantization_config = config.get('quantization', {})

    quantization_type = quantization_config.get('format', DEFAULT_QUANTIZATION_TYPE)
    input_quantization_type = adapter_config.get(
        'input_quantization_type',
        quantization_config.get('input_format', None),
    )
    layer_config = quantization_config.get('layers', adapter_config.get('layers', None))
    layer_config = _merge_cache_sim_overrides(
        layer_config,
        quantization_config,
        quantization_type,
        input_quantization_type,
    )

    quantized_ops = adapter_config.get('quantized_ops', ['all'])
    default_build_quantized = _build_quantized_default(adapter_config, quantized_ops, layer_config)

    return {
        'model_config': model_config,
        'adapter_config': adapter_config,
        'adapter_type': adapter_config.get('type', 'generic'),
        'model_name': model_config.get('name', 'resnet18'),
        'model_source': model_config.get('source', 'auto'),
        'weights': model_config.get('weights', None),
        'input_quantization': adapter_config.get('input_quantization', True),
        'weight_quantization': adapter_config.get('weight_quantization', True),
        'output_quantization': adapter_config.get('output_quantization', False),
        'quantize_first_layer': adapter_config.get('quantize_first_layer', False),
        'quantized_ops': quantized_ops,
        'excluded_ops': adapter_config.get('excluded_ops', []),
        'fold_layers': adapter_config.get('fold_layers', False),
        'skip_calibration': adapter_config.get('skip_calibration', False),
        'quantization_type': quantization_type,
        'quantization_bias': quantization_config.get('bias', None),
        'input_quantization_type': input_quantization_type,
        'quant_mode': quantization_config.get('mode', 'tensor'),
        'chunk_size': quantization_config.get('chunk_size', None),
        'weight_mode': quantization_config.get('weight_mode', 'channel'),
        'weight_chunk_size': quantization_config.get('weight_chunk_size', None),
        'act_mode': quantization_config.get('act_mode', 'tensor'),
        'act_chunk_size': quantization_config.get('act_chunk_size', None),
        'output_quantization_type': quantization_config.get('output_format', None),
        'output_mode': quantization_config.get('output_mode', 'tensor'),
        'output_chunk_size': quantization_config.get('output_chunk_size', None),
        'input_chunk_size': adapter_config.get('input_chunk_size', quantization_config.get('chunk_size', None)),
        'rounding': quantization_config.get('rounding', 'nearest'),
        'simulate_tf32_accum': quantization_config.get('simulate_tf32_accum', False),
        'layer_config': layer_config,
        'per_chunk_format': quantization_config.get('per_chunk_format', False),
        'strict_format_check': quantization_config.get('strict_format_check', False),
        'build_quantized': adapter_config.get('build_quantized', default_build_quantized),
        'run_id': config.get('output_name', 'default'),
        'unsigned_input_sources': quantization_config.get('unsigned_input_sources', []),
        'input_size': adapter_config.get('input_size', None),
        'enable_fx_quantization': adapter_config.get('enable_fx_quantization', True),
    }


def _common_adapter_kwargs(params: dict) -> dict:
    keys = (
        'model_name', 'model_source', 'weights', 'input_quantization',
        'weight_quantization', 'output_quantization', 'quantize_first_layer',
        'quantized_ops',
        'excluded_ops', 'quantization_type', 'quantization_bias',
        'layer_config', 'input_quantization_type', 'output_quantization_type',
        'quant_mode',
        'chunk_size', 'weight_mode', 'weight_chunk_size', 'act_mode',
        'act_chunk_size', 'output_mode', 'output_chunk_size',
        'fold_layers', 'simulate_tf32_accum', 'rounding',
        'input_chunk_size', 'input_size', 'enable_fx_quantization',
    )
    return {key: params[key] for key in keys}


def create_adapter(config: dict) -> BaseAdapter:
    """
    Create an adapter from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with model and adapter settings.
        
    Returns:
        A configured adapter instance.
        
    Example config:
        {
            'model': {
                'name': 'resnet18',
                'source': 'torchvision',
                'weights': 'IMAGENET1K_V1'
            },
            'adapter': {
                'type': 'generic',
                'quantize_first_layer': True,
                'quantized_ops': ['Conv2d', 'Linear', 'BatchNorm2d']
            }
        }
    """
    # Validate config
    validate_config(config)

    params = _resolve_adapter_inputs(config)
    model_config = params['model_config']
    adapter_config = params['adapter_config']
    adapter_type = params['adapter_type']

    if adapter_type == 'generic' or adapter_type == 'resnet':
        from .generic_adapter import GenericAdapter
        resolved_kwargs = _common_adapter_kwargs(params)
        resolved_kwargs.update(
            per_chunk_format=params['per_chunk_format'],
            run_id=params['run_id'],
            skip_calibration=params['skip_calibration'],
            build_quantized=params['build_quantized'],
            strict_format_check=params['strict_format_check'],
            unsigned_input_sources=params['unsigned_input_sources'],
            input_size=params['input_size'],
            enable_fx_quantization=params['enable_fx_quantization'],
        )
        if _should_print_adapter_config(config):
            _print_adapter_config_snapshot(config, adapter_type, resolved_kwargs)
        return GenericAdapter(**resolved_kwargs)
    
    # elif adapter_type == 'slm':
    #     from .slm_adapter import SLMAdapter
    #     resolved_kwargs = _common_adapter_kwargs(params)
    #     if _should_print_adapter_config(config):
    #         _print_adapter_config_snapshot(config, adapter_type, resolved_kwargs)
    #     return SLMAdapter(**resolved_kwargs)

    elif adapter_type == 'feature_matching':
        from .feature_matching_adapter import FeatureMatchingAdapter
        resolved_kwargs = dict(
            pipeline_name=model_config.get('pipeline', model_config['name']),
            model_cfg=model_config,
            quantization_type=params['quantization_type'],
            quantized_ops=params['quantized_ops'],
            excluded_ops=params['excluded_ops'],
            quantize_first_layer=params['quantize_first_layer'],
            weight_quantization=params['weight_quantization'],
            input_quantization=params['input_quantization'],
            output_quantization=params['output_quantization'],
            layer_config=params['layer_config'],
            quant_mode=params['quant_mode'],
            chunk_size=params['chunk_size'],
            weight_mode=params['weight_mode'],
            weight_chunk_size=params['weight_chunk_size'],
            output_quantization_type=params['output_quantization_type'],
            output_mode=params['output_mode'],
            output_chunk_size=params['output_chunk_size'],
            rounding=params['rounding'],
            run_id=params['run_id'],
            skip_calibration=params['skip_calibration'],
            build_quantized=params['build_quantized'],
            quantize_components=adapter_config.get('quantize_components', []),
            strict_format_check=params['strict_format_check'],
        )
        if _should_print_adapter_config(config):
            _print_adapter_config_snapshot(config, adapter_type, resolved_kwargs)
        return FeatureMatchingAdapter(**resolved_kwargs)

    else:
        raise ValueError(
            f"Unknown adapter type: '{adapter_type}'. "
            f"Available types: 'generic', 'resnet', 'slm', 'feature_matching'"
        )


def load_reference_model(config: dict):
    """
    Load a reference (FP32) model for comparison based on config.
    
    Args:
        config: Configuration dictionary with model settings.
        
    Returns:
        A PyTorch model instance.
    """
    from .generic_adapter import GenericAdapter

    model_config = config.get('model', {})
    adapter = GenericAdapter(
        model_name=model_config.get('name', 'resnet18'),
        model_source=model_config.get('source', 'auto'),
        weights=model_config.get('weights', None),
        build_quantized=False,
    )
    return adapter.model


def validate_config(config: dict):
    """
    Validates the configuration dictionary against a schema.
    Warns about unknown keys.
    """
    import warnings
    
    # Check top-level keys
    allowed_top_level = list(ADAPTER_SCHEMA.keys()) + ['output_name', 'meta', 'debug', 'experiment', 'target_pipeline']
    for key in config:
        if key not in allowed_top_level:
            warnings.warn(f"Unknown top-level config key: '{key}'")
            
    # Check nested keys
    for section, allowed_keys in ADAPTER_SCHEMA.items():
        if section in config:
            section_config = config[section]
            if isinstance(section_config, dict):
                for key in section_config:
                    if key not in allowed_keys:
                        warnings.warn(f"Unknown config key in '{section}': '{key}'. Allowed: {allowed_keys}")
