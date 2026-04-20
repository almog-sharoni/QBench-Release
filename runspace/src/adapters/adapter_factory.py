"""
Adapter Factory

Creates adapters from configuration dictionaries.
Provides a single entry point for adapter instantiation.
"""

import json
import os

from .base_adapter import BaseAdapter


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

    model_config = config.get('model', {})
    adapter_config = config.get('adapter', {})
    
    # Get adapter type (default to 'generic')
    adapter_type = adapter_config.get('type', 'generic')
    
    # Extract common parameters
    model_name = model_config.get('name', 'resnet18')
    model_source = model_config.get('source', 'auto')
    weights = model_config.get('weights', None)
    
    quantize_first_layer = adapter_config.get('quantize_first_layer', False)
    quantized_ops = adapter_config.get('quantized_ops', ['all'])
    excluded_ops = adapter_config.get('excluded_ops', [])
    fold_layers = adapter_config.get('fold_layers', False)
    weight_quantization = adapter_config.get('weight_quantization', True)
    # input_quantization is now True by default, unless explicitly disabled in adapter config (which we removed from base config but keep support for overrides if needed, or just force True as per request)
    # User said "we asking id to quant inputs - this shouldnt be a question". So we default to True.
    input_quantization = adapter_config.get('input_quantization', True)
    skip_calibration = adapter_config.get('skip_calibration', False)
    
    # Check for quantization type in adapter config first, then in quantization section
    quantization_config = config.get('quantization', {})
    # quantization_type is now solely determined by quantization.format
    quantization_type = quantization_config.get('format', 'fp8_e4m3')
    quantization_bias = quantization_config.get('bias', None)
    
    # Extract global input format if present
    input_quantization_type = adapter_config.get('input_quantization_type', quantization_config.get('input_format', None))
    
    # Extract quantization modes
    quant_mode = quantization_config.get('mode', 'tensor')
    chunk_size = quantization_config.get('chunk_size', None)
    weight_mode = quantization_config.get('weight_mode', 'channel') # Default weights to channel
    weight_chunk_size = quantization_config.get('weight_chunk_size', None)
    act_mode = quantization_config.get('act_mode', 'tensor') # Default activations to tensor

    act_chunk_size = quantization_config.get('act_chunk_size', None)
    
    # Extract input chunk size from adapter config if present
    input_chunk_size = adapter_config.get('input_chunk_size', chunk_size)
    
    # Extract rounding mode
    rounding = quantization_config.get('rounding', 'nearest')
    
    # Extract TF32 simulation flag
    simulate_tf32_accum = quantization_config.get('simulate_tf32_accum', False)
    
    # Extract layer-specific config
    # Can be in quantization.layers or adapter.layers (prefer quantization.layers)
    layer_config = quantization_config.get('layers', adapter_config.get('layers', None))

    # Merge forced-FP8 overrides from a cache simulation result file.
    # quantization.cache_simulation_path points to a simulation_results.json produced by
    # simulate_cache.py.  The forced_fp8_layers block in that file is merged on top of any
    # existing layer_config entries so those layers are always pinned to FP8 regardless of
    # the global quantization format chosen for the run.
    cache_sim_path = quantization_config.get('cache_simulation_path', None)
    if cache_sim_path:
        abs_path = os.path.abspath(cache_sim_path)
        if not os.path.isfile(abs_path):
            import warnings
            warnings.warn(f"cache_simulation_path '{abs_path}' not found — skipping FP8 overrides.")
        else:
            with open(abs_path, 'r') as _f:
                _sim = json.load(_f)
            _off_chip = _sim.get('off_chip_layers', [])
            if _off_chip:
                # Build a layer config entry for each off-chip layer using this run's
                # global format and modes so the format decision stays with the runner.
                _sim_overrides = {
                    name: {
                        'format':      quantization_type,
                        'input_format': input_quantization_type or quantization_type,
                        'mode':        quant_mode,
                        'weight_mode': weight_mode,
                    }
                    for name in _off_chip
                }
                # Explicit layer_config entries take priority over simulation overrides.
                layer_config = dict(_sim_overrides, **(layer_config or {}))
                print(f"[CacheSim] Applied {len(_off_chip)} off-chip layer overrides "
                      f"(format={quantization_type}, mode={quant_mode}) from {abs_path}")
    
    # Check if per-chunk format mode is enabled
    per_chunk_format = quantization_config.get('per_chunk_format', False)

    # If nothing is requested to be quantized, avoid the quantized build path.
    # This keeps pure FP32 / file-backed state-dict runs on a simpler, safer path.
    qops_list = quantized_ops if isinstance(quantized_ops, list) else [quantized_ops]
    has_qops = any(bool(x) for x in qops_list)
    has_layer_overrides = isinstance(layer_config, dict) and len(layer_config) > 0
    default_build_quantized = bool(quantize_first_layer or has_qops or has_layer_overrides)
    build_quantized = adapter_config.get('build_quantized', default_build_quantized)
    
    # Extract output_name as run_id for tracking
    run_id = config.get('output_name', 'default')

    if adapter_type == 'generic' or adapter_type == 'resnet':
        from .generic_adapter import GenericAdapter
        # For 'resnet' type, we just use GenericAdapter with defaults or config overrides
        resolved_kwargs = dict(
            model_name=model_name,
            model_source=model_source,
            weights=weights,
            input_quantization=input_quantization,
            weight_quantization=weight_quantization,
            quantize_first_layer=quantize_first_layer,
            quantized_ops=quantized_ops,
            excluded_ops=excluded_ops,
            quantization_type=quantization_type,
            quantization_bias=quantization_bias,
            layer_config=layer_config,
            input_quantization_type=input_quantization_type,
            quant_mode=quant_mode,
            chunk_size=chunk_size,
            weight_mode=weight_mode,
            weight_chunk_size=weight_chunk_size,
            act_mode=act_mode,
            act_chunk_size=act_chunk_size,
            fold_layers=fold_layers,
            simulate_tf32_accum=simulate_tf32_accum,
            rounding=rounding,
            per_chunk_format=per_chunk_format,
            input_chunk_size=input_chunk_size,
            run_id=run_id,
            skip_calibration=skip_calibration,
            build_quantized=build_quantized,
        )
        if _should_print_adapter_config(config):
            _print_adapter_config_snapshot(config, adapter_type, resolved_kwargs)
        return GenericAdapter(**resolved_kwargs)
    
    elif adapter_type == 'slm':
        from .slm_adapter import SLMAdapter
        resolved_kwargs = dict(
            model_name=model_name,
            model_source=model_source,
            weights=weights,
            input_quantization=input_quantization,
            weight_quantization=weight_quantization,
            quantize_first_layer=quantize_first_layer,
            quantized_ops=quantized_ops,
            excluded_ops=excluded_ops,
            quantization_type=quantization_type,
            quantization_bias=quantization_bias,
            layer_config=layer_config,
            input_quantization_type=input_quantization_type,
            quant_mode=quant_mode,
            chunk_size=chunk_size,
            weight_mode=weight_mode,
            weight_chunk_size=weight_chunk_size,
            act_mode=act_mode,
            act_chunk_size=act_chunk_size,
            fold_layers=fold_layers,
            simulate_tf32_accum=simulate_tf32_accum,
            rounding=rounding,
            input_chunk_size=input_chunk_size
        )
        if _should_print_adapter_config(config):
            _print_adapter_config_snapshot(config, adapter_type, resolved_kwargs)
        return SLMAdapter(**resolved_kwargs)

        

    else:
        raise ValueError(
            f"Unknown adapter type: '{adapter_type}'. "
            f"Available types: 'generic', 'resnet', 'slm'"
        )


def load_reference_model(config: dict):
    """
    Load a reference (FP32) model for comparison based on config.
    
    Args:
        config: Configuration dictionary with model settings.
        
    Returns:
        A PyTorch model instance.
    """
    import torchvision.models as models
    
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'resnet18')
    model_source = model_config.get('source', 'auto')
    weights = model_config.get('weights', None)
    
    if model_source == 'auto':
        if hasattr(models, model_name):
            model_source = 'torchvision'
        else:
            try:
                import timm as _timm
                model_source = 'timm' if _timm.is_model(model_name) else 'torchvision'
            except ImportError:
                model_source = 'torchvision'

    if model_source == 'torchvision':
        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not found in torchvision.models.")

        model_fn = getattr(models, model_name)
        if not weights or str(weights).strip().lower() in ('none', 'false', '0', 'null', ''):
            return model_fn(weights=None)

        # Resolve weight enums safely (no deprecated weights=True / string fallbacks).
        try:
            from torchvision.models import get_model_weights
            weights_cls = get_model_weights(model_fn)
            token = str(weights).strip()
            token_l = token.lower()
            if token_l in ('default', 'true', '1'):
                return model_fn(weights=weights_cls.DEFAULT if hasattr(weights_cls, 'DEFAULT') else None)
            if hasattr(weights_cls, token):
                return model_fn(weights=getattr(weights_cls, token))
            token_up = token.upper()
            for attr in dir(weights_cls):
                if attr.upper() == token_up:
                    return model_fn(weights=getattr(weights_cls, attr))
            if hasattr(weights_cls, 'DEFAULT'):
                return model_fn(weights=weights_cls.DEFAULT)
        except Exception:
            pass

        # Last resort: run uninitialized rather than using deprecated bool path.
        print(
            f"Warning: Could not resolve torchvision weights '{weights}' for "
            f"{model_name}; using weights=None."
        )
        return model_fn(weights=None)
    elif model_source == 'timm':
        try:
            import timm
        except ImportError:
            raise ImportError(
                "The 'timm' package is required to load this model. "
                "Install it with: pip install timm"
            )
        pretrained = bool(weights) and str(weights).lower() not in ('none', 'false', '0')
        return timm.create_model(model_name, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model source: {model_source}")


def validate_config(config: dict):
    """
    Validates the configuration dictionary against a schema.
    Warns about unknown keys.
    """
    import warnings
    
    # Define allowed keys
    schema = {
        'model': ['name', 'source', 'weights'],
        'adapter': ['type', 'quantize_first_layer', 'quantized_ops', 'excluded_ops', 'input_quantization', 'weight_quantization', 'quantization_type', 'layers', 'fold_layers', 'input_quantization_type', 'input_chunk_size', 'skip_calibration', 'build_quantized'],
        'quantization': ['format', 'bias', 'calib_method', 'layers', 'type', 'enabled', 'input_format', 'mode', 'chunk_size', 'weight_mode', 'weight_chunk_size', 'act_mode', 'act_chunk_size', 'simulate_tf32_accum', 'rounding', 'per_chunk_format', 'cache_simulation_path'], # 'type' and 'enabled' for backward compat/fp4 example
        'dataset': ['name', 'path', 'batch_size', 'num_workers'],
        'evaluation': ['mode', 'compare_batches', 'dataset', 'batch_size', 'max_samples', 'generate_graph_svg', 'save_histograms', 'max_batches', 'graph_only', 'dynamic_input_quant', 'input_quant'] # dataset/batch_size allowed here too?
    }
    
    # Check top-level keys
    allowed_top_level = list(schema.keys()) + ['output_name', 'meta', 'debug', 'experiment']  # metadata/debug keys used by runners
    for key in config:
        if key not in allowed_top_level:
            warnings.warn(f"Unknown top-level config key: '{key}'")
            
    # Check nested keys
    for section, allowed_keys in schema.items():
        if section in config:
            section_config = config[section]
            if isinstance(section_config, dict):
                for key in section_config:
                    if key not in allowed_keys:
                        warnings.warn(f"Unknown config key in '{section}': '{key}'. Allowed: {allowed_keys}")
