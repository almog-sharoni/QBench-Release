"""
Adapter Factory

Creates adapters from configuration dictionaries.
Provides a single entry point for adapter instantiation.
"""

from .base_adapter import BaseAdapter


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
    model_source = model_config.get('source', 'torchvision')
    weights = model_config.get('weights', None)
    
    quantize_first_layer = adapter_config.get('quantize_first_layer', False)
    quantized_ops = adapter_config.get('quantized_ops', ['Conv2d'])
    excluded_ops = adapter_config.get('excluded_ops', [])
    fold_layers = adapter_config.get('fold_layers', False)
    # input_quantization is now True by default, unless explicitly disabled in adapter config (which we removed from base config but keep support for overrides if needed, or just force True as per request)
    # User said "we asking id to quant inputs - this shouldnt be a question". So we default to True.
    input_quantization = adapter_config.get('input_quantization', True)
    
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
    
    # Check if per-chunk format mode is enabled
    per_chunk_format = quantization_config.get('per_chunk_format', False)
    
    # Extract output_name as run_id for tracking
    run_id = config.get('output_name', 'default')

    if adapter_type == 'generic' or adapter_type == 'resnet':
        from .generic_adapter import GenericAdapter
        # For 'resnet' type, we just use GenericAdapter with defaults or config overrides
        return GenericAdapter(
            model_name=model_name,
            model_source=model_source,
            weights=weights,
            input_quantization=input_quantization,
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
            run_id=run_id
        )
    
    elif adapter_type == 'slm':
        from .slm_adapter import SLMAdapter
        return SLMAdapter(
            model_name=model_name,
            model_source=model_source,
            weights=weights,
            input_quantization=input_quantization,
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
    model_source = model_config.get('source', 'torchvision')
    weights = model_config.get('weights', None)
    
    if model_source == 'torchvision':
        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not found in torchvision.models.")
        
        model_fn = getattr(models, model_name)
        
        if weights:
            # Try to load with weights
            return model_fn(weights=weights if weights != 'DEFAULT' else 'DEFAULT')
        else:
            return model_fn(weights=None)
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
        'adapter': ['type', 'quantize_first_layer', 'quantized_ops', 'excluded_ops', 'input_quantization', 'quantization_type', 'layers', 'fold_layers', 'input_quantization_type', 'input_chunk_size'],
        'quantization': ['format', 'bias', 'calib_method', 'layers', 'type', 'enabled', 'input_format', 'mode', 'chunk_size', 'weight_mode', 'weight_chunk_size', 'act_mode', 'act_chunk_size', 'simulate_tf32_accum', 'rounding', 'per_chunk_format'], # 'type' and 'enabled' for backward compat/fp4 example
        'dataset': ['name', 'path', 'batch_size', 'num_workers'],
        'evaluation': ['mode', 'compare_batches', 'dataset', 'batch_size', 'max_samples', 'generate_graph_svg', 'save_histograms', 'max_batches'] # dataset/batch_size allowed here too?
    }
    
    # Check top-level keys
    allowed_top_level = list(schema.keys()) + ['output_name']  # output_name used by runner
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
