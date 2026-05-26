import os
import sys
import json
import torch
import torch.nn as nn
import argparse
import math
from datetime import datetime
from runspace.src.registry.op_registry import OpRegistry

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def get_footprint_elements(num_elements: int, metadata_bits: int) -> int:
    """
    Calculate total element-equivalent footprint including metadata overhead.

    Tensors are allocated in 128-element chunks. If a tensor uses any part
    of a chunk, it occupies the full chunk.

    Metadata bytes are counted as element-equivalents (1 byte = 1 FP8 element)
    and are also computed per chunk.
    """
    if num_elements <= 0:
        return 0
    chunk_size = 128
    num_chunks = math.ceil(num_elements / chunk_size)
    chunk_elems = num_chunks * chunk_size
    metadata_elems = math.ceil(num_chunks * metadata_bits / 8)
    return chunk_elems + metadata_elems


def round_to_banks(size_elems: int, bank_size: int) -> int:
    """Round up to the nearest bank boundary (in elements)."""
    if size_elems <= 0:
        return 0
    if bank_size <= 0:
        return size_elems
    return math.ceil(size_elems / bank_size) * bank_size


def fmt_elems(n: int) -> str:
    """Human-readable element count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Rule system (imported from rules.py)
# ---------------------------------------------------------------------------
try:
    from runspace.experiments.asic_cache_simulation.rules import RULES, LAYER_RULES
except ImportError:
    from rules import RULES, LAYER_RULES


def _next_layer_viable(next_layer: dict, metadata_bits: int, bank_size: int,
                       cache_elements: int, xin_in_cache: bool) -> bool:
    """
    Check whether the next layer has at least one valid rule (1-level lookahead).

    xin_in_cache: whether the next layer's xin arrives from cache (True) or
                  external memory (False). Only rules whose xin_from_cache matches
                  are considered.
    """
    if next_layer is None:
        return True
    ni = round_to_banks(get_footprint_elements(next_layer['input_elems'],  metadata_bits), bank_size)
    no = round_to_banks(get_footprint_elements(next_layer['output_elems'], metadata_bits), bank_size)
    nw = round_to_banks(get_footprint_elements(next_layer['weight_elems'], metadata_bits), bank_size)
    ctx = {
        'input_banked':   ni, 'output_banked': no, 'weight_banked': nw,
        'cache_elements': cache_elements, 'bank_size': bank_size,
        'jump_back_size_in_banks': next_layer.get('jump_back_size_in_banks', 0),
    }
    layer_type = next_layer.get('type', '__default__')
    rule_keys  = LAYER_RULES.get(layer_type, LAYER_RULES['__default__'])
    for key in rule_keys:
        rule = RULES[key]
        if rule['xin_from_cache'] != xin_in_cache:
            continue
        guard = rule.get('ctx_guard')
        if guard is not None and not guard(ctx):
            continue
        if rule['stay'](ctx):
            return True
    return False


def evaluate_stay(layer: dict, ctx: dict,
                  next_layer, metadata_bits: int, bank_size: int, cache_elements: int) -> tuple:
    """
    Returns (stay_on_chip, perm_elems, possible, rule_name).

    Looks up the layer type in LAYER_RULES and tries each rule key in order.
    A rule is confirmed if ctx_guard passes (if present), stay() passes, and
    the next layer has a compatible rule given the xin source implied by on_chip.
    If no rule confirms → stay_on_chip=False, rule_name='FLAGGED'.
    """
    layer_type = layer.get('type', '__default__')
    rule_keys  = LAYER_RULES.get(layer_type, LAYER_RULES['__default__'])
    for key in rule_keys:
        rule  = RULES[key]
        guard = rule.get('ctx_guard')
        if guard is not None and not guard(ctx):
            continue
        if not rule['stay'](ctx):
            continue
        on_chip = rule['on_chip']
        if not _next_layer_viable(next_layer, metadata_bits, bank_size, cache_elements,
                                  xin_in_cache=on_chip):
            continue
        return on_chip, rule['perm'](ctx), True, key
    return False, 0, False, 'FLAGGED'


def is_collapsible(layer_type: str) -> bool:
    """Check if the layer is an activation, layer norm, or softmax that should be collapsed."""
    _COLLAPSIBLE_TYPES = {
        # Norm types
        'QuantLayerNorm','LayerNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'GroupNorm',
        # Softmax types
        'Softmax', 'QuantSoftmax',
    }
    if layer_type in _COLLAPSIBLE_TYPES:
        return True
    
    # Check if registered as an activation in OpRegistry
    try:
        if OpRegistry.is_activation(layer_type):
            return True
    except Exception:
        pass

    # Fallback to standard activation name patterns
    act_names = {'relu', 'relu6', 'gelu', 'silu', 'hardswish', 'hardsigmoid', 'tanh', 'sigmoid', 'softmax'}
    cleaned_type = layer_type.lower()
    if cleaned_type.startswith("quant"):
        cleaned_type = cleaned_type[5:]
    if cleaned_type in act_names:
        return True
    
    return False


def is_registry_activation(layer_type: str) -> bool:
    """Check whether a layer type is marked as activation in OpRegistry."""
    try:
        if OpRegistry.is_activation(layer_type):
            return True
        for original_cls, quantized_cls in OpRegistry.get_supported_ops().items():
            if (
                quantized_cls.__name__ == layer_type
                and OpRegistry.is_activation(quantized_cls.__name__)
            ):
                return True
            if (
                original_cls.__name__ == layer_type
                and OpRegistry.is_activation(quantized_cls.__name__)
            ):
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------

COMPUTE_PU_WIDTH = 128


def _cycles_for_ops(num_ops: float) -> int:
    if num_ops <= 0:
        return 0
    return math.ceil(num_ops / COMPUTE_PU_WIDTH)


def _cycles_for_reduction_outputs(num_outputs: int, reduction_dim: float) -> int:
    if num_outputs <= 0 or reduction_dim <= 0:
        return 0
    chunks_per_output = math.ceil(reduction_dim / COMPUTE_PU_WIDTH)
    return int(num_outputs) * chunks_per_output + 1


def _numel_from_shape(shape) -> int:
    if not shape:
        return 0
    return math.prod(shape)


def _collect_tensor_shapes(value) -> list[tuple]:
    shapes = []
    if isinstance(value, torch.Tensor):
        shapes.append(tuple(value.shape))
    elif isinstance(value, (tuple, list)):
        for item in value:
            shapes.extend(_collect_tensor_shapes(item))
    return shapes


def _compute_layer_cycles(layer: dict) -> float:
    """Compute cycles with 128-wide chunks, including collapsed children."""
    l_type = layer['type']
    if is_registry_activation(l_type):
        return 0.0

    compute_cycles = 0

    if 'Conv' in l_type:
        in_c = layer.get('in_channels', 0)
        groups = layer.get('groups', 1)
        fh = layer.get('filter_height', 0)
        fw = layer.get('filter_width', 0)
        output_elems = layer.get('output_elems', 0)
        reduction_dim = (in_c / groups) * fh * fw if groups else 0
        compute_cycles = _cycles_for_reduction_outputs(output_elems, reduction_dim)
    elif 'Linear' in l_type:
        in_features = layer.get('in_features', 0)
        out_features = layer.get('out_features', 0)
        weight_elems = layer.get('weight_elems', 0)
        if not in_features and out_features:
            in_features = weight_elems / out_features
        output_elems = layer.get('output_elems', 0)
        compute_cycles = _cycles_for_reduction_outputs(output_elems, in_features)
        if compute_cycles == 0 and weight_elems:
            compute_cycles = _cycles_for_reduction_outputs(out_features or 1, in_features or weight_elems)
    elif l_type in ('QuantMatMul', 'QuantBMM'):
        input_shapes = layer.get('input_shapes', [])
        output_shape = layer.get('output_shape')
        output_elems = layer.get('output_elems', _numel_from_shape(output_shape))
        reduction_dim = input_shapes[0][-1] if input_shapes and input_shapes[0] else 0
        compute_cycles = _cycles_for_reduction_outputs(output_elems, reduction_dim)
    elif l_type in ('QuantAdd', 'QuantSub', 'QuantMul', 'QuantDiv', 'Residual'):
        compute_cycles = _cycles_for_ops(layer.get('output_elems', 0))
    elif l_type == 'QuantCat':
        compute_cycles = 0
    else:
        in_elems = layer.get('input_elems', 0)
        out_elems = layer.get('output_elems', 0)
        compute_cycles = _cycles_for_ops(max(in_elems, out_elems))

    for collapsed in layer.get('collapsed_layers', []):
        if is_registry_activation(collapsed.get('type', '')):
            continue
        collapsed_out_elems = collapsed.get('output_elems', 0)
        compute_cycles += _cycles_for_ops(collapsed_out_elems)

    return float(compute_cycles)


def optimize_layer_bits(layer: dict, bandwidth: float,
                        need_input_transfer: bool,
                        need_weight_transfer: bool,
                        need_output_transfer: bool,
                        min_bits: int = 3, max_bits: int = 8):
    """
    Layer-wide bit-width optimization for BW-limited transfers.

    Starting from *max_bits* for every transferred component, if the layer is
    overall BW-limited (total_transfer > compute), all transferred components are
    reduced by 1 bit simultaneously. Repeats until the layer becomes compute-
    limited or all transferred components reach *min_bits*.

    Returns
    -------
    input_bits : int
    weight_bits : int
    output_bits : int
    total_cycles : float   – max(compute_cycles, total_transfer_cycles)
    """
    compute = _compute_layer_cycles(layer)

    def _transfer_cycles(name, bits):
        elems = 0
        if name == 'weight':
            if not need_weight_transfer:
                return 0.0
            elems = layer.get('weight_elems', 0)
        elif name == 'input':
            if not need_input_transfer:
                return 0.0
            elems = layer.get('input_elems', 0)
        elif name == 'output':
            if not need_output_transfer:
                return 0.0
            elems = layer.get('output_elems', 0)
        if elems <= 0:
            return 0.0
        num_chunks = math.ceil(elems / 128)
        bytes_per_chunk = 16 * bits
        return (num_chunks * bytes_per_chunk) / bandwidth

    input_bits = max_bits
    weight_bits = max_bits
    output_bits = max_bits

    while True:
        w_t = _transfer_cycles('weight', weight_bits)
        i_t = _transfer_cycles('input', input_bits)
        o_t = _transfer_cycles('output', output_bits)
        total_transfer = w_t + i_t + o_t

        if total_transfer <= compute:
            break

        reduced = False
        if need_weight_transfer and weight_bits > min_bits:
            weight_bits -= 1
            reduced = True
        if need_input_transfer and input_bits > min_bits:
            input_bits -= 1
            reduced = True
        if need_output_transfer and output_bits > min_bits:
            output_bits -= 1
            reduced = True

        if not reduced:
            break

    total_transfer = (
        _transfer_cycles('weight', weight_bits) +
        _transfer_cycles('input', input_bits) +
        _transfer_cycles('output', output_bits)
    )
    total_cycles = max(compute, total_transfer)

    return input_bits, weight_bits, output_bits, total_cycles


def analyze_model(model_cfg_or_name, batch_size: int, device: str = "cpu", adapter_cfg: dict = None,
                  cache_elements: int = 0, bank_size: int = 0, metadata_bits: int = 0):
    """Trace model to get layer element counts in execution order."""
    import torch.nn.functional as F
    import yaml
    from runspace.src.adapters.adapter_factory import create_adapter

    if isinstance(model_cfg_or_name, dict):
        config = {
            'model': model_cfg_or_name,
            'adapter': dict({'type': 'generic', 'build_quantized': True}, **(adapter_cfg or {}))
        }
    elif isinstance(model_cfg_or_name, str) and (model_cfg_or_name.endswith('.yaml') or model_cfg_or_name.endswith('.yml') or os.path.isfile(model_cfg_or_name)):
        with open(model_cfg_or_name, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            if isinstance(config, list):
                item = config[0]
                config = {'model': item if isinstance(item, dict) else {'name': item, 'weights': None}}
            else:
                raise ValueError(f"Loaded YAML from {model_cfg_or_name} is not a valid dictionary or list.")
        if 'model' not in config:
            config = {'model': config}
        if 'adapter' not in config:
            config['adapter'] = {}
        config['adapter']['build_quantized'] = True
    else:
        config = {
            'model': {'name': model_cfg_or_name, 'weights': None},
            'adapter': {'type': 'generic', 'build_quantized': True}
        }

    adapter = create_adapter(config)
    model = adapter.model
    model.eval()
    model.to(device)

    execution_order = []
    hooks        = []
    _scope_stack = []   # tracks innermost active module name for functional-op naming

    _residual_block_types = []
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck
        _residual_block_types = [BasicBlock, Bottleneck]
    except ImportError:
        pass

    _POOL_TYPES = (
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
        nn.MaxPool1d, nn.AvgPool1d,
    )
    _NORM_TYPES = (
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm,
    )

    def _current_scope() -> str:
        return _scope_stack[-1] if _scope_stack else 'unknown'

    # --- Op identification ---
    # Register hooks on modules that match the DynamicInputQuantizer's hooking strategy.
    # We want to hook "leaf" operations and exclude high-level containers like DecomposedMultiheadAttention.
    supported_ops = set(OpRegistry.get_supported_ops().values())
    functional_ops = []
    # These match DynamicInputQuantizer._FUNCTIONAL_OP_NAMES
    for op_name in ["QuantMatMul", "QuantBMM", "QuantAdd", "QuantSub", "QuantMul", "QuantDiv", "QuantCat"]:
        try:
            functional_ops.append(OpRegistry.get(op_name))
        except Exception:
            continue
    
    # Filter out decomposed containers that are NOT hooked by the quantizer.
    # Note: We now exclude DecomposedMultiheadAttention as well, because we recurse into its 
    # sub-blocks (ScaledDotProduct, etc.) to hook the individual matmuls.
    EXCLUDED_CONTAINERS = ("DecomposedMlpBlock", "DecomposedQkvAttention", "DecomposedMultiheadAttention")
    quantized_types = tuple(
        cls for cls in set(list(supported_ops) + functional_ops)
        if cls.__name__ not in EXCLUDED_CONTAINERS
    )

    # --- module recording hooks ---

    def hook_fn(module, input, output):
        if isinstance(module, (nn.Conv2d, nn.Linear) + quantized_types):
            info = {
                'name':         getattr(module, 'layer_name', 'unknown'),
                'type':         module.__class__.__name__,
                'weight_elems': module.weight.numel() if getattr(module, 'weight', None) is not None else 0,
                'input_elems':  input[0].numel() if isinstance(input[0], torch.Tensor) else 0,
                'output_elems': output.numel()   if isinstance(output,   torch.Tensor) else 0,
                'input_shapes': _collect_tensor_shapes(input),
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
            }
            if isinstance(module, nn.Conv2d):
                ks = module.kernel_size
                if isinstance(ks, tuple):
                    filter_height, filter_width = ks
                else:
                    filter_height = filter_width = ks
                in_t = input[0] if isinstance(input[0], torch.Tensor) else None
                out_t = output   if isinstance(output,   torch.Tensor) else None

                info['in_channels']            = module.in_channels
                info['out_channels']           = module.out_channels
                info['filter_height']          = filter_height
                info['filter_width']           = filter_width
                info['kernel_size']            = ks[0] if isinstance(ks, tuple) else ks
                info['groups']                 = module.groups
                info['input_channel_height']   = in_t.shape[-2] if in_t is not None and in_t.ndim >= 4 else 0
                info['input_channel_width']    = in_t.shape[-1] if in_t is not None and in_t.ndim >= 4 else 0
                info['output_channel_height']  = out_t.shape[-2] if out_t is not None and out_t.ndim >= 4 else 0
                info['output_channel_width']   = out_t.shape[-1] if out_t is not None and out_t.ndim >= 4 else 0
                info['jump_back_size_in_banks'] = round_to_banks(info['filter_width'] * info['in_channels'] * (info['input_channel_width'])//128 * 128, bank_size)
            elif isinstance(module, nn.Linear):
                info['in_features']  = module.in_features
                info['out_features'] = module.out_features
            execution_order.append(info)
        elif isinstance(module, _POOL_TYPES):
            execution_order.append({
                'name':         getattr(module, 'layer_name', 'unknown'),
                'type':         module.__class__.__name__,
                'weight_elems': 0,
                'input_elems':  input[0].numel() if isinstance(input[0], torch.Tensor) else 0,
                'output_elems': output.numel()   if isinstance(output,   torch.Tensor) else 0,
                'input_shapes': _collect_tensor_shapes(input),
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
            })
        elif isinstance(module, _NORM_TYPES):
            in_t  = input[0] if isinstance(input[0], torch.Tensor) else None
            out_t = output   if isinstance(output,   torch.Tensor) else None
            wt    = module.weight.numel() if getattr(module, 'weight', None) is not None else 0
            execution_order.append({
                'name':         getattr(module, 'layer_name', 'unknown'),
                'type':         module.__class__.__name__,
                'weight_elems': wt,
                'input_elems':  in_t.numel()  if in_t  is not None else 0,
                'output_elems': out_t.numel() if out_t is not None else 0,
                'input_shapes': _collect_tensor_shapes(input),
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
            })

    def residual_hook_fn(module, input, output):
        skip = input[0] if isinstance(input[0], torch.Tensor) else None
        execution_order.append({
            'name':           getattr(module, 'layer_name', 'unknown'),
            'type':           'Residual',
            'weight_elems':   0,
            'input_elems':    skip.numel() if skip is not None else 0,
            'output_elems':   output.numel() if isinstance(output, torch.Tensor) else 0,
            'input_shapes':    _collect_tensor_shapes(input),
            'output_shape':    tuple(output.shape) if isinstance(output, torch.Tensor) else None,
            'has_downsample': module.downsample is not None,
        })

    # --- scope tracking: push/pop for every module so functional ops get sensible names ---
    for name, module in model.named_modules():
        def _make_scope_hooks(n):
            def _pre(mod, inp):
                _scope_stack.append(n)
            def _post(mod, inp, out):
                if _scope_stack and _scope_stack[-1] == n:
                    _scope_stack.pop()
            return _pre, _post
        _pre, _post = _make_scope_hooks(name)
        hooks.append(module.register_forward_pre_hook(_pre))
        hooks.append(module.register_forward_hook(_post))

    # --- recording hooks (registered after scope hooks so scope is still live during forward) ---

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, _POOL_TYPES, _NORM_TYPES) + quantized_types):
            module.layer_name = name
            hooks.append(module.register_forward_hook(hook_fn))
        elif _residual_block_types and isinstance(module, tuple(_residual_block_types)):
            module.layer_name = name
            hooks.append(module.register_forward_hook(residual_hook_fn))

    # --- functional op patching ---
    # Patches Python-level calls to torch.matmul / torch.bmm / softmax.
    # Catches custom attention implementations (e.g. MobileViT) that use these directly.
    # nn.MultiheadAttention's C++ fast path is handled above by mha_hook_fn instead.
    # nn.Linear uses F.linear → torch.addmm (not matmul), so no double-counting.
    from runspace.src.utils.model_input_utils import resolve_model_input_size
    input_shape = resolve_model_input_size(model, batch_size=batch_size)
    dummy_input = torch.randn(*input_shape).to(device)

    try:
        with torch.no_grad():
            try:
                # Try standard single tensor input first
                model(dummy_input)
            except Exception as e1:
                try:
                    # Fallback for models requiring (x, None) tuple
                    model((dummy_input, None))
                except Exception as e2:
                    print(f"Warning: Dummy forward failed with both tensor and tuple.")
                    print(f"  Tensor error: {e1}")
                    print(f"  Tuple error: {e2}")
    finally:
        for h in hooks:
            h.remove()

    collapsed_order = []
    for layer in execution_order:
        if is_collapsible(layer['type']) and collapsed_order:
            prev_layer = collapsed_order[-1]
            prev_layer['output_elems'] = layer['output_elems']
            prev_layer['weight_elems'] += layer.get('weight_elems', 0)
            if 'collapsed_layers' not in prev_layer:
                prev_layer['collapsed_layers'] = []
            prev_layer['collapsed_layers'].append({
                'name': layer['name'],
                'type': layer['type'],
                'weight_elems': layer.get('weight_elems', 0),
                'output_elems': layer['output_elems'],
                'input_shapes': layer.get('input_shapes', []),
                'output_shape': layer.get('output_shape'),
            })
        else:
            collapsed_order.append(layer)
    execution_order = collapsed_order

    return execution_order


def _invert_layer_rules() -> dict:
    """Build rule_name -> list of layer types that include it (from LAYER_RULES)."""
    applies: dict = {}
    for layer_type, rule_keys in LAYER_RULES.items():
        for key in rule_keys:
            applies.setdefault(key, [])
            if layer_type not in applies[key]:
                applies[key].append(layer_type)
    return applies


def serialize_rules() -> list:
    """Return serializable rule metadata in RULES insertion order.

    'applies_to' is derived by inverting LAYER_RULES, so it stays in sync
    with LAYER_RULES without any duplication.
    """
    applies = _invert_layer_rules()
    result  = []
    for name, rule in RULES.items():
        layer_types = applies.get(name, [])
        if '__default__' in layer_types:
            applies_to = 'All layers'
        else:
            applies_to = ', '.join(layer_types) or 'N/A'
        result.append({
            'name':            name,
            'on_chip':         rule['on_chip'],
            'xin_from_cache':  rule['xin_from_cache'],
            'applies_to':      applies_to,
            'stay_condition':  rule.get('stay_condition', ''),
            'permanents':      rule.get('permanents', ''),
            'pipeline_banks':  rule.get('pipeline_banks', 0),
            'notes':           rule.get('notes', ''),
        })
    return result


def run_simulation(args):
    if getattr(args, 'model_config', None):
        args.model_name = args.model_config

    cache_elements = int(args.cache_size * 1_000_000)
    bank_size      = cache_elements // args.num_banks   # elements per bank

    adapter_override = {}
    if args.fold_layers is not None:
        adapter_override['fold_layers'] = args.fold_layers
    if args.fold_input_norm is not None:
        adapter_override['fold_input_norm'] = args.fold_input_norm
    if args.quantize_first_layer is not None:
        adapter_override['quantize_first_layer'] = args.quantize_first_layer

    # Resolve and load models to run
    models_to_run = []
    is_yaml = args.model_name.endswith('.yaml') or args.model_name.endswith('.yml') or os.path.isfile(args.model_name)
    if is_yaml:
        import yaml
        try:
            with open(args.model_name, 'r') as f:
                cfg = yaml.safe_load(f)
            
            if isinstance(cfg, list):
                for item in cfg:
                    if isinstance(item, dict):
                        models_to_run.append({
                            'config_path': args.model_name,
                            'name': item.get('name'),
                            'model_cfg': item,
                            'adapter_cfg': {}
                        })
                    elif isinstance(item, str):
                        models_to_run.append({
                            'config_path': args.model_name,
                            'name': item,
                            'model_cfg': {'name': item, 'weights': None},
                            'adapter_cfg': {}
                        })
            elif isinstance(cfg, dict):
                if 'model' in cfg and isinstance(cfg['model'], dict):
                    model_cfg = cfg['model']
                    name = model_cfg.get('name', args.model_name)
                    adapter_cfg = dict(cfg.get('adapter', {}), **adapter_override)
                else:
                    model_cfg = cfg
                    name = cfg.get('name', args.model_name)
                    adapter_cfg = dict({}, **adapter_override)
                
                models_to_run.append({
                    'config_path': args.model_name,
                    'name': name,
                    'model_cfg': model_cfg,
                    'adapter_cfg': adapter_cfg
                })
        except Exception as e:
            print(f"Warning: Failed to parse YAML file {args.model_name}: {e}")
            models_to_run.append({
                'config_path': None,
                'name': args.model_name,
                'model_cfg': {'name': args.model_name, 'weights': None},
                'adapter_cfg': dict({}, **adapter_override)
            })
    else:
        models_to_run.append({
            'config_path': None,
            'name': args.model_name,
            'model_cfg': {'name': args.model_name, 'weights': None},
            'adapter_cfg': dict({'type': 'generic', 'build_quantized': True}, **adapter_override)
        })

    total_models = len(models_to_run)
    print(f"--- ASIC Cache Simulation ---")
    print(f"Loaded {total_models} model(s) to simulate.")
    print(f"Cache Size:    {fmt_elems(cache_elements)} elements  ({args.num_banks} banks × {fmt_elems(bank_size)} elements)")
    print(f"Metadata Bits: {args.metadata_bits} per 128-bit chunk")
    print(f"Batch Size:    {args.batch_size}")
    print(f"-----------------------------")

    for idx, model_info in enumerate(models_to_run, 1):
        model_display_name = model_info['name']
        config_path = model_info['config_path']
        model_cfg = model_info['model_cfg']
        adapter_cfg = model_info['adapter_cfg']

        print(f"\n[{idx}/{total_models}] Simulating model: {model_display_name}" + (f" (from config: {config_path})" if config_path else ""))

        try:
            layers = analyze_model(model_cfg, args.batch_size, args.device, adapter_cfg,
                                   cache_elements, bank_size, args.metadata_bits)
        except Exception as e:
            print(f"Error: Failed to analyze model {model_display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not layers:
            print(f"No layers found to analyze for model {model_display_name}.")
            continue

        results = []
        prev_stay_on_chip = False

        for i, layer in enumerate(layers):
            next_layer = layers[i + 1] if i + 1 < len(layers) else None

            weight_elems = get_footprint_elements(layer['weight_elems'], args.metadata_bits)
            input_elems  = get_footprint_elements(layer['input_elems'],  args.metadata_bits)
            output_elems = get_footprint_elements(layer['output_elems'], args.metadata_bits)

            next_xin_elems = (
                get_footprint_elements(next_layer['input_elems'], args.metadata_bits)
                if next_layer else 0
            )

            output_banked   = round_to_banks(output_elems,   bank_size)
            input_banked    = round_to_banks(input_elems,    bank_size)
            weight_banked   = round_to_banks(weight_elems,   bank_size)
            next_xin_banked = round_to_banks(next_xin_elems, bank_size)

            ctx = {
                'input_banked':    input_banked,
                'output_banked':   output_banked,
                'weight_banked':   weight_banked,
                'next_xin_banked': next_xin_banked,
                'cache_elements':  cache_elements,
                'bank_size':       bank_size,
                'filter_height':   layer.get('filter_height', 0),
                'filter_width':    layer.get('filter_width', 0),
                'in_channels':     layer.get('in_channels', 0),
                'out_channels':    layer.get('out_channels', 0),
                'input_channel_height':  layer.get('input_channel_height', 0),
                'input_channel_width':   layer.get('input_channel_width', 0),
                'output_channel_height': layer.get('output_channel_height', 0),
                'output_channel_width':  layer.get('output_channel_width', 0),
                'jump_back_size_in_banks': layer.get('jump_back_size_in_banks', 0),
            }

            stay_on_chip, perm_elems, possible, rule_name = evaluate_stay(
                layer, ctx, next_layer, args.metadata_bits, bank_size, cache_elements
            )

            rule_xin_from_cache = RULES.get(rule_name, {}).get('xin_from_cache', True)
            need_input = (i == 0 or not prev_stay_on_chip or not rule_xin_from_cache)
            need_output = not stay_on_chip
            in_b, w_b, out_b, cycle_count = optimize_layer_bits(
                layer, args.bandwidth, need_input, True, need_output, min_bits=3
            )
            compute_cycles = _compute_layer_cycles(layer)

            input_bw_limited  = need_input  and in_b < 8
            weight_bw_limited = True         and w_b < 8
            output_bw_limited = need_output and out_b < 8

            results.append({
                'name':             layer['name'],
                'type':             layer['type'],
                'input_elems':      input_elems,
                'weight_elems':     weight_elems,
                'output_elems':     output_elems,
                'output_banked':    output_banked,
                'perm_elems':       perm_elems,
                'next_xin_banked':  next_xin_banked,
                'footprint_banks':  output_banked // bank_size,
                'next_xin_banks':   next_xin_banked // bank_size,
                'next_layer_name':  next_layer['name'] if next_layer else None,
                'total_required':   output_banked + next_xin_banked,
                'filter_height':    layer.get('filter_height', 0),
                'filter_width':     layer.get('filter_width', 0),
                'in_channels':      layer.get('in_channels', 0),
                'out_channels':     layer.get('out_channels', 0),
                'input_channel_height':  layer.get('input_channel_height', 0),
                'input_channel_width':   layer.get('input_channel_width', 0),
                'output_channel_height': layer.get('output_channel_height', 0),
                'output_channel_width':  layer.get('output_channel_width', 0),
                'stay_on_chip':     stay_on_chip,
                'xin_from_cache':   rule_xin_from_cache,
                'rule':             rule_name,
                'reason': (
                    rule_name if stay_on_chip
                    else f"no rule fits (flagged)" if rule_name == 'FLAGGED'
                    else f"{rule_name} — output to external"
                ),
                'collapsed_layers': layer.get('collapsed_layers', []),
                'input_bits':       in_b,
                'weight_bits':      w_b,
                'output_bits':      out_b,
                'input_bw_limited':   input_bw_limited,
                'weight_bw_limited':  weight_bw_limited,
                'output_bw_limited':  output_bw_limited,
                'compute_cycles':   compute_cycles,
                'total_cycles':     cycle_count,
            })
            prev_stay_on_chip = stay_on_chip

        # --- Console output ---
        COL = 11
        BWCOL = 6
        header = (
            f"{'Layer Name':<45} | {'Type':<14}"
            f" | {'Input':>{COL}} | {'Weights':>{COL}}"
            f" | {'Output':>{COL}} | {'Banked':>{COL}}"
            f" | {'NextXin':>{COL}} | {'OnChip':<7}"
            f" | {'inB':>{BWCOL}} | {'wB':>{BWCOL}} | {'outB':>{BWCOL}}"
            f" | Reason"
        )
        sep = "-" * len(header)
        print(f"\n{header}\n{sep}")

        quantize_count = flagged_count = 0
        for res in results:
            on_chip_str = "yes" if res['stay_on_chip'] else "no"
            print(
                f"{res['name']:<45} | {res['type']:<14}"
                f" | {fmt_elems(res['input_elems']):>{COL}}"
                f" | {fmt_elems(res['weight_elems']):>{COL}}"
                f" | {fmt_elems(res['output_elems']):>{COL}}"
                f" | {fmt_elems(res['output_banked']):>{COL}}"
                f" | {fmt_elems(res['next_xin_banked']):>{COL}}"
                f" | {on_chip_str:<7}"
                f" | {res['input_bits']:>{BWCOL}} | {res['weight_bits']:>{BWCOL}} | {res['output_bits']:>{BWCOL}}"
                f" | {res['reason']}"
            )
            if not res['stay_on_chip'] and res['rule'] != 'FLAGGED':
                quantize_count += 1
            elif res['rule'] == 'FLAGGED':
                flagged_count += 1

        print(sep)
        print(f"Total layers:              {len(results)}")
        print(f"Layers marked QUANTIZE:    {quantize_count}")
        print(f"Layers FLAGGED (no rule):  {flagged_count}")

        # --- off_chip_layers: names only ---
        off_chip_layers = [res['name'] for res in results if not res['stay_on_chip']]

        # --- Structured JSON output ---
        output = {
            'metadata': {
                'model':          model_display_name,
                'model_config':   config_path,
                'cache_elements': cache_elements,
                'cache_size_M':   args.cache_size,
                'num_banks':      args.num_banks,
                'bank_size':      bank_size,
                'metadata_bits':  args.metadata_bits,
                'batch_size':     args.batch_size,
                'bandwidth':      args.bandwidth,
                'timestamp':      datetime.utcnow().isoformat() + 'Z',
            },
            'summary': {
                'total_layers':   len(results),
                'quantize_count': quantize_count,
                'flagged_count':  flagged_count,
            },
            'layers': results,
            'off_chip_layers': off_chip_layers,
            'rules': serialize_rules(),
        }

        out_dir  = os.path.dirname(os.path.abspath(__file__))
        
        # Save standard output file
        out_path_std = os.path.join(out_dir, "simulation_results.json")
        with open(out_path_std, 'w') as f:
            json.dump(output, f, indent=2)

        # Save model-specific output file
        sanitized_model_name = "".join([c if c.isalnum() else "_" for c in model_display_name])
        out_path_model = os.path.join(out_dir, f"simulation_results_{sanitized_model_name}.json")
        with open(out_path_model, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {out_path_model} and {out_path_std}")

        # Upload to DB
        try:
            from runspace.src.database.handler import RunDatabase
            RunDatabase().store_cache_simulation(output)
            print(f"[CacheSim] Successfully stored simulation for {model_display_name} to DB.")
        except Exception as e:
            print(f"[CacheSim] Warning: could not upload to DB for {model_display_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str,   default="resnet18",
                        help="Model name (e.g., resnet18) or path to a model config .yaml file")
    parser.add_argument("--model_config",  type=str,   default=None,
                        help="Path to a model config .yaml file (overrides --model_name)")
    parser.add_argument("--cache_size",    type=float, default=2.0,
                        help="Cache size in millions of elements (e.g. 2.0 = 2,000,000 elements)")
    parser.add_argument("--num_banks",     type=int,   default=16)
    parser.add_argument("--metadata_bits", type=int,   default=0)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--bandwidth",     type=float, default=1.0,
                        help="Memory bandwidth in bytes/cycle for BW-limitation analysis")
    parser.add_argument("--fold_layers", action="store_true", dest="fold_layers",
                        default=True,
                        help="Fold batchnorm/conv layers during model build")
    parser.add_argument("--no_fold_layers", action="store_false", dest="fold_layers",
                        help="Disable layer folding during model build")
    parser.add_argument("--fold_input_norm", action="store_true", dest="fold_input_norm",
                        default=True,
                        help="Fold input normalization into the first layer")
    parser.add_argument("--no_fold_input_norm", action="store_false", dest="fold_input_norm",
                        help="Disable folding of input normalization")
    parser.add_argument("--quantize_first_layer", action="store_true", dest="quantize_first_layer",
                        default=True,
                        help="Quantize the first layer's input/weights")
    parser.add_argument("--no_quantize_first_layer", action="store_false", dest="quantize_first_layer",
                        help="Disable quantization of the first layer")
    args = parser.parse_args()

    run_simulation(args)
