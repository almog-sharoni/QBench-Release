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

    Each FP8 element is 8 bits, so 16 elements form one 128-bit chunk.
    Metadata bytes (ceil(metadata_bits/8)) are counted as element-equivalents
    (1 byte = 1 FP8 element).
    """
    if num_elements <= 0:
        return 0
    num_chunks = math.ceil(num_elements / 16)          # 16 FP8 elements per 128-bit chunk
    metadata_elems = math.ceil(num_chunks * metadata_bits / 8)
    return num_elements + metadata_elems


def round_to_banks(size_elems: int, bank_size: int) -> int:
    """Round up to the nearest bank boundary (in elements)."""
    if size_elems <= 0:
        return 0
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


# ---------------------------------------------------------------------------

def analyze_model(model_cfg_or_name, batch_size: int, device: str = "cpu", adapter_cfg: dict = None):
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
            }
            if isinstance(module, nn.Conv2d):
                ks = module.kernel_size
                info['in_channels']  = module.in_channels
                info['out_channels'] = module.out_channels
                info['kernel_size']  = ks[0] if isinstance(ks, tuple) else ks
                info['groups']       = module.groups
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
            })

    def residual_hook_fn(module, input, output):
        skip = input[0] if isinstance(input[0], torch.Tensor) else None
        execution_order.append({
            'name':           getattr(module, 'layer_name', 'unknown'),
            'type':           'Residual',
            'weight_elems':   0,
            'input_elems':    skip.numel() if skip is not None else 0,
            'output_elems':   output.numel() if isinstance(output, torch.Tensor) else 0,
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
                    adapter_cfg = cfg.get('adapter', {})
                else:
                    model_cfg = cfg
                    name = cfg.get('name', args.model_name)
                    adapter_cfg = {}
                
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
                'adapter_cfg': {}
            })
    else:
        models_to_run.append({
            'config_path': None,
            'name': args.model_name,
            'model_cfg': {'name': args.model_name, 'weights': None},
            'adapter_cfg': {}
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
            layers = analyze_model(model_cfg, args.batch_size, args.device, adapter_cfg)
        except Exception as e:
            print(f"Error: Failed to analyze model {model_display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not layers:
            print(f"No layers found to analyze for model {model_display_name}.")
            continue

        results = []

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
            }

            stay_on_chip, perm_elems, possible, rule_name = evaluate_stay(
                layer, ctx, next_layer, args.metadata_bits, bank_size, cache_elements
            )

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
                'stay_on_chip':     stay_on_chip,
                'rule':             rule_name,
                'reason': (
                    rule_name if stay_on_chip
                    else f"no rule fits (flagged)" if rule_name == 'FLAGGED'
                    else f"{rule_name} — output to external"
                ),
            })

        # --- Console output ---
        COL = 11
        header = (
            f"{'Layer Name':<45} | {'Type':<14}"
            f" | {'Input':>{COL}} | {'Weights':>{COL}}"
            f" | {'Output':>{COL}} | {'Banked':>{COL}}"
            f" | {'NextXin':>{COL}} | {'OnChip':<7} | Reason"
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
                f" | {on_chip_str:<7} | {res['reason']}"
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
    args = parser.parse_args()

    run_simulation(args)
