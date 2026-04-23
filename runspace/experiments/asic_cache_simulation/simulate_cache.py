import os
import sys
import json
import torch
import torch.nn as nn
import argparse
import math
from datetime import datetime

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
# Rule system
#
# RULES: dict of rule_name -> rule definition.  Each rule has:
#   'on_chip'       : bool  — True  → output stays in cache after this rule
#                             False → output streamed to external memory (QUANTIZE)
#   'xin_from_cache': bool  — True  → rule requires xin fully resident in cache
#                             False → rule streams xin from external (2-bank buffer)
#   'ctx_guard'     : (ctx) -> bool  — optional runtime condition (e.g. xout >= xin);
#                                      evaluated before stay(); rule skipped if False
#   'perm'          : (ctx) -> int   — elements resident after execution
#   'stay'          : (ctx) -> bool  — capacity check for this rule
#
# ctx fields (all in bank-aligned elements unless noted):
#   input_banked, output_banked, weight_banked, cache_elements, bank_size
#
# LAYER_RULES: dict of layer_type -> ordered list of rule keys to try.
#   '__default__' is the fallback for any type not explicitly listed.
#
# Evaluation:
#   1. Look up layer type in LAYER_RULES (fallback to '__default__')
#   2. Iterate rule keys in order; for each:
#      a. If ctx_guard present and returns False → skip
#      b. If stay(ctx) is False → skip
#      c. Run 1-level lookahead: next layer must have a compatible rule given
#         the xin source implied by on_chip:
#           on_chip=True  → next xin from cache  (xin_from_cache=True rules only)
#           on_chip=False → next xin from external (xin_from_cache=False rules only)
#   3. First confirmed rule wins.  No confirmation → FLAGGED.
# ---------------------------------------------------------------------------

RULES = {

    'global_fit': {
        'on_chip':        True,
        'xin_from_cache': True,
        'stay_condition': 'xin + xout + 2 banks ≤ cache',
        'permanents':     'xout (→ next xin)',
        'notes':          'Everything fits; both tensors stay on chip simultaneously.',
        'perm':  lambda ctx: ctx['input_banked'],
        'stay':  lambda ctx: (
            ctx['input_banked'] + ctx['output_banked'] + 2 * ctx['bank_size']
            <= ctx['cache_elements']
        ),
    },

    'residual': {
        'on_chip':        True,
        'xin_from_cache': True,
        'stay_condition': 'xin + 4 banks (2 xout, 2 x_residual) ≤ cache',
        'permanents':     'xin (skip tensor)',
        'notes':          'Skip tensor held in cache; x_residual + xout use 4 banks (2 xout, 2 x_residual), xout is written on xin.',
        'perm':  lambda ctx: ctx['input_banked'],
        'stay':  lambda ctx: (
            ctx['input_banked'] + 4 * ctx['bank_size']
            <= ctx['cache_elements']
        ),
    },

    'conv_output_dominated': {
        'on_chip':        True,
        'xin_from_cache': True,
        'ctx_guard':      lambda ctx: ctx['output_banked'] >= ctx['input_banked'],
        'stay_condition': 'xout + weights + 1 bank ≤ cache',
        'permanents':     'weights + xout',
        'pipeline_banks': 1,
        'notes':          'xout is the larger tensor; xin is written onto xout\'s space. 1 bank overhead for read/write pipeline boundary.',
        'perm':  lambda ctx: ctx['weight_banked'] + ctx['output_banked'],
        'stay':  lambda ctx: (
            ctx['output_banked'] + 1 * ctx['bank_size'] + ctx['weight_banked']
            <= ctx['cache_elements']
        ),
    },

    'conv_input_dominated': {
        'on_chip':        True,
        'xin_from_cache': True,
        'ctx_guard':      lambda ctx: ctx['input_banked'] > ctx['output_banked'],
        'stay_condition': 'xin + weights + 1 bank ≤ cache',
        'permanents':     'weights + xin',
        'pipeline_banks': 1,
        'notes':          'xin is the larger tensor; xout is written onto xin\'s space. 1 bank overhead for read/write pipeline boundary.',
        'perm':  lambda ctx: ctx['weight_banked'] + ctx['input_banked'],
        'stay':  lambda ctx: (
            ctx['input_banked'] + 1 * ctx['bank_size'] + ctx['weight_banked']
            <= ctx['cache_elements']
        ),
    },

    'pool': {
        'on_chip':        True,
        'xin_from_cache': True,
        'stay_condition': 'xin + xout + 2 banks ≤ cache',
        'permanents':     'xout',
        'notes':          'No weights; output stays on chip. Shows size reduction from spatial downsampling.',
        'perm':  lambda ctx: ctx['output_banked'],
        'stay':  lambda ctx: (
            ctx['input_banked'] + ctx['output_banked'] + 2 * ctx['bank_size']
            <= ctx['cache_elements']
        ),
    },

    'stream_xin_keep_xout': {
        'on_chip':        True,
        'xin_from_cache': False,
        'stay_condition': 'xout + weights + 2 banks ≤ cache',
        'permanents':     'weights + xout',
        'notes':          'xin streams from external (prev layer evicted); xout still kept on chip.',
        'perm':  lambda ctx: ctx['weight_banked'] + ctx['output_banked'],
        'stay':  lambda ctx: (
            ctx['output_banked'] + ctx['weight_banked'] + 2 * ctx['bank_size']
            <= ctx['cache_elements']
        ),
    },

    'fallback': {
        'on_chip':        False,
        'xin_from_cache': False,
        'stay_condition': 'weights + 4 banks ≤ cache',
        'permanents':     'weights only',
        'notes':          'Full streaming: xin and xout both via external memory → output marked QUANTIZE.',
        'perm':  lambda ctx: ctx['weight_banked'],
        'stay':  lambda ctx: (
            ctx['weight_banked'] + 4 * ctx['bank_size']
            <= ctx['cache_elements']
        ),
    },

    'linear_stream_xout': {
        'on_chip':        False,
        'xin_from_cache': True,
        'stay_condition': 'xin + 4 banks ≤ cache',
        'permanents':     'xin',
        'notes':          'xin in cache, weights streamed in, xout is computed and streamed out.',
        'perm':  lambda ctx: ctx['input_banked'],
        'stay':  lambda ctx: (
            ctx['input_banked'] + 4 * ctx['bank_size']
            <= ctx['cache_elements']
        ),
    },

    'conv_stream_xin_out': {
        'on_chip':        False,
        'xin_from_cache': True,
        'stay_condition': 'xin + weights + 2 banks ≤ cache',
        'permanents':     'xin + weights (held while streaming xout)',
        'notes':          'xin from cache, weights resident, xout streamed to external.',
        'perm':  lambda ctx: ctx['input_banked'],
        'stay':  lambda ctx: (
            ctx['input_banked'] + ctx['weight_banked'] + 2 * ctx['bank_size']
            <= ctx['cache_elements']
        ),
    },
}


# Layer type → ordered list of rule keys to try (priority order).
# Linear intentionally skips conv/pool-specific rules and falls through to fallback.
LAYER_RULES = {
    'Conv2d':            ['conv_output_dominated', 'conv_input_dominated', 'global_fit',
                          'stream_xin_keep_xout', 'conv_stream_xin_out', 'fallback'],
    'Linear':            ['global_fit', 'linear_stream_xout', 'fallback'],
    'Residual':          ['residual', 'fallback'],
    'MaxPool2d':         ['global_fit', 'pool', 'fallback'],
    'AvgPool2d':         ['global_fit', 'pool', 'fallback'],
    'AdaptiveAvgPool2d': ['global_fit', 'pool', 'fallback'],
    'MaxPool1d':         ['global_fit', 'pool', 'fallback'],
    'AvgPool1d':         ['global_fit', 'pool', 'fallback'],
    # --- placeholder rules (edit to model actual hardware behaviour) ---
    'Matmul':            ['global_fit', 'fallback'],
    'BMM':               ['global_fit', 'fallback'],
    'Softmax':           ['global_fit', 'fallback'],
    'LayerNorm':         ['global_fit', 'fallback'],
    'BatchNorm1d':       ['global_fit', 'fallback'],
    'BatchNorm2d':       ['global_fit', 'fallback'],
    'BatchNorm3d':       ['global_fit', 'fallback'],
    'GroupNorm':         ['global_fit', 'fallback'],
    '__default__':       ['global_fit', 'fallback'],
}


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

def analyze_model(model_name: str, batch_size: int, device: str = "cpu"):
    """Trace model to get layer element counts in execution order."""
    import torch.nn.functional as F
    from runspace.src.adapters.adapter_factory import create_adapter
    config = {
        'model': {'name': model_name, 'weights': None},
        'adapter': {'type': 'generic', 'build_quantized': False}
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

    # --- module recording hooks ---

    def hook_fn(module, input, output):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            info = {
                'name':         getattr(module, 'layer_name', 'unknown'),
                'type':         module.__class__.__name__,
                'weight_elems': module.weight.numel(),
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

    def mha_hook_fn(module, input, output):
        # nn.MultiheadAttention uses a C++ fast path (F.multi_head_attention_forward) that
        # calls torch.bmm and softmax at C++ level, bypassing Python-level patches.
        # We decompose the 5 sub-ops explicitly from input shape + module params.
        name  = getattr(module, 'layer_name', 'unknown')
        query = input[0] if isinstance(input[0], torch.Tensor) else None
        if query is None:
            return
        if module.batch_first:
            batch_size, seq_len, _ = query.shape
        else:
            seq_len, batch_size, _ = query.shape
        embed_dim = module.embed_dim
        num_heads = module.num_heads
        head_dim  = embed_dim // num_heads
        bh        = batch_size * num_heads

        # 1. In-projection  (fused QKV or separate Q/K/V weights)
        if module.in_proj_weight is not None:
            execution_order.append({
                'name': f'{name}.in_proj', 'type': 'Linear',
                'weight_elems': module.in_proj_weight.numel(),
                'input_elems':  batch_size * seq_len * embed_dim,
                'output_elems': batch_size * seq_len * embed_dim * 3,
                'in_features': embed_dim, 'out_features': embed_dim * 3,
            })
        else:
            for proj, attr in (('q_proj', 'q_proj_weight'),
                               ('k_proj', 'k_proj_weight'),
                               ('v_proj', 'v_proj_weight')):
                wt = getattr(module, attr, None)
                if wt is not None:
                    execution_order.append({
                        'name': f'{name}.{proj}', 'type': 'Linear',
                        'weight_elems': wt.numel(),
                        'input_elems':  batch_size * seq_len * embed_dim,
                        'output_elems': batch_size * seq_len * embed_dim,
                        'in_features': embed_dim, 'out_features': embed_dim,
                    })

        # 2. Q @ K^T
        execution_order.append({
            'name': f'{name}.attn_qk', 'type': 'BMM',
            'weight_elems': 0,
            'input_elems':  bh * seq_len * head_dim,
            'output_elems': bh * seq_len * seq_len,
        })
        # 3. Softmax on attention weights
        execution_order.append({
            'name': f'{name}.attn_softmax', 'type': 'Softmax',
            'weight_elems': 0,
            'input_elems':  bh * seq_len * seq_len,
            'output_elems': bh * seq_len * seq_len,
        })
        # 4. Attn @ V
        execution_order.append({
            'name': f'{name}.attn_av', 'type': 'BMM',
            'weight_elems': 0,
            'input_elems':  bh * seq_len * seq_len,
            'output_elems': bh * seq_len * head_dim,
        })
        # 5. Output projection (NonDynamicallyQuantizableLinear, also bypassed in fast path)
        out_wt = module.out_proj.weight.numel() if getattr(module, 'out_proj', None) is not None else 0
        execution_order.append({
            'name': f'{name}.out_proj', 'type': 'Linear',
            'weight_elems': out_wt,
            'input_elems':  batch_size * seq_len * embed_dim,
            'output_elems': batch_size * seq_len * embed_dim,
            'in_features': embed_dim, 'out_features': embed_dim,
        })

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            module.layer_name = name
            hooks.append(module.register_forward_hook(mha_hook_fn))
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            module.layer_name = name
            hooks.append(module.register_forward_hook(hook_fn))
        elif isinstance(module, _POOL_TYPES):
            module.layer_name = name
            hooks.append(module.register_forward_hook(hook_fn))
        elif isinstance(module, _NORM_TYPES):
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
    _counters      = {}
    _orig_matmul   = torch.matmul
    _orig_bmm      = torch.bmm
    _orig_softmax  = F.softmax
    _orig_tsoftmax = getattr(torch, 'softmax', None)

    def _patched_matmul(input, other, *args, **kwargs):
        out = _orig_matmul(input, other, *args, **kwargs)
        n   = _counters.get('matmul', 0); _counters['matmul'] = n + 1
        scope = _current_scope()
        execution_order.append({
            'name':         f'{scope}.matmul_{n}' if scope != 'unknown' else f'matmul_{n}',
            'type':         'Matmul',
            'weight_elems': 0,
            'input_elems':  input.numel() if isinstance(input, torch.Tensor) else 0,
            'output_elems': out.numel()   if isinstance(out,   torch.Tensor) else 0,
        })
        return out

    def _patched_bmm(input, mat2, *args, **kwargs):
        out = _orig_bmm(input, mat2, *args, **kwargs)
        n   = _counters.get('bmm', 0); _counters['bmm'] = n + 1
        scope = _current_scope()
        execution_order.append({
            'name':         f'{scope}.bmm_{n}' if scope != 'unknown' else f'bmm_{n}',
            'type':         'BMM',
            'weight_elems': 0,
            'input_elems':  input.numel() if isinstance(input, torch.Tensor) else 0,
            'output_elems': out.numel()   if isinstance(out,   torch.Tensor) else 0,
        })
        return out

    def _patched_softmax(input, dim=None, _stacklevel=3, dtype=None):
        out   = _orig_softmax(input, dim=dim, dtype=dtype)
        n     = _counters.get('softmax', 0); _counters['softmax'] = n + 1
        scope = _current_scope()
        execution_order.append({
            'name':         f'{scope}.softmax_{n}' if scope != 'unknown' else f'softmax_{n}',
            'type':         'Softmax',
            'weight_elems': 0,
            'input_elems':  input.numel() if isinstance(input, torch.Tensor) else 0,
            'output_elems': out.numel()   if isinstance(out,   torch.Tensor) else 0,
        })
        return out

    torch.matmul = _patched_matmul
    torch.bmm    = _patched_bmm
    F.softmax    = _patched_softmax
    if _orig_tsoftmax is not None:
        torch.softmax = _patched_softmax

    from runspace.src.utils.model_input_utils import resolve_model_input_size
    input_shape = resolve_model_input_size(model, batch_size=batch_size)
    dummy_input = torch.randn(*input_shape).to(device)

    try:
        with torch.no_grad():
            try:
                model((dummy_input, None))
            except Exception:
                try:
                    model(dummy_input)
                except Exception as e:
                    print(f"Warning: Dummy forward failed: {e}.")
    finally:
        torch.matmul = _orig_matmul
        torch.bmm    = _orig_bmm
        F.softmax    = _orig_softmax
        if _orig_tsoftmax is not None:
            torch.softmax = _orig_tsoftmax
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
    cache_elements = int(args.cache_size * 1_000_000)
    bank_size      = cache_elements // args.num_banks   # elements per bank

    print(f"--- ASIC Cache Simulation ---")
    print(f"Model:         {args.model_name}")
    print(f"Cache Size:    {fmt_elems(cache_elements)} elements  ({args.num_banks} banks × {fmt_elems(bank_size)} elements)")
    print(f"Metadata Bits: {args.metadata_bits} per 128-bit chunk")
    print(f"Batch Size:    {args.batch_size}")

    layers = analyze_model(args.model_name, args.batch_size, args.device)
    if not layers:
        print("No layers found to analyze.")
        return

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

    # --- off_chip_layers: names only — format is resolved at run time by the runner ---
    off_chip_layers = [res['name'] for res in results if not res['stay_on_chip']]

    # --- Structured JSON output ---
    output = {
        'metadata': {
            'model':          args.model_name,
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
        # Layer names whose outputs must be quantized for external memory transfer.
        # The runner stamps these with whatever format+mode is active for the run.
        'off_chip_layers': off_chip_layers,
        # Serialized rule metadata — captured at run time so the dashboard always
        # reflects the rules that were actually used for this simulation.
        'rules': serialize_rules(),
    }

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "simulation_results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Upload to DB
    try:
        from runspace.src.database.handler import RunDatabase
        RunDatabase().store_cache_simulation(output)
    except Exception as e:
        print(f"[CacheSim] Warning: could not upload to DB: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str,   default="resnet18")
    parser.add_argument("--cache_size",    type=float, default=2.0,
                        help="Cache size in millions of elements (e.g. 2.0 = 2,000,000 elements)")
    parser.add_argument("--num_banks",     type=int,   default=16)
    parser.add_argument("--metadata_bits", type=int,   default=0)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--device",        type=str,   default="cuda")
    args = parser.parse_args()

    run_simulation(args)
