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
    'QuantConv2d':       ['conv_output_dominated', 'conv_input_dominated', 'global_fit',
                          'stream_xin_keep_xout', 'conv_stream_xin_out', 'fallback'],
    'Conv2d':            ['conv_output_dominated', 'conv_input_dominated', 'global_fit',
                          'stream_xin_keep_xout', 'conv_stream_xin_out', 'fallback'],
    'QuantLinear':       ['global_fit', 'linear_stream_xout', 'fallback'],
    'Linear':            ['global_fit', 'linear_stream_xout', 'fallback'],
    'Residual':          ['residual', 'fallback'],
    'QuantAdd':          ['residual', 'global_fit', 'fallback'],
    'QuantMaxPool2d':    ['global_fit', 'pool', 'fallback'],
    'QuantAvgPool2d':    ['global_fit', 'pool', 'fallback'],
    'QuantMaxPool1d':    ['global_fit', 'pool', 'fallback'],
    'QuantAvgPool1d':    ['global_fit', 'pool', 'fallback'],
    'QuantMatMul':       ['global_fit', 'fallback'],
    'QuantSoftmax':      ['global_fit', 'fallback'],
    'LayerNorm':         ['global_fit', 'fallback'],
    'BatchNorm1d':       ['global_fit', 'fallback'],
    'BatchNorm2d':       ['global_fit', 'fallback'],
    'BatchNorm3d':       ['global_fit', 'fallback'],
    'GroupNorm':         ['global_fit', 'fallback'],
    'QuantGELU':         ['global_fit', 'fallback'],
    'QuantDropout':      ['global_fit', 'fallback'],
    'DecomposedMultiheadAttention': ['global_fit', 'fallback'],
    '__default__':       ['global_fit', 'fallback'],
}
