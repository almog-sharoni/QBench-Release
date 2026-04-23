"""
Verify that analyze_model captures every meaningful op in a given model.

Outputs two tables:
  1. Captured ops — type + count as recorded by analyze_model
  2. Module census — every nn.Module subtype in the model, whether captured (✓) or not (✗)

For attention matmul/bmm/softmax (which are functional, not nn.Module) a separate
section shows how many were captured per functional op type.

Usage:
    python verify_coverage.py --model_name vit_b_16
    python verify_coverage.py --model_name mobilevit_s --batch_size 1 --device cpu
"""

import os
import sys
import argparse
import collections

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_FUNCTIONAL_TYPES = {'Matmul', 'BMM', 'Softmax'}

_SKIP_MODULE_TYPES = {
    # containers — no compute of their own
    'Sequential', 'ModuleList', 'ModuleDict',
    # activation functions — not memory-bandwidth bottlenecks
    'ReLU', 'ReLU6', 'GELU', 'SiLU', 'Hardswish', 'Hardsigmoid',
    'Tanh', 'Sigmoid', 'LeakyReLU', 'PReLU', 'ELU', 'SELU',
    # dropout — identity at eval time
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',
    # tiny wrappers
    'Identity', 'Flatten',
}

# These module types are decomposed into sub-ops by the MHA hook (not directly hookable).
_MHA_DECOMPOSED_TYPES = {
    'MultiheadAttention',             # hooked explicitly; emits in_proj/attn_qk/softmax/attn_av/out_proj
    'NonDynamicallyQuantizableLinear', # out_proj inside MHA, called via F.linear in C++ fast path
}

# Pure container modules in common architectures — no direct compute, just call children.
_CONTAINER_TYPES = {
    'Encoder', 'EncoderBlock', 'MLPBlock',            # torchvision ViT
    'MobileViTBlock', 'InvertedResidual', 'MV2Block', # MobileViT
}


def _model_module_census(model) -> dict[str, int]:
    """Count every nn.Module subtype present in the model (excluding root)."""
    census: dict[str, int] = collections.Counter()
    root_cls = model.__class__.__name__
    for name, module in model.named_modules():
        if not name:   # skip the root module itself
            continue
        t = module.__class__.__name__
        if t != root_cls:
            census[t] += 1
    return census


def _load_model(model_name: str):
    from runspace.src.adapters.adapter_factory import create_adapter
    config = {
        'model':   {'name': model_name, 'weights': None},
        'adapter': {'type': 'generic', 'build_quantized': False},
    }
    return create_adapter(config).model.eval()


def run_verification(model_name: str, batch_size: int = 1, device: str = 'cpu'):
    from simulate_cache import analyze_model

    print(f"\n{'='*60}")
    print(f"Coverage verification: {model_name}")
    print(f"{'='*60}")

    # --- module census (ground truth for module-level ops) ---
    model  = _load_model(model_name)
    census = _model_module_census(model)
    del model   # free memory before re-loading inside analyze_model

    # --- captured ops ---
    captured = analyze_model(model_name, batch_size, device)
    cap_types: dict[str, int] = collections.Counter(layer['type'] for layer in captured)

    # --- report: captured ops ---
    print(f"\nCaptured ops ({sum(cap_types.values())} total):")
    print(f"  {'Type':<30} {'Count':>6}")
    print(f"  {'-'*36}")
    for t, n in sorted(cap_types.items()):
        marker = '  (functional)' if t in _FUNCTIONAL_TYPES else ''
        print(f"  {t:<30} {n:>6}{marker}")

    # --- report: module coverage ---
    # For MHA modules: each emits 5 sub-ops (in_proj, attn_qk, attn_softmax, attn_av, out_proj).
    # NonDynamicallyQuantizableLinear (out_proj) is emitted by the parent MHA hook, not directly.
    mha_module_count = census.get('MultiheadAttention', 0)
    mha_subops       = mha_module_count * 5   # in_proj + qk + softmax + av + out_proj

    print(f"\nModule coverage ({sum(census.values())} modules in model):")
    print(f"  {'Status':<8} {'Type':<38} {'In model':>8}  {'Captured':>8}")
    print(f"  {'-'*62}")

    missed_important = []
    for t, model_count in sorted(census.items()):
        if t in _SKIP_MODULE_TYPES:
            status    = '  skip   '
            cap_str   = '-'
        elif t == 'MultiheadAttention':
            status    = '  mha    '
            cap_str   = f'{mha_subops} sub-ops'
        elif t in _MHA_DECOMPOSED_TYPES:
            status    = '  mha    '
            cap_str   = '(via parent)'
        elif t in _CONTAINER_TYPES:
            status    = '  ctnr   '
            cap_str   = '-'
        elif t in cap_types:
            status    = '  ✓      '
            cap_str   = str(cap_types[t])
        else:
            status    = '  ✗      '
            cap_str   = '0'
            missed_important.append((t, model_count))
        print(f"  {status} {t:<38} {model_count:>8}  {cap_str:>12}")

    # --- report: functional op sanity check ---
    print(f"\nFunctional op capture:")
    for ft in ('Matmul', 'BMM', 'Softmax'):
        n = cap_types.get(ft, 0)
        print(f"  {ft:<30} {n:>6}")

    # --- summary ---
    if missed_important:
        print(f"\n[!] Uncaptured module types ({len(missed_important)}):")
        for t, n in missed_important:
            print(f"    {t}: {n} instances")
    else:
        print(f"\n[✓] All non-trivial module types are captured.")

    return cap_types, census


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device',     type=str, default='cpu')
    args = parser.parse_args()

    run_verification(args.model_name, args.batch_size, args.device)
