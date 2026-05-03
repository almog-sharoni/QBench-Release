import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.adapters.generic_adapter import GenericAdapter
from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

adapter = GenericAdapter(model_name='vit_b_16')
model = adapter.build_model(quantized=True)
quantizer = DynamicInputQuantizer(model)
quantizer.register_hooks()
print(f'Total hooked modules: {len(quantizer.hooked_modules)}')
types = {}
for m in quantizer.hooked_modules:
    t = type(m).__name__
    types[t] = types.get(t, 0) + 1
for t, c in sorted(types.items()):
    print(f'{t}: {c}')
