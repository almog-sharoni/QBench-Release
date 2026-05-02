import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.adapters.generic_adapter import GenericAdapter

adapter = GenericAdapter(model_name='vit_b_16')
model = adapter.build_model(quantized=True)
print(f'Total modules: {len(list(model.named_modules()))}')
for name, mod in model.named_modules():
    if 'matmul' in type(mod).__name__.lower() or 'quantadd' in type(mod).__name__.lower():
        print(f'{name}: {type(mod).__name__}')
