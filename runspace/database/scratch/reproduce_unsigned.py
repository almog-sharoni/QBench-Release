import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.registry.op_registry import OpRegistry

config = {
    'model': {'name': 'vit_b_16', 'weights': 'DEFAULT'},
    'adapter': {
        'type': 'generic',
        'unsigned_input_sources': ['quantsoftmax', 'softmax']
    },
    'quantization': {'format': 'fp8_e4m3'}
}
adapter = create_adapter(config)
model = adapter.model

print('Searching for matmul modules...')
found = False
for name, module in model.named_modules():
    if 'matmul' in name.lower():
        print(f"  Found: {name} ({module.__class__.__name__})")
        found = True

if not found:
    print("  No matmul modules found!")
