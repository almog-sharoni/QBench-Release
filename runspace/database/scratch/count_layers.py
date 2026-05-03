import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.adapters.adapter_factory import create_adapter

config = {
    'model': {'name': 'vit_b_16', 'weights': 'DEFAULT'},
    'adapter': {'type': 'generic'},
    'quantization': {'format': 'fp8_e4m3'}
}
adapter = create_adapter(config)
model = adapter.model

print(f"Total layers: {len(list(model.named_modules()))}")
