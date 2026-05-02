import os
import sys
import torch
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.core.runner import Runner
runner = Runner()
config = {
    'model': {'name': 'vit_b_16', 'weights': 'DEFAULT'},
    'adapter': {'type': 'generic'},
    'quantization': {'format': 'fp8_e4m3'}
}
# We need to build the model to see the names
from runspace.src.adapters.adapter_factory import create_adapter
adapter = create_adapter(config)
model = adapter.model
print('Model modules:')
for name, module in model.named_modules():
    if 'matmul' in name.lower() or 'div' in name.lower() or 'attention' in name.lower():
        print(f"  {name}: {module.__class__.__name__}")
