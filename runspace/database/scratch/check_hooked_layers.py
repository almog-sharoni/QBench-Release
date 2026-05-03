import os
import sys
import torch
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

config = {
    'model': {'name': 'vit_b_16', 'weights': 'DEFAULT'},
    'adapter': {'type': 'generic'},
    'quantization': {
        'format': 'fp8_e4m3',
        'input_quantization': {
            'enabled': True,
            'mode': 'dynamic_mse',
            'format': 'fp8_e4m3'
        }
    }
}

adapter = create_adapter(config)
model = adapter.model

quantizer = DynamicInputQuantizer(
    model,
    metric='mse',
    candidate_formats=['fp4_e2m1'] # just to initialize
)
quantizer.register_hooks()

print(f"Total layers hooked: {len(quantizer.hooks)}")

# Check for matmuls in hooked modules
matmul_hooked = [m for m in quantizer.hooked_modules if 'QuantMatMul' in m.__class__.__name__]
print(f"Matmuls hooked: {len(matmul_hooked)}")

# Print some names
for h in quantizer.hooks[:10]:
    # We can't easily get the name from the hook itself without some effort
    pass

# Actually, let's just look at hooked_modules and their layer names if assigned
for m in quantizer.hooked_modules:
    if 'matmul' in getattr(m, 'layer_name', '').lower():
        print(f"Hooked matmul: {m.layer_name}")
