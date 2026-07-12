import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.adapters.generic_adapter import GenericAdapter
from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

adapter = GenericAdapter(
    model_name='vit_b_16',
    input_quantization=False,
    output_quantization=False,
)
model = adapter.model
quantizer = DynamicInputQuantizer(model)
quantizer.register_hooks()
plan = quantizer._transport_runtime.plan
print(f'Total activation transport stages: {len(plan.stages)}')
types = {}
for stage in plan.stages:
    t = stage.kind.value
    types[t] = types.get(t, 0) + 1
for t, c in sorted(types.items()):
    print(f'{t}: {c}')
quantizer.cleanup()
