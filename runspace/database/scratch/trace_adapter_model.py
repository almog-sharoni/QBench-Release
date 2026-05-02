import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.utils.fx_trace_utils import trace_quant_aware

config = {
    'model': {'name': 'vit_b_16', 'weights': 'DEFAULT'},
    'adapter': {'type': 'generic'},
    'quantization': {'format': 'fp8_e4m3'}
}
adapter = create_adapter(config)
model = adapter.model

print('Tracing model...')
# Note: adapter.model might already be a GraphModule if build_model called _fx_quantize
# But we want to see if we can trace it again or if the first trace failed.
tracer, graph, gm = trace_quant_aware(model)

print(f"Total nodes: {len(graph.nodes)}")
found_matmul = False
for node in graph.nodes:
    if node.op == 'call_function' and 'matmul' in str(node.target).lower():
        # print(f"  Found matmul node: {node.name}")
        found_matmul = True

if found_matmul:
    print("  Matmul nodes FOUND in trace.")
else:
    print("  No matmul nodes found in trace.")
