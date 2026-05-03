import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.utils.fx_trace_utils import trace_quant_aware
import torchvision

model = torchvision.models.vit_b_16(weights=None)
print('Tracing VisionTransformer...')
tracer, graph, gm = trace_quant_aware(model)

print(f"Total nodes: {len(graph.nodes)}")
found_matmul = False
for node in graph.nodes:
    if node.op == 'call_function' and 'matmul' in str(node.target).lower():
        print(f"  Found matmul node: {node.name}")
        found_matmul = True

if not found_matmul:
    print("  No matmul nodes found in whole-model trace.")
