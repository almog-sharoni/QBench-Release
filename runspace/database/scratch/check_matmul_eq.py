import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.utils.fx_trace_utils import trace_quant_aware
from runspace.src.ops.quant_mha import ScaledDotProduct

model = ScaledDotProduct(64)
tracer, graph, gm = trace_quant_aware(model)

print(f"torch.matmul: {torch.matmul}")
for node in graph.nodes:
    if node.op == 'call_function':
        print(f"  Node target: {node.target}")
        print(f"  Equal to torch.matmul? {node.target == torch.matmul}")
        if node.target == torch.matmul:
            print("    MATCH!")
