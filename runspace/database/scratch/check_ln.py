import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.registry.op_registry import OpRegistry

ops = OpRegistry.get_supported_ops()
print(f'LayerNorm in supported ops: {nn.LayerNorm in ops}')
if nn.LayerNorm in ops:
    print(f'Mapped to: {ops[nn.LayerNorm]}')
