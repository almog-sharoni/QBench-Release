import os
import sys
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.registry.op_registry import OpRegistry
ops = OpRegistry.get_supported_ops()
print(f'Ops: {list(ops.keys())}')
for k, v in ops.items():
    print(f'  {k}: {v.__name__}')
