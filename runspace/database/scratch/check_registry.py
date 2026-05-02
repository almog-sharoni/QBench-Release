import os
import sys
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.registry.op_registry import OpRegistry
print('LayerNorm' in OpRegistry.get_supported_ops())
print('QuantLayerNorm' in OpRegistry.get_supported_ops())
print('QuantLayerNorm' in OpRegistry.list_registered_ops())
