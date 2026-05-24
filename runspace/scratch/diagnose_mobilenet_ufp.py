
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.generic_adapter import GenericAdapter
from runspace.src.utils.fx_trace_utils import trace_quant_aware

def diagnose_mobilenet_v3():
    print("=== Scanning MobileNetV3 Module Classes ===")
    
    adapter = GenericAdapter(
        model_name="mobilenet_v3_large",
        unsigned_input_sources=["relu", "relu6", "hardsigmoid"],
        input_quantization=True,
        enable_fx_quantization=True
    )
    
    model = adapter.build_model(quantized=True)
    
    _, _, gm = trace_quant_aware(model)
    modules = dict(gm.named_modules())
    
    unique_classes = set()
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            m = modules.get(node.target)
            if m:
                unique_classes.add(m.__class__.__name__)
    
    print("\nUnique Classes in FX Graph:")
    for c in sorted(unique_classes):
        print(f" - {c}")

if __name__ == "__main__":
    diagnose_mobilenet_v3()
