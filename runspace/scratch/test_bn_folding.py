
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.generic_adapter import GenericAdapter

def test_folding():
    model = mobilenet_v3_large(weights=None)
    model.eval()
    adapter = GenericAdapter(fold_layers=True)
    
    print("Initial module count:", len(list(model.named_modules())))
    
    # Manually trigger folding logic
    named_modules = list(model.named_modules())
    modules_to_fuse = []
    i = 0
    while i < len(named_modules) - 1:
        name_curr, mod_curr = named_modules[i]
        name_next, mod_next = named_modules[i+1]

        if (
            isinstance(mod_curr, (nn.Conv2d, nn.Linear)) and 
            isinstance(mod_next, (nn.BatchNorm2d, nn.BatchNorm1d))
        ):
            modules_to_fuse.append([name_curr, name_next])
            i += 2
            continue
        i += 1
    
    print(f"Modules to fuse (adjacent pairs): {len(modules_to_fuse)}")
    
    # Test with torch.ao.quantization.fuse_modules (modern path)
    try:
        import torch.ao.quantization as ao_quant
        ao_quant.fuse_modules(model, modules_to_fuse, inplace=True)
        print("Fusion (ao_quant) successful.")
    except Exception as e:
        print(f"Fusion (ao_quant) failed: {e}")
        try:
            torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
            print("Fusion (torch.quantization) successful.")
        except Exception as e2:
            print(f"Fusion (torch.quantization) failed: {e2}")

    # Check if they are still there
    new_named_modules = dict(model.named_modules())
    fused_count = 0
    for pair in modules_to_fuse:
        # After fusion, the second module is usually replaced by Identity
        if isinstance(new_named_modules.get(pair[1]), nn.Identity):
            fused_count += 1
    
    print(f"BN modules successfully replaced by Identity: {fused_count}")

if __name__ == "__main__":
    test_folding()
