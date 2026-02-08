
import yaml
import torchvision
from torchvision.models import get_model_weights
import re
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), "models.yaml")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "models_best_acc1.yaml")

def get_family(name):
    # Special handlings
    if name.startswith("wide_resnet"): return "wide_resnet"
    if name.startswith("resnext"): return "resnext"
    if name.startswith("regnet"):
        # Group regnet_x and regnet_y separately or together? 
        # User said "from each type". RegNetX and RegNetY are usually treated as variants.
        # But let's check if they are distinct enough. RegNetY has SE.
        # I'll group them as "regnet" so we pick the absolute best RegNet.
        return "regnet" 
    if name.startswith("vit"): return "vit"
    if name.startswith("swin"): return "swin"
    if name.startswith("mobilenet"): return "mobilenet"
    if name.startswith("shufflenet"): return "shufflenet"
    if name.startswith("squeezenet"): return "squeezenet"
    if name.startswith("efficientnet"): return "efficientnet"
    if name.startswith("densenet"): return "densenet"
    if name.startswith("vgg"): return "vgg"
    if name.startswith("mnasnet"): return "mnasnet"
    
    # Fallback to simple prefix matching (everything before first underscore if present, else full name)
    # Actually, names like 'convnext_base' should be 'convnext'
    match = re.match(r"^([a-z]+)", name)
    if match:
        return match.group(1)
    return name

def get_acc1(model_name):
    try:
        weights_enum = get_model_weights(model_name)
        default_weights = weights_enum.DEFAULT
        if hasattr(default_weights, 'meta'):
            meta = default_weights.meta
            # Try to find acc@1
            # Structure observed: _metrics -> ImageNet-1K -> acc@1
            if '_metrics' in meta:
                metrics = meta['_metrics']
                if 'ImageNet-1K' in metrics:
                    return metrics['ImageNet-1K'].get('acc@1', 0.0)
            
            # Fallback for older/different structures
            if 'acc@1' in meta: return float(meta['acc@1'])
            if 'categories' in meta and isinstance(meta['categories'], dict) and 'acc@1' in meta['categories']:
                return float(meta['categories']['acc@1'])
            
            print(f"Warning: No acc@1 found for {model_name}")
            return 0.0
    except Exception as e:
        print(f"Error fetching weights for {model_name}: {e}")
        return 0.0
    return 0.0

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        models = yaml.safe_load(f)
    
    if not models:
        print("No models found.")
        return

    print(f"Found {len(models)} models.")
    
    best_models = {} # family -> (model_entry, acc1)
    
    for model_entry in models:
        name = model_entry.get('name')
        if not name: continue
        
        family = get_family(name)
        acc1 = get_acc1(name)
        
        print(f"Model: {name}, Family: {family}, Acc@1: {acc1}")
        
        if family not in best_models:
            best_models[family] = (model_entry, acc1)
        else:
            current_best_acc = best_models[family][1]
            if acc1 > current_best_acc:
                best_models[family] = (model_entry, acc1)
            elif acc1 == current_best_acc:
                # Tie breaking? Maybe prefer newer/larger?
                # Just keep existing or replace?
                pass
    
    # Collect results
    final_list = [entry for entry, acc in best_models.values()]
    
    # Sort by name for cleanliness
    final_list.sort(key=lambda x: x['name'])
    
    print(f"Filtered down to {len(final_list)} models.")
    
    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(final_list, f, sort_keys=False)
    
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
