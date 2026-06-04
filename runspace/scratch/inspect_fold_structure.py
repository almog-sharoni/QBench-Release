import os, sys
ROOT='/data/almog/Projects/QBench-Release'
sys.path.insert(0, os.path.join(ROOT,'runspace'))
from runspace.src.adapters.adapter_factory import create_adapter
cfg={'model':{'name':'resnet18','weights':None},'adapter':{'type':'generic','build_quantized':True,'fold_layers':True,'fold_input_norm':True,'quantize_first_layer':True}}
adapter=create_adapter(cfg)
model=adapter.model
for name,module in model.named_modules():
    if 'conv1_1x1' in name or 'bn' in name:
        print(name, type(module).__name__, list(module._modules.keys()) if hasattr(module,'_modules') else None)
