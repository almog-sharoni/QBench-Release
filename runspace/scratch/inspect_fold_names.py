import os, sys
ROOT='/data/almog/Projects/QBench-Release'
sys.path.insert(0, os.path.join(ROOT,'runspace'))
from runspace.src.adapters.adapter_factory import create_adapter
cfg={'model':{'name':'resnet18','weights':None},'adapter':{'type':'generic','build_quantized':True,'fold_layers':True,'fold_input_norm':True,'quantize_first_layer':True}}
adapter=create_adapter(cfg)
model=adapter.model
names = [name for name,module in model.named_modules()]
print('total', len(names))
print('first20', names[:20])
print('bn count', sum(1 for name in names if 'bn' in name.lower()))
print('conv1x1 count', sum(1 for name in names if 'conv1_1x1' in name))
