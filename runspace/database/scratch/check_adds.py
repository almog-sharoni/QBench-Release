import sys, os
sys.path.append('.')
from runspace.src.adapters.adapter_factory import create_adapter
config = {
    'model': {'name': 'vit_b_16', 'weights': None},
    'adapter': {'type': 'generic', 'build_quantized': True}
}
adapter = create_adapter(config)
model = adapter.model
for name, mod in model.named_modules():
    if 'add' in name.lower() or 'quantadd' in type(mod).__name__.lower():
        print(f'{name}: {type(mod).__name__}')
