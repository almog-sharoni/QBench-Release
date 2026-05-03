import sys, os
sys.path.append('.')
from runspace.src.adapters.adapter_factory import create_adapter
config = {
    'model': {'name': 'vit_b_16', 'weights': None},
    'adapter': {'type': 'generic', 'build_quantized': True}
}
adapter = create_adapter(config)
model = adapter.model
print(f'Total modules: {len(list(model.named_modules()))}')
for name, mod in list(model.named_modules())[:20]:
    print(f'{name}: {type(mod).__name__}')
