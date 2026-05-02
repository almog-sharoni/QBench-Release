import os
import sys
import torch
import torch.nn as nn
PROJECT_ROOT = '/data/almog/Projects/QBench-Release'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from runspace.src.adapters.adapter_factory import create_adapter

config = {
    'model': {'name': 'vit_b_16', 'weights': 'DEFAULT'},
    'adapter': {'type': 'generic'},
    'quantization': {'format': 'fp8_e4m3'}
}
adapter = create_adapter(config)
model = adapter.model

# Print first encoder layer's attention
try:
    attn = model.encoder.layers.encoder_layer_0.self_attention
    print(f"Attention type: {type(attn)}")
    print(f"Children: {list(attn.named_children())}")
except Exception as e:
    print(f"Error: {e}")
