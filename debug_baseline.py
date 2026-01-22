
import os
import sys
import torch
import torch.nn as nn
from runspace.src.adapters.adapter_factory import create_adapter

def test_baseline_execution():
    formats = ['fp8_e4m3', 'fp8_e5m2', 'fp8_e3m4', 'fp8_e2m5', 'fp8_e1m6', 'fp8_e6m1', 'fp8_e7m0', 'fp4_e2m1', 'fp4_e3m0']
    
    print("Testing baseline formats execution...")
    
    data = torch.randn(2, 3, 224, 224)
    
    for fmt in formats:
        print(f"Testing format: {fmt} ... ", end="")
        try:
            config = {
                'model': {'name': 'resnet18', 'weights': None},
                'adapter': {
                    'type': 'generic',
                    'quantized_ops': ['Conv2d', 'Linear'],
                    'input_quantization': True
                },
                'quantization': {
                    'format': fmt,
                    'weight_mode': 'channel',
                    'enabled': True
                }
            }
            
            adapter = create_adapter(config)
            model = adapter.model
            model.eval()
            
            # Forward pass
            out = model(data)
            print("Success")
            
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_baseline_execution()
