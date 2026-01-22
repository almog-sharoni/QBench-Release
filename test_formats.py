
import torch
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.quantization.quantizer import quantize

def test_formats():
    formats = ['fp8_e4m3', 'fp8_e5m2', 'fp8_e3m4', 'fp8_e2m5', 'fp8_e1m6', 'fp8_e6m1', 'fp8_e7m0', 'fp4_e2m1', 'fp4_e3m0', 'int8', 'int4']
    
    tensor = torch.randn(10, 10)
    
    print("Testing quantization formats...")
    for fmt in formats:
        try:
            print(f"Testing {fmt}...", end="")
            q = quantize(tensor, q_type=fmt)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_formats()
