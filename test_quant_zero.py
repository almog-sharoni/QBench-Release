import torch
import sys
import os

sys.path.append(os.getcwd())
from runspace.src.quantization.quantizer import quantize, get_fp4_e2m1_table

def test_efp_hybrid():
    print("Testing Hybrid EFP (Zero Preserved + NegZero as MaxExt)...")
    
    # 1. Zero Preservation Test
    # Small values should map to 0, not 0.5 (smallest pos)
    print("\n--- Zero Preservation ---")
    vals_zero = torch.tensor([0.0, 0.1, 0.25, 0.5], device='cpu')
    # efp4_e2m1 (E=2, M=1, Bias=1). Smallest pos norm = 1.0 * 2^(1-1) = 1.0? 
    # Wait, e2m1 subnorms? Exp=0. 0.1 * 2^(1-1) = 0.5.
    # So 0.0 -> 0.0. 0.1 -> 0.0? 0.25 -> 0.0 or 0.5?
    
    q_zero = quantize(vals_zero, q_type='efp4_e2m1', validate=False)
    print(f"Input:    {vals_zero.tolist()}")
    print(f"Quantized:{q_zero.tolist()}")
    
    # 2. Extended Max Test
    # efp4_e2m1: Max Std = 6.0. Max Ext (Index 8) -> Exp=4, Mant=0 -> 1.0*2^3 = 8.0.
    print("\n--- Extended Range ---")
    vals_ext = torch.tensor([6.0, 7.0, 8.0, 10.0], device='cpu')
    # 6.0 is Max Std. 8.0 is Max Ext.
    # 7.0 is midpoint.
    # 10.0 should clamp to 8.0.
    
    q_ext = quantize(vals_ext, q_type='efp4_e2m1', validate=False)
    print(f"Input:    {vals_ext.tolist()}")
    print(f"Quantized:{q_ext.tolist()}")
    
    # 3. Negative Integers Test (Standard Negatives)
    print("\n--- Standard Negatives ---")
    vals_neg = torch.tensor([-0.5, -2.0, -6.0], device='cpu')
    q_neg = quantize(vals_neg, q_type='efp4_e2m1', validate=False)
    print(f"Input:    {vals_neg.tolist()}")
    print(f"Quantized:{q_neg.tolist()}")

    # 4. Table Verification
    print("\n--- Table Dump (First 10 and Last 5) ---")
    try:
         # Manually invoke generator for EFP
         from runspace.src.quantization.quantizer import _generate_efp_generic_table
         table = _generate_efp_generic_table(device=torch.device('cpu'), total_bits=4, exp_bits=2, mant_bits=1, bias=1)
         # 16 values
         print(f"Table (16 values): {table.tolist()}")
         
         # Verification
         # Index 0: 0.0
         # Index 8 (1000): Should be 8.0
         if table[0] == 0.0: print("SUCCESS: Index 0 is 0.0")
         else: print(f"FAILURE: Index 0 is {table[0]}")
         
         if table[8] == 8.0: print("SUCCESS: Index 8 is 8.0 (Extended Max)")
         else: print(f"FAILURE: Index 8 is {table[8]}")
         
    except Exception as e:
        print(f"Table gen failed: {e}")

if __name__ == "__main__":
    test_efp_hybrid()
