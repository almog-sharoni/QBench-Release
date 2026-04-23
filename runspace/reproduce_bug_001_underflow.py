"""
Reproduces BUG-001: Quantizer does not clamp underflow values to zero

This script demonstrates that quantize_fp_generic preserves values smaller than
the minimum representable subnormal of the target format, instead of clamping
them to zero. These values then fail the FP8 compliance check.

Minimum subnormal values:
  - FP8 E4M3: 2^(-9) = 0.001953125
  - FP8 E5M2: 2^(-16) = 0.0000153
  - FP8 E1M6: 2^(-7) = 0.0078125
"""

import torch
import sys
import os

# Set up path to import runspace modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantization.quantizer import quantize_fp_generic
from src.eval.metrics import check_fp8_compliance

def test_bug_001_underflow_clamping():
    print("=" * 80)
    print("BUG-001: Quantizer does not clamp underflow values to zero")
    print("=" * 80)
    
    # Test cases: (name, exp_bits, mant_bits, bias, problematic_value, min_subnormal)
    test_cases = [
        ("FP8 E4M3", 4, 3, 7, 0.00048828125, 2**(-9)),  # 2^(-11) - below min subnormal
        ("FP8 E5M2", 5, 2, 15, 3.8147e-6, 2**(-16)),    # 2^(-18) - below min subnormal
        ("FP8 E1M6", 1, 6, 0, 7.46e-22, 2**(-7)),       # 2^(-70) - below min subnormal
    ]
    
    for name, exp_bits, mant_bits, bias, test_value, min_subnormal in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Testing {name} (exp_bits={exp_bits}, mant_bits={mant_bits}, bias={bias})")
        print(f"{'Minimum subnormal:':<30} {min_subnormal:e}")
        print(f"{'Test value:':<30} {test_value:e}")
        print(f"{'Below threshold:':<30} {test_value < min_subnormal}")
        
        # Create test tensor
        input_tensor = torch.tensor([test_value, -test_value, 0.0, 1.0, min_subnormal])
        print(f"\nInput tensor: {input_tensor}")
        
        # Quantize using quantize_fp_generic
        quantized = quantize_fp_generic(
            input_tensor, 
            exp_bits=exp_bits, 
            mant_bits=mant_bits,
            rounding="nearest"
        )
        print(f"Quantized tensor: {quantized}")
        
        # Check what happened to underflow values
        print("\nValue Analysis:")
        for i, (orig, quant) in enumerate(zip(input_tensor, quantized)):
            orig_val = orig.item()
            quant_val = quant.item()
            
            if orig_val != 0 and abs(orig_val) < min_subnormal:
                status = "❌ NOT CLAMPED" if quant_val != 0 else "✅ CORRECTLY CLAMPED"
                print(f"  [{i}] {orig_val:e} → {quant_val:e}  {status}")
            elif orig_val == 0:
                status = "✅ CORRECT" if quant_val == 0 else "❌ WRONG"
                print(f"  [{i}] {orig_val} → {quant_val}  {status}")
            else:
                print(f"  [{i}] {orig_val:e} → {quant_val:e}  (normal magnitude)")
        
        # Run compliance check - should fail for underflow values
        print("\nCompliance Check:")
        
        # Table source of truth: get_format_table returns None for simulated fp
        # formats (they must use mantissa-precision check instead). int8/int4
        # still return finite tables.
        from src.quantization.quantizer import get_format_table
        if name == "FP8 E4M3":
            q_type = "fp8_e4m3"
        elif name == "FP8 E5M2":
            q_type = "fp8_e5m2"
        else:
            q_type = "fp8_e1m6"
        valid_values = get_format_table(q_type, quantized.device)
        
        if valid_values is not None:
            passed, invalid_count, examples = check_fp8_compliance(quantized, valid_values)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  Result: {status}")
            if not passed:
                print(f"  Invalid values found: {invalid_count}")
                print(f"  Examples: {examples}")
        else:
            print(f"  Skipped (no pre-built table for {q_type} - see BUG-002)")
    
    print(f"\n{'=' * 80}")
    print("Summary:")
    print("  - Underflow values below minimum subnormal should be clamped to zero")
    print("  - Currently, quantize_fp_generic preserves them (with reduced precision)")
    print("  - This causes compliance checks to fail on valid quantizations")
    print("  - Fix: Add explicit subnormal clamping after mantissa truncation")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    test_bug_001_underflow_clamping()
