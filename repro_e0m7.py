
import torch
from runspace.src.quantization.quantizer import quantize

def test_e0m7_values():
    print("Testing FP8 E0M7 Quantization...")
    
    # E0M7: 1 sign, 0 exponent, 7 mantissa.
    # Bias = 0? (Implementation uses '0' for exp_bits=0)
    # Effective Exp = 1 - Bias. If Bias=0 -> Exp=1.
    # Value = Mantissa * 2^(1 - 0) = Mantissa * 2.
    # Mantissa is 0.xxxxxxx (7 bits).
    # Smallest > 0: 2^-7.
    # Value: 2^-7 * 2 = 2^-6 = 0.015625?
    # Wait, implementation uses: 
    #   if exp_bits=0: bias=0.
    #   Re-bias: exp32 - (127 - 0) = exp32 - 127.
    #   Subnormal handling: fp8_exp < 1.
    #   Input 1.0 (Exp=0). fp8_exp = 0 - 127 = -127.
    #   Shift = (23 - 7) + (1 - (-127)) = 16 + 128 = 144.
    #   Mant >> 144 -> 0.
    
    # Let's verify what values are representable.
    # E0M7 is basically fixed point in a small range or very subnormal?
    # Usually E0 formats represent numbers in range (-1, 1) or similar.
    # If standard bias is used (e.g. 63 like E7M0?), then range shifts.
    # Current implementation uses Bias=0.
    # This implies range is very small numbers?
    
    test_vals = [
        (1.0, "1.0"),
        (0.5, "0.5"),
        (0.015625, "2^-6"),
        (0.0078125, "2^-7"),
        (2.0, "2.0"), # Should saturate?
        (-0.5, "-0.5"),
        (0.0, "0.0")
    ]
    
    vals_tensor = torch.tensor([v[0] for v in test_vals], dtype=torch.float32)
    
    # Run with default bias (0)
    print("\n--- Bias = 0 (Default) ---")
    try:
        q = quantize(vals_tensor, q_type='fp8_e0m7')
        
        print(f"{'Original':<25} {'Quantized':<25} {'Error':<20} {'Desc'}")
        print("-" * 80)
        
        for i, (val, desc) in enumerate(test_vals):
            q_val = q[i].item()
            err = abs(val - q_val)
            print(f"{val:<25.8f} {q_val:<25.8f} {err:<20.8e} {desc}")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_e0m7_values()
