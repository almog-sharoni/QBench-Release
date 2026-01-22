
import torch
from runspace.src.quantization.quantizer import quantize_fp8_e5m2, quantize_fp8_e4m3, quantize_fp8_generic

def compare_implementations():
    print("Comparing FP8 Specialized vs Generic implementations...")
    
    # Test Data: Range covering subnormal, normal, and saturation
    test_tensor = torch.tensor([
        0.0, 
        1e-5, # Subnormal E5M2
        1.0, 
        2.0, 
        57344.0, # Max Normal E5M2
        60000.0, # Overflow E5M2
        1e5,    # Large Overflow
        float('inf'), # Inf
        448.0, # Max E4M3
        500.0 # Overflow E4M3
    ], dtype=torch.float32)
    
    print("\n--- E5M2 Comparison ---")
    specialized = quantize_fp8_e5m2(test_tensor)
    # E5M2 specialized clamps to exp=30, mant=3
    generic = quantize_fp8_generic(test_tensor, exp_bits=5, mant_bits=2, clip_max_exp=30, clip_max_mant=3)
    
    diff = (specialized - generic).abs()
    if diff.max() == 0:
        print("E5M2: MATCH ✅")
    else:
        print("E5M2: MISMATCH ❌")
        for i in range(len(test_tensor)):
            v = test_tensor[i].item()
            s = specialized[i].item()
            g = generic[i].item()
            d = diff[i].item()
            if d > 0:
                print(f"Val: {v:<10} Spec: {s:<10} Gen: {g:<10} Diff: {d}")

    print("\n--- E4M3 Comparison ---")
    specialized_e4 = quantize_fp8_e4m3(test_tensor)
    # E4M3 specialized clamps to exp=15, mant=6
    generic_e4 = quantize_fp8_generic(test_tensor, exp_bits=4, mant_bits=3, clip_max_exp=15, clip_max_mant=6)
    
    diff_e4 = (specialized_e4 - generic_e4).abs()
    if diff_e4.max() == 0:
        print("E4M3: MATCH ✅")
    else:
        print("E4M3: MISMATCH ❌")
        for i in range(len(test_tensor)):
            v = test_tensor[i].item()
            s = specialized_e4[i].item()
            g = generic_e4[i].item()
            d = diff_e4[i].item()
            if d > 0:
                print(f"Val: {v:<10} Spec: {s:<10} Gen: {g:<10} Diff: {d}")

if __name__ == "__main__":
    compare_implementations()
