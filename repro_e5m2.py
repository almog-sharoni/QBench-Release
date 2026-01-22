
import torch
from runspace.src.quantization.quantizer import quantize

def test_e5m2_values():
    print("Testing FP8 E5M2 Quantization...")
    
    # Test cases: [Expected Value, Description]
    # Bias = 15
    # Normal: 1.0 = 1.0 * 2^0. Exp=15 (01111). Mant=0 (00).
    # Smallest Normal: 1.0 * 2^(1-15) = 2^-14 = 6.1035e-5
    # Largest Subnormal: 0.11_2 * 2^-14 = 0.75 * 2^-14 = 4.5776e-5
    # Smallest Subnormal: 0.01_2 * 2^-14 = 0.25 * 2^-14 = 1.5258e-5
    
    test_vals = [
        (1.0, "1.0"),
        (2.0, "2.0"),
        (1.75, "1.75 (1.11_2)"),
        (0.5, "0.5"),
        (6.1035156e-5, "Smallest Normal (2^-14)"),
        (4.5776367e-5, "Largest Subnormal (0.75 * 2^-14)"),
        (1.5258789e-5, "Smallest Subnormal (0.25 * 2^-14)"),
        (57344.0, "Max Value"),
        (-1.0, "-1.0"),
        (0.0, "0.0")
    ]
    
    vals_tensor = torch.tensor([v[0] for v in test_vals], dtype=torch.float32)
    
    q = quantize(vals_tensor, q_type='fp8_e5m2', bias=15)
    
    print(f"{'Original':<25} {'Quantized':<25} {'Error':<20} {'Desc'}")
    print("-" * 80)
    
    for i, (val, desc) in enumerate(test_vals):
        q_val = q[i].item()
        err = abs(val - q_val)
        print(f"{val:<25.8f} {q_val:<25.8f} {err:<20.8e} {desc}")
        
    # Randomized Test
    print("\nRandomized Range Test...")
    rand_tensor = torch.randn(1000) * 10 # Normal distribution
    q_rand = quantize(rand_tensor, q_type='fp8_e5m2')
    mse = (rand_tensor - q_rand).pow(2).mean().item()
    print(f"MSE on Random Normal (mean=0, std=10): {mse}")
    
    # Check for unexpected zeros in normal range
    mask_zeros = (q_rand == 0) & (rand_tensor.abs() > 1e-4)
    if mask_zeros.any():
        print("WARNING: Found zeros where input was significant!")
        print(rand_tensor[mask_zeros][:5])
        
if __name__ == "__main__":
    test_e5m2_values()
