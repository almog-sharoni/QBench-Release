
import torch
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.ops.quant_base import quantize_tensor

def test_ufp_equivalence():
    print("=== Testing UFP vs FP Equivalence (Positive Values) ===")
    
    # Test values in [0.001, 100.0]
    x = torch.logspace(-3, 2, 20, device='cuda')
    
    print(f"{'Value':>12} | {'FP8_E4M3':>15} | {'UFP8_E4M4':>15} | {'Diff (%)':>10}")
    print("-" * 60)
    
    for val in x:
        v_tensor = val.unsqueeze(0).float()
        
        # 1. Quantize with signed fp8_e4m3
        q_signed, _ = quantize_tensor(v_tensor, q_type="fp8_e4m3", mode="tensor")
        
        # 2. Quantize with unsigned ufp8_e4m4 (should have more precision)
        q_unsigned, _ = quantize_tensor(v_tensor, q_type="ufp8_e4m4", mode="tensor")
        
        err_signed = (q_signed - v_tensor).abs().item()
        err_unsigned = (q_unsigned - v_tensor).abs().item()
        
        rel_diff = (q_unsigned - q_signed).item() / (val.item() + 1e-9) * 100
        
        print(f"{val.item():12.6f} | {q_signed.item():15.6f} | {q_unsigned.item():15.6f} | {rel_diff:10.2f}%")

    # Check for systematic bias
    q_s_all, _ = quantize_tensor(x, q_type="fp8_e4m3", mode="tensor")
    q_u_all, _ = quantize_tensor(x, q_type="ufp8_e4m4", mode="tensor")
    
    mse_s = torch.mean((q_s_all - x)**2).item()
    mse_u = torch.mean((q_u_all - x)**2).item()
    
    print(f"\nOverall MSE:")
    print(f"  FP8_E4M3:  {mse_s:.8f}")
    print(f"  UFP8_E4M4: {mse_u:.8f}")
    
    if mse_u > mse_s * 1.1:
        print("\n[FAILURE] UFP8_E4M4 is LESS accurate than FP8_E4M3!")
    else:
        print("\n[SUCCESS] UFP8_E4M4 is at least as accurate as FP8_E4M3.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
    else:
        test_ufp_equivalence()
