
import torch
from runspace.src.quantization.quantizer import quantize_int8_manual

def test_manual_int8():
    print("Testing Manual INT8 Quantization...")
    
    # Range of values:
    # 0, integers, small fractions, large fractions, negative, clamp limits
    test_vals = [
        0.0, 
        0.5, # Tie?
        0.500001, # Round up
        0.499999, # Round down
        1.0, -1.0,
        126.9, 127.0, 127.1, 128.0, 200.0,
        -127.0, -127.1, -128.0, -200.0,
        # Strange
        1e-5, # Close to 0
        1.5, 2.5, 3.5 # Ties to even in torch.round? Manual uses ties away from zero?
    ]
    
    input_tensor = torch.tensor(test_vals, dtype=torch.float32)
    
    # Compare with reference behavior: round to nearest integer, clamp
    # Note: torch.round uses "round to nearest even" for ties (x.5)
    # Our manual impl uses x + 0.5 floor -> "round half up" (ties to +inf) or similar?
    # Let's see differences.
    
    manual = quantize_int8_manual(input_tensor, rounding="nearest")
    ref = torch.round(input_tensor).clamp(-127, 127)
    
    print(f"{'Input':<15} {'Manual':<15} {'Ref (Torch)':<15} {'Diff':<15}")
    print("-" * 60)
    
    for i in range(len(test_vals)):
        v = test_vals[i]
        m = manual[i].item()
        r = ref[i].item()
        d = abs(m - r)
        note = ""
        if d > 0:
            note = "*" # Flag differences
        print(f"{v:<15.6f} {m:<15.1f} {r:<15.1f} {d:<15.1f} {note}")

    # Check Truncate
    print("\nTesting Truncate...")
    manual_trunc = quantize_int8_manual(input_tensor, rounding="truncate")
    # Reference truncate: floor(abs(x)) * sign(x) ? or just int(x)? torch.trunc
    ref_trunc = torch.trunc(input_tensor).clamp(-127, 127)
    
    diff_trunc = (manual_trunc - ref_trunc).abs().sum().item()
    if diff_trunc == 0:
        print("Truncate MATCH ✅")
    else:
        print(f"Truncate MISMATCH ❌ (Diff: {diff_trunc})")
        
        for i in range(len(test_vals)):
            v = test_vals[i]
            m = manual_trunc[i].item()
            r = ref_trunc[i].item()
            if m != r:
                 print(f"{v:<15.6f} {m:<15.1f} {r:<15.1f}")

if __name__ == "__main__":
    test_manual_int8()
