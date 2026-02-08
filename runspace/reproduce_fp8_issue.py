
import torch
import sys
import os

# Set up path to import runspace modules
sys.path.append('/home/almog/Projects/QBench-Release/runspace')

from src.ops.quant_base import quantize_tensor, calculate_scale

def test_quantization_difference():
    print("Testing FP8 E1M6 vs E0M7 Difference")
    # Value that reveals precision diff
    # E0M7 step is 1/64 = 0.015625
    # E1M6 step is 1/32 = 0.03125 (in subnormal range [0, 2))
    
    # Choose a value that E0M7 can represent but E1M6 cannot
    # e.g., 1.0 + 1/64 = 1.015625
    val = 1.0 + 1.0/64.0
    input_tensor = torch.tensor([val])
    
    print(f"\nInput Value: {val}")
    
    # 1. FP8 E1M6
    print("\n--- FP8 E1M6 ---")
    
    # Calculate scale (auto logic)
    s_e1m6 = calculate_scale(input_tensor, 'fp8_e1m6')
    print(f"Calculated Scale: {s_e1m6.item()}")
    
    # Quantize
    q_e1m6, _ = quantize_tensor(input_tensor, 'fp8_e1m6')
    print(f"Quantized Value: {q_e1m6.item()}")
    print(f"Error: { abs(q_e1m6.item() - val) }")
    
    # 2. FP8 E0M7
    print("\n--- FP8 E0M7 ---")
    
    # Calculate scale
    s_e0m7 = calculate_scale(input_tensor, 'fp8_e0m7')
    print(f"Calculated Scale: {s_e0m7.item()}")
    
    # Quantize
    q_e0m7, _ = quantize_tensor(input_tensor, 'fp8_e0m7')
    print(f"Quantized Value: {q_e0m7.item()}")
    print(f"Error: { abs(q_e0m7.item() - val) }")

    # Analysis
    print("\nAnalysis:")
    if abs(q_e0m7.item() - val) < 1e-9 and abs(q_e1m6.item() - val) > 1e-9:
        print("CONFIRMED: E0M7 represented the value perfectly, E1M6 had error.")
    elif abs(q_e0m7.item() - val) < 1e-9 and abs(q_e1m6.item() - val) < 1e-9:
        print("SUCCESS: Both formats represented the value perfectly.")
    else:
        print("Inconclusive.")

if __name__ == "__main__":
    test_quantization_difference()
