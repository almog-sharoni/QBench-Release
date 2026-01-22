
import torch
from runspace.src.ops.quant_base import calculate_scale, quantize_tensor

def check_int8_scaling():
    print("Checking INT8 Scaling Logic...")
    
    # Test case: Value causing potential overflow
    # Max value = 250. 
    # log2(250) approx 7.96.
    # Current logic: floor(7.96) - 7 = 7 - 7 = 0. Scale = 2^0 = 1.
    # Scaled value = 250 / 1 = 250.
    # INT8 clip limit = 127. 
    # Result: 127. Massive error.
    
    vals = [120.0, 130.0, 250.0, 255.0]
    tensor = torch.tensor(vals)
    
    # Calculate scale manually using quant_base
    # Mocking max_val per channel behavior (scalar max for test)
    max_val = tensor.abs().max()
    print(f"Max Value: {max_val.item()}")
    
    scale = calculate_scale(max_val, 'int8', bias=7)
    print(f"Calculated Scale (bias=7): {scale.item()}")
    
    scaled = tensor / scale
    print(f"Scaled Values: {scaled.tolist()}")
    
    quantized = torch.round(scaled).clamp(-127, 127)
    print(f"Quantized (Internal): {quantized.tolist()}")
    
    recovered = quantized * scale
    print(f"Recovered: {recovered.tolist()}")
    
    # Check E0M7 for comparison
    # Bias=0.
    # Logic: floor(log2(250)) - 0 = 7. Scale = 2^7 = 128.
    # Scaled = 250 / 128 = 1.95.
    # E0M7 max = 1.98. fits.
    print("\n--- E0M7 Comparison ---")
    scale_fp8 = calculate_scale(max_val, 'fp8_e0m7', bias=0)
    print(f"E0M7 Scale (bias=0): {scale_fp8.item()}")
    scaled_fp8 = tensor / scale_fp8
    print(f"E0M7 Scaled: {scaled_fp8.tolist()}")
    print(f"Fit check: {scaled_fp8.max().item()} < 1.98? {scaled_fp8.max().item() < 1.984375}")

if __name__ == "__main__":
    check_int8_scaling()
