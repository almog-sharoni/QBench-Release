import torch
import torch.nn as nn
import sys
import os

# Set PYTHONPATH to include the project root
sys.path.append('/data/almog/Projects/QBench-Release/')

from runspace.src.ops.quant_ln import QuantLayerNorm
from runspace.src.quantization.dynamic_input_quantizer import DynamicInputQuantizer

def test_dynamic_chunks():
    print("=== Validating Dynamic Input Chunking ===")
    
    # 1. Setup Layer and Dummy Model
    # Use a small normalized_shape but enough chunks. 
    # chunk_size is 128 by default in DynamicInputQuantizer.
    normalized_shape = 512 
    # Create a dummy model to hold the layer
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = QuantLayerNorm(normalized_shape, q_type="fp8_e4m3")
            self.ln.capture_activations = True
        def forward(self, x):
            return self.ln(x)
    
    model = Model().cuda()
    
    # 2. Setup Dynamic Quantizer
    candidates = ['fp8_e4m3', 'fp8_e1m6', 'fp8_e7m0', 'fp8_e5m2', 'fp8_e2m5']
    dq = DynamicInputQuantizer(
        model, 
        chunk_size=128, 
        candidate_formats=candidates
    )
    dq.register_hooks()
    
    # 3. Create Heterogeneous Input
    input_tensor = torch.randn(4, normalized_shape, device='cuda')
    
    # Force different properties in different chunks
    with torch.no_grad():
        # Chunk 0: standard normal
        
        # Chunk 1: Uniform [0, 1]
        input_tensor[0, 128:256] = torch.rand(128, device='cuda')
        
        # Chunk 2: Wide range log-normal
        input_tensor[0, 256:384] = torch.exp(torch.randn(128, device='cuda') * 2.0)
        
        # Chunk 3: Sparse (mostly zeros, one large value)
        input_tensor[0, 384:512] = 0.0
        input_tensor[0, 384] = 100.0
        input_tensor[0, 385:400] = 0.1
    
    # 4. Run Forward Pass
    print("Running forward pass...")
    output = model(input_tensor)
    
    # 5. Debugging: Check MSE manually for Chunk 1
    print("\n--- Manual MSE check for Chunk 1 (Wide Distribution) ---")
    chunk1 = input_tensor[0, 128:256].unsqueeze(0).unsqueeze(0) # [1, 1, 128]
    for i, fmt in enumerate(candidates):
        from runspace.src.ops.quant_base import quantize_tensor
        q_tensor, _ = quantize_tensor(chunk1, q_type=fmt, mode='chunk', chunk_size=128)
        mse = torch.mean((chunk1 - q_tensor)**2).item()
        print(f"Format: {fmt:10} | MSE: {mse:.4e}")

    # 6. Check selected formats
    best_indices = model.ln.input_chunk_format_indices
    best_candidates = model.ln.input_chunk_candidates
    
    print(f"Selected format indices (TENSOR): {best_indices}")
    print(f"Candidates: {best_candidates}")
    
    unique_indices = torch.unique(best_indices)
    print(f"Unique indices chosen: {unique_indices.tolist()}")
    
    if len(unique_indices) > 1:
        print("SUCCESS: DynamicInputQuantizer chose DIFFERENT formats for different chunks!")
    else:
        print("WARNING: Only one format was chosen. This might happen if one format is strictly better for all tested chunks, or if the distribution isn't diverse enough.")

    # 6. Check internal quantization stages
    print(f"\nLayer input_quantization state: {model.ln.input_quantization}")
    
    unscaled_list = getattr(model.ln, 'last_quant_inputs_unscaled', [])
    print(f"Captured internal unscaled stages: {len(unscaled_list)}")
    for i, stage in enumerate(unscaled_list):
        if stage is not None:
            print(f"  Stage {i}: Shape {tuple(stage.shape)}, Min {stage.min():.4f}, Max {stage.max():.4f}")
        else:
            print(f"  Stage {i}: None")
    
    if len(unscaled_list) > 0 and unscaled_list[0] is not None:
        print("SUCCESS: Internal quantization stages were captured!")
    else:
        print("WARNING: Internal quantization stages are missing or empty.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run test.")
    else:
        test_dynamic_chunks()
