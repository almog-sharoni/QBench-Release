import torch
import torch.nn as nn
import sys
import os

# Set PYTHONPATH to include the project root
sys.path.append('/data/almog/Projects/QBench-Release/')

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
            self.ln = nn.LayerNorm(normalized_shape)
        def forward(self, x):
            return self.ln(x)
    
    model = Model().cuda()
    
    # 2. Setup Dynamic Quantizer
    candidates = ['fp8_e4m3', 'fp8_e1m6', 'fp8_e7m0', 'fp8_e5m2', 'fp8_e2m5']
    observations = []

    def observe_chunks(*, layer_name, candidates, best_indices, **_kwargs):
        observations.append(
            {
                'stage': layer_name,
                'candidates': tuple(candidates),
                'best_indices': best_indices.detach().cpu(),
            }
        )

    dq = DynamicInputQuantizer(
        model, 
        chunk_size=128, 
        candidate_formats=candidates,
        chunk_observer=observe_chunks,
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
    chunk1 = input_tensor[0, 128:256].unsqueeze(0)  # [1, 128]
    for i, fmt in enumerate(candidates):
        from runspace.src.ops.quant_base import quantize_tensor
        q_tensor, _ = quantize_tensor(chunk1, q_type=fmt, mode='chunk', chunk_size=128)
        mse = torch.mean((chunk1 - q_tensor)**2).item()
        print(f"Format: {fmt:10} | MSE: {mse:.4e}")

    # 6. Check selected formats from the hardware transport observer.
    for observation in observations:
        print(
            f"Stage {observation['stage']}: indices="
            f"{observation['best_indices'].tolist()} "
            f"candidates={observation['candidates']}"
        )

    all_indices = torch.cat(
        [observation['best_indices'].reshape(-1) for observation in observations]
    )
    unique_indices = torch.unique(all_indices)
    print(f"Unique indices chosen: {unique_indices.tolist()}")
    
    if len(unique_indices) > 1:
        print("SUCCESS: DynamicInputQuantizer chose DIFFERENT formats for different chunks!")
    else:
        print("WARNING: Only one format was chosen. This might happen if one format is strictly better for all tested chunks, or if the distribution isn't diverse enough.")

    stats = dq.get_final_stats()
    print(
        "Hardware transport: "
        f"packets={stats['packet_count']} decode_reads={stats['decode_reads']} "
        f"stages={stats['stage_count']}"
    )
    assert stats['packet_count'] > 0
    assert stats['decode_reads'] > 0
    dq.cleanup()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run test.")
    else:
        test_dynamic_chunks()
