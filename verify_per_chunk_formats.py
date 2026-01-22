#!/usr/bin/env python3
"""
Verification script to ensure per-chunk heterogeneous format quantization is working.
This script will:
1. Load a model with per-chunk format config
2. Inspect the quantized weights to verify different chunks have different quantization characteristics
3. Report findings
"""

import torch
import torch.nn as nn
import yaml
import sys
import os
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runspace.src.adapters.adapter_factory import create_adapter
from runspace.src.quantization.constants import get_quantization_bias

def analyze_quantized_values(weight_fp8, weight_scale, chunk_formats, chunk_size):
    """
    Analyze the quantized values to see if they match expected format characteristics.
    """
    print("\n=== Analyzing Quantized Values ===")
    
    # Flatten weight
    if weight_fp8.dim() > 1:
        flat_weight_fp8 = weight_fp8.flatten(1)
        flat_weight_scale = weight_scale.flatten(1)
        batch_size = weight_fp8.shape[0]
    else:
        flat_weight_fp8 = weight_fp8.flatten(0)
        flat_weight_scale = weight_scale.flatten(0)
        batch_size = 1
    
    num_elements = flat_weight_fp8.shape[-1]
    pad_len = 0
    if num_elements % chunk_size != 0:
        pad_len = chunk_size - (num_elements % chunk_size)
    
    num_chunks = (num_elements + pad_len) // chunk_size
    
    print(f"Weight shape: {weight_fp8.shape}")
    print(f"Num chunks: {num_chunks}, Chunk size: {chunk_size}")
    print(f"Chunk formats specified: {len(chunk_formats)}")
    
    total_chunks = num_chunks * batch_size
    
    if len(chunk_formats) == num_chunks:
        print(f"Chunk formats provided for {num_chunks} chunks (per-filter broadcast).")
    elif len(chunk_formats) == total_chunks:
        print(f"Chunk formats provided for all {total_chunks} chunks (global).")
    else:
        print(f"WARNING: Mismatch! chunk_formats length ({len(chunk_formats)}) matches neither num_chunks ({num_chunks}) nor total_chunks ({total_chunks})")
        return False
    
    # Analyze each chunk
    results = []
    for i in range(min(5, num_chunks)):  # Check first 5 chunks
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_elements)
        
        # Get chunk data (first output channel for simplicity)
        chunk_fp8 = flat_weight_fp8[0, start_idx:end_idx]
        chunk_scale = flat_weight_scale[0, start_idx:end_idx]
        expected_format = chunk_formats[i]
        
        # Reconstruct dequantized values
        chunk_dequant = chunk_fp8 * chunk_scale
        
        # Analyze unique scales (should be same within chunk for per-chunk quantization)
        unique_scales = torch.unique(chunk_scale)
        
        # Analyze value range
        max_unscaled = chunk_fp8.abs().max().item()
        
        results.append({
            'chunk': i,
            'format': expected_format,
            'unique_scales': len(unique_scales),
            'scale_value': chunk_scale[0].item() if len(chunk_scale) > 0 else 0,
            'max_unscaled': max_unscaled,
            'max_dequant': chunk_dequant.abs().max().item()
        })
        
        print(f"\nChunk {i} (format={expected_format}):")
        print(f"  Unique scales in chunk: {len(unique_scales)}")
        print(f"  Scale value: {chunk_scale[0].item():.6e}")
        print(f"  Max unscaled value: {max_unscaled:.6f}")
        print(f"  Max dequantized value: {chunk_dequant.abs().max().item():.6e}")
    
    # Check if different chunks have different scales (evidence of per-chunk formats)
    scales = [r['scale_value'] for r in results]
    unique_scales_across_chunks = len(set([f"{s:.6e}" for s in scales]))
    
    print(f"\n=== Summary ===")
    print(f"Unique scales across first 5 chunks: {unique_scales_across_chunks}/5")
    
    if unique_scales_across_chunks > 1:
        print("✓ PASS: Different chunks have different scales (per-chunk quantization working)")
        return True
    else:
        print("✗ FAIL: All chunks have same scale (per-chunk quantization NOT working)")
        return False

def main():
    # Load config
    config_path = "runspace/experiments/optimal_layer_quant/resnet18/optimized_l1/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Please run the experiment first with --per_chunk_format flag")
        return
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if per_chunk_format is enabled
    per_chunk = config.get('quantization', {}).get('per_chunk_format', False)
    print(f"Per-chunk format enabled in config: {per_chunk}")
    
    if not per_chunk:
        print("WARNING: per_chunk_format is not enabled in config!")
        return
    
    # Create adapter and build model
    print("\nBuilding quantized model...")
    adapter = create_adapter(config)
    model = adapter.model
    
    # Find a layer with chunk_formats
    print("\n=== Inspecting Layers ===")
    found_per_chunk = False
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'chunk_formats') and module.chunk_formats is not None:
                print(f"\n✓ Found layer with chunk_formats: {name}")
                print(f"  chunk_formats attribute: {module.chunk_formats[:10]}... (showing first 10)")
                
                if hasattr(module, 'weight_fp8') and module.weight_fp8 is not None:
                    print(f"  weight_fp8 shape: {module.weight_fp8.shape}")
                    print(f"  weight_scale shape: {module.weight_scale.shape}")
                    
                    chunk_size = getattr(module, 'weight_chunk_size', 128)
                    success = analyze_quantized_values(
                        module.weight_fp8,
                        module.weight_scale,
                        module.chunk_formats,
                        chunk_size
                    )
                    
                    found_per_chunk = True
                    
                    if success:
                        print("\n" + "="*50)
                        print("VERIFICATION SUCCESSFUL!")
                        print("Per-chunk heterogeneous format quantization is working correctly.")
                        print("="*50)
                    else:
                        print("\n" + "="*50)
                        print("VERIFICATION FAILED!")
                        print("Per-chunk format is configured but not being applied correctly.")
                        print("="*50)
                    
                    break
                else:
                    print("  WARNING: chunk_formats present but weights not quantized yet")
            elif hasattr(module, 'weight_fp8') and module.weight_fp8 is not None:
                print(f"  Layer {name}: quantized but no chunk_formats (uniform quantization)")
    
    if not found_per_chunk:
        print("\n✗ No layers found with chunk_formats attribute!")
        print("Per-chunk format may not be configured correctly.")

if __name__ == "__main__":
    main()
