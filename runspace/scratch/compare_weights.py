import torch
import torch.nn as nn
from runspace.src.adapters.generic_adapter import GenericAdapter

def compare_resnet18_weights():
    # Use standard ImageNet constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    print("--- Comparing ResNet18 Weights With and Without Input Normalization Folding ---")
    
    # 1. Without folding
    adapter_no_fold = GenericAdapter(
        model_name='resnet18',
        model_source='torchvision',
        weights='IMAGENET1K_V1',
        fold_layers=False,
        fold_input_norm=False,
        build_quantized=False
    )
    model_no_fold = adapter_no_fold.model
    conv1_no_fold = model_no_fold.conv1
    
    # 2. With folding
    adapter_fold = GenericAdapter(
        model_name='resnet18',
        model_source='torchvision',
        weights='IMAGENET1K_V1',
        fold_layers=False,
        fold_input_norm=True,
        input_mean=mean,
        input_std=std,
        build_quantized=False
    )
    model_fold = adapter_fold.model
    conv1_fold = model_fold.conv1

    print("\n[Layer: conv1]")
    print(f"No Folding Bias exists: {conv1_no_fold.bias is not None}")
    print(f"Folding Bias exists: {conv1_fold.bias is not None}")
    
    # Print first 5 weights from the first output channel, first input channel
    print("\nFirst 5 weights of output channel 0, input channel 0 (R):")
    print(f"  Original: {conv1_no_fold.weight[0, 0, 0, :5].detach().tolist()}")
    print(f"  Folded:   {conv1_fold.weight[0, 0, 0, :5].detach().tolist()}")
    
    # Expected: Folded = Original / std[0] = Original / 0.229
    ratio_w = conv1_fold.weight[0, 0, 0, 0] / conv1_no_fold.weight[0, 0, 0, 0]
    print(f"  Weight Ratio (Folded/Original): {ratio_w.item():.4f} (Expected: 1/0.229 = {1/0.229:.4f})")

    # Print bias
    if conv1_fold.bias is not None:
        print(f"\nBias (first 5 output channels):")
        if conv1_no_fold.bias is not None:
            print(f"  Original: {conv1_no_fold.bias[:5].detach().tolist()}")
        else:
            print(f"  Original: [0.0, 0.0, 0.0, 0.0, 0.0] (Implicit)")
        print(f"  Folded:   {conv1_fold.bias[:5].detach().tolist()}")

    # Manual check of bias folding formula
    # b_new = b_old - sum(W_new * mean)
    with torch.no_grad():
        w_new = conv1_fold.weight
        m_t = torch.tensor(mean).view(1, 3, 1, 1).to(w_new.device)
        correction = (w_new * m_t).sum(dim=(1, 2, 3))
        
        b_old = conv1_no_fold.bias if conv1_no_fold.bias is not None else torch.zeros_like(correction)
        expected_bias = b_old - correction
        
        print(f"\nManual Bias Check (Channel 0):")
        print(f"  Original Bias:      {b_old[0].item():.6f}")
        print(f"  Correction:         {correction[0].item():.6f}")
        print(f"  Actual Folded Bias: {conv1_fold.bias[0].item():.6f}")
        print(f"  Expected Bias:      {expected_bias[0].item():.6f}")

if __name__ == "__main__":
    compare_resnet18_weights()
