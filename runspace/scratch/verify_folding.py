import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

def fold_norm(model, mean, std):
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)
    
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            first_conv = module
            print(f"Folding into {name}")
            break
    
    if first_conv is None:
        return
    
    with torch.no_grad():
        std_view = std_t.view(1, 3, 1, 1)
        first_conv.weight.copy_(first_conv.weight / std_view)
        
        correction = (first_conv.weight * mean_t.view(1, 3, 1, 1)).sum(dim=(1, 2, 3))
        if first_conv.bias is None:
            first_conv.bias = nn.Parameter(torch.zeros(first_conv.out_channels))
        first_conv.bias.copy_(first_conv.bias - correction)

def main():
    model_name = "mobilenet_v3_large"
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    
    model = models.mobilenet_v3_large(weights=weights)
    model.eval()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create dummy input
    torch.manual_seed(42)
    x_raw = torch.rand(1, 3, 224, 224)
    
    # Normalized input
    x_norm = (x_raw - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    
    # Run original model on normalized input
    with torch.no_grad():
        y_orig = model(x_norm)
    
    # Run folded model on raw input
    model_folded = models.mobilenet_v3_large(weights=weights)
    model_folded.eval()
    fold_norm(model_folded, mean, std)
    
    with torch.no_grad():
        y_folded = model_folded(x_raw)
    
    diff = (y_orig - y_folded).abs().max().item()
    print(f"Max difference in logits: {diff:.2e}")
    
    # Check weight range
    w_orig = model.features[0][0].weight
    w_folded = model_folded.features[0][0].weight
    print(f"Orig weight range: [{w_orig.min():.4f}, {w_orig.max():.4f}]")
    print(f"Folded weight range: [{w_folded.min():.4f}, {w_folded.max():.4f}]")

if __name__ == "__main__":
    main()
