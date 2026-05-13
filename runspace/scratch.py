import torch
from torchvision.models import resnet18, vit_b_16, mobilenet_v3_large

def count_layers(model):
    num_weights = sum(1 for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
    print(f"{model.__class__.__name__}: Conv2d+Linear weights = {num_weights}")

count_layers(resnet18())
count_layers(vit_b_16())

from transformers import MobileViTForImageClassification
try:
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    count_layers(model)
except Exception as e:
    print(f"MobileViT: {e}")

