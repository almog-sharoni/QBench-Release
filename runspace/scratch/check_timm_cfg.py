import timm
import torch

model_name = "mobilevit_s"
model = timm.create_model(model_name, pretrained=False)
cfg = model.default_cfg
print(f"Model: {model_name}")
print(f"Input size: {cfg.get('input_size')}")
print(f"Crop pct: {cfg.get('crop_pct')}")
print(f"Interpolation: {cfg.get('interpolation')}")
