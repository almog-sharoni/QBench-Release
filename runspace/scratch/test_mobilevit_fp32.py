import timm
import torch

model_name = "mobilevit_s"
try:
    model = timm.create_model(model_name, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"FP32 Success with 224x224! Output shape: {out.shape}")
except Exception as e:
    print(f"FP32 Failed with 224x224: {e}")
