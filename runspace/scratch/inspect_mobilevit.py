import torch
import torchvision.models as models

model = models.mobilevit_s(weights=None)
print(model)
